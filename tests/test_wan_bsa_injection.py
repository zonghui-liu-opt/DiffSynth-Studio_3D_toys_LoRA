import json

import pytest
import torch
import torch.nn as nn

from diffsynth.models.wan_video_bsa import (
    BSAContext,
    WanBSAConfig,
    WanBSASelfAttention,
    bsa_config_manifest,
    bsa_sparsity_for_step,
    collect_wan_bsa_runtime_info,
    inject_wan_bsa,
    load_wan_bsa_adapter,
    save_wan_bsa_adapter,
    validate_bsa_manifest,
    wan_bsa_gate_state_dict,
    write_student_model_info,
)
from diffsynth.models.wan_video_dit import DiTBlock, precompute_freqs_cis_3d
from diffsynth.pipelines.wan_video import model_fn_wan_video


class TinyWan(nn.Module):
    def __init__(self, layers=3, dim=24, heads=3):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList(
            [DiTBlock(False, dim, heads, dim * 2) for _ in range(layers)]
        )


def _freqs(grid, head_dim):
    f, h, w = grid
    frequencies = precompute_freqs_cis_3d(head_dim, end=max(grid))
    return torch.cat(
        [
            frequencies[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            frequencies[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            frequencies[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)


def test_injection_reuses_dense_modules_and_preserves_state_paths():
    model = TinyWan()
    original = [block.self_attn for block in model.blocks]
    original_q = [module.q for module in original]
    summary = inject_wan_bsa(model, WanBSAConfig(backend="eager_math"), expected_layers=3)

    assert summary["injected_bsa_modules"] == 3
    assert summary["cross_attention_bsa_modules"] == 0
    assert summary["gate_granularity"] == "block"
    assert all(isinstance(block.self_attn, WanBSASelfAttention) for block in model.blocks)
    assert all(block.self_attn.q is q for block, q in zip(model.blocks, original_q))
    keys = set(model.state_dict())
    assert "blocks.0.self_attn.q.weight" in keys
    assert "blocks.0.self_attn.bsa_gate_down.weight" in keys
    assert "blocks.0.self_attn.bsa_gate_up.weight" in keys
    assert not any("cross_attn.bsa_" in key for key in keys)
    with pytest.raises(RuntimeError, match="Repeated BSA injection"):
        inject_wan_bsa(model, WanBSAConfig(backend="eager_math"), expected_layers=3)


def test_zero_gate_dense_context_matches_preinjection_output_exactly():
    torch.manual_seed(21)
    model = TinyWan(layers=1).double()
    dense = model.blocks[0].self_attn
    grid = (2, 3, 4)
    x = torch.randn(1, 24, 24, dtype=torch.float64)
    freqs = _freqs(grid, dense.head_dim)
    expected = dense(x, freqs)
    inject_wan_bsa(model, WanBSAConfig(backend="eager_math"), expected_layers=1)
    context = BSAContext.from_config(
        WanBSAConfig(backend="eager_math"), sparsity=0.0
    ).with_grid(grid)
    actual = model.blocks[0].self_attn(x, freqs, context)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_forward_records_consistent_block_runtime_shapes():
    torch.manual_seed(22)
    model = TinyWan(layers=2).double()
    config = WanBSAConfig(backend="eager_math", query_block_chunk=1)
    inject_wan_bsa(model, config, expected_layers=2)
    grid = (2, 3, 4)
    context = BSAContext.from_config(config, sparsity=0.5).with_grid(grid)
    x = torch.randn(1, 24, 24, dtype=torch.float64)
    freqs = _freqs(grid, 8)
    for block in model.blocks:
        x = block.self_attn(x, freqs, context)

    runtime = collect_wan_bsa_runtime_info(model)
    assert runtime["runtime_grid"] == grid
    assert runtime["num_blocks"] == 1
    assert runtime["gate_input_shape"] == (1, 1, 24)
    assert runtime["gate_output_shape"] == (1, 1, 24)
    assert runtime["materialized_token_gate_tensor"] is False


def test_gate_checkpoint_roundtrip_and_manifest_granularity(tmp_path):
    torch.manual_seed(23)
    config = WanBSAConfig(backend="eager_math")
    model = TinyWan(layers=2)
    inject_wan_bsa(model, config, expected_layers=2)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.ndim == 2:
                parameter.add_(0.1)
    expected = {key: value.clone() for key, value in wan_bsa_gate_state_dict(model).items()}
    adapter_path = tmp_path / "bsa_adapter.safetensors"
    save_wan_bsa_adapter(model, str(adapter_path))

    restored = TinyWan(layers=2)
    inject_wan_bsa(restored, config, expected_layers=2)
    load_wan_bsa_adapter(restored, str(adapter_path))
    for key, value in wan_bsa_gate_state_dict(restored).items():
        torch.testing.assert_close(value, expected[key])

    manifest = bsa_config_manifest(config, completed_optimizer_steps=10)
    validate_bsa_manifest(manifest, config)
    invalid = dict(manifest)
    invalid.pop("gate_granularity")
    with pytest.raises(ValueError, match="missing required"):
        validate_bsa_manifest(invalid, config)
    invalid = dict(manifest, gate_granularity="token")
    with pytest.raises(ValueError, match="block-gate"):
        validate_bsa_manifest(invalid, config)


def test_student_info_atomic_json_and_schedule(tmp_path):
    model = TinyWan(layers=1)
    summary = inject_wan_bsa(model, WanBSAConfig(backend="eager_math"), expected_layers=1)
    path = tmp_path / "student_model_info.json"
    write_student_model_info(str(path), summary, {"runtime_grid": [41, 15, 26]})
    payload = json.loads(path.read_text())
    assert payload["gate_granularity"] == "block"
    assert payload["runtime"]["runtime_grid"] == [41, 15, 26]
    assert not (tmp_path / "student_model_info.json.tmp").exists()

    assert bsa_sparsity_for_step(0, 1000) == 0.0
    assert bsa_sparsity_for_step(50, 1000) == 0.1
    assert bsa_sparsity_for_step(399, 1000) == 0.7
    assert bsa_sparsity_for_step(400, 1000) == 0.8


def test_config_rejects_wrong_capacity_and_gate_granularity():
    with pytest.raises(ValueError, match="capacity 72"):
        WanBSAConfig(block_size=(2, 2, 2))
    with pytest.raises(ValueError, match="block gate"):
        WanBSAConfig(gate_granularity="token")


def test_checkpoint_on_off_sparse_output_and_gradients_match():
    torch.manual_seed(24)
    config = WanBSAConfig(backend="eager_math", query_block_chunk=1)
    model = TinyWan(layers=1).double()
    inject_wan_bsa(model, config, expected_layers=1)
    attention = model.blocks[0].self_attn
    grid = (2, 3, 4)
    context = BSAContext.from_config(config, sparsity=0.5).with_grid(grid)
    freqs = _freqs(grid, attention.head_dim)
    reference_input = torch.randn(1, 24, 24, dtype=torch.float64, requires_grad=True)
    checkpoint_input = reference_input.detach().clone().requires_grad_(True)

    reference = attention(reference_input, freqs, context)
    reference_grad = torch.autograd.grad(reference.square().mean(), reference_input)[0]
    checkpointed = torch.utils.checkpoint.checkpoint(
        lambda value: attention(value, freqs, context),
        checkpoint_input,
        use_reentrant=False,
    )
    checkpoint_grad = torch.autograd.grad(checkpointed.square().mean(), checkpoint_input)[0]

    torch.testing.assert_close(checkpointed, reference)
    torch.testing.assert_close(checkpoint_grad, reference_grad)


class _CaptureBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def forward(self, x, context, t_mod, freqs, bsa_context=None):
        self.contexts.append(bsa_context)
        return x


class _IdentityHead(nn.Module):
    def forward(self, x, timestep):
        return x


class _ModelFnWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.seperated_timestep = False
        self.freq_dim = 8
        self.dim = 4
        self.require_vae_embedding = False
        self.require_clip_embedding = False
        self.time_embedding = nn.Linear(8, 4, bias=False)
        self.time_projection = nn.Linear(4, 24, bias=False)
        self.text_embedding = nn.Identity()
        self.blocks = nn.ModuleList([_CaptureBlock()])
        self.head = _IdentityHead()
        self.freqs = precompute_freqs_cis_3d(4, end=8)

    def patchify(self, x, control_camera_latents_input=None):
        return x

    def unpatchify(self, x, grid):
        f, h, w = grid
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], f, h, w)


def test_model_fn_passes_real_patch_grid_and_bsa_usp_fails_fast():
    config = WanBSAConfig(backend="eager_math")
    context = BSAContext.from_config(config, sparsity=0.8)
    dit = _ModelFnWan()
    output = model_fn_wan_video(
        dit=dit,
        latents=torch.randn(1, 4, 2, 3, 4),
        timestep=torch.tensor([1.0]),
        context=torch.randn(1, 2, 4),
        bsa_context=context,
    )
    assert output.shape == (1, 4, 2, 3, 4)
    captured = dit.blocks[0].contexts[0]
    assert captured.grid_size == (2, 3, 4)
    assert captured.top_k == 1

    with pytest.raises(ValueError, match="unified sequence parallel"):
        model_fn_wan_video(
            dit=dit,
            latents=torch.randn(1, 4, 2, 3, 4),
            timestep=torch.tensor([1.0]),
            context=torch.randn(1, 2, 4),
            bsa_context=context,
            use_unified_sequence_parallel=True,
        )
