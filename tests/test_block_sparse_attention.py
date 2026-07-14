import math
from types import SimpleNamespace

import pytest
import torch

from diffsynth.core.attention.block_sparse_attention import (
    BlockSparseAttention,
    EagerMathBackend,
    SDPAGatherBackend,
    bsa_coarse_attention,
    build_bsa_metadata,
    compute_bsa_top_k,
    deterministic_topk,
    pack_bsa_tokens,
    probe_bsa_backend,
    unpack_bsa_tokens,
)


def test_target_geometry_and_valid_count_histogram():
    metadata = build_bsa_metadata((21, 15, 26), (4, 3, 6), use_cache=False)

    assert metadata.block_grid == (6, 5, 5)
    assert metadata.num_blocks == 150
    assert metadata.block_capacity == 72
    assert metadata.padded_grid == (24, 15, 30)
    assert metadata.sequence_length == 8190
    assert metadata.padded_tokens == 10800
    assert compute_bsa_top_k(metadata.num_blocks, 0.8) == 30
    assert metadata.padding_ratio == pytest.approx(0.2416666667)
    values, counts = torch.unique(metadata.valid_count, return_counts=True)
    assert dict(zip(values.tolist(), counts.tolist())) == {6: 5, 18: 20, 24: 25, 72: 100}


def test_pack_unpack_is_reversible_and_uses_w_fastest_order():
    metadata = build_bsa_metadata((5, 4, 7), (4, 3, 6), use_cache=False)
    tokens = torch.arange(metadata.sequence_length).reshape(1, -1, 1).float()
    packed = pack_bsa_tokens(tokens, metadata)

    assert torch.equal(unpack_bsa_tokens(packed, metadata), tokens)
    assert packed[0, 0, :6, 0].tolist() == [0, 1, 2, 3, 4, 5]
    assert packed[0, 0, 6, 0].item() == 7
    assert torch.count_nonzero(packed[~metadata.valid_mask.unsqueeze(0)]) == 0


def test_deterministic_topk_breaks_ties_by_ascending_block_id():
    score = torch.tensor([[[[1.0, 2.0, 2.0, 2.0, 0.0]]]])
    indices = deterministic_topk(score, 3)
    assert indices.tolist() == [[[[1, 2, 3]]]]


def test_count_bias_adds_log_valid_capacity_prior_only_to_kv_blocks():
    metadata = build_bsa_metadata((3, 3, 7), (2, 2, 3), use_cache=False)
    blocks = torch.ones(1, 1, metadata.num_blocks, metadata.block_capacity, 2)
    unbiased = bsa_coarse_attention(
        blocks, blocks, blocks, metadata.valid_mask, top_k=2, count_bias=False
    )
    biased = bsa_coarse_attention(
        blocks, blocks, blocks, metadata.valid_mask, top_k=2, count_bias=True
    )
    expected = torch.log(metadata.valid_count.float() / metadata.block_capacity)
    torch.testing.assert_close(
        biased.score - unbiased.score,
        expected.reshape(1, 1, 1, -1).expand_as(unbiased.score),
    )


def _backend_fixture(dtype=torch.float64):
    torch.manual_seed(7)
    metadata = build_bsa_metadata((3, 3, 7), (2, 2, 3), use_cache=False)
    batch, heads, head_dim = 1, 2, 4
    shape = (batch, heads, metadata.sequence_length, head_dim)
    q = torch.randn(shape, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, dtype=dtype, requires_grad=True)
    q_blocks = pack_bsa_tokens(q, metadata)
    k_blocks = pack_bsa_tokens(k, metadata)
    v_blocks = pack_bsa_tokens(v, metadata)
    coarse = bsa_coarse_attention(
        q_blocks,
        k_blocks,
        v_blocks,
        metadata.valid_mask,
        top_k=2,
        scale=1 / math.sqrt(head_dim),
    )
    return metadata, (q, k, v), (q_blocks, k_blocks, v_blocks), coarse.topk_indices


@pytest.mark.parametrize("mask_mode", ["bool", "additive"])
def test_sdpa_gather_matches_eager_forward_and_gradients(mask_mode):
    metadata, tensors, blocks, topk = _backend_fixture()
    q, k, v = tensors
    q_blocks, k_blocks, v_blocks = blocks
    kwargs = dict(
        topk_indices=topk,
        q_valid_mask=metadata.valid_mask,
        kv_valid_mask=metadata.valid_mask,
        scale=1 / math.sqrt(q.shape[-1]),
        query_block_chunk=2,
        mask_mode=mask_mode,
    )
    eager = EagerMathBackend()(q_blocks, k_blocks, v_blocks, **kwargs)
    eager_grads = torch.autograd.grad(eager.square().sum(), (q, k, v), retain_graph=True)
    sdpa = SDPAGatherBackend()(q_blocks, k_blocks, v_blocks, **kwargs)
    sdpa_grads = torch.autograd.grad(sdpa.square().sum(), (q, k, v))

    torch.testing.assert_close(sdpa, eager, rtol=2e-5, atol=2e-7)
    for actual, expected in zip(sdpa_grads, eager_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_query_chunk_does_not_change_output_or_gradient():
    metadata, tensors, blocks, topk = _backend_fixture()
    q, k, v = tensors
    outputs = []
    gradients = []
    for chunk in (1, 2, 4):
        output = EagerMathBackend()(
            *blocks,
            topk,
            metadata.valid_mask,
            metadata.valid_mask,
            scale=1 / math.sqrt(q.shape[-1]),
            query_block_chunk=chunk,
            mask_mode="bool",
        )
        outputs.append(output)
        gradients.append(torch.autograd.grad(output.sum(), (q, k, v), retain_graph=True))
    for output in outputs[1:]:
        torch.testing.assert_close(output, outputs[0], rtol=1e-6, atol=3e-7)
    for gradient_tuple in gradients[1:]:
        for actual, expected in zip(gradient_tuple, gradients[0]):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=3e-7)


def test_block_gate_shapes_zero_init_and_first_backward():
    torch.manual_seed(11)
    model = BlockSparseAttention(
        model_dim=16,
        num_heads=2,
        block_size=(2, 2, 3),
        gate_rank=4,
        backend="eager_math",
    ).double()
    grid = (3, 3, 7)
    length = math.prod(grid)
    hidden = torch.randn(1, length, 16, dtype=torch.float64, requires_grad=True)
    q = torch.randn(1, 2, length, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    output = model(
        q,
        k,
        v,
        hidden,
        grid_size=grid,
        sparsity=0.5,
        query_block_chunk=2,
        mask_mode="bool",
    )
    info = model.last_runtime_info
    assert model.gate_is_zero_initialized()
    assert info["gate_input_shape"] == (1, 12, 16)
    assert info["gate_hidden_shape"] == (1, 12, 4)
    assert info["gate_output_shape"] == (1, 12, 16)
    assert info["materialized_token_gate_tensor"] is False
    output.square().mean().backward()
    assert torch.count_nonzero(model.gate_up.weight.grad) > 0
    assert torch.count_nonzero(model.gate_down.weight.grad) == 0


def test_dense_special_case_matches_dense_attention_at_zero_gate():
    torch.manual_seed(13)
    model = BlockSparseAttention(
        model_dim=8,
        num_heads=2,
        block_size=(2, 2, 2),
        gate_rank=2,
        backend="eager_math",
    ).double()
    grid = (3, 3, 3)
    length = math.prod(grid)
    hidden = torch.randn(1, length, 8, dtype=torch.float64)
    q = torch.randn(1, 2, length, 4, dtype=torch.float64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    actual = model(q, k, v, hidden, grid_size=grid, sparsity=0.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_runtime_grid_length_mismatch_fails():
    model = BlockSparseAttention(8, 2, block_size=(2, 2, 2), gate_rank=2)
    q = torch.randn(1, 2, 7, 4)
    with pytest.raises(ValueError, match="Runtime grid"):
        model(q, q, q, torch.randn(1, 7, 8), grid_size=(2, 2, 2), sparsity=0.5)


@pytest.mark.parametrize("backend", ["eager_math", "sdpa_gather"])
@pytest.mark.parametrize("mask_mode", ["bool", "additive"])
def test_backend_capability_probe_checks_forward_and_backward(backend, mask_mode):
    result = probe_bsa_backend(
        backend,
        device="cpu",
        dtype=torch.float32,
        num_heads=2,
        head_dim=4,
        block_capacity=6,
        mask_mode=mask_mode,
    )
    assert result["forward_backward_finite"] is True
    assert result["output_max_abs_error"] <= 2e-5
    assert result["gradient_max_abs_error"] <= 2e-5


def test_compact_ragged_matches_fixed_padded_output_and_gradients():
    metadata, tensors, blocks, topk = _backend_fixture()
    q, k, v = tensors
    kwargs = dict(
        topk_indices=topk,
        q_valid_mask=metadata.valid_mask,
        kv_valid_mask=metadata.valid_mask,
        scale=1 / math.sqrt(q.shape[-1]),
        query_block_chunk=2,
        mask_mode="bool",
    )
    backend = EagerMathBackend()
    fixed = backend(
        *blocks, **kwargs, context=SimpleNamespace(boundary_mode="fixed_padded")
    )
    fixed_grad = torch.autograd.grad(fixed.square().sum(), (q, k, v), retain_graph=True)
    compact = backend(
        *blocks, **kwargs, context=SimpleNamespace(boundary_mode="compact_ragged")
    )
    compact_grad = torch.autograd.grad(compact.square().sum(), (q, k, v))
    torch.testing.assert_close(compact, fixed, rtol=1e-6, atol=3e-7)
    for actual, expected in zip(compact_grad, fixed_grad):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-7)
