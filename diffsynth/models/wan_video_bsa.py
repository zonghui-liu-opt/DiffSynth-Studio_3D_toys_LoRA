"""Wan video self-attention integration for portable 3D BSA."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from ..core.attention import BlockSparseAttention, build_bsa_metadata, compute_bsa_top_k
from .wan_video_dit import rope_apply


@dataclass(frozen=True)
class WanBSAConfig:
    block_size: Tuple[int, int, int] = (4, 3, 6)
    target_sparsity: float = 0.8
    backend: str = "sdpa_gather"
    query_block_chunk: int = 4
    mask_mode: str = "additive"
    boundary_mode: str = "fixed_padded"
    count_bias: bool = False
    gate_type: str = "low_rank_dynamic"
    gate_granularity: str = "block"
    gate_rank: int = 32
    gate_alpha: float = 32.0
    fail_on_backend_fallback: bool = True

    def __post_init__(self):
        block_size = tuple(int(value) for value in self.block_size)
        object.__setattr__(self, "block_size", block_size)
        if len(block_size) != 3 or min(block_size) <= 0:
            raise ValueError("BSA block_size must contain three positive integers.")
        if math.prod(block_size) != 72:
            raise ValueError(
                f"P0 Wan BSA requires block capacity 72, received {block_size} ({math.prod(block_size)})."
            )
        if not 0 <= float(self.target_sparsity) < 1:
            raise ValueError("BSA target_sparsity must be in [0, 1).")
        if self.backend not in {"sdpa_gather", "eager_math"}:
            raise ValueError("P0 BSA backend must be sdpa_gather or eager_math.")
        if int(self.query_block_chunk) <= 0:
            raise ValueError("BSA query_block_chunk must be positive.")
        if self.mask_mode not in {"additive", "bool"}:
            raise ValueError("BSA mask_mode must be additive or bool.")
        if self.boundary_mode not in {"fixed_padded", "compact_ragged"}:
            raise ValueError("BSA boundary_mode must be fixed_padded or compact_ragged.")
        if self.gate_type != "low_rank_dynamic" or self.gate_granularity != "block":
            raise ValueError("P0 BSA requires a low_rank_dynamic block gate.")
        if int(self.gate_rank) <= 0 or float(self.gate_alpha) <= 0:
            raise ValueError("BSA gate rank and alpha must be positive.")


@dataclass(frozen=True)
class BSAContext:
    enabled: bool
    grid_size: Optional[Tuple[int, int, int]]
    block_size: Tuple[int, int, int]
    sparsity: float
    top_k: Optional[int]
    backend: str
    query_block_chunk: int
    mask_mode: str
    boundary_mode: str
    count_bias: bool
    gate_granularity: str
    denoise_progress_id: int = 0
    optimizer_step: int = 0

    @classmethod
    def from_config(
        cls,
        config: WanBSAConfig,
        *,
        sparsity: Optional[float] = None,
        optimizer_step: int = 0,
        denoise_progress_id: int = 0,
    ):
        return cls(
            enabled=True,
            grid_size=None,
            block_size=config.block_size,
            sparsity=config.target_sparsity if sparsity is None else float(sparsity),
            top_k=None,
            backend=config.backend,
            query_block_chunk=config.query_block_chunk,
            mask_mode=config.mask_mode,
            boundary_mode=config.boundary_mode,
            count_bias=config.count_bias,
            gate_granularity=config.gate_granularity,
            optimizer_step=int(optimizer_step),
            denoise_progress_id=int(denoise_progress_id),
        )

    def with_grid(self, grid_size):
        grid_size = tuple(int(value) for value in grid_size)
        metadata = build_bsa_metadata(grid_size, self.block_size)
        top_k = compute_bsa_top_k(metadata.num_blocks, self.sparsity)
        return replace(self, grid_size=grid_size, top_k=top_k)


def bsa_sparsity_for_step(completed_optimizer_steps: int, max_optimizer_steps: int) -> float:
    """Piecewise-constant 0→80% schedule keyed only by completed optimizer steps."""
    completed_optimizer_steps = int(completed_optimizer_steps)
    max_optimizer_steps = int(max_optimizer_steps)
    if completed_optimizer_steps < 0 or max_optimizer_steps <= 0:
        raise ValueError("Optimizer step counters must be non-negative with a positive maximum.")
    progress = min(completed_optimizer_steps / max_optimizer_steps, 1.0)
    if progress < 0.05:
        return 0.0
    if progress < 0.10:
        return 0.1
    if progress < 0.15:
        return 0.2
    if progress < 0.20:
        return 0.3
    if progress < 0.25:
        return 0.4
    if progress < 0.30:
        return 0.5
    if progress < 0.35:
        return 0.6
    if progress < 0.40:
        return 0.7
    return 0.8


class WanBSASelfAttention(nn.Module):
    """Drop-in replacement that reuses every dense projection/norm module."""

    def __init__(self, dense_attention: nn.Module, config: WanBSAConfig):
        super().__init__()
        required = ("q", "k", "v", "o", "norm_q", "norm_k", "attn", "dim", "num_heads")
        missing = [name for name in required if not hasattr(dense_attention, name)]
        if missing:
            raise TypeError(f"Dense Wan self-attention is missing required attributes: {missing}")
        self.dim = int(dense_attention.dim)
        self.num_heads = int(dense_attention.num_heads)
        self.head_dim = self.dim // self.num_heads
        self.q = dense_attention.q
        self.k = dense_attention.k
        self.v = dense_attention.v
        self.o = dense_attention.o
        self.norm_q = dense_attention.norm_q
        self.norm_k = dense_attention.norm_k
        self.attn = dense_attention.attn
        self.bsa_config = config
        # In inference the dense DiT may already live on CUDA when BSA is injected.
        # Keep routing math in its trained FP32 precision, but colocate new gates
        # with the reused projections so the first sparse forward is device-safe.
        gate_device = self.q.weight.device
        self.bsa_gate_down = nn.Linear(self.dim, config.gate_rank, bias=False).to(
            device=gate_device, dtype=torch.float32
        )
        self.bsa_gate_up = nn.Linear(config.gate_rank, self.dim, bias=False).to(
            device=gate_device, dtype=torch.float32
        )
        nn.init.normal_(self.bsa_gate_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.bsa_gate_up.weight)
        self.bsa_engine = BlockSparseAttention(
            self.dim,
            self.num_heads,
            block_size=config.block_size,
            gate_rank=config.gate_rank,
            gate_alpha=config.gate_alpha,
            backend=config.backend,
            use_internal_gate=False,
        )

    @classmethod
    def from_dense(cls, dense_attention: nn.Module, config: WanBSAConfig):
        if isinstance(dense_attention, cls):
            raise RuntimeError("Wan BSA injection cannot wrap an already injected self-attention.")
        return cls(dense_attention, config)

    def _dense_attention_heads(self, q, k, v):
        batch, heads, length, head_dim = q.shape
        q_flat = q.permute(0, 2, 1, 3).reshape(batch, length, heads * head_dim)
        k_flat = k.permute(0, 2, 1, 3).reshape(batch, length, heads * head_dim)
        v_flat = v.permute(0, 2, 1, 3).reshape(batch, length, heads * head_dim)
        output = self.attn(q_flat, k_flat, v_flat)
        return output.reshape(batch, length, heads, head_dim).permute(0, 2, 1, 3)

    def forward(self, x, freqs, bsa_context: Optional[BSAContext] = None):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        if bsa_context is None or not bsa_context.enabled:
            return self.o(self.attn(q, k, v))
        if bsa_context.grid_size is None:
            raise ValueError("BSAContext must carry the runtime patch grid before self-attention.")
        if bsa_context.block_size != self.bsa_config.block_size:
            raise ValueError("BSAContext block size does not match the injected student config.")
        if bsa_context.backend != self.bsa_config.backend:
            raise ValueError("BSAContext backend does not match the injected student config.")
        if bsa_context.gate_granularity != "block":
            raise ValueError("P0 Wan BSA rejects non-block gate granularity.")
        batch, length, _ = q.shape
        q_heads = q.reshape(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_heads = k.reshape(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v_heads = v.reshape(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        output = self.bsa_engine(
            q_heads,
            k_heads,
            v_heads,
            x,
            grid_size=bsa_context.grid_size,
            sparsity=bsa_context.sparsity,
            query_block_chunk=bsa_context.query_block_chunk,
            mask_mode=bsa_context.mask_mode,
            count_bias=bsa_context.count_bias,
            dense_attention=self._dense_attention_heads,
            gate_down=self.bsa_gate_down,
            gate_up=self.bsa_gate_up,
            context=bsa_context,
        )
        output = output.permute(0, 2, 1, 3).reshape(batch, length, self.dim)
        return self.o(output)


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    digest = hashlib.sha256()
    with open(os.path.expanduser(path), "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_wan_bsa_provenance(
    manifest: Dict,
    *,
    figurine_lora_path: Optional[str] = None,
    direct_distill_warmstart_path: Optional[str] = None,
) -> Dict[str, str]:
    """Fail fast when inference/resume inputs differ from checkpoint provenance."""
    requested = {
        "figurine_lora_sha256": figurine_lora_path,
        "direct_distill_warmstart_sha256": direct_distill_warmstart_path,
    }
    requested = {key: path for key, path in requested.items() if path is not None}
    if not requested:
        raise ValueError("Wan BSA provenance validation requires at least one input path.")
    missing = sorted(
        key for key in requested if not isinstance(manifest.get(key), str)
    )
    if missing:
        raise ValueError(f"BSA manifest is missing required provenance hashes: {missing}")
    actual = {key: _sha256_file(path) for key, path in requested.items()}
    mismatches = {
        key: {"expected": manifest[key], "actual": actual[key]}
        for key in requested
        if manifest[key] != actual[key]
    }
    if mismatches:
        raise ValueError(f"BSA checkpoint provenance mismatch: {mismatches}")
    return actual


def inject_wan_bsa(
    model: nn.Module,
    config: WanBSAConfig,
    *,
    expected_layers: Optional[int] = None,
    figurine_lora_path: Optional[str] = None,
    direct_distill_warmstart_path: Optional[str] = None,
) -> Dict:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("Wan BSA injection requires model.blocks as nn.ModuleList.")
    expected_layers = len(blocks) if expected_layers is None else int(expected_layers)
    if len(blocks) != expected_layers:
        raise RuntimeError(
            f"Wan BSA expected {expected_layers} DiT blocks, found {len(blocks)}."
        )
    injected_paths = []
    for index, block in enumerate(blocks):
        if isinstance(block.self_attn, WanBSASelfAttention):
            raise RuntimeError(f"Repeated BSA injection detected at blocks.{index}.self_attn.")
        block.self_attn = WanBSASelfAttention.from_dense(block.self_attn, config)
        injected_paths.append(f"blocks.{index}.self_attn")
    cross_count = sum(
        isinstance(getattr(block, "cross_attn", None), WanBSASelfAttention) for block in blocks
    )
    gate_up_zero = all(
        torch.count_nonzero(block.self_attn.bsa_gate_up.weight.detach()) == 0 for block in blocks
    )
    if len(injected_paths) != expected_layers or cross_count or not gate_up_zero:
        raise RuntimeError("Wan BSA injection invariant failed.")
    summary = {
        "model_class": model.__class__.__name__,
        "dim": int(getattr(model, "dim", blocks[0].self_attn.dim)),
        "num_heads": int(blocks[0].self_attn.num_heads),
        "head_dim": int(blocks[0].self_attn.head_dim),
        "num_layers": len(blocks),
        "expected_self_attention_modules": expected_layers,
        "injected_bsa_modules": len(injected_paths),
        "failed_modules": [],
        "cross_attention_bsa_modules": cross_count,
        "block_size": list(config.block_size),
        "block_capacity": math.prod(config.block_size),
        "target_sparsity": config.target_sparsity,
        "backend": config.backend,
        "query_block_chunk": config.query_block_chunk,
        "boundary_mode": config.boundary_mode,
        "count_bias": config.count_bias,
        "gate_type": config.gate_type,
        "gate_granularity": config.gate_granularity,
        "gate_rank": config.gate_rank,
        "gate_up_zero_initialized": gate_up_zero,
        "figurine_lora_fused": bool(figurine_lora_path),
        "figurine_lora_sha256": _sha256_file(figurine_lora_path),
        "dense_direct_distill_warmstart_path": direct_distill_warmstart_path,
        "dense_direct_distill_warmstart_sha256": _sha256_file(direct_distill_warmstart_path),
        "warmstart_missing_keys": [],
        "warmstart_unexpected_keys": [],
        "injected_module_paths": injected_paths,
    }
    model._wan_bsa_config = config
    model._wan_bsa_student_info = summary
    return summary


def collect_wan_bsa_runtime_info(model: nn.Module) -> Dict:
    modules = [
        block.self_attn for block in model.blocks if isinstance(block.self_attn, WanBSASelfAttention)
    ]
    if not modules or any(module.bsa_engine.last_runtime_info is None for module in modules):
        raise RuntimeError("Wan BSA runtime info is unavailable before every injected layer runs once.")
    first = modules[0].bsa_engine.last_runtime_info
    geometry_keys = (
        "runtime_grid", "sequence_length", "block_grid", "num_blocks", "top_k",
        "actual_sparsity", "padded_tokens", "padding_ratio", "gate_input_shape",
        "gate_hidden_shape", "gate_output_shape", "materialized_token_gate_tensor",
    )
    for module in modules[1:]:
        current = module.bsa_engine.last_runtime_info
        if any(current[key] != first[key] for key in geometry_keys):
            raise RuntimeError("Injected Wan BSA layers observed inconsistent runtime geometry/context.")
    return dict(first)


def wan_bsa_gate_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".bsa_gate_down.weight" in name or ".bsa_gate_up.weight" in name
    }
    expected = 2 * len(getattr(model, "blocks", ()))
    if len(state) != expected:
        raise RuntimeError(f"Expected {expected} Wan BSA gate tensors, found {len(state)}.")
    return state


def bsa_config_manifest(
    config: WanBSAConfig, *, schema_version: int = 1, **training_state
) -> Dict:
    manifest = {
        "schema_version": int(schema_version),
        **asdict(config),
        "block_size": list(config.block_size),
        "block_capacity": math.prod(config.block_size),
    }
    manifest.update(training_state)
    return manifest


def validate_bsa_manifest(manifest: Dict, config: Optional[WanBSAConfig] = None):
    required = {
        "schema_version",
        "block_size",
        "block_capacity",
        "target_sparsity",
        "gate_type",
        "gate_granularity",
        "gate_rank",
        "gate_alpha",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        raise ValueError(f"BSA manifest is missing required fields: {missing}")
    if manifest["schema_version"] not in (1, 2):
        raise ValueError(f"Unsupported BSA manifest schema: {manifest['schema_version']}")
    if manifest["schema_version"] == 2:
        schedule_fields = {
            "completed_optimizer_steps",
            "schedule_spec",
            "schedule_sha256",
            "schedule_runtime",
        }
        missing_schedule = sorted(schedule_fields.difference(manifest))
        if missing_schedule:
            raise ValueError(
                f"BSA schema-v2 manifest is missing schedule fields: {missing_schedule}"
            )
        from diffsynth.diffusion.bsa_schedule import compiled_bsa_schedule_from_dict

        compiled = compiled_bsa_schedule_from_dict(
            manifest, target_sparsity=manifest["target_sparsity"]
        )
        completed_steps = manifest["completed_optimizer_steps"]
        if (
            isinstance(completed_steps, bool)
            or int(completed_steps) != completed_steps
            or not 0 <= int(completed_steps) <= compiled.total_optimizer_steps
        ):
            raise ValueError("BSA completed_optimizer_steps is outside the saved schedule.")
        if manifest.get("max_optimizer_steps") != compiled.total_optimizer_steps:
            raise ValueError(
                "BSA max_optimizer_steps does not match the saved schedule runtime."
            )
    if manifest["gate_type"] != "low_rank_dynamic" or manifest["gate_granularity"] != "block":
        raise ValueError("BSA checkpoint is not a low_rank_dynamic block-gate checkpoint.")
    if math.prod(manifest["block_size"]) != manifest["block_capacity"] or manifest["block_capacity"] != 72:
        raise ValueError("BSA checkpoint block geometry is invalid.")
    if config is not None:
        semantic_fields = (
            "block_size",
            "target_sparsity",
            "boundary_mode",
            "count_bias",
            "gate_type",
            "gate_granularity",
            "gate_rank",
            "gate_alpha",
        )
        config_values = asdict(config)
        config_values["block_size"] = list(config.block_size)
        conflicts = {
            key: (manifest.get(key), config_values[key])
            for key in semantic_fields
            if manifest.get(key) != config_values[key]
        }
        if conflicts:
            raise ValueError(f"BSA checkpoint semantic config conflicts with runtime: {conflicts}")
    return manifest


def save_wan_bsa_adapter(model: nn.Module, path: str):
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_file(wan_bsa_gate_state_dict(model), path)


def load_wan_bsa_adapter(model: nn.Module, path: str):
    expected = wan_bsa_gate_state_dict(model)
    loaded = load_file(os.path.expanduser(path), device="cpu")
    if set(loaded) != set(expected):
        raise RuntimeError(
            "BSA adapter does not exactly cover injected gates: "
            f"missing={sorted(set(expected) - set(loaded))}, "
            f"unexpected={sorted(set(loaded) - set(expected))}."
        )
    shape_mismatch = {
        key: (tuple(loaded[key].shape), tuple(expected[key].shape))
        for key in expected
        if loaded[key].shape != expected[key].shape
    }
    if shape_mismatch:
        raise RuntimeError(f"BSA adapter gate shape mismatch: {shape_mismatch}")
    target = model.state_dict()
    with torch.no_grad():
        for key, value in loaded.items():
            target[key].copy_(value.to(device=target[key].device, dtype=target[key].dtype))


def write_student_model_info(path: str, summary: Dict, runtime_info: Optional[Dict] = None):
    payload = dict(summary)
    if runtime_info is not None:
        payload["runtime"] = runtime_info
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def resolve_bsa_checkpoint(checkpoint_dir: str, config: Optional[WanBSAConfig] = None) -> Dict:
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    required = {
        "direct_distill_lora": checkpoint_dir / "direct_distill_lora.safetensors",
        "bsa_adapter": checkpoint_dir / "bsa_adapter.safetensors",
        "config": checkpoint_dir / "bsa_config.json",
        "complete": checkpoint_dir / "checkpoint_complete",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete BSA checkpoint {checkpoint_dir}: missing {missing}."
        )
    manifest = json.loads(required["config"].read_text(encoding="utf-8"))
    validate_bsa_manifest(manifest, config)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "direct_distill_lora": str(required["direct_distill_lora"]),
        "bsa_adapter": str(required["bsa_adapter"]),
        "manifest": manifest,
    }


def save_composite_bsa_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    checkpoint_dir: str,
    manifest: Dict,
):
    """Atomically save the P0 weight-continuation checkpoint.

    Optimizer/RNG restoration is intentionally not implied. The manifest marks
    this explicitly so a continuation cannot be mistaken for exact resumption.
    """
    validate_bsa_manifest(manifest)
    checkpoint_dir = Path(checkpoint_dir)
    temporary = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if checkpoint_dir.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite BSA checkpoint: {checkpoint_dir}")
    temporary.mkdir(parents=True)
    lora = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if ".lora_A." in key or ".lora_B." in key
    }
    gate = {
        key: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if ".bsa_gate_down.weight" in key or ".bsa_gate_up.weight" in key
    }
    if not lora or not gate:
        raise RuntimeError(
            f"Composite BSA checkpoint requires LoRA and gate tensors; found {len(lora)} and {len(gate)}."
        )
    save_file(lora, str(temporary / "direct_distill_lora.safetensors"))
    save_file(gate, str(temporary / "bsa_adapter.safetensors"))
    manifest = dict(manifest)
    manifest["resume_semantics"] = "weight_continuation"
    manifest["optimizer_state_saved"] = False
    (temporary / "bsa_config.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trainer_state = temporary / "trainer_state"
    trainer_state.mkdir()
    (trainer_state / "state.json").write_text(
        json.dumps(
            {
                "completed_optimizer_steps": manifest.get("completed_optimizer_steps", 0),
                "resume_semantics": "weight_continuation",
                "optimizer_state_saved": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (temporary / "checkpoint_complete").write_text("complete\n", encoding="utf-8")
    os.replace(temporary, checkpoint_dir)
