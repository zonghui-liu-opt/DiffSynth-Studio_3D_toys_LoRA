"""Portable 3D block-sparse attention primitives.

The module deliberately contains no CUDA/NPU-specific code. Geometry, routing,
and coarse/fine fusion are device independent; only the fine provider is
replaceable. The production provider gathers selected KV blocks before calling
PyTorch SDPA, so it never constructs a global ``[L, L]`` sparse mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Int3 = Tuple[int, int, int]


@dataclass(frozen=True)
class BSAStaticMetadata:
    grid_size: Int3
    block_size: Int3
    block_grid: Int3
    block_capacity: int
    num_blocks: int
    padded_grid: Int3
    sequence_length: int
    padded_tokens: int
    pack_indices: torch.Tensor
    unpack_indices: torch.Tensor
    valid_mask: torch.Tensor
    valid_count: torch.Tensor
    block_coordinates: torch.Tensor

    @property
    def padding_ratio(self) -> float:
        return 1.0 - self.sequence_length / self.padded_tokens


_METADATA_CACHE: Dict[Tuple[Int3, Int3, str], BSAStaticMetadata] = {}


def _positive_int3(value, name: str) -> Int3:
    value = tuple(int(item) for item in value)
    if len(value) != 3 or min(value) <= 0:
        raise ValueError(f"{name} must contain three positive integers; received {value!r}.")
    return value


def build_bsa_metadata(
    grid_size: Int3,
    block_size: Int3 = (4, 3, 6),
    *,
    device: Optional[torch.device] = None,
    use_cache: bool = True,
) -> BSAStaticMetadata:
    """Build immutable W-fastest pack/unpack geometry for a runtime 3D grid."""
    grid_size = _positive_int3(grid_size, "grid_size")
    block_size = _positive_int3(block_size, "block_size")
    device = torch.device("cpu") if device is None else torch.device(device)
    cache_key = (grid_size, block_size, str(device))
    if use_cache and cache_key in _METADATA_CACHE:
        return _METADATA_CACHE[cache_key]

    t, h, w = grid_size
    bt, bh, bw = block_size
    block_grid = (
        math.ceil(t / bt),
        math.ceil(h / bh),
        math.ceil(w / bw),
    )
    padded_grid = tuple(block_grid[i] * block_size[i] for i in range(3))
    block_capacity = math.prod(block_size)
    num_blocks = math.prod(block_grid)
    sequence_length = math.prod(grid_size)
    padded_tokens = num_blocks * block_capacity

    pack_indices = torch.full(
        (num_blocks, block_capacity), -1, dtype=torch.long, device=device
    )
    valid_mask = torch.zeros(
        (num_blocks, block_capacity), dtype=torch.bool, device=device
    )
    block_coordinates = torch.empty((num_blocks, 3), dtype=torch.long, device=device)
    unpack_indices = torch.empty(sequence_length, dtype=torch.long, device=device)

    block_id = 0
    for block_t in range(block_grid[0]):
        for block_h in range(block_grid[1]):
            for block_w in range(block_grid[2]):
                block_coordinates[block_id] = torch.tensor(
                    (block_t, block_h, block_w), dtype=torch.long, device=device
                )
                lane = 0
                for local_t in range(bt):
                    token_t = block_t * bt + local_t
                    for local_h in range(bh):
                        token_h = block_h * bh + local_h
                        for local_w in range(bw):
                            token_w = block_w * bw + local_w
                            if token_t < t and token_h < h and token_w < w:
                                token_id = (token_t * h + token_h) * w + token_w
                                pack_indices[block_id, lane] = token_id
                                unpack_indices[token_id] = block_id * block_capacity + lane
                                valid_mask[block_id, lane] = True
                            lane += 1
                block_id += 1

    metadata = BSAStaticMetadata(
        grid_size=grid_size,
        block_size=block_size,
        block_grid=block_grid,
        block_capacity=block_capacity,
        num_blocks=num_blocks,
        padded_grid=padded_grid,
        sequence_length=sequence_length,
        padded_tokens=padded_tokens,
        pack_indices=pack_indices,
        unpack_indices=unpack_indices,
        valid_mask=valid_mask,
        valid_count=valid_mask.sum(dim=-1),
        block_coordinates=block_coordinates,
    )
    if use_cache:
        _METADATA_CACHE[cache_key] = metadata
    return metadata


def pack_bsa_tokens(x: torch.Tensor, metadata: BSAStaticMetadata) -> torch.Tensor:
    """Pack ``[..., L, D]`` into ``[..., N, C, D]`` and zero invalid lanes."""
    if x.ndim < 2 or x.shape[-2] != metadata.sequence_length:
        raise ValueError(
            f"Expected sequence length {metadata.sequence_length}, received shape {tuple(x.shape)}."
        )
    safe_indices = metadata.pack_indices.clamp_min(0).flatten()
    packed = x.index_select(-2, safe_indices).unflatten(
        -2, (metadata.num_blocks, metadata.block_capacity)
    )
    mask_shape = (1,) * (packed.ndim - 3) + metadata.valid_mask.shape + (1,)
    return packed * metadata.valid_mask.reshape(mask_shape).to(dtype=packed.dtype)


def unpack_bsa_tokens(x: torch.Tensor, metadata: BSAStaticMetadata) -> torch.Tensor:
    """Unpack ``[..., N, C, D]`` back to the valid ``[..., L, D]`` sequence."""
    if x.ndim < 3 or x.shape[-3:-1] != (
        metadata.num_blocks,
        metadata.block_capacity,
    ):
        raise ValueError(
            "Packed tensor geometry does not match metadata: "
            f"shape={tuple(x.shape)}, expected N,C={(metadata.num_blocks, metadata.block_capacity)}."
        )
    flattened = x.flatten(-3, -2)
    return flattened.index_select(-2, metadata.unpack_indices)


def masked_block_mean(
    blocks: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """FP32 masked mean over block lanes for ``[..., N, C, D]`` inputs."""
    if blocks.ndim < 3:
        raise ValueError("blocks must have at least N,C,D dimensions.")
    if tuple(blocks.shape[-3:-1]) != tuple(valid_mask.shape[-2:]):
        raise ValueError(
            f"Mask {tuple(valid_mask.shape)} does not match blocks {tuple(blocks.shape)}."
        )
    mask_shape = (1,) * (blocks.ndim - 3) + tuple(valid_mask.shape) + (1,)
    mask = valid_mask.reshape(mask_shape)
    count_shape = (1,) * (blocks.ndim - 3) + (valid_mask.shape[-2], 1)
    counts = valid_mask.sum(dim=-1).reshape(count_shape).clamp_min(1).float()
    return (blocks.float() * mask).sum(dim=-2) / counts


def compute_bsa_top_k(num_blocks: int, sparsity: float) -> int:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive.")
    sparsity = float(sparsity)
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], received {sparsity}.")
    return max(1, min(num_blocks, math.ceil((1.0 - sparsity) * num_blocks)))


def deterministic_topk(score: torch.Tensor, top_k: int) -> torch.Tensor:
    """Select descending scores, breaking exact ties by ascending KV block id."""
    if score.ndim < 2 or score.shape[-1] <= 0:
        raise ValueError("score must have a non-empty KV-block dimension.")
    if not 1 <= int(top_k) <= score.shape[-1]:
        raise ValueError(f"top_k={top_k} is invalid for {score.shape[-1]} KV blocks.")
    # Stable sort preserves the original ascending block-id order for equal scores.
    return torch.argsort(score, dim=-1, descending=True, stable=True)[..., : int(top_k)]


@dataclass
class BSACoarseResult:
    score: torch.Tensor
    coarse: torch.Tensor
    topk_indices: torch.Tensor


def bsa_coarse_attention(
    q_blocks: torch.Tensor,
    k_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int,
    scale: Optional[float] = None,
    count_bias: bool = False,
) -> BSACoarseResult:
    """Masked block-level global attention and deterministic routing."""
    if q_blocks.shape != k_blocks.shape or q_blocks.shape != v_blocks.shape:
        raise ValueError("q_blocks, k_blocks, and v_blocks must have identical shapes.")
    head_dim = q_blocks.shape[-1]
    scale = 1.0 / math.sqrt(head_dim) if scale is None else float(scale)
    q_coarse = masked_block_mean(q_blocks, valid_mask)
    k_coarse = masked_block_mean(k_blocks, valid_mask)
    v_coarse = masked_block_mean(v_blocks, valid_mask)
    score = torch.matmul(q_coarse, k_coarse.transpose(-1, -2)) * scale
    if count_bias:
        counts = valid_mask.sum(dim=-1).float()
        score = score + torch.log(counts / valid_mask.shape[-1]).reshape(
            (1,) * (score.ndim - 1) + (counts.numel(),)
        )
    topk_indices = deterministic_topk(score, top_k)
    probability = torch.softmax(score, dim=-1)
    coarse = torch.matmul(probability, v_coarse).to(dtype=v_blocks.dtype)
    return BSACoarseResult(score=score, coarse=coarse, topk_indices=topk_indices)


class BlockSparseAttentionBackend(nn.Module):
    """Interface implemented by portable fine-attention providers."""

    def forward(
        self,
        q_blocks: torch.Tensor,
        k_blocks: torch.Tensor,
        v_blocks: torch.Tensor,
        topk_indices: torch.Tensor,
        q_valid_mask: torch.Tensor,
        kv_valid_mask: torch.Tensor,
        *,
        scale: float,
        query_block_chunk: int,
        mask_mode: str,
        block_residual: Optional[torch.Tensor] = None,
        context=None,
    ) -> torch.Tensor:
        raise NotImplementedError


def _selected_kv(
    blocks: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[B,H,Q,K]`` block ids as ``[B*Q,H,K*C,D]``."""
    batch, heads, _, capacity, head_dim = blocks.shape
    query_blocks = indices.shape[2]
    batch_index = torch.arange(batch, device=blocks.device)[:, None, None, None]
    head_index = torch.arange(heads, device=blocks.device)[None, :, None, None]
    selected = blocks[batch_index, head_index, indices]
    return selected.permute(0, 2, 1, 3, 4, 5).reshape(
        batch * query_blocks, heads, -1, head_dim
    )


def _selected_valid_mask(
    valid_mask: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather validity as ``[B*Q,H,1,K*C]``."""
    batch, heads, query_blocks, _ = indices.shape
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0).expand(batch, -1, -1)
    if valid_mask.shape[0] != batch:
        raise ValueError("valid_mask batch dimension does not match routing indices.")
    expanded = valid_mask[:, None].expand(-1, heads, -1, -1)
    batch_index = torch.arange(batch, device=indices.device)[:, None, None, None]
    head_index = torch.arange(heads, device=indices.device)[None, :, None, None]
    selected = expanded[batch_index, head_index, indices]
    return selected.permute(0, 2, 1, 3, 4).reshape(
        batch * query_blocks, heads, 1, -1
    )


class _GatherBackend(BlockSparseAttentionBackend):
    def __init__(self, provider: str):
        super().__init__()
        if provider not in {"sdpa_gather", "eager_math"}:
            raise ValueError(f"Unsupported BSA provider: {provider}")
        self.provider = provider

    def _attention(self, q, k, v, allowed, scale, mask_mode):
        if self.provider == "eager_math":
            logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
            logits = logits.masked_fill(~allowed, -torch.inf)
            probability = torch.softmax(logits, dim=-1)
            return torch.matmul(probability, v.float()).to(dtype=v.dtype)
        if mask_mode == "bool":
            attention_mask = allowed
        elif mask_mode == "additive":
            attention_mask = torch.zeros(
                allowed.shape, dtype=q.dtype, device=q.device
            ).masked_fill(~allowed, -torch.inf)
        else:
            raise ValueError(f"Unsupported BSA mask mode: {mask_mode}")
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )

    def _compact_attention(self, q, k, v, allowed, scale, mask_mode):
        """P1 reference layout: remove invalid selected KV lanes per head/chunk."""
        batch_queries, heads, query_tokens, head_dim = q.shape
        selected_tokens = k.shape[-2]
        rows = batch_queries * heads
        q_rows = q.reshape(rows, 1, query_tokens, head_dim)
        k_rows = k.reshape(rows, selected_tokens, head_dim)
        v_rows = v.reshape(rows, selected_tokens, head_dim)
        allowed_rows = allowed.reshape(rows, selected_tokens)
        order = torch.argsort(
            allowed_rows.to(dtype=torch.int8), dim=-1, descending=True, stable=True
        )
        gather_index = order.unsqueeze(-1).expand(-1, -1, head_dim)
        k_rows = torch.gather(k_rows, 1, gather_index)
        v_rows = torch.gather(v_rows, 1, gather_index)
        valid_count = allowed_rows.sum(dim=-1)
        # P1 is an explicit diagnostic path; this synchronization is why it is
        # not the default performance provider. A future fused provider may
        # bucket counts without a host round-trip.
        max_valid = int(valid_count.max().detach().cpu())
        k_rows = k_rows[:, :max_valid].unsqueeze(1)
        v_rows = v_rows[:, :max_valid].unsqueeze(1)
        compact_allowed = (
            torch.arange(max_valid, device=q.device).reshape(1, 1, 1, max_valid)
            < valid_count.reshape(rows, 1, 1, 1)
        )
        output = self._attention(
            q_rows, k_rows, v_rows, compact_allowed, scale, mask_mode
        )
        return output.reshape(batch_queries, heads, query_tokens, head_dim)

    def forward(
        self,
        q_blocks,
        k_blocks,
        v_blocks,
        topk_indices,
        q_valid_mask,
        kv_valid_mask,
        *,
        scale,
        query_block_chunk,
        mask_mode,
        block_residual=None,
        context=None,
    ):
        if q_blocks.ndim != 5 or topk_indices.ndim != 4:
            raise ValueError("Expected q_blocks [B,H,N,C,D] and topk_indices [B,H,N,K].")
        batch, heads, num_blocks, capacity, head_dim = q_blocks.shape
        if k_blocks.shape != v_blocks.shape or k_blocks.shape[:3] != (
            batch,
            heads,
            num_blocks,
        ):
            raise ValueError("K/V block geometry must match Q block geometry.")
        query_block_chunk = int(query_block_chunk)
        if query_block_chunk <= 0:
            raise ValueError("query_block_chunk must be positive.")
        if q_valid_mask.ndim == 2:
            q_valid_mask = q_valid_mask.unsqueeze(0).expand(batch, -1, -1)

        output_chunks = []
        for start in range(0, num_blocks, query_block_chunk):
            end = min(start + query_block_chunk, num_blocks)
            chunk_indices = topk_indices[:, :, start:end]
            chunk_size = end - start
            q_chunk = q_blocks[:, :, start:end].permute(0, 2, 1, 3, 4).reshape(
                batch * chunk_size, heads, capacity, head_dim
            )
            k_selected = _selected_kv(k_blocks, chunk_indices)
            v_selected = _selected_kv(v_blocks, chunk_indices)
            allowed = _selected_valid_mask(kv_valid_mask, chunk_indices)
            boundary_mode = getattr(context, "boundary_mode", "fixed_padded")
            if boundary_mode == "fixed_padded":
                output = self._attention(
                    q_chunk, k_selected, v_selected, allowed, float(scale), mask_mode
                )
            elif boundary_mode == "compact_ragged":
                output = self._compact_attention(
                    q_chunk, k_selected, v_selected, allowed, float(scale), mask_mode
                )
            else:
                raise ValueError(f"Unsupported BSA boundary mode: {boundary_mode}")
            output = output.reshape(batch, chunk_size, heads, capacity, head_dim).permute(
                0, 2, 1, 3, 4
            )
            if block_residual is not None:
                residual = block_residual[:, :, start:end].unsqueeze(-2)
                output = output + residual
            q_mask = q_valid_mask[:, start:end].reshape(
                batch, 1, chunk_size, capacity, 1
            )
            output_chunks.append(output * q_mask.to(dtype=output.dtype))
        return torch.cat(output_chunks, dim=2)


class SDPAGatherBackend(_GatherBackend):
    def __init__(self):
        super().__init__("sdpa_gather")


class EagerMathBackend(_GatherBackend):
    def __init__(self):
        super().__init__("eager_math")


def create_bsa_backend(name: str) -> BlockSparseAttentionBackend:
    if name == "sdpa_gather":
        return SDPAGatherBackend()
    if name == "eager_math":
        return EagerMathBackend()
    raise ValueError(
        f"Unsupported portable BSA backend {name!r}; expected 'sdpa_gather' or 'eager_math'."
    )


def probe_bsa_backend(
    backend: str,
    *,
    device,
    dtype=torch.float32,
    num_heads: int = 2,
    head_dim: int = 8,
    block_capacity: int = 72,
    mask_mode: str = "additive",
) -> Dict[str, object]:
    """Run a deterministic portable-provider forward/backward reference probe."""
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(20260713)
    shape = (1, int(num_heads), 4, int(block_capacity), int(head_dim))
    base = [
        torch.randn(shape, device=device, dtype=dtype, generator=generator)
        for _ in range(3)
    ]
    valid_mask = torch.ones((4, block_capacity), dtype=torch.bool, device=device)
    valid_mask[-1, block_capacity // 3:] = False
    topk = torch.tensor([0, 3], dtype=torch.long, device=device).reshape(1, 1, 1, 2)
    topk = topk.expand(1, num_heads, 4, 2)

    def run(provider):
        tensors = [tensor.detach().clone().requires_grad_(True) for tensor in base]
        output = provider(
            *tensors,
            topk,
            valid_mask,
            valid_mask,
            scale=1.0 / math.sqrt(head_dim),
            query_block_chunk=2,
            mask_mode=mask_mode,
        )
        loss = output.float().square().mean()
        gradients = torch.autograd.grad(loss, tensors)
        if not torch.isfinite(output).all() or not all(torch.isfinite(item).all() for item in gradients):
            raise RuntimeError(f"BSA backend {backend} produced non-finite forward/backward values.")
        return output.detach(), [item.detach() for item in gradients]

    reference_output, reference_gradients = run(EagerMathBackend())
    output, gradients = run(create_bsa_backend(backend))
    output_error = (output.float() - reference_output.float()).abs().max()
    gradient_error = max(
        (actual.float() - expected.float()).abs().max()
        for actual, expected in zip(gradients, reference_gradients)
    )
    tolerance = 5e-2 if dtype in {torch.float16, torch.bfloat16} else 2e-5
    if output_error > tolerance or gradient_error > tolerance:
        raise RuntimeError(
            f"BSA backend probe mismatch: output={float(output_error):.6g}, "
            f"gradient={float(gradient_error):.6g}, tolerance={tolerance}."
        )
    return {
        "backend": backend,
        "device": str(device),
        "dtype": str(dtype),
        "mask_mode": mask_mode,
        "shape": shape,
        "output_max_abs_error": float(output_error),
        "gradient_max_abs_error": float(gradient_error),
        "forward_backward_finite": True,
    }


class BlockSparseAttention(nn.Module):
    """Coarse router, low-rank block gate, and portable fine attention."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        block_size: Int3 = (4, 3, 6),
        gate_rank: int = 32,
        gate_alpha: Optional[float] = None,
        backend: str = "sdpa_gather",
        use_internal_gate: bool = True,
    ):
        super().__init__()
        model_dim = int(model_dim)
        num_heads = int(num_heads)
        gate_rank = int(gate_rank)
        if model_dim <= 0 or num_heads <= 0 or model_dim % num_heads:
            raise ValueError("model_dim must be positive and divisible by num_heads.")
        if gate_rank <= 0:
            raise ValueError("gate_rank must be positive.")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.block_size = _positive_int3(block_size, "block_size")
        self.gate_rank = gate_rank
        self.gate_alpha = float(gate_rank if gate_alpha is None else gate_alpha)
        self.backend_name = backend
        self.backend = create_bsa_backend(backend)
        self.gate_down = None
        self.gate_up = None
        if use_internal_gate:
            self.gate_down = nn.Linear(model_dim, gate_rank, bias=False)
            self.gate_up = nn.Linear(gate_rank, model_dim, bias=False)
            nn.init.normal_(self.gate_down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.gate_up.weight)
        self.last_runtime_info = None

    @property
    def gate_scale(self) -> float:
        return self.gate_alpha / self.gate_rank

    def gate_is_zero_initialized(self) -> bool:
        if self.gate_up is None:
            raise RuntimeError("This BSA engine uses an external gate.")
        return bool(torch.count_nonzero(self.gate_up.weight.detach()) == 0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_input: torch.Tensor,
        *,
        grid_size: Int3,
        sparsity: float,
        query_block_chunk: int = 4,
        mask_mode: str = "additive",
        count_bias: bool = False,
        dense_attention: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        gate_down: Optional[nn.Module] = None,
        gate_up: Optional[nn.Module] = None,
        context=None,
    ) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise ValueError("Q/K/V must have identical [B,H,L,D] shapes.")
        if attention_input.shape != (
            q.shape[0],
            q.shape[2],
            self.model_dim,
        ):
            raise ValueError(
                "attention_input must have [B,L,model_dim] shape; received "
                f"{tuple(attention_input.shape)}."
            )
        metadata = build_bsa_metadata(
            grid_size, self.block_size, device=q.device
        )
        if q.shape[2] != metadata.sequence_length:
            raise ValueError(
                f"Runtime grid {metadata.grid_size} has {metadata.sequence_length} tokens, "
                f"but attention received L={q.shape[2]}."
            )
        q_blocks = pack_bsa_tokens(q, metadata)
        k_blocks = pack_bsa_tokens(k, metadata)
        v_blocks = pack_bsa_tokens(v, metadata)
        hidden_blocks = pack_bsa_tokens(attention_input, metadata)
        top_k = compute_bsa_top_k(metadata.num_blocks, sparsity)
        coarse_result = bsa_coarse_attention(
            q_blocks,
            k_blocks,
            v_blocks,
            metadata.valid_mask,
            top_k=top_k,
            scale=1.0 / math.sqrt(self.head_dim),
            count_bias=count_bias,
        )
        gate_down = self.gate_down if gate_down is None else gate_down
        gate_up = self.gate_up if gate_up is None else gate_up
        if gate_down is None or gate_up is None:
            raise RuntimeError("BlockSparseAttention requires internal or explicitly supplied gate modules.")
        hidden_coarse = masked_block_mean(hidden_blocks, metadata.valid_mask).to(
            dtype=gate_down.weight.dtype
        )
        gate_hidden = gate_down(hidden_coarse)
        gate = gate_up(gate_hidden) * self.gate_scale
        gate = gate.to(dtype=q.dtype)
        coarse_merged = coarse_result.coarse.permute(0, 2, 1, 3).reshape(
            q.shape[0], metadata.num_blocks, self.model_dim
        )
        residual = (gate * coarse_merged).reshape(
            q.shape[0], metadata.num_blocks, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        if top_k == metadata.num_blocks:
            if dense_attention is None:
                fine = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=False
                )
            else:
                fine = dense_attention(q, k, v)
            fine_blocks = pack_bsa_tokens(fine, metadata)
            fused_blocks = fine_blocks + residual.unsqueeze(-2)
            mask = metadata.valid_mask.reshape(
                1, 1, metadata.num_blocks, metadata.block_capacity, 1
            )
            fused_blocks = fused_blocks * mask.to(dtype=fused_blocks.dtype)
        else:
            fused_blocks = self.backend(
                q_blocks,
                k_blocks,
                v_blocks,
                coarse_result.topk_indices,
                metadata.valid_mask,
                metadata.valid_mask,
                scale=1.0 / math.sqrt(self.head_dim),
                query_block_chunk=query_block_chunk,
                mask_mode=mask_mode,
                block_residual=residual,
                context=context,
            )
        output = unpack_bsa_tokens(fused_blocks, metadata)
        selected_valid_count = metadata.valid_count[coarse_result.topk_indices]
        self.last_runtime_info = {
            "runtime_grid": metadata.grid_size,
            "sequence_length": metadata.sequence_length,
            "block_grid": metadata.block_grid,
            "num_blocks": metadata.num_blocks,
            "top_k": top_k,
            "actual_sparsity": 1.0 - top_k / metadata.num_blocks,
            "padded_tokens": metadata.padded_tokens,
            "padding_ratio": metadata.padding_ratio,
            "gate_input_shape": tuple(hidden_coarse.shape),
            "gate_hidden_shape": tuple(gate_hidden.shape),
            "gate_output_shape": tuple(gate.shape),
            "materialized_token_gate_tensor": False,
            "gate_input_rms": hidden_coarse.detach().float().square().mean().sqrt(),
            "gate_output_rms": gate.detach().float().square().mean().sqrt(),
            "selected_valid_token_ratio": (
                selected_valid_count.detach().float().mean() / metadata.block_capacity
            ),
        }
        return output
