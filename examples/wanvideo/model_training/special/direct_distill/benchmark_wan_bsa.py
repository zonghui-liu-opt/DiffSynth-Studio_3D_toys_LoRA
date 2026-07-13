#!/usr/bin/env python3
"""Independent-process Wan BSA single-layer prefilter benchmark.

This is not an end-to-end speed claim. Final selection must still use the full
four-step DirectDistill forward/backward/optimizer step described in the NOTE.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from diffsynth.core.attention import BlockSparseAttention


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def device_memory(device):
    if device.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    return {
        "max_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / gib,
        "max_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / gib,
        "device_free_gib": free / gib,
        "device_total_gib": total / gib,
    }


def worker(args):
    torch.manual_seed(20260713)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    grid = tuple(args.grid)
    length = math.prod(grid)
    model_dim = args.heads * args.head_dim
    model = BlockSparseAttention(
        model_dim,
        args.heads,
        block_size=tuple(args.block_size),
        gate_rank=args.gate_rank,
        gate_alpha=args.gate_rank,
        backend=args.backend,
    ).to(device=device)
    q = torch.randn(1, args.heads, length, args.head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    hidden = torch.randn(1, length, model_dim, device=device, dtype=dtype, requires_grad=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    def step():
        output = model(
            q,
            k,
            v,
            hidden,
            grid_size=grid,
            sparsity=args.sparsity,
            query_block_chunk=args.query_block_chunk,
            mask_mode=args.mask_mode,
            count_bias=args.count_bias,
            context=SimpleNamespace(boundary_mode=args.boundary_mode),
        )
        output.float().square().mean().backward()
        for tensor in (q, k, v, hidden):
            tensor.grad = None
        model.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    timings = []
    for _ in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
    result = {
        "scope": "single_layer_forward_backward_prefilter",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "grid": grid,
        "block_size": tuple(args.block_size),
        "backend": args.backend,
        "mask_mode": args.mask_mode,
        "boundary_mode": args.boundary_mode,
        "sparsity": args.sparsity,
        "query_block_chunk": args.query_block_chunk,
        "median_step_s": statistics.median(timings),
        "p10_step_s": percentile(timings, 0.10),
        "p90_step_s": percentile(timings, 0.90),
        **device_memory(device),
        "runtime": model.last_runtime_info,
        "end_to_end_validated": False,
        "sdpa_kernel_name": "requires_target_profiler" if args.backend == "sdpa_gather" else "eager_reference",
    }
    result["runtime"] = {
        key: (float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value)
        for key, value in result["runtime"].items()
    }
    print(json.dumps(result, allow_nan=False))


def parent(args):
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = []
    for chunk in args.chunks:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--device", args.device,
            "--dtype", args.dtype,
            "--grid", *(str(value) for value in args.grid),
            "--block-size", *(str(value) for value in args.block_size),
            "--heads", str(args.heads),
            "--head-dim", str(args.head_dim),
            "--gate-rank", str(args.gate_rank),
            "--backend", args.backend,
            "--mask-mode", args.mask_mode,
            "--boundary-mode", args.boundary_mode,
            "--sparsity", str(args.sparsity),
            "--query-block-chunk", str(chunk),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
        ]
        if args.count_bias:
            command.append("--count-bias")
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            candidates.append({"query_block_chunk": chunk, "status": "failed", "error": completed.stderr[-4000:]})
            continue
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        hard_stop = (
            result.get("max_memory_allocated_gib", 0) >= args.memory_hard_stop_gib
            or result.get("max_memory_reserved_gib", 0) >= args.memory_hard_stop_gib
            or result.get("device_free_gib", math.inf) < args.min_device_free_gib
        )
        result["status"] = "memory_gate_failed" if hard_stop else "passed_prefilter"
        candidates.append(result)
    passed = [item for item in candidates if item["status"] == "passed_prefilter"]
    selected = min(passed, key=lambda item: item["median_step_s"]) if passed else None
    manifest = {
        "scope": "single_layer_prefilter_only",
        "candidates": candidates,
        "selected_prefilter": selected,
        "required_next_step": "Run full 4-step forward+backward+optimizer benchmarks in independent processes.",
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if selected is None:
        raise SystemExit(2)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--grid", nargs=3, type=int, default=(41, 15, 26))
    parser.add_argument("--block-size", nargs=3, type=int, default=(4, 3, 6))
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--gate-rank", type=int, default=32)
    parser.add_argument("--backend", choices=("sdpa_gather", "eager_math"), default="sdpa_gather")
    parser.add_argument("--mask-mode", choices=("additive", "bool"), default="additive")
    parser.add_argument("--boundary-mode", choices=("fixed_padded", "compact_ragged"), default="fixed_padded")
    parser.add_argument("--count-bias", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.8)
    parser.add_argument("--query-block-chunk", type=int, default=4)
    parser.add_argument("--chunks", type=lambda value: [int(item) for item in value.split(",")], default=[4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--memory-hard-stop-gib", type=float, default=75)
    parser.add_argument("--min-device-free-gib", type=float, default=4)
    parser.add_argument("--output", default="h100_bsa_single_layer_prefilter.json")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    worker(arguments) if arguments.worker else parent(arguments)
