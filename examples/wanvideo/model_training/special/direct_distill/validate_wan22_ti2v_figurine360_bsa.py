#!/usr/bin/env python3
"""Validate teacher, dense warm-start, joint-dense, and joint-sparse students."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from safetensors.torch import save_file

from validate_wan22_ti2v_figurine360 import (
    build_pipeline,
    decode_latents,
    load_single_sample,
    require_local_file,
    run_pipeline_with_forward_count,
    sample_pipeline_kwargs,
    validate_latent_tensor,
)
from diffsynth.models.wan_video_bsa import (
    BSAContext,
    WanBSAConfig,
    collect_wan_bsa_runtime_info,
    inject_wan_bsa,
    load_wan_bsa_adapter,
    resolve_bsa_checkpoint,
)
from diffsynth.utils.data import save_video


def release_pipeline(pipe):
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_and_figurine(args):
    pipe = build_pipeline(args.model_path, args.tokenizer_path, args.device, args.torch_dtype)
    matches = pipe.load_lora(
        pipe.dit, str(require_local_file(args.figurine360_lora, "figurine360 LoRA")), hotload=False
    )
    if matches is None or matches <= 0:
        raise RuntimeError("figurine360 LoRA matched zero modules.")
    return pipe


def fuse_lora(pipe, path, label):
    matches = pipe.load_lora(pipe.dit, str(require_local_file(path, label)), hotload=False)
    if matches is None or matches <= 0:
        raise RuntimeError(f"{label} matched zero modules.")


def run_four_step(pipe, sample, bsa_context=None):
    latents, count = run_pipeline_with_forward_count(
        pipe,
        **sample_pipeline_kwargs(sample),
        num_inference_steps=4,
        cfg_scale=1,
        sigma_shift=5,
        bsa_context=bsa_context,
    )
    if count != 4:
        raise RuntimeError(f"Expected 4 student forwards, observed {count}.")
    validate_latent_tensor(latents, "Student")
    return latents, count


def save_result(pipe, sample, output_dir, name, latents):
    save_file({"latents": latents.detach().cpu().contiguous()}, output_dir / f"{name}.safetensors")
    frames = decode_latents(
        pipe,
        latents,
        tiled=sample["tiled"],
        tile_size=sample["tile_size"],
        tile_stride=sample["tile_stride"],
    )
    save_video(frames, str(output_dir / f"{name}.mp4"), fps=15, quality=5)


def config_from_manifest(manifest):
    return WanBSAConfig(
        block_size=tuple(manifest["block_size"]),
        target_sparsity=float(manifest["target_sparsity"]),
        backend=manifest.get("backend", "sdpa_gather"),
        query_block_chunk=int(manifest.get("query_block_chunk", 4)),
        mask_mode=manifest.get("mask_mode", "additive"),
        boundary_mode=manifest.get("boundary_mode", "fixed_padded"),
        count_bias=bool(manifest.get("count_bias", False)),
        gate_type=manifest["gate_type"],
        gate_granularity=manifest["gate_granularity"],
        gate_rank=int(manifest["gate_rank"]),
        gate_alpha=float(manifest["gate_alpha"]),
    )


def latent_mse(prediction, target):
    value = torch.nn.functional.mse_loss(
        prediction[:, :, 1:].float().cpu(), target[:, :, 1:].float().cpu()
    ).item()
    if not math.isfinite(value):
        raise RuntimeError("Validation latent MSE is non-finite.")
    return value


@torch.no_grad()
def validate(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    sample = load_single_sample(args)

    teacher_pipe = load_base_and_figurine(args)
    teacher = teacher_pipe(
        **sample_pipeline_kwargs(sample),
        num_inference_steps=sample["teacher_steps"],
        cfg_scale=sample["teacher_cfg"],
        sigma_shift=sample["teacher_shift"],
    )
    validate_latent_tensor(teacher, "Teacher")
    save_result(teacher_pipe, sample, output_dir, "teacher", teacher)
    release_pipeline(teacher_pipe)

    dense_pipe = load_base_and_figurine(args)
    fuse_lora(dense_pipe, args.dense_warmstart_lora, "Dense DirectDistill warm-start")
    dense_warmstart, dense_count = run_four_step(dense_pipe, sample)
    save_result(dense_pipe, sample, output_dir, "dense_warmstart", dense_warmstart)
    release_pipeline(dense_pipe)

    checkpoint = resolve_bsa_checkpoint(args.bsa_checkpoint)
    config = config_from_manifest(checkpoint["manifest"])
    joint_pipe = load_base_and_figurine(args)
    fuse_lora(joint_pipe, checkpoint["direct_distill_lora"], "Joint DirectDistill LoRA")
    summary = inject_wan_bsa(
        joint_pipe.dit,
        config,
        expected_layers=30,
        figurine_lora_path=args.figurine360_lora,
        direct_distill_warmstart_path=checkpoint["direct_distill_lora"],
    )
    load_wan_bsa_adapter(joint_pipe.dit, checkpoint["bsa_adapter"])
    joint_dense, joint_dense_count = run_four_step(
        joint_pipe, sample, BSAContext.from_config(config, sparsity=0.0)
    )
    save_result(joint_pipe, sample, output_dir, "joint_dense", joint_dense)
    joint_sparse, joint_sparse_count = run_four_step(
        joint_pipe, sample, BSAContext.from_config(config, sparsity=config.target_sparsity)
    )
    save_result(joint_pipe, sample, output_dir, "joint_sparse", joint_sparse)
    runtime = collect_wan_bsa_runtime_info(joint_pipe.dit)
    runtime = {
        key: (float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value)
        for key, value in runtime.items()
    }

    first_frame_equal = all(
        torch.equal(teacher[:, :, :1].cpu(), item[:, :, :1].cpu())
        for item in (dense_warmstart, joint_dense, joint_sparse)
    )
    if not first_frame_equal:
        raise RuntimeError("Teacher/student first-frame latents are not bitwise identical.")
    report = {
        "sample": {key: sample[key] for key in ("prompt", "seed", "height", "width", "num_frames")},
        "forward_counts": {
            "dense_warmstart": dense_count,
            "joint_dense": joint_dense_count,
            "joint_sparse": joint_sparse_count,
        },
        "first_frame_bitwise_equal": True,
        "endpoint_latent_mse": {
            "dense_warmstart_vs_teacher": latent_mse(dense_warmstart, teacher),
            "joint_dense_vs_teacher": latent_mse(joint_dense, teacher),
            "joint_sparse_vs_teacher": latent_mse(joint_sparse, teacher),
            "joint_sparse_vs_joint_dense": latent_mse(joint_sparse, joint_dense),
        },
        "bsa_summary": summary,
        "bsa_runtime": runtime,
        "quality_status": "待H100人工检查身份、360度运动、尾帧和右边缘质量",
    }
    (output_dir / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", action="append", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--figurine360_lora", required=True)
    parser.add_argument("--dense_warmstart_lora", required=True)
    parser.add_argument("--bsa_checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata_path")
    source.add_argument("--input_image")
    parser.add_argument("--dataset_base_path", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--prompt", default="a studio figurine rotating through a complete 360 degree turn")
    parser.add_argument("--negative_prompt", default="overexposed, flicker, incomplete rotation, deformation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rand_device", default="cpu")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--teacher_num_inference_steps", type=int, default=50)
    parser.add_argument("--teacher_cfg_scale", type=float, default=5.0)
    parser.add_argument("--teacher_sigma_shift", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main():
    parser = build_parser()
    try:
        args = parser.parse_args()
        report = validate(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
