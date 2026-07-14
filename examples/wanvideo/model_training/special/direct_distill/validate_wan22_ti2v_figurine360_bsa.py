#!/usr/bin/env python3
"""Validate teacher, dense warm-start, joint-dense, and joint-sparse students."""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from validate_wan22_ti2v_figurine360 import (
    build_sample,
    build_pipeline,
    compose_labeled_panels,
    decode_latents,
    load_sample_from_row,
    load_single_sample,
    read_metadata_rows,
    require_local_file,
    run_teacher,
    run_pipeline_with_forward_count,
    sample_pipeline_kwargs,
    select_batch_indices,
    validate_latent_tensor,
)
from diffsynth.models.wan_video_bsa import (
    BSAContext,
    WanBSAConfig,
    collect_wan_bsa_runtime_info,
    inject_wan_bsa,
    load_wan_bsa_adapter,
    resolve_bsa_checkpoint,
    validate_wan_bsa_provenance,
)
from diffsynth.utils.data import save_video


COMPARISON_VIDEO_NAME = "teacher_vs_student_vs_student_bsa.mp4"


def release_device_memory():
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


def save_latents(output_dir, name, latents):
    save_file({"latents": latents.detach().cpu().contiguous()}, output_dir / f"{name}.safetensors")


def load_latents(output_dir, name, label):
    path = output_dir / f"{name}.safetensors"
    tensors = load_file(path, device="cpu")
    if set(tensors) != {"latents"}:
        raise ValueError(f"{label} latent file must contain only `latents`: {path}")
    latents = tensors["latents"]
    validate_latent_tensor(latents, label)
    return latents


def render_result(pipe, sample, output_dir, name, latents, fps):
    frames = decode_latents(
        pipe,
        latents,
        tiled=sample["tiled"],
        tile_size=sample["tile_size"],
        tile_stride=sample["tile_stride"],
    )
    save_video(frames, str(output_dir / f"{name}.mp4"), fps=fps, quality=5)
    return frames


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


def _manifest_sha256(manifest):
    payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_validation_checkpoint(args):
    checkpoint = resolve_bsa_checkpoint(args.bsa_checkpoint)
    validate_wan_bsa_provenance(
        checkpoint["manifest"],
        figurine_lora_path=args.figurine360_lora,
        direct_distill_warmstart_path=args.dense_warmstart_lora,
    )
    checkpoint = dict(checkpoint)
    checkpoint["manifest_sha256"] = _manifest_sha256(checkpoint["manifest"])
    return checkpoint, config_from_manifest(checkpoint["manifest"])


def latent_mse(prediction, target):
    value = torch.nn.functional.mse_loss(
        prediction[:, :, 1:].float().cpu(), target[:, :, 1:].float().cpu()
    ).item()
    if not math.isfinite(value):
        raise RuntimeError("Validation latent MSE is non-finite.")
    return value


def _to_cpu_latents(latents, label, expected=None):
    validate_latent_tensor(
        latents,
        label,
        expected_shape=None if expected is None else expected.shape,
        expected_dtype=None if expected is None else expected.dtype,
    )
    return latents.detach().cpu().contiguous()


def _runtime_to_json(runtime):
    return {
        key: (float(value.detach().cpu()) if isinstance(value, torch.Tensor) else value)
        for key, value in runtime.items()
    }


def _write_sample_outputs(
    pipe,
    args,
    sample,
    output_dir,
    teacher,
    dense_warmstart,
    joint_dense,
    joint_sparse,
    dense_count,
    joint_dense_count,
    joint_sparse_count,
    config,
    checkpoint_manifest_sha256,
    bsa_summary,
    bsa_runtime,
):
    first_frame_equal = all(
        torch.equal(teacher[:, :, :1], item[:, :, :1])
        for item in (dense_warmstart, joint_dense, joint_sparse)
    )
    if not first_frame_equal:
        raise RuntimeError("Teacher/student first-frame latents are not bitwise identical.")

    teacher_frames = render_result(
        pipe, sample, output_dir, "teacher", teacher, args.fps
    )
    dense_frames = render_result(
        pipe,
        sample,
        output_dir,
        "dense_warmstart",
        dense_warmstart,
        args.fps,
    )
    render_result(
        pipe, sample, output_dir, "joint_dense", joint_dense, args.fps
    )
    sparse_frames = render_result(
        pipe, sample, output_dir, "joint_sparse", joint_sparse, args.fps
    )
    comparison_frames = compose_labeled_panels(
        [
            (f"Teacher {sample['teacher_steps']}-step", teacher_frames),
            ("Student 4-step", dense_frames),
            ("Student BSA 4-step", sparse_frames),
        ]
    )
    save_video(
        comparison_frames,
        str(output_dir / COMPARISON_VIDEO_NAME),
        fps=args.fps,
        quality=5,
    )

    report = {
        "sample": {
            key: sample[key]
            for key in ("prompt", "seed", "height", "width", "num_frames")
        },
        "teacher": {
            "steps": sample["teacher_steps"],
            "cfg_scale": sample["teacher_cfg"],
            "sigma_shift": sample["teacher_shift"],
        },
        "comparison": {
            "teacher": "teacher",
            "student": "dense_warmstart",
            "student_bsa": "joint_sparse",
            "student_bsa_sparsity": config.target_sparsity,
            "video": COMPARISON_VIDEO_NAME,
        },
        "bsa_checkpoint": str(Path(args.bsa_checkpoint).expanduser().resolve()),
        "bsa_checkpoint_manifest_sha256": checkpoint_manifest_sha256,
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
        "bsa_summary": bsa_summary,
        "bsa_runtime": bsa_runtime,
        "quality_status": "待H100人工检查身份、360度运动、尾帧和右边缘质量",
    }
    # Written last: batch --skip-existing treats this file as the completion marker.
    _write_json_atomic(output_dir / "validation.json", report)
    return report


@torch.no_grad()
def _validate_jobs(
    args, job_factory, on_complete=None, checkpoint=None, config=None
):
    if checkpoint is None or config is None:
        checkpoint, config = _resolve_validation_checkpoint(args)
    dense_counts = {}

    # Teacher and the original dense DirectDistill student share one base pipeline.
    dense_pipe = load_base_and_figurine(args)
    try:
        for key, sample, output_dir in job_factory():
            print(f"[teacher] sample {key}", flush=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            teacher = _to_cpu_latents(run_teacher(dense_pipe, sample), "Teacher")
            save_latents(output_dir, "teacher", teacher)

        fuse_lora(dense_pipe, args.dense_warmstart_lora, "Dense DirectDistill warm-start")
        for key, sample, output_dir in job_factory():
            print(f"[student] sample {key}", flush=True)
            teacher = load_latents(output_dir, "teacher", "Teacher")
            dense_warmstart, dense_count = run_four_step(dense_pipe, sample)
            dense_warmstart = _to_cpu_latents(
                dense_warmstart, "Dense DirectDistill student", expected=teacher
            )
            if not torch.equal(teacher[:, :, :1], dense_warmstart[:, :, :1]):
                raise RuntimeError(
                    "Teacher/dense DirectDistill first-frame latents are not bitwise identical."
                )
            save_latents(output_dir, "dense_warmstart", dense_warmstart)
            dense_counts[key] = dense_count
    finally:
        del dense_pipe
        release_device_memory()

    # The checkpoint contains a different, jointly trained DirectDistill LoRA, so it
    # needs a fresh base pipeline. Joint-dense and joint-sparse reuse that pipeline.
    joint_pipe = load_base_and_figurine(args)
    try:
        fuse_lora(joint_pipe, checkpoint["direct_distill_lora"], "Joint DirectDistill LoRA")
        bsa_summary = inject_wan_bsa(
            joint_pipe.dit,
            config,
            expected_layers=30,
            figurine_lora_path=args.figurine360_lora,
            direct_distill_warmstart_path=args.dense_warmstart_lora,
        )
        load_wan_bsa_adapter(joint_pipe.dit, checkpoint["bsa_adapter"])
        reports = {}
        for key, sample, output_dir in job_factory():
            print(f"[student_bsa] sample {key}", flush=True)
            teacher = load_latents(output_dir, "teacher", "Teacher")
            dense_warmstart = load_latents(
                output_dir, "dense_warmstart", "Dense DirectDistill student"
            )
            joint_dense, joint_dense_count = run_four_step(
                joint_pipe, sample, BSAContext.from_config(config, sparsity=0.0)
            )
            joint_dense = _to_cpu_latents(
                joint_dense, "Joint-dense BSA student", expected=teacher
            )
            save_latents(output_dir, "joint_dense", joint_dense)
            joint_sparse, joint_sparse_count = run_four_step(
                joint_pipe,
                sample,
                BSAContext.from_config(config, sparsity=config.target_sparsity),
            )
            joint_sparse = _to_cpu_latents(
                joint_sparse, "Joint-sparse BSA student", expected=teacher
            )
            save_latents(output_dir, "joint_sparse", joint_sparse)
            runtime = _runtime_to_json(collect_wan_bsa_runtime_info(joint_pipe.dit))
            report = _write_sample_outputs(
                joint_pipe,
                args,
                sample,
                output_dir,
                teacher,
                dense_warmstart,
                joint_dense,
                joint_sparse,
                dense_counts[key],
                joint_dense_count,
                joint_sparse_count,
                config,
                checkpoint["manifest_sha256"],
                bsa_summary,
                runtime,
            )
            reports[key] = report
            if on_complete is not None:
                on_complete(key, report)
    finally:
        del joint_pipe
        release_device_memory()
    return reports


def validate(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation.json").unlink(missing_ok=True)
    sample = load_single_sample(args)

    def jobs():
        yield 0, sample, output_dir

    return _validate_jobs(args, jobs)[0]


def _write_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_batch_summary(output_root, summary):
    _write_json_atomic(output_root / "batch_summary.json", summary)


def _is_complete_bsa_sample(
    output_root, index, bsa_checkpoint, checkpoint_manifest_sha256
):
    sample_dir = output_root / f"sample-{index}"
    report_path = sample_dir / "validation.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    comparison = report.get("comparison") if isinstance(report, dict) else None
    report_checkpoint = report.get("bsa_checkpoint") if isinstance(report, dict) else None
    report_manifest_sha256 = (
        report.get("bsa_checkpoint_manifest_sha256")
        if isinstance(report, dict)
        else None
    )
    return (
        isinstance(comparison, dict)
        and comparison.get("video") == COMPARISON_VIDEO_NAME
        and (sample_dir / COMPARISON_VIDEO_NAME).is_file()
        and isinstance(report_checkpoint, str)
        and Path(report_checkpoint).expanduser().resolve()
        == Path(bsa_checkpoint).expanduser().resolve()
        and report_manifest_sha256 == checkpoint_manifest_sha256
    )


def validate_batch(args):
    if not args.metadata_path:
        raise ValueError("Batch validation requires --metadata_path.")
    metadata_path, rows = read_metadata_rows(args.metadata_path)
    checkpoint, config = _resolve_validation_checkpoint(args)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requested, _ = select_batch_indices(
        len(rows),
        args.batch_start,
        args.batch_end,
        output_root,
        skip_existing=False,
    )
    skipped = (
        [
            index
            for index in requested
            if _is_complete_bsa_sample(
                output_root,
                index,
                args.bsa_checkpoint,
                checkpoint["manifest_sha256"],
            )
        ]
        if args.skip_existing
        else []
    )
    skipped_set = set(skipped)
    indices = [index for index in requested if index not in skipped_set]
    summary = {
        "metadata_path": str(metadata_path),
        "batch_start": 0 if args.batch_start is None else args.batch_start,
        "batch_end": len(rows) if args.batch_end is None else args.batch_end,
        "selected": len(indices),
        "skipped": skipped,
        "completed": [],
        "model_load_count": 0 if not indices else 2,
        "figurine_lora_fusion_count": 0 if not indices else 2,
        "dense_warmstart_lora_fusion_count": 0 if not indices else 1,
        "joint_direct_distill_lora_fusion_count": 0 if not indices else 1,
        "bsa_adapter_load_count": 0 if not indices else 1,
    }
    _write_batch_summary(output_root, summary)
    if not indices:
        return summary
    # A forced rerun must not leave a stale completion marker if it is interrupted.
    for index in indices:
        (output_root / f"sample-{index}" / "validation.json").unlink(missing_ok=True)

    def jobs():
        for index in indices:
            row, image, source = load_sample_from_row(
                rows[index], metadata_path, args.dataset_base_path
            )
            yield index, build_sample(args, row, image, source), output_root / f"sample-{index}"

    def completed(index, _report):
        summary["completed"].append(index)
        _write_batch_summary(output_root, summary)

    _validate_jobs(
        args,
        jobs,
        on_complete=completed,
        checkpoint=checkpoint,
        config=config,
    )
    return summary


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
    parser.add_argument(
        "--batch_start",
        type=int,
        default=None,
        help="Inclusive metadata index for single-process batch validation.",
    )
    parser.add_argument(
        "--batch_end",
        type=int,
        default=None,
        help="Exclusive metadata index for single-process batch validation.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip batch samples whose validation.json completion marker exists.",
    )
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
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--output_dir", required=True)
    return parser


def main():
    parser = build_parser()
    try:
        args = parser.parse_args()
        batch_mode = args.batch_start is not None or args.batch_end is not None
        report = validate_batch(args) if batch_mode else validate(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
