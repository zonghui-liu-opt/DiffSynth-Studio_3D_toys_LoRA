#!/usr/bin/env python3
"""Compare the figurine360 teacher with the four-step DirectDistill student."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file


def require_local_file(path, label):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_first_video_frame(path):
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        return Image.fromarray(reader.get_data(0)).convert("RGB")
    finally:
        reader.close()


def compose_labeled_panels(panels):
    """Compose equally sized videos horizontally without re-encoding inputs first."""
    if not panels:
        raise ValueError("At least one labeled video panel is required.")
    labels = [label for label, _ in panels]
    videos = [frames for _, frames in panels]
    frame_count = len(videos[0])
    if frame_count == 0 or any(len(frames) != frame_count for frames in videos):
        raise ValueError("All video panels must contain the same non-zero number of frames.")

    output = []
    for frame_index in range(frame_count):
        frames = [video[frame_index].convert("RGB") for video in videos]
        width, height = frames[0].size
        if any(frame.size != (width, height) for frame in frames[1:]):
            raise ValueError("All video panel frame sizes must match.")
        canvas = Image.new("RGB", (width * len(frames), height), "black")
        draw = ImageDraw.Draw(canvas)
        for panel_index, (label, frame) in enumerate(zip(labels, frames)):
            x = panel_index * width
            canvas.paste(frame, (x, 0))
            label_width = min(width, max(130, 12 + 7 * len(label)))
            draw.rectangle((x, 0, x + label_width, 24), fill="black")
            draw.text((x + 6, 5), label, fill="white")
        output.append(canvas)
    return output


def compose_side_by_side(teacher_frames, student_frames):
    return compose_labeled_panels(
        [
            ("Teacher 50-step", teacher_frames),
            ("Student 4-step", student_frames),
        ]
    )


def _resolve_metadata_path(value, base_path):
    path = Path(value)
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def read_metadata_rows(metadata_path):
    metadata_path = require_local_file(metadata_path, "Metadata CSV")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Metadata CSV has no rows: {metadata_path}")
    return metadata_path, rows


def load_sample_from_row(row, metadata_path, dataset_base_path=None):
    metadata_path = Path(metadata_path).expanduser().resolve()
    base_path = Path(dataset_base_path).expanduser().resolve() if dataset_base_path else metadata_path.parent
    input_image = row.get("input_image")
    if input_image:
        image_path = _resolve_metadata_path(input_image, base_path)
        image = Image.open(image_path).convert("RGB")
    else:
        video_path = _resolve_metadata_path(row["video"], base_path)
        image = load_first_video_frame(video_path)
        image_path = video_path
    return row, image, image_path


def load_sample_from_metadata(metadata_path, sample_index=0, dataset_base_path=None):
    metadata_path, rows = read_metadata_rows(metadata_path)
    if sample_index < 0 or sample_index >= len(rows):
        raise IndexError(f"sample_index={sample_index} is outside [0, {len(rows) - 1}].")
    return load_sample_from_row(rows[sample_index], metadata_path, dataset_base_path)


@torch.no_grad()
def decode_latents(pipe, latents, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        latents,
        device=pipe.device,
        tiled=tiled,
        tile_size=tile_size,
        tile_stride=tile_stride,
    )
    frames = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    return frames


def run_pipeline_with_forward_count(pipe, **kwargs):
    original_model_fn = pipe.model_fn
    calls = 0

    def counted_model_fn(*args, **model_kwargs):
        nonlocal calls
        calls += 1
        return original_model_fn(*args, **model_kwargs)

    pipe.model_fn = counted_model_fn
    try:
        output = pipe(**kwargs)
    finally:
        pipe.model_fn = original_model_fn
    return output, calls


def parse_model_path_groups(values):
    groups = []
    for value in values:
        value = value.strip()
        if value.startswith("["):
            parsed = json.loads(value)
            if (
                not isinstance(parsed, list)
                or not parsed
                or not all(isinstance(item, str) for item in parsed)
            ):
                raise ValueError("JSON --model_path must be a non-empty list of shard paths.")
            groups.append(tuple(require_local_file(item, "Model shard") for item in parsed))
        else:
            groups.append(require_local_file(value, "Model path"))
    return groups


def validate_latent_tensor(latents, label, expected_shape=None, expected_dtype=None):
    if not isinstance(latents, torch.Tensor) or latents.ndim != 5:
        shape = None if not isinstance(latents, torch.Tensor) else tuple(latents.shape)
        raise TypeError(f"{label} latents must be a B,C,T,H,W tensor; received {shape}.")
    if latents.shape[2] <= 1:
        raise ValueError(f"{label} latents must contain at least one non-fixed frame.")
    if expected_shape is not None and tuple(latents.shape) != tuple(expected_shape):
        raise ValueError(
            f"{label} latent shape {tuple(latents.shape)} does not match {tuple(expected_shape)}."
        )
    if expected_dtype is not None and latents.dtype != expected_dtype:
        raise ValueError(
            f"{label} latent dtype {latents.dtype} does not match {expected_dtype}."
        )
    if not torch.is_floating_point(latents) or not bool(torch.isfinite(latents).all()):
        raise ValueError(f"{label} latents must be finite floating-point values.")


def build_pipeline(model_paths, tokenizer_path, device="cuda", torch_dtype="bfloat16"):
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]
    groups = parse_model_path_groups(model_paths)
    model_configs = [
        ModelConfig(
            path=str(group) if isinstance(group, Path) else [str(shard) for shard in group],
            skip_download=True,
        )
        for group in groups
    ]
    tokenizer_config = ModelConfig(
        path=str(require_local_file(tokenizer_path, "Tokenizer path")), skip_download=True
    )
    return WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )


def validate_student_schedule(args):
    if args.student_num_inference_steps != 4:
        raise ValueError("Student validation requires exactly 4 denoising steps.")
    if args.student_cfg_scale != 1:
        raise ValueError("Student validation requires cfg_scale=1.")
    if args.student_sigma_shift != 5:
        raise ValueError("Student validation requires sigma_shift=5.")


def build_sample(args, row, input_image, input_source):
    def value(name, cli_value, cast):
        raw = row.get(name)
        return cast(raw) if raw not in (None, "") else cast(cli_value)

    def pair_value(name, fallback):
        raw = row.get(name)
        if raw in (None, ""):
            return tuple(fallback)
        parts = raw.replace("x", ",").split(",")
        parsed = tuple(int(part.strip()) for part in parts if part.strip())
        if len(parsed) != 2:
            raise ValueError(f"Metadata `{name}` must contain two integers.")
        return parsed

    prompt = row.get("prompt") or args.prompt
    negative_prompt = row.get("negative_prompt") or args.negative_prompt
    seed = value("seed", args.seed, int)
    rand_device = row.get("rand_device") or args.rand_device
    height = value("height", args.height, int)
    width = value("width", args.width, int)
    num_frames = value("num_frames", args.num_frames, int)
    teacher_steps = value("teacher_num_inference_steps", args.teacher_num_inference_steps, int)
    teacher_cfg = value("teacher_cfg_scale", args.teacher_cfg_scale, float)
    teacher_shift = value("teacher_sigma_shift", args.teacher_sigma_shift, float)
    tiled = str(row.get("tiled", str(int(args.tiled)))).strip().lower() in ("1", "true", "yes")
    tile_size = pair_value("tile_size", (30, 52))
    tile_stride = pair_value("tile_stride", (15, 26))
    return {
        "input_image": input_image,
        "input_source": input_source,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "rand_device": rand_device,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "teacher_steps": teacher_steps,
        "teacher_cfg": teacher_cfg,
        "teacher_shift": teacher_shift,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,
    }


def load_single_sample(args):
    if args.metadata_path:
        row, input_image, input_source = load_sample_from_metadata(
            args.metadata_path, args.sample_index, args.dataset_base_path
        )
    else:
        row = {}
        input_source = require_local_file(args.input_image, "Input image")
        input_image = Image.open(input_source).convert("RGB")
    return build_sample(args, row, input_image, input_source)


def sample_pipeline_kwargs(sample):
    return {
        "prompt": sample["prompt"],
        "negative_prompt": sample["negative_prompt"],
        "input_image": sample["input_image"],
        "seed": sample["seed"],
        "rand_device": sample["rand_device"],
        "height": sample["height"],
        "width": sample["width"],
        "num_frames": sample["num_frames"],
        "tiled": sample["tiled"],
        "tile_size": sample["tile_size"],
        "tile_stride": sample["tile_stride"],
        "return_latents": True,
        "progress_bar_cmd": lambda values: values,
    }


def load_teacher_pipeline(args):
    figurine_lora = require_local_file(args.figurine360_lora, "figurine360 LoRA")
    direct_distill_lora = require_local_file(args.direct_distill_lora, "DirectDistill LoRA")
    if figurine_lora == direct_distill_lora:
        raise ValueError("figurine360 and DirectDistill LoRA paths must be different files.")

    pipe = build_pipeline(
        args.model_path, args.tokenizer_path, args.device, args.torch_dtype
    )
    figurine_matches = pipe.load_lora(pipe.dit, str(figurine_lora), hotload=False)
    if figurine_matches is None or figurine_matches <= 0:
        raise RuntimeError("figurine360 LoRA matched zero Wan2.2-TI2V-5B modules.")
    return pipe, direct_distill_lora


def fuse_direct_distill_lora(pipe, direct_distill_lora):
    # The second fusion creates base + frozen figurine360 + distilled adapter.
    direct_distill_matches = pipe.load_lora(
        pipe.dit, str(direct_distill_lora), hotload=False
    )
    if direct_distill_matches is None or direct_distill_matches <= 0:
        raise RuntimeError("DirectDistill LoRA matched zero Wan2.2-TI2V-5B modules.")


def run_teacher(pipe, sample):
    teacher_latents = pipe(
        **sample_pipeline_kwargs(sample),
        num_inference_steps=sample["teacher_steps"],
        cfg_scale=sample["teacher_cfg"],
        sigma_shift=sample["teacher_shift"],
    )
    validate_latent_tensor(teacher_latents, "Teacher")
    return teacher_latents


def finish_student_validation(
    pipe, args, sample, teacher_latents, output_dir, teacher_frames=None
):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_latent_tensor(teacher_latents, "Teacher")
    if teacher_frames is None:
        teacher_frames = decode_latents(
            pipe,
            teacher_latents,
            tiled=sample["tiled"],
            tile_size=sample["tile_size"],
            tile_stride=sample["tile_stride"],
        )

    student_latents, student_forward_count = run_pipeline_with_forward_count(
        pipe,
        **sample_pipeline_kwargs(sample),
        num_inference_steps=args.student_num_inference_steps,
        cfg_scale=args.student_cfg_scale,
        sigma_shift=args.student_sigma_shift,
    )
    if student_forward_count != 4:
        raise RuntimeError(
            f"Expected exactly 4 student DiT forwards, observed {student_forward_count}."
        )
    validate_latent_tensor(
        student_latents,
        "Student",
        expected_shape=teacher_latents.shape,
        expected_dtype=teacher_latents.dtype,
    )
    teacher_latents = teacher_latents.to(
        device=student_latents.device, dtype=student_latents.dtype
    )
    if not torch.equal(teacher_latents[:, :, :1], student_latents[:, :, :1]):
        raise RuntimeError("Teacher/student first-frame latents are not bitwise identical.")
    student_frames = decode_latents(
        pipe,
        student_latents,
        tiled=sample["tiled"],
        tile_size=sample["tile_size"],
        tile_stride=sample["tile_stride"],
    )
    paired_frames = compose_side_by_side(teacher_frames, student_frames)

    from diffsynth.utils.data import save_video

    save_video(teacher_frames, str(output_dir / "teacher.mp4"), fps=args.fps, quality=5)
    save_video(student_frames, str(output_dir / "student.mp4"), fps=args.fps, quality=5)
    save_video(paired_frames, str(output_dir / "teacher_vs_student.mp4"), fps=args.fps, quality=5)
    save_file({"latents": teacher_latents.detach().cpu().contiguous()}, output_dir / "teacher_latents.safetensors")
    save_file({"latents": student_latents.detach().cpu().contiguous()}, output_dir / "student_latents.safetensors")

    mse = torch.nn.functional.mse_loss(
        student_latents[:, :, 1:].float(), teacher_latents[:, :, 1:].float()
    ).item()
    if not math.isfinite(mse):
        raise RuntimeError("Teacher/student non-first-frame latent MSE is not finite.")
    report = {
        "input_source": str(sample["input_source"]),
        "prompt": sample["prompt"],
        "seed": sample["seed"],
        "rand_device": sample["rand_device"],
        "height": sample["height"],
        "width": sample["width"],
        "num_frames": sample["num_frames"],
        "teacher": {
            "steps": sample["teacher_steps"],
            "cfg_scale": sample["teacher_cfg"],
            "sigma_shift": sample["teacher_shift"],
        },
        "student": {
            "steps": 4,
            "cfg_scale": 1,
            "sigma_shift": 5,
            "model_forward_count": student_forward_count,
        },
        "latent_shape": list(student_latents.shape),
        "first_frame_bitwise_equal": True,
        "non_first_frame_latent_mse": mse,
        "visual_acceptance": "待 H100 人工检查 360°、身份一致性、周期闪烁与过曝",
    }
    with (output_dir / "validation.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return report


def validate(args):
    validate_student_schedule(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = load_single_sample(args)
    pipe, direct_distill_lora = load_teacher_pipeline(args)
    teacher_latents = run_teacher(pipe, sample)
    teacher_frames = decode_latents(
        pipe,
        teacher_latents,
        tiled=sample["tiled"],
        tile_size=sample["tile_size"],
        tile_stride=sample["tile_stride"],
    )
    fuse_direct_distill_lora(pipe, direct_distill_lora)
    return finish_student_validation(
        pipe, args, sample, teacher_latents, output_dir, teacher_frames=teacher_frames
    )


def select_batch_indices(row_count, start, end, output_dir, skip_existing=True):
    start = 0 if start is None else start
    end = row_count if end is None else end
    if start < 0 or end < 0 or start >= end or end > row_count:
        raise ValueError(
            f"Batch range [{start}, {end}) is invalid for metadata with {row_count} rows."
        )
    requested = list(range(start, end))
    if not skip_existing:
        return requested, []
    skipped = [
        index
        for index in requested
        if (Path(output_dir) / f"sample-{index}" / "validation.json").is_file()
    ]
    skipped_set = set(skipped)
    return [index for index in requested if index not in skipped_set], skipped


def validate_batch(args):
    validate_student_schedule(args)
    if not args.metadata_path:
        raise ValueError("Batch validation requires --metadata_path.")

    metadata_path, rows = read_metadata_rows(args.metadata_path)
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    indices, skipped = select_batch_indices(
        len(rows),
        args.batch_start,
        args.batch_end,
        output_root,
        skip_existing=args.skip_existing,
    )
    summary = {
        "metadata_path": str(metadata_path),
        "batch_start": 0 if args.batch_start is None else args.batch_start,
        "batch_end": len(rows) if args.batch_end is None else args.batch_end,
        "selected": len(indices),
        "skipped": skipped,
        "completed": [],
        "model_load_count": 0,
        "figurine_lora_fusion_count": 0,
        "direct_distill_lora_fusion_count": 0,
    }
    if not indices:
        with (output_root / "batch_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        return summary

    pipe, direct_distill_lora = load_teacher_pipeline(args)
    summary["model_load_count"] = 1
    summary["figurine_lora_fusion_count"] = 1

    # Phase 1: all teachers must run before the DirectDistill LoRA is fused.
    for index in indices:
        print(f"[teacher] sample {index}", flush=True)
        row, image, source = load_sample_from_row(
            rows[index], metadata_path, args.dataset_base_path
        )
        sample = build_sample(args, row, image, source)
        teacher_latents = run_teacher(pipe, sample)
        sample_dir = output_root / f"sample-{index}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            {"latents": teacher_latents.detach().cpu().contiguous()},
            sample_dir / "teacher_latents.safetensors",
        )
        del teacher_latents, sample, image

    # Phase 2: fuse once, then run every student without rebuilding the pipeline.
    fuse_direct_distill_lora(pipe, direct_distill_lora)
    summary["direct_distill_lora_fusion_count"] = 1
    for index in indices:
        print(f"[student] sample {index}", flush=True)
        row, image, source = load_sample_from_row(
            rows[index], metadata_path, args.dataset_base_path
        )
        sample = build_sample(args, row, image, source)
        sample_dir = output_root / f"sample-{index}"
        teacher_tensors = load_file(
            sample_dir / "teacher_latents.safetensors", device="cpu"
        )
        if set(teacher_tensors) != {"latents"}:
            raise ValueError(
                f"Teacher latent file for sample {index} must contain only `latents`."
            )
        finish_student_validation(
            pipe, args, sample, teacher_tensors["latents"], sample_dir
        )
        summary["completed"].append(index)
        with (output_root / "batch_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
        del teacher_tensors, sample, image
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        action="append",
        required=True,
        help="Repeat for separate models; pass one JSON list for all shards of a model.",
    )
    parser.add_argument("--tokenizer_path", required=True, help="Local tokenizer directory.")
    parser.add_argument("--figurine360_lora", required=True)
    parser.add_argument("--direct_distill_lora", required=True)
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
        help="Skip batch samples whose validation.json already exists.",
    )
    parser.add_argument("--prompt", default="a studio figurine rotating through a complete 360 degree turn")
    parser.add_argument("--negative_prompt", default="overexposed, flicker, incomplete rotation, deformation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rand_device", default="cpu")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--teacher_num_inference_steps", type=int, default=50)
    parser.add_argument("--teacher_cfg_scale", type=float, default=5.0)
    parser.add_argument("--teacher_sigma_shift", type=float, default=5.0)
    parser.add_argument("--student_num_inference_steps", type=int, default=4)
    parser.add_argument("--student_cfg_scale", type=float, default=1.0)
    parser.add_argument("--student_sigma_shift", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch_dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
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
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
