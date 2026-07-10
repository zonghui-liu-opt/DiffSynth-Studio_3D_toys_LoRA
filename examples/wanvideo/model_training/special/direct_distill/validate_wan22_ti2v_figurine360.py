#!/usr/bin/env python3
"""Compare the figurine360 teacher with the four-step DirectDistill student."""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from safetensors.torch import save_file


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


def compose_side_by_side(teacher_frames, student_frames):
    if len(teacher_frames) != len(student_frames) or not teacher_frames:
        raise ValueError("Teacher and student must contain the same non-zero number of frames.")
    output = []
    for teacher, student in zip(teacher_frames, student_frames):
        teacher = teacher.convert("RGB")
        student = student.convert("RGB")
        if teacher.size != student.size:
            raise ValueError("Teacher and student frame sizes must match.")
        width, height = teacher.size
        canvas = Image.new("RGB", (width * 2, height), "black")
        canvas.paste(teacher, (0, 0))
        canvas.paste(student, (width, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 130, 24), fill="black")
        draw.rectangle((width, 0, width + 130, 24), fill="black")
        draw.text((6, 5), "Teacher 50-step", fill="white")
        draw.text((width + 6, 5), "Student 4-step", fill="white")
        output.append(canvas)
    return output


def _resolve_metadata_path(value, base_path):
    path = Path(value)
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def load_sample_from_metadata(metadata_path, sample_index=0, dataset_base_path=None):
    metadata_path = require_local_file(metadata_path, "Metadata CSV")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Metadata CSV has no rows: {metadata_path}")
    if sample_index < 0 or sample_index >= len(rows):
        raise IndexError(f"sample_index={sample_index} is outside [0, {len(rows) - 1}].")
    row = rows[sample_index]
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


def validate(args):
    if args.student_num_inference_steps != 4:
        raise ValueError("Student validation requires exactly 4 denoising steps.")
    if args.student_cfg_scale != 1:
        raise ValueError("Student validation requires cfg_scale=1.")
    if args.student_sigma_shift != 5:
        raise ValueError("Student validation requires sigma_shift=5.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.metadata_path:
        row, input_image, input_source = load_sample_from_metadata(
            args.metadata_path, args.sample_index, args.dataset_base_path
        )
    else:
        row = {}
        input_source = require_local_file(args.input_image, "Input image")
        input_image = Image.open(input_source).convert("RGB")

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
    shared = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "input_image": input_image,
        "seed": seed,
        "rand_device": rand_device,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "return_latents": True,
        "progress_bar_cmd": lambda values: values,
    }

    teacher_latents = pipe(
        **shared,
        num_inference_steps=teacher_steps,
        cfg_scale=teacher_cfg,
        sigma_shift=teacher_shift,
    )
    validate_latent_tensor(teacher_latents, "Teacher")
    teacher_frames = decode_latents(
        pipe, teacher_latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
    )

    # The second fusion creates base + frozen figurine360 + distilled adapter.
    direct_distill_matches = pipe.load_lora(
        pipe.dit, str(direct_distill_lora), hotload=False
    )
    if direct_distill_matches is None or direct_distill_matches <= 0:
        raise RuntimeError("DirectDistill LoRA matched zero Wan2.2-TI2V-5B modules.")
    student_latents, student_forward_count = run_pipeline_with_forward_count(
        pipe,
        **shared,
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
    if not torch.equal(teacher_latents[:, :, :1], student_latents[:, :, :1]):
        raise RuntimeError("Teacher/student first-frame latents are not bitwise identical.")
    student_frames = decode_latents(
        pipe, student_latents, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride
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
        "input_source": str(input_source),
        "prompt": prompt,
        "seed": seed,
        "rand_device": rand_device,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "teacher": {"steps": teacher_steps, "cfg_scale": teacher_cfg, "sigma_shift": teacher_shift},
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
        report = validate(parser.parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
