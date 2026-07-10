#!/usr/bin/env python3
"""Prepare Wan2.2-TI2V-5B teacher latents for Figurine360 DirectDistill.

The command only accepts local model/tokenizer/LoRA paths. Dataset bookkeeping,
artifact validation, and resume behavior are kept independent from model loading
so they can be tested without Wan weights or a CUDA device.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

DIRECT_DISTILL_FIELDS = (
    "video",
    "input_image",
    "prompt",
    "negative_prompt",
    "teacher_latent",
    "seed",
    "rand_device",
    "num_inference_steps",
    "cfg_scale",
    "sigma_shift",
    "teacher_num_inference_steps",
    "teacher_cfg_scale",
    "teacher_sigma_shift",
    "height",
    "width",
    "num_frames",
    "tiled",
    "tile_size",
    "tile_stride",
    "object_id",
    "sample_id",
    "teacher_fingerprint",
    "source_fingerprint",
    "split",
)


@dataclass(frozen=True)
class PreparationConfig:
    base_path: Path
    output_path: Path
    seeds: tuple[int, ...] = (1,)
    validation_seeds: tuple[int, ...] = ()
    height: int = 480
    width: int = 832
    num_frames: int = 49
    rand_device: str = "cpu"
    student_num_inference_steps: int = 4
    student_cfg_scale: float = 1.0
    student_sigma_shift: float = 5.0
    teacher_num_inference_steps: int = 50
    teacher_cfg_scale: float = 5.0
    teacher_sigma_shift: float = 5.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    validation_fraction: float = 0.1
    split_salt: str = "wan22-ti2v5b-figurine360-v1"
    teacher_provenance: str = "unspecified-local-artifacts"
    max_samples: Optional[int] = None
    verify_determinism: bool = False
    tiled: bool = False
    tile_size: tuple[int, int] = (30, 52)
    tile_stride: tuple[int, int] = (15, 26)

    def __post_init__(self):
        if not self.seeds:
            raise ValueError("At least one seed is required.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be unique.")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("Validation seeds must be unique.")
        if self.validation_seeds and set(self.seeds) & set(self.validation_seeds):
            raise ValueError("Training and held-out validation seeds must be disjoint.")
        if self.height <= 0 or self.width <= 0 or self.num_frames <= 0:
            raise ValueError("height, width, and num_frames must be positive.")
        if self.num_frames % 4 != 1:
            raise ValueError("Wan video num_frames must satisfy num_frames % 4 == 1.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided.")
        if self.student_num_inference_steps != 4:
            raise ValueError("This task requires exactly four student inference steps.")
        if self.student_cfg_scale != 1.0:
            raise ValueError("This task requires student cfg_scale=1.")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = ""
    latents: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ImageCacheResult:
    image: Image.Image
    reused: bool
    invalid_reason: str = ""
    quarantined_path: Optional[Path] = None


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_video_key(video: str) -> str:
    value = str(video).strip().replace("\\", "/")
    if not value:
        raise ValueError("video path is empty")
    return posixpath.normpath(value)


def stable_object_id(video: str, source_fingerprint: str = "") -> str:
    payload = f"{normalize_video_key(video)}:{source_fingerprint}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"obj_{digest}"


def stable_sample_id(
    video: str, prompt: str, seed: int, teacher_fingerprint: str = ""
) -> str:
    payload = {
        "video": normalize_video_key(video),
        "prompt": str(prompt).strip(),
        "seed": int(seed),
        "teacher_fingerprint": str(teacher_fingerprint),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]
    return f"sample_{digest}"


def teacher_generation_fingerprint(config: PreparationConfig) -> str:
    payload = {
        "height": config.height,
        "width": config.width,
        "num_frames": config.num_frames,
        "rand_device": config.rand_device,
        "teacher_num_inference_steps": config.teacher_num_inference_steps,
        "teacher_cfg_scale": config.teacher_cfg_scale,
        "teacher_sigma_shift": config.teacher_sigma_shift,
        "negative_prompt": config.negative_prompt,
        "tiled": config.tiled,
        "tile_size": config.tile_size,
        "tile_stride": config.tile_stride,
        "teacher_provenance": config.teacher_provenance,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _local_path_stat_records(path: Path) -> list[dict]:
    path = path.resolve()
    candidates = [path] if path.is_file() else sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    if not candidates:
        raise ValueError(f"Local artifact contains no files: {path}")
    records = []
    for candidate in candidates:
        stat = candidate.stat()
        records.append({
            "path": str(candidate),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return records


def source_file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_local_teacher_provenance(
    model_paths: Sequence[Path | Sequence[Path]],
    tokenizer_path: Path,
    figurine_lora_path: Path,
    torch_dtype: str,
) -> str:
    model_groups = []
    for group in model_paths:
        paths = (group,) if isinstance(group, Path) else tuple(group)
        model_groups.append([
            record
            for path in paths
            for record in _local_path_stat_records(path)
        ])
    payload = {
        "model_groups": model_groups,
        "tokenizer": _local_path_stat_records(tokenizer_path),
        "figurine_lora": _local_path_stat_records(figurine_lora_path),
        "torch_dtype": torch_dtype,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def assign_object_split(
    object_id: str,
    validation_fraction: float = 0.1,
    salt: str = "wan22-ti2v5b-figurine360-v1",
) -> str:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    digest = hashlib.sha256(f"{salt}:{object_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "val" if bucket < validation_fraction else "train"


def seeds_for_object_split(config: PreparationConfig, split: str) -> tuple[int, ...]:
    if split == "val" and config.validation_seeds:
        return config.validation_seeds
    return config.seeds


def parse_model_path_values(values: Sequence[str]) -> list[Path | tuple[Path, ...]]:
    """Parse model path groups; a JSON list represents shards of one model."""
    paths: list[Path | tuple[Path, ...]] = []
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
            paths.append(tuple(Path(item).expanduser() for item in parsed))
        else:
            paths.append(Path(value).expanduser())
    if not paths:
        raise ValueError("At least one local --model_path is required.")
    return paths


def require_existing_path(path: Path, label: str, directory: Optional[bool] = None) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if directory is True and not path.is_dir():
        raise NotADirectoryError(f"{label} must be a directory: {path}")
    if directory is False and not path.is_file():
        raise FileNotFoundError(f"{label} must be a file: {path}")
    return path.resolve()


def require_existing_model_group(
    group: Path | Sequence[Path], label: str = "model_path"
) -> Path | tuple[Path, ...]:
    if isinstance(group, Path):
        return require_existing_path(group, label)
    resolved = tuple(require_existing_path(path, label) for path in group)
    if not resolved:
        raise ValueError(f"{label} shard group cannot be empty")
    return resolved


def resolve_data_path(base_path: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def read_source_metadata(metadata_path: Path) -> list[dict[str, str]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        missing = {"video", "prompt"} - fields
        if missing:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def read_video_first_frame(video_path: Path) -> Image.Image:
    """Read only frame zero through imageio's random-access reader."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    return Image.fromarray(frame).convert("RGB")


def _load_valid_png(
    path: Path, expected_size: Optional[tuple[int, int]] = None
) -> tuple[Optional[Image.Image], str]:
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                return None, f"expected PNG, found {image.format}"
            if expected_size is not None and image.size != expected_size:
                return None, f"PNG size mismatch: {image.size} != {expected_size}"
            image.verify()
        with Image.open(path) as image:
            return image.convert("RGB").copy(), ""
    except Exception as error:
        return None, f"invalid PNG: {error}"


def quarantine_invalid_file(path: Path) -> Path:
    token = uuid.uuid4().hex[:10]
    quarantined = path.with_name(f"{path.stem}.corrupt-{token}{path.suffix}")
    os.replace(path, quarantined)
    return quarantined


def _save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        image.convert("RGB").save(temp_path, format="PNG", optimize=False)
        loaded, reason = _load_valid_png(temp_path)
        if loaded is None:
            raise ValueError(reason)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing PNG: {path}")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def resize_first_frame_for_training(image: Image.Image, height: int, width: int) -> Image.Image:
    """Use the exact resize/crop operator used by WanTrainingModule input_image loading."""
    from diffsynth.core.data.operators import ImageCropAndResize

    return ImageCropAndResize(
        height=height,
        width=width,
        max_pixels=None,
        height_division_factor=16,
        width_division_factor=16,
    )(image.convert("RGB"))


def ensure_first_frame_png(
    video_path: Path,
    png_path: Path,
    frame_reader: Callable[[Path], Image.Image] = read_video_first_frame,
    image_processor: Optional[Callable[[Image.Image], Image.Image]] = None,
    expected_size: Optional[tuple[int, int]] = None,
) -> ImageCacheResult:
    if png_path.exists():
        image, reason = _load_valid_png(png_path, expected_size=expected_size)
        if image is not None:
            return ImageCacheResult(image=image, reused=True)
        quarantined = quarantine_invalid_file(png_path)
    else:
        reason = ""
        quarantined = None

    image = frame_reader(video_path)
    if not isinstance(image, Image.Image):
        raise TypeError("frame_reader must return PIL.Image.Image")
    image = image.convert("RGB")
    if image_processor is not None:
        image = image_processor(image)
    if expected_size is not None and image.size != expected_size:
        raise ValueError(f"processed first-frame size mismatch: {image.size} != {expected_size}")
    _save_png_atomic(image, png_path)
    return ImageCacheResult(
        image=image.copy(),
        reused=False,
        invalid_reason=reason,
        quarantined_path=quarantined,
    )


def validate_latent_tensor(
    latents: object,
    expected_shape: Optional[tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    expected_first_frame: Optional[torch.Tensor] = None,
) -> ValidationResult:
    if not isinstance(latents, torch.Tensor):
        return ValidationResult(False, f"latents must be a tensor, got {type(latents).__name__}")
    if latents.ndim != 5:
        return ValidationResult(False, f"latents must be B,C,T,H,W; got shape {tuple(latents.shape)}")
    if expected_shape is not None and tuple(latents.shape) != tuple(expected_shape):
        return ValidationResult(False, f"shape mismatch: {tuple(latents.shape)} != {tuple(expected_shape)}")
    if expected_dtype is not None and latents.dtype != expected_dtype:
        return ValidationResult(False, f"dtype mismatch: {latents.dtype} != {expected_dtype}")
    if not torch.is_floating_point(latents):
        return ValidationResult(False, f"latents must be floating point, got {latents.dtype}")
    if not bool(torch.isfinite(latents).all()):
        return ValidationResult(False, "latents contain NaN or Inf")
    if expected_first_frame is not None:
        actual = latents[:, :, :1].detach().to("cpu")
        expected = expected_first_frame.detach().to(dtype=latents.dtype, device="cpu")
        if actual.shape != expected.shape:
            return ValidationResult(False, f"first-frame shape mismatch: {actual.shape} != {expected.shape}")
        if not torch.equal(actual, expected):
            return ValidationResult(False, "first-frame latent is not bitwise equal to the input-image latent")
    return ValidationResult(True, latents=latents)


def validate_latent_file(
    path: Path,
    expected_shape: Optional[tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    expected_first_frame: Optional[torch.Tensor] = None,
    expected_metadata: Optional[dict[str, str]] = None,
) -> ValidationResult:
    if not path.is_file():
        return ValidationResult(False, "file does not exist")
    try:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            keys = set(file.keys())
            metadata = file.metadata() or {}
            tensors = {key: file.get_tensor(key) for key in keys}
    except Exception as error:
        return ValidationResult(False, f"cannot read safetensors: {error}")
    if set(tensors) != {"latents"}:
        return ValidationResult(False, f"expected only key='latents', found {sorted(tensors)}")
    result = validate_latent_tensor(
        tensors["latents"],
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_first_frame=expected_first_frame,
    )
    if not result.valid:
        return result
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if str(actual) != str(expected):
                return ValidationResult(
                    False,
                    f"metadata mismatch for {key}: {actual!r} != {str(expected)!r}",
                )
    return ValidationResult(True, latents=tensors["latents"])


def save_latents_atomic(
    path: Path,
    latents: torch.Tensor,
    expected_shape: Optional[tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    expected_first_frame: Optional[torch.Tensor] = None,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    result = validate_latent_tensor(latents, expected_shape, expected_dtype, expected_first_frame)
    if not result.valid:
        raise ValueError(result.reason)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing latent: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tensor = latents.detach().to("cpu").contiguous()
        save_file({"latents": tensor}, str(temp_path), metadata=metadata)
        saved = validate_latent_file(
            temp_path,
            expected_shape,
            expected_dtype,
            expected_first_frame,
            expected_metadata=metadata,
        )
        if not saved.valid:
            raise ValueError(f"Saved latent failed validation: {saved.reason}")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_metadata_atomic(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=DIRECT_DISTILL_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def expected_latent_shape(pipe, config: PreparationConfig) -> tuple[int, int, int, int, int]:
    vae = pipe.vae
    z_dim = getattr(vae, "z_dim", None)
    if z_dim is None and hasattr(vae, "model"):
        z_dim = getattr(vae.model, "z_dim", None)
    upsampling_factor = getattr(vae, "upsampling_factor", None)
    if z_dim is None or upsampling_factor is None:
        raise AttributeError("pipe.vae must expose z_dim and upsampling_factor")
    height_divisor = getattr(pipe, "height_division_factor", upsampling_factor)
    width_divisor = getattr(pipe, "width_division_factor", upsampling_factor)
    if config.height % height_divisor or config.width % width_divisor:
        raise ValueError(
            "height and width must already satisfy the pipeline division factors; "
            "implicit pipeline rounding would invalidate metadata and latent shapes"
        )
    if config.height % upsampling_factor or config.width % upsampling_factor:
        raise ValueError("height and width must be divisible by the VAE upsampling factor")
    latent_frames = (config.num_frames - 1) // 4 + 1
    return (
        1,
        int(z_dim),
        latent_frames,
        config.height // int(upsampling_factor),
        config.width // int(upsampling_factor),
    )


@torch.no_grad()
def encode_first_frame_latents(pipe, image: Image.Image, config: PreparationConfig) -> torch.Tensor:
    pipe.load_models_to_device(["vae"])
    image_tensor = pipe.preprocess_image(image.resize((config.width, config.height))).transpose(0, 1)
    latents = pipe.vae.encode(
        [image_tensor],
        device=pipe.device,
        tiled=config.tiled,
        tile_size=config.tile_size,
        tile_stride=config.tile_stride,
    )
    return latents.to(dtype=pipe.torch_dtype, device=pipe.device).detach()


@torch.no_grad()
def generate_teacher_latents(
    pipe,
    image: Image.Image,
    prompt: str,
    seed: int,
    config: PreparationConfig,
) -> torch.Tensor:
    latents = pipe(
        prompt=prompt,
        negative_prompt=config.negative_prompt,
        input_image=image,
        seed=int(seed),
        rand_device=config.rand_device,
        height=config.height,
        width=config.width,
        num_frames=config.num_frames,
        cfg_scale=config.teacher_cfg_scale,
        num_inference_steps=config.teacher_num_inference_steps,
        sigma_shift=config.teacher_sigma_shift,
        tiled=config.tiled,
        tile_size=config.tile_size,
        tile_stride=config.tile_stride,
        return_latents=True,
    )
    if not isinstance(latents, torch.Tensor):
        raise TypeError("WanVideoPipeline(return_latents=True) did not return a tensor")
    return latents.detach()


def build_direct_distill_row(
    *,
    video_path: Path,
    input_image_relative: Path,
    teacher_latent_relative: Path,
    prompt: str,
    seed: int,
    object_id: str,
    sample_id: str,
    split: str,
    teacher_fingerprint: str,
    source_fingerprint: str,
    config: PreparationConfig,
) -> dict:
    return {
        "video": str(video_path),
        "input_image": input_image_relative.as_posix(),
        "prompt": prompt,
        "negative_prompt": config.negative_prompt,
        "teacher_latent": teacher_latent_relative.as_posix(),
        "seed": int(seed),
        "rand_device": config.rand_device,
        "num_inference_steps": config.student_num_inference_steps,
        "cfg_scale": config.student_cfg_scale,
        "sigma_shift": config.student_sigma_shift,
        "teacher_num_inference_steps": config.teacher_num_inference_steps,
        "teacher_cfg_scale": config.teacher_cfg_scale,
        "teacher_sigma_shift": config.teacher_sigma_shift,
        "height": config.height,
        "width": config.width,
        "num_frames": config.num_frames,
        "tiled": int(config.tiled),
        "tile_size": f"{config.tile_size[0]},{config.tile_size[1]}",
        "tile_stride": f"{config.tile_stride[0]},{config.tile_stride[1]}",
        "object_id": object_id,
        "sample_id": sample_id,
        "teacher_fingerprint": teacher_fingerprint,
        "source_fingerprint": source_fingerprint,
        "split": split,
    }


def prepare_records(
    records: Sequence[dict[str, str]],
    pipe,
    config: PreparationConfig,
    *,
    frame_reader: Callable[[Path], Image.Image] = read_video_first_frame,
    first_frame_encoder: Callable = encode_first_frame_latents,
    teacher_generator: Callable = generate_teacher_latents,
) -> dict[str, int]:
    output_path = config.output_path
    first_frame_dir = output_path / "first_frames"
    latent_dir = output_path / "teacher_latents"
    failure_path = output_path / "failures.jsonl"
    metadata_output_path = output_path / "metadata_direct_distill.csv"
    metadata_train_path = output_path / "metadata_direct_distill_train.csv"
    metadata_validation_path = output_path / "metadata_direct_distill_validation.csv"
    first_frame_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)

    expected_shape = expected_latent_shape(pipe, config)
    expected_dtype = pipe.torch_dtype
    teacher_fingerprint = teacher_generation_fingerprint(config)
    output_rows: dict[str, dict] = {}
    summary = {
        "generated": 0,
        "skipped": 0,
        "invalid_detected": 0,
        "failed": 0,
        "metadata_rows": 0,
    }

    selected_records = records if config.max_samples is None else records[: config.max_samples]
    for source_index, source in enumerate(selected_records):
        video_value = str(source.get("video", "")).strip()
        prompt = str(source.get("prompt", "")).strip()
        if not video_value or not prompt:
            summary["failed"] += 1
            _append_jsonl(failure_path, {
                "stage": "metadata_validation",
                "source_index": source_index,
                "video": video_value,
                "error": "video and prompt must both be non-empty",
            })
            continue

        video_path = resolve_data_path(config.base_path, video_value)
        if not video_path.is_file():
            summary["failed"] += 1
            _append_jsonl(failure_path, {
                "stage": "source_validation",
                "source_index": source_index,
                "video": str(video_path),
                "error": "video file does not exist",
            })
            continue
        source_fingerprint = source_file_fingerprint(video_path)
        object_id = stable_object_id(video_value, source_fingerprint)
        split = assign_object_split(object_id, config.validation_fraction, config.split_salt)
        object_seeds = seeds_for_object_split(config, split)

        input_image_relative = Path("first_frames") / f"{object_id}.png"
        input_image_path = output_path / input_image_relative
        try:
            cache = ensure_first_frame_png(
                video_path,
                input_image_path,
                frame_reader=frame_reader,
                image_processor=lambda image: resize_first_frame_for_training(
                    image, config.height, config.width
                ),
                expected_size=(config.width, config.height),
            )
            if cache.invalid_reason:
                summary["invalid_detected"] += 1
                _append_jsonl(failure_path, {
                    "stage": "existing_first_frame_validation",
                    "source_index": source_index,
                    "object_id": object_id,
                    "video": str(video_path),
                    "error": cache.invalid_reason,
                    "quarantined_path": str(cache.quarantined_path),
                    "recovered": True,
                })
            first_frame_latents = first_frame_encoder(pipe, cache.image, config).detach().to("cpu")
            expected_first_shape = (
                expected_shape[0], expected_shape[1], 1, expected_shape[3], expected_shape[4]
            )
            first_result = validate_latent_tensor(
                first_frame_latents,
                expected_shape=expected_first_shape,
                expected_dtype=expected_dtype,
            )
            if not first_result.valid:
                raise ValueError(f"Invalid first-frame latent: {first_result.reason}")
        except Exception as error:
            summary["failed"] += len(object_seeds)
            _append_jsonl(failure_path, {
                "stage": "first_frame_preparation",
                "source_index": source_index,
                "object_id": object_id,
                "video": str(video_path),
                "error": f"{type(error).__name__}: {error}",
            })
            continue

        for seed in object_seeds:
            sample_id = stable_sample_id(
                video_value,
                prompt,
                seed,
                f"{teacher_fingerprint}:{source_fingerprint}",
            )
            if sample_id in output_rows:
                continue
            teacher_latent_relative = Path("teacher_latents") / f"{sample_id}.safetensors"
            teacher_latent_path = output_path / teacher_latent_relative
            latent_metadata = {
                "object_id": object_id,
                "sample_id": sample_id,
                "seed": str(seed),
                "rand_device": config.rand_device,
                "teacher_fingerprint": teacher_fingerprint,
                "source_fingerprint": source_fingerprint,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            try:
                existing = validate_latent_file(
                    teacher_latent_path,
                    expected_shape=expected_shape,
                    expected_dtype=expected_dtype,
                    expected_first_frame=first_frame_latents,
                    expected_metadata=latent_metadata,
                )
                if existing.valid:
                    if config.verify_determinism:
                        rerun = teacher_generator(pipe, cache.image, prompt, seed, config)
                        rerun_result = validate_latent_tensor(
                            rerun, expected_shape, expected_dtype, first_frame_latents
                        )
                        if not rerun_result.valid:
                            raise ValueError(f"Determinism rerun is invalid: {rerun_result.reason}")
                        if not torch.equal(existing.latents, rerun.detach().to("cpu")):
                            raise RuntimeError("Determinism rerun differs from the saved teacher latent")
                    summary["skipped"] += 1
                else:
                    if teacher_latent_path.exists():
                        summary["invalid_detected"] += 1
                        quarantined = quarantine_invalid_file(teacher_latent_path)
                        _append_jsonl(failure_path, {
                            "stage": "existing_latent_validation",
                            "source_index": source_index,
                            "object_id": object_id,
                            "sample_id": sample_id,
                            "seed": seed,
                            "video": str(video_path),
                            "error": existing.reason,
                            "quarantined_path": str(quarantined),
                            "recovered": True,
                        })

                    latents = teacher_generator(pipe, cache.image, prompt, seed, config)
                    generated_result = validate_latent_tensor(
                        latents, expected_shape, expected_dtype, first_frame_latents
                    )
                    if not generated_result.valid:
                        raise ValueError(f"Generated teacher latent is invalid: {generated_result.reason}")
                    if config.verify_determinism:
                        rerun = teacher_generator(pipe, cache.image, prompt, seed, config)
                        rerun_result = validate_latent_tensor(
                            rerun, expected_shape, expected_dtype, first_frame_latents
                        )
                        if not rerun_result.valid:
                            raise ValueError(f"Determinism rerun is invalid: {rerun_result.reason}")
                        if not torch.equal(latents.detach().to("cpu"), rerun.detach().to("cpu")):
                            raise RuntimeError("Teacher pipeline is not bitwise deterministic")

                    save_latents_atomic(
                        teacher_latent_path,
                        latents,
                        expected_shape=expected_shape,
                        expected_dtype=expected_dtype,
                        expected_first_frame=first_frame_latents,
                        metadata=latent_metadata,
                    )
                    summary["generated"] += 1

                output_rows[sample_id] = build_direct_distill_row(
                    video_path=video_path,
                    input_image_relative=input_image_relative,
                    teacher_latent_relative=teacher_latent_relative,
                    prompt=prompt,
                    seed=seed,
                    object_id=object_id,
                    sample_id=sample_id,
                    split=split,
                    teacher_fingerprint=teacher_fingerprint,
                    source_fingerprint=source_fingerprint,
                    config=config,
                )
            except Exception as error:
                summary["failed"] += 1
                _append_jsonl(failure_path, {
                    "stage": "teacher_latent_preparation",
                    "source_index": source_index,
                    "object_id": object_id,
                    "sample_id": sample_id,
                    "seed": seed,
                    "video": str(video_path),
                    "error": f"{type(error).__name__}: {error}",
                })

    ordered_rows = sorted(
        output_rows.values(), key=lambda row: (row["object_id"], int(row["seed"]), row["sample_id"])
    )
    _write_metadata_atomic(metadata_output_path, ordered_rows)
    _write_metadata_atomic(
        metadata_train_path, [row for row in ordered_rows if row["split"] == "train"]
    )
    _write_metadata_atomic(
        metadata_validation_path, [row for row in ordered_rows if row["split"] == "val"]
    )
    summary["metadata_rows"] = len(ordered_rows)
    try:
        pipe.load_models_to_device([])
    except (AttributeError, TypeError):
        pass
    return summary


def build_teacher_pipeline(
    model_paths: Sequence[Path | Sequence[Path]],
    tokenizer_path: Path,
    figurine_lora_path: Path,
    *,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
):
    """Build from local paths only; ModelConfig never receives a model_id."""
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[torch_dtype]
    model_configs = [
        ModelConfig(
            path=str(path) if isinstance(path, Path) else [str(shard) for shard in path],
            skip_download=True,
        )
        for path in model_paths
    ]
    tokenizer_config = ModelConfig(path=str(tokenizer_path), skip_download=True)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )
    matched_lora_layers = pipe.load_lora(
        pipe.dit, str(figurine_lora_path), alpha=1, hotload=False
    )
    if matched_lora_layers is None or matched_lora_layers <= 0:
        raise RuntimeError(
            "figurine360 LoRA matched zero Wan2.2-TI2V-5B modules; teacher generation aborted"
        )
    return pipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_path", required=True, type=Path)
    parser.add_argument("--base_path", required=True, type=Path)
    parser.add_argument("--output_path", required=True, type=Path)
    parser.add_argument(
        "--model_path",
        required=True,
        action="append",
        help=(
            "Local model file/directory. Repeat for separate models; pass a JSON list "
            "to keep all shards of one model in a single ModelConfig."
        ),
    )
    parser.add_argument("--tokenizer_path", required=True, type=Path)
    parser.add_argument("--figurine_lora_path", required=True, type=Path)
    parser.add_argument("--seed", action="append", type=int, help="Repeat for multiple seeds; default: 1")
    parser.add_argument(
        "--validation_seed",
        action="append",
        type=int,
        help="Optional held-out seed(s) used only for validation objects; must not overlap --seed.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--rand_device", default="cpu")
    parser.add_argument("--teacher_num_inference_steps", type=int, default=50)
    parser.add_argument("--teacher_cfg_scale", type=float, default=5.0)
    parser.add_argument("--teacher_sigma_shift", type=float, default=5.0)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--split_salt", default="wan22-ti2v5b-figurine360-v1")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional H100 smoke limit applied to source metadata rows before seed expansion.",
    )
    parser.add_argument("--verify_determinism", action="store_true")
    parser.add_argument("--tiled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tile_size", type=int, nargs=2, default=(30, 52), metavar=("H", "W"))
    parser.add_argument("--tile_stride", type=int, nargs=2, default=(15, 26), metavar=("H", "W"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        metadata_path = require_existing_path(args.metadata_path, "metadata_path", directory=False)
        base_path = require_existing_path(args.base_path, "base_path", directory=True)
        tokenizer_path = require_existing_path(args.tokenizer_path, "tokenizer_path")
        figurine_lora_path = require_existing_path(
            args.figurine_lora_path, "figurine_lora_path", directory=False
        )
        model_paths = [
            require_existing_model_group(group)
            for group in parse_model_path_values(args.model_path)
        ]
        records = read_source_metadata(metadata_path)
        config = PreparationConfig(
            base_path=base_path,
            output_path=args.output_path.expanduser().resolve(),
            seeds=tuple(args.seed or (1,)),
            validation_seeds=tuple(args.validation_seed or ()),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            rand_device=args.rand_device,
            teacher_num_inference_steps=args.teacher_num_inference_steps,
            teacher_cfg_scale=args.teacher_cfg_scale,
            teacher_sigma_shift=args.teacher_sigma_shift,
            negative_prompt=args.negative_prompt,
            validation_fraction=args.validation_fraction,
            split_salt=args.split_salt,
            teacher_provenance=build_local_teacher_provenance(
                model_paths,
                tokenizer_path,
                figurine_lora_path,
                args.torch_dtype,
            ),
            max_samples=args.max_samples,
            verify_determinism=args.verify_determinism,
            tiled=args.tiled,
            tile_size=tuple(args.tile_size),
            tile_stride=tuple(args.tile_stride),
        )
        config.output_path.mkdir(parents=True, exist_ok=True)
        pipe = build_teacher_pipeline(
            model_paths,
            tokenizer_path,
            figurine_lora_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
        summary = prepare_records(records, pipe, config)
    except Exception as error:
        parser.error(str(error))
        return 2

    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
