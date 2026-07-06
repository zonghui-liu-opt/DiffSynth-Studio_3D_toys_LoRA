"""Lightweight video metrics used by Stage B validation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _as_float_rgb(video: np.ndarray) -> np.ndarray:
    video = np.asarray(video)
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError("video must have shape [frames, height, width, 3].")
    if np.issubdtype(video.dtype, np.integer):
        video = video.astype(np.float32) / 255.0
    else:
        video = video.astype(np.float32)
    return np.clip(video, 0.0, 1.0)


def _histogram(values: np.ndarray, bins: int) -> np.ndarray:
    hist, _ = np.histogram(values.reshape(-1), bins=bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total == 0:
        return hist.astype(np.float64)
    return hist.astype(np.float64) / float(total)


def _total_variation_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(a - b).sum())


def _brightness(video: np.ndarray) -> np.ndarray:
    return video.mean(axis=-1)


def _saturation(video: np.ndarray) -> np.ndarray:
    max_rgb = video.max(axis=-1)
    min_rgb = video.min(axis=-1)
    saturation = np.zeros_like(max_rgb)
    nonzero = max_rgb > 1e-6
    saturation[nonzero] = (max_rgb[nonzero] - min_rgb[nonzero]) / max_rgb[nonzero]
    return saturation


def compute_basic_video_metrics(teacher_video: np.ndarray, student_video: np.ndarray, *, bins: int = 64) -> dict:
    teacher = _as_float_rgb(teacher_video)
    student = _as_float_rgb(student_video)
    if teacher.shape != student.shape:
        raise ValueError(f"teacher/student shape mismatch: {teacher.shape} vs {student.shape}")

    teacher_brightness = _brightness(teacher)
    student_brightness = _brightness(student)
    teacher_saturation = _saturation(teacher)
    student_saturation = _saturation(student)

    teacher_frame_brightness = teacher_brightness.mean(axis=(1, 2))
    student_frame_brightness = student_brightness.mean(axis=(1, 2))

    return {
        "num_frames": int(teacher.shape[0]),
        "height": int(teacher.shape[1]),
        "width": int(teacher.shape[2]),
        "brightness_hist_shift": _total_variation_distance(
            _histogram(teacher_brightness, bins),
            _histogram(student_brightness, bins),
        ),
        "saturation_hist_shift": _total_variation_distance(
            _histogram(teacher_saturation, bins),
            _histogram(student_saturation, bins),
        ),
        "teacher_interframe_brightness_std": float(np.std(teacher_frame_brightness)),
        "student_interframe_brightness_std": float(np.std(student_frame_brightness)),
        "mean_brightness_shift": float(abs(student_brightness.mean() - teacher_brightness.mean())),
        "mean_saturation_shift": float(abs(student_saturation.mean() - teacher_saturation.mean())),
    }


def compute_video_diff_metrics(left_video: np.ndarray, right_video: np.ndarray) -> dict:
    left = _as_float_rgb(left_video)
    right = _as_float_rgb(right_video)
    if left.shape != right.shape:
        raise ValueError(f"video shape mismatch: {left.shape} vs {right.shape}")
    diff = np.abs(left - right)
    return {
        "max_pixel_diff": float(diff.max()),
        "max_pixel_diff_255": float(diff.max() * 255.0),
        "pixel_mse": float(np.mean((left - right) ** 2)),
    }


def read_video(path: str | Path) -> np.ndarray:
    import imageio.v3 as iio

    return np.asarray(iio.imread(path))


def write_metrics_json(metrics: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
