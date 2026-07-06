"""Config helpers for figurine360 Wan2.2 DMD training."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _save_yaml(data: dict, path: str | Path):
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def latent_shape(height: int, width: int, num_frames: int) -> list[int]:
    if num_frames % 4 != 1:
        raise ValueError("Wan2.2 TI2V num_frames must equal 1 mod 4.")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError("Wan2.2 TI2V height and width must be divisible by 16.")
    return [1, (num_frames - 1) // 4 + 1, 48, height // 16, width // 16]


def load_dmd_config_summary(path: str | Path) -> dict:
    cfg = _load_yaml(path)
    lora = cfg.get("lora", {})
    return {
        "trainer": cfg["trainer"],
        "distribution_loss": cfg["distribution_loss"],
        "generator_type": cfg["generator_type"],
        "denoising_step_list": list(cfg["denoising_step_list"]),
        "warp_denoising_step": bool(cfg["warp_denoising_step"]),
        "denoising_loss_type": cfg["denoising_loss_type"],
        "dfake_gen_update_ratio": int(cfg["dfake_gen_update_ratio"]),
        "real_guidance_scale": float(cfg["real_guidance_scale"]),
        "fake_guidance_scale": float(cfg["fake_guidance_scale"]),
        "lora": {
            "enabled": bool(lora.get("enabled", False)),
            "rank": int(lora.get("rank", 0)),
            "alpha": int(lora.get("alpha", 0)),
            "target_modules": list(lora.get("target_modules", [])),
        },
        "attention_backend": cfg.get("attention_backend", None),
        "gan_loss_weight": float(cfg.get("gan_loss_weight", 0.0)),
        "load_video_latent": bool(cfg.get("load_video_latent", False)),
        "max_iters": int(cfg.get("max_iters", 0)),
        "validation_interval": int(cfg.get("validation_interval", cfg.get("log_iters", 0))),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 8)),
        "enable_orientation_buckets": bool(cfg.get("enable_orientation_buckets", False)),
    }


def write_runtime_config(
    *,
    template_path: str | Path,
    output_path: str | Path,
    height: int,
    width: int,
    num_frames: int,
    max_iters: int,
    log_iters: int,
    batch_size: int = 1,
    lr: float | None = None,
    lr_critic: float | None = None,
    ema_weight: float | None = None,
    validation_interval: int | None = None,
    dataloader_num_workers: int | None = None,
    enable_orientation_buckets: bool | int | None = None,
) -> dict:
    cfg = _load_yaml(template_path)
    cfg["h"] = int(height)
    cfg["w"] = int(width)
    cfg["num_frames"] = int(num_frames)
    cfg["num_training_frames"] = latent_shape(height, width, num_frames)[1]
    cfg["num_frame_per_block"] = cfg["num_training_frames"]
    cfg["image_or_video_shape"] = latent_shape(height, width, num_frames)
    cfg["max_iters"] = int(max_iters)
    cfg["log_iters"] = int(log_iters)
    if validation_interval is not None:
        cfg["validation_interval"] = int(validation_interval)
    cfg["batch_size"] = int(batch_size)
    if dataloader_num_workers is not None:
        cfg["dataloader_num_workers"] = int(dataloader_num_workers)
    if enable_orientation_buckets is not None:
        cfg["enable_orientation_buckets"] = bool(enable_orientation_buckets)
    if lr is not None:
        cfg["lr"] = float(lr)
    if lr_critic is not None:
        cfg["lr_critic"] = float(lr_critic)
    if ema_weight is not None:
        cfg["ema_weight"] = float(ema_weight)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_yaml(cfg, output_path)
    return load_dmd_config_summary(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize a Wan2.2 DMD YAML with runtime dimensions.")
    parser.add_argument("--template_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--max_iters", type=int, required=True)
    parser.add_argument("--log_iters", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_critic", type=float, default=None)
    parser.add_argument("--ema_weight", type=float, default=None)
    parser.add_argument("--validation_interval", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--enable_orientation_buckets", type=int, choices=(0, 1), default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = write_runtime_config(**vars(args))
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
