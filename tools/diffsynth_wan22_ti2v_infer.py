#!/usr/bin/env python3
"""Run DiffSynth Wan2.2-TI2V-5B inference from a local model directory."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_root", type=Path, required=True, help="Local merged Wan2.2-TI2V-5B directory.")
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--timing_json", type=Path, default=None)
    return parser.parse_args()


def _required_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required model file: {path}")
    return str(path)


def _required_glob(pattern: Path) -> list[str]:
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def main():
    args = parse_args()

    import torch
    from PIL import Image

    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    model_root = args.model_root
    model_configs = [
        ModelConfig(path=_required_file(model_root / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=_required_glob(model_root / "diffusion_pytorch_model*.safetensors")),
        ModelConfig(path=_required_file(model_root / "Wan2.2_VAE.pth")),
    ]
    tokenizer_config = ModelConfig(path=str(model_root / "google/umt5-xxl/"))

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    input_image = None
    if args.image is not None:
        input_image = Image.open(args.image).convert("RGB").resize((args.width, args.height))

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        seed=args.seed,
        tiled=True,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        sigma_shift=args.sigma_shift,
    )
    inference_seconds = time.time() - start_time

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(args.output_path), fps=args.fps, quality=args.quality)

    if args.timing_json is not None:
        args.timing_json.parent.mkdir(parents=True, exist_ok=True)
        timing = {
            "output_path": str(args.output_path),
            "model_root": str(args.model_root),
            "prompt": args.prompt,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "sigma_shift": args.sigma_shift,
            "inference_seconds": inference_seconds,
            "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
        args.timing_json.write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
