import argparse
import glob
import os
from pathlib import Path

import torch
from PIL import Image

from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video


# ====== Stage B: only edit this block on the H100 machine ======
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/path/to/local/wan"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", str(MODEL_ROOT / "google" / "umt5-xxl")))
LORA_PATH = Path(os.environ.get("LORA_PATH", "./models/train/Wan2.2-TI2V-5B_figurine360_lora/epoch-0.safetensors"))
IMAGE_PATH = Path(os.environ.get("IMAGE_PATH", "/path/to/figurine_input.jpg"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "./outputs/figurine360_validate.mp4"))
PROMPT = os.environ.get(
    "PROMPT",
    "手办360度水平旋转展示，单个精致手办，固定相机，白色背景，平滑匀速水平旋转",
)
NEGATIVE_PROMPT = os.environ.get(
    "NEGATIVE_PROMPT",
    "低质量，模糊，抖动，变形，主体漂移，背景杂乱，文字，水印",
)
HEIGHT = int(os.environ.get("HEIGHT", "480"))
WIDTH = int(os.environ.get("WIDTH", "832"))
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "121"))
SEED = int(os.environ.get("SEED", "1"))
FPS = int(os.environ.get("FPS", "15"))
# ===============================================================


def require_path(path, label, is_dir=False):
    path = Path(path)
    exists = path.is_dir() if is_dir else path.is_file()
    if not exists:
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def build_model_configs(model_root=MODEL_ROOT, tokenizer_path=TOKENIZER_PATH):
    model_root = Path(model_root)
    tokenizer_path = Path(tokenizer_path)
    dit_paths = sorted(glob.glob(str(model_root / "diffusion_pytorch_model*.safetensors")))
    if not dit_paths:
        raise FileNotFoundError(f"No DiT safetensors found at {model_root / 'diffusion_pytorch_model*.safetensors'}")

    text_encoder = require_path(model_root / "models_t5_umt5-xxl-enc-bf16.pth", "T5 text encoder weights")
    vae = require_path(model_root / "Wan2.2_VAE.pth", "Wan2.2 VAE weights")
    tokenizer = require_path(tokenizer_path, "tokenizer directory", is_dir=True)

    return [
        ModelConfig(path=dit_paths),
        ModelConfig(path=str(text_encoder)),
        ModelConfig(path=str(vae)),
    ], ModelConfig(path=str(tokenizer))


def video_output_path(path=OUTPUT_PATH, seed=SEED, create_parent=True):
    path = Path(path)
    if path.suffix != ".mp4":
        path = path / f"wan22_ti2v5b_figurine360_seed{seed}.mp4"
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Wan2.2-TI2V-5B figurine 360 LoRA inference.")
    parser.add_argument("--dry_run", action="store_true", help="Print resolved config without loading model weights.")
    parser.add_argument("--model_root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--tokenizer_path", type=Path, default=TOKENIZER_PATH)
    parser.add_argument("--lora_path", type=Path, default=LORA_PATH)
    parser.add_argument("--image_path", type=Path, default=IMAGE_PATH)
    parser.add_argument("--output_path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--prompt", type=str, default=PROMPT)
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fps", type=int, default=FPS)
    return parser.parse_args()


def print_dry_run(args):
    print("dry_run: true")
    print("feature: Wan2.2-TI2V-5B_figurine360")
    print(f"model_root: {args.model_root}")
    print(f"tokenizer_path: {args.tokenizer_path}")
    print(f"lora_path: {args.lora_path}")
    print(f"image_path: {args.image_path}")
    print(f"output_path: {video_output_path(args.output_path, args.seed, create_parent=False)}")
    print(f"height: {args.height}")
    print(f"width: {args.width}")
    print(f"num_frames: {args.num_frames}")
    print(f"seed: {args.seed}")
    print(f"fps: {args.fps}")
    print(f"prompt: {args.prompt}")


def main():
    args = parse_args()
    if args.dry_run:
        print_dry_run(args)
        return

    require_path(args.lora_path, "LoRA checkpoint")
    require_path(args.image_path, "input image")
    model_configs, tokenizer_config = build_model_configs(args.model_root, args.tokenizer_path)

    from diffsynth.pipelines.wan_video import WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    pipe.load_lora(pipe.dit, str(args.lora_path), alpha=1)

    input_image = Image.open(args.image_path).convert("RGB").resize((args.width, args.height))
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        input_image=input_image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=True,
    )
    save_video(video, str(video_output_path(args.output_path, args.seed)), fps=args.fps, quality=5)


if __name__ == "__main__":
    main()
