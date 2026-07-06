from pipeline import Wan22FewstepInferencePipeline
from diffusers.utils import export_to_video
from omegaconf import OmegaConf
import argparse
import json
import torch
import os
import time
import torchvision.transforms.functional as TF
from PIL import Image

from dmd.wan22_lora import (
    LoRALinear,
    apply_configured_lora,
    import_lora_state_dict_for_runtime,
    load_generator_lora_checkpoint,
    load_lora_state_dict,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str, default=None)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--lora_source", choices=["auto", "ema", "generator"], default="auto")
parser.add_argument("--strict_lora", action="store_true")
parser.add_argument("--output_path", type=str)
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--image", type=str, default=None)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--h", type=int, default=704)
parser.add_argument("--w", type=int, default=1280)
parser.add_argument("--num_frames", type=int, default=121)
parser.add_argument("--timing_json", type=str, default=None)
args = parser.parse_args()
assert args.num_frames % 4 == 1, "num_frames must be 1 more than a multiple of 4"


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)


def _normalize_full_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "")
        new_key = new_key.replace("_checkpoint_wrapped_module.", "")
        new_key = new_key.replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    return new_state_dict


def _ensure_generator_lora(pipe, config):
    if any(isinstance(module, LoRALinear) for module in pipe.generator.model.modules()):
        return
    report = apply_configured_lora(pipe.generator, config, role="generator")
    if report is None:
        raise RuntimeError("Config must contain lora.enabled=true when --lora_path or a LoRA checkpoint is used.")


def _load_lora_into_generator(pipe, config, state_dict):
    _ensure_generator_lora(pipe, config)
    runtime_state = import_lora_state_dict_for_runtime(state_dict)
    load_lora_state_dict(pipe.generator.model, runtime_state, strict_lora=args.strict_lora)


def _load_checkpoint_if_requested(pipe, config):
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None and args.checkpoint_folder is not None:
        checkpoint_path = os.path.join(args.checkpoint_folder, "model.pt")
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and (
        "generator_lora" in checkpoint or "generator_ema_lora" in checkpoint
    ):
        state_dict = load_generator_lora_checkpoint(checkpoint_path, source=args.lora_source)
        _load_lora_into_generator(pipe, config, state_dict)
        return

    if isinstance(checkpoint, dict) and "generator" in checkpoint:
        checkpoint = checkpoint["generator"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    new_state_dict = _normalize_full_state_dict(checkpoint)
    _, unexpected = pipe.generator.load_state_dict(new_state_dict, strict=False)
    assert len(unexpected) == 0, f"Unexpected keys in state_dict: {unexpected}"


pipe = Wan22FewstepInferencePipeline(config)
_load_checkpoint_if_requested(pipe, config)
if args.lora_path is not None:
    lora_state = load_generator_lora_checkpoint(args.lora_path, source=args.lora_source)
    _load_lora_into_generator(pipe, config, lora_state)
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)

output_dir = os.path.dirname(args.output_path)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

if args.image is not None:
    img = Image.open(args.image).convert("RGB")
    img = img.resize((args.w, args.h), Image.LANCZOS)
    img = TF.to_tensor(img).sub_(0.5).div_(0.5).to("cuda").unsqueeze(1).to(dtype=torch.bfloat16)
    wan22_image_latent = pipe.vae.encode_to_latent(img.unsqueeze(0))
else:
    wan22_image_latent = None

torch.cuda.reset_peak_memory_stats()
start_time = time.time()
video = (
    pipe.inference(
        noise=torch.randn(
            1,
            (args.num_frames - 1) // 4 + 1,
            48,
            args.h // 16,
            args.w // 16,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        text_prompts=[args.prompt],
        wan22_image_latent=wan22_image_latent,
    )[0]
    .permute(0, 2, 3, 1)
    .cpu()
    .numpy()
)
inference_seconds = time.time() - start_time

export_to_video(video, args.output_path, fps=24)

if args.timing_json is not None:
    timing_dir = os.path.dirname(args.timing_json)
    if timing_dir:
        os.makedirs(timing_dir, exist_ok=True)
    timing = {
        "output_path": args.output_path,
        "prompt": args.prompt,
        "seed": args.seed,
        "height": args.h,
        "width": args.w,
        "num_frames": args.num_frames,
        "num_steps": len(config.denoising_step_list),
        "inference_seconds": inference_seconds,
        "peak_cuda_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "lora_path": args.lora_path,
        "checkpoint_path": args.checkpoint_path,
        "checkpoint_folder": args.checkpoint_folder,
    }
    with open(args.timing_json, "w", encoding="utf-8") as handle:
        json.dump(timing, handle, ensure_ascii=False, indent=2)
