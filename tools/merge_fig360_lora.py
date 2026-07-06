#!/usr/bin/env python3
"""Merge figurine360 LoRA weights into a local Wan2.2-TI2V-5B DiT directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


DEFAULT_PREFIXES_TO_STRIP = ("pipe.dit.",)
LORA_SUFFIX_PAIRS = (
    (".lora_A.default.weight", ".lora_B.default.weight"),
    (".lora_A.weight", ".lora_B.weight"),
    (".lora_down.default.weight", ".lora_up.default.weight"),
    (".lora_down.weight", ".lora_up.weight"),
)
DEFAULT_TARGET_MODULES = ("q", "k", "v", "o", "ffn.0", "ffn.2")


def strip_known_prefix(key: str, prefixes: Iterable[str] = DEFAULT_PREFIXES_TO_STRIP) -> str:
    for prefix in prefixes:
        if prefix and key.startswith(prefix):
            return key[len(prefix) :]
    return key


def split_lora_key(key: str):
    for suffix_a, suffix_b in LORA_SUFFIX_PAIRS:
        if key.endswith(suffix_a):
            return key[: -len(suffix_a)], "A"
        if key.endswith(suffix_b):
            return key[: -len(suffix_b)], "B"
    if key.endswith(".alpha"):
        return key[: -len(".alpha")], "alpha"
    return None, None


def tensor_to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().reshape(-1)[0].item())
    return float(value)


def normalize_lora_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) == 4:
        return tensor.squeeze(3).squeeze(2)
    return tensor


def collect_lora_layers(lora_state_dict, prefixes_to_strip=DEFAULT_PREFIXES_TO_STRIP):
    layers = {}
    for original_key, tensor in lora_state_dict.items():
        key = strip_known_prefix(original_key, prefixes_to_strip)
        target, kind = split_lora_key(key)
        if target is None:
            continue
        layer = layers.setdefault(target, {"target": target})
        if kind == "A":
            layer["A"] = tensor
            layer["A_key"] = original_key
        elif kind == "B":
            layer["B"] = tensor
            layer["B_key"] = original_key
        elif kind == "alpha":
            layer["alpha"] = tensor_to_float(tensor)
            layer["alpha_key"] = original_key

    complete_layers = []
    incomplete_layers = []
    for target, layer in sorted(layers.items()):
        if "A" in layer and "B" in layer:
            complete_layers.append(layer)
        else:
            incomplete_layers.append(target)
    return complete_layers, incomplete_layers


def categorize_target(target: str, target_modules=DEFAULT_TARGET_MODULES) -> str:
    for module in sorted(target_modules, key=len, reverse=True):
        if target == module or target.endswith(f".{module}"):
            return module
    return "<other>"


def merge_lora_state_dict(
    base_state_dict,
    lora_state_dict,
    *,
    lora_strength: float = 1.0,
    prefixes_to_strip=DEFAULT_PREFIXES_TO_STRIP,
    strict: bool = True,
    target_modules=DEFAULT_TARGET_MODULES,
):
    merged = dict(base_state_dict)
    layers, incomplete_layers = collect_lora_layers(lora_state_dict, prefixes_to_strip)
    report_layers = []
    unmatched = []
    changed_keys = []

    for layer in layers:
        target = layer["target"]
        base_key = f"{target}.weight"
        if base_key not in merged:
            unmatched.append(target)
            continue

        base_weight = merged[base_key]
        lora_a = normalize_lora_tensor(layer["A"])
        lora_b = normalize_lora_tensor(layer["B"])
        rank = int(lora_a.shape[0])
        alpha = float(layer.get("alpha", rank))
        scale = float(lora_strength) * alpha / rank
        delta = torch.mm(lora_b.float(), lora_a.float()) * scale
        if tuple(delta.shape) != tuple(base_weight.shape):
            raise ValueError(
                f"LoRA shape mismatch for {target}: delta {tuple(delta.shape)} vs base {tuple(base_weight.shape)}"
            )

        merged[base_key] = (base_weight.float() + delta).to(dtype=base_weight.dtype).contiguous()
        changed_keys.append(base_key)
        report_layers.append(
            {
                "target": target,
                "base_key": base_key,
                "rank": rank,
                "alpha": alpha,
                "scale": scale,
                "category": categorize_target(target, target_modules),
                "dtype": str(base_weight.dtype).replace("torch.", ""),
                "shape": list(base_weight.shape),
            }
        )

    if strict and (unmatched or incomplete_layers):
        details = {
            "unmatched_lora_layers": unmatched,
            "incomplete_lora_layers": incomplete_layers,
        }
        raise ValueError(f"LoRA merge failed strict validation: {json.dumps(details, ensure_ascii=False)}")

    category_counts = {}
    for layer in report_layers:
        category_counts[layer["category"]] = category_counts.get(layer["category"], 0) + 1

    report = {
        "hit_lora_layers": len(report_layers),
        "unmatched_lora_layers": unmatched,
        "incomplete_lora_layers": incomplete_layers,
        "changed_keys": changed_keys,
        "lora_layers": report_layers,
        "category_counts": category_counts,
    }
    return merged, report


def find_dit_files(base_model_dir: Path, dit_glob: str):
    files = sorted(base_model_dir.glob(dit_glob))
    if not files:
        raise FileNotFoundError(f"No DiT safetensors found at {base_model_dir / dit_glob}")
    return files


def safetensors_keys(path: Path):
    with safe_open(path, framework="pt", device="cpu") as handle:
        return list(handle.keys())


def copy_or_link_file(src: Path, dst: Path, copy_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        os.symlink(src, dst)
    elif copy_mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown copy mode: {copy_mode}")


def copy_non_dit_assets(base_model_dir: Path, output_dir: Path, dit_files, copy_mode: str):
    dit_paths = {path.resolve() for path in dit_files}
    for src in base_model_dir.rglob("*"):
        if src.resolve() in dit_paths:
            continue
        rel = src.relative_to(base_model_dir)
        dst = output_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif src.is_file() or src.is_symlink():
            copy_or_link_file(src, dst, copy_mode)


def verify_unchanged_tensors(base_file: Path, output_file: Path, changed_keys: set[str]) -> int:
    base = load_file(base_file)
    output = load_file(output_file)
    verified = 0
    for key, base_tensor in base.items():
        if key in changed_keys:
            continue
        if key not in output:
            raise AssertionError(f"Missing unchanged tensor in merged output: {key}")
        if not torch.equal(base_tensor, output[key]):
            raise AssertionError(f"Unchanged tensor differs after merge: {key}")
        verified += 1
    return verified


def merge_lora_into_directory(
    *,
    base_model_dir,
    lora_path,
    output_dir,
    dit_glob="diffusion_pytorch_model*.safetensors",
    copy_mode="hardlink",
    lora_strength: float = 1.0,
    prefixes_to_strip=DEFAULT_PREFIXES_TO_STRIP,
    strict: bool = True,
    verify_output: bool = True,
    dry_run: bool = False,
    target_modules=DEFAULT_TARGET_MODULES,
):
    base_model_dir = Path(base_model_dir)
    lora_path = Path(lora_path)
    output_dir = Path(output_dir)
    if output_dir.resolve() == base_model_dir.resolve():
        raise ValueError("output_dir must be different from base_model_dir")
    if not base_model_dir.is_dir():
        raise FileNotFoundError(f"Missing base model directory: {base_model_dir}")
    if not lora_path.is_file():
        raise FileNotFoundError(f"Missing LoRA file: {lora_path}")

    dit_files = find_dit_files(base_model_dir, dit_glob)
    lora_state_dict = load_file(lora_path)
    aggregate_report = {
        "base_model_dir": str(base_model_dir),
        "lora_path": str(lora_path),
        "output_dir": str(output_dir),
        "dit_glob": dit_glob,
        "copy_mode": copy_mode,
        "lora_strength": float(lora_strength),
        "dry_run": bool(dry_run),
        "dit_files": [path.name for path in dit_files],
        "hit_lora_layers": 0,
        "unmatched_lora_layers": [],
        "incomplete_lora_layers": [],
        "changed_keys": [],
        "lora_layers": [],
        "category_counts": {},
        "unchanged_tensors_verified": 0,
    }
    matched_targets = set()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        copy_non_dit_assets(base_model_dir, output_dir, dit_files, copy_mode)

    for dit_file in dit_files:
        base_state_dict = load_file(dit_file)
        merged_state_dict, report = merge_lora_state_dict(
            base_state_dict,
            lora_state_dict,
            lora_strength=lora_strength,
            prefixes_to_strip=prefixes_to_strip,
            strict=False,
            target_modules=target_modules,
        )
        changed_keys = set(report["changed_keys"])
        matched_targets.update(layer["target"] for layer in report["lora_layers"])
        aggregate_report["changed_keys"].extend(report["changed_keys"])
        aggregate_report["lora_layers"].extend(report["lora_layers"])

        if not dry_run:
            save_file(merged_state_dict, output_dir / dit_file.name)
            if verify_output:
                aggregate_report["unchanged_tensors_verified"] += verify_unchanged_tensors(
                    dit_file,
                    output_dir / dit_file.name,
                    changed_keys,
                )

    all_layers, incomplete_layers = collect_lora_layers(lora_state_dict, prefixes_to_strip)
    unmatched = [layer["target"] for layer in all_layers if layer["target"] not in matched_targets]
    if strict and (unmatched or incomplete_layers):
        details = {
            "unmatched_lora_layers": unmatched,
            "incomplete_lora_layers": incomplete_layers,
        }
        raise ValueError(f"LoRA merge failed strict validation: {json.dumps(details, ensure_ascii=False)}")

    category_counts = {}
    for layer in aggregate_report["lora_layers"]:
        category_counts[layer["category"]] = category_counts.get(layer["category"], 0) + 1

    aggregate_report["hit_lora_layers"] = len(aggregate_report["lora_layers"])
    aggregate_report["unmatched_lora_layers"] = unmatched
    aggregate_report["incomplete_lora_layers"] = incomplete_layers
    aggregate_report["category_counts"] = category_counts
    return aggregate_report


def validate_expectations(report, expected_lora_layers=None, expected_layers_per_target=None):
    if expected_lora_layers is not None and report["hit_lora_layers"] != expected_lora_layers:
        raise AssertionError(
            f"Expected {expected_lora_layers} merged LoRA layers, got {report['hit_lora_layers']}"
        )
    if expected_layers_per_target is not None:
        for target in DEFAULT_TARGET_MODULES:
            actual = report["category_counts"].get(target, 0)
            if actual != expected_layers_per_target:
                raise AssertionError(
                    f"Expected {expected_layers_per_target} layers for target {target}, got {actual}"
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Merge figurine360 LoRA into local Wan2.2-TI2V-5B DiT shards.")
    parser.add_argument("--base_model_dir", type=Path, required=True)
    parser.add_argument("--lora_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dit_glob", default="diffusion_pytorch_model*.safetensors")
    parser.add_argument("--copy_mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--lora_strength", type=float, default=1.0)
    parser.add_argument("--strip_prefix", action="append", default=list(DEFAULT_PREFIXES_TO_STRIP))
    parser.add_argument("--allow_unmatched", action="store_true")
    parser.add_argument("--skip_output_verify", action="store_true")
    parser.add_argument("--expected_lora_layers", type=int, default=None)
    parser.add_argument("--expected_layers_per_target", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = merge_lora_into_directory(
        base_model_dir=args.base_model_dir,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        dit_glob=args.dit_glob,
        copy_mode=args.copy_mode,
        lora_strength=args.lora_strength,
        prefixes_to_strip=tuple(args.strip_prefix),
        strict=not args.allow_unmatched,
        verify_output=not args.skip_output_verify,
        dry_run=args.dry_run,
    )
    validate_expectations(
        report,
        expected_lora_layers=args.expected_lora_layers,
        expected_layers_per_target=args.expected_layers_per_target,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
