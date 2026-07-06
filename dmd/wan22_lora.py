"""LoRA helpers used by the vendored Wan2.2 DMD runtime.

The module is intentionally import-safe on CPU: it only depends on torch and
does not import the Wan model stack or initialize CUDA.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


DEFAULT_TARGET_MODULES = ("q", "k", "v", "o", "ffn.0", "ffn.2")


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int | float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32), persistent=True)
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        base = self.base_layer(x)
        delta = self.lora_B(self.lora_A(x)) * self.scaling
        return base + delta.to(dtype=base.dtype)


def _target_matches(module_name: str, target: str) -> bool:
    return module_name == target or module_name.endswith(f".{target}")


def should_patch_lora(module_name: str, target_modules: Iterable[str]) -> bool:
    return any(_target_matches(module_name, target) for target in target_modules)


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_model(
    model: nn.Module,
    *,
    rank: int,
    alpha: int | float,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
    freeze_base: bool = True,
):
    if freeze_base:
        model.requires_grad_(False)
    patched = []
    target_modules = tuple(target_modules)
    for name, module in list(model.named_modules()):
        if not name or isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and should_patch_lora(name, target_modules):
            parent, leaf = _get_parent_module(model, name)
            setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha))
            patched.append(name)
    if not patched:
        raise ValueError(f"No nn.Linear modules matched LoRA targets: {target_modules}")
    return {
        "patched_count": len(patched),
        "patched_modules": patched,
        "rank": int(rank),
        "alpha": float(alpha),
        "target_modules": list(target_modules),
    }


def _strip_prefix(name: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if prefix and name.startswith(prefix):
            return name[len(prefix) :]
    return name


def normalize_state_key(name: str) -> str:
    for token in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod."):
        name = name.replace(token, "")
    return name


def is_lora_state_key(name: str) -> bool:
    return (
        name.endswith(".lora_A.weight")
        or name.endswith(".lora_B.weight")
        or name.endswith(".lora_A.default.weight")
        or name.endswith(".lora_B.default.weight")
        or name.endswith(".alpha")
    )


def filter_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        normalize_state_key(name): tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if is_lora_state_key(name)
    }


def export_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    strip_prefixes: Iterable[str] = (),
    include_alpha: bool = True,
) -> dict[str, torch.Tensor]:
    exported = {}
    alpha_by_target = {}
    lora_targets = set()

    for name, tensor in state_dict.items():
        name = normalize_state_key(name)
        if not is_lora_state_key(name):
            continue
        name = _strip_prefix(name, strip_prefixes)
        if name.endswith(".lora_A.weight"):
            name = name.replace(".lora_A.weight", ".lora_A.default.weight")
        elif name.endswith(".lora_B.weight"):
            name = name.replace(".lora_B.weight", ".lora_B.default.weight")
        elif name.endswith(".lora_A.default.weight") or name.endswith(".lora_B.default.weight"):
            pass
        elif name.endswith(".alpha"):
            alpha_by_target[name[: -len(".alpha")]] = tensor.detach().cpu()
            continue
        else:
            continue
        lora_targets.add(name.split(".lora_")[0])
        exported[name] = tensor.detach().cpu().contiguous()

    if include_alpha:
        ranks = defaultdict(lambda: None)
        for name, tensor in exported.items():
            if name.endswith(".lora_A.default.weight"):
                ranks[name.split(".lora_A.")[0]] = tensor.shape[0]
        for target in sorted(lora_targets):
            alpha = alpha_by_target.get(target)
            if alpha is None:
                alpha = torch.tensor(float(ranks[target]), dtype=torch.float32)
            exported[f"{target}.alpha"] = alpha.reshape(()).detach().cpu()

    return dict(sorted(exported.items()))


def import_lora_state_dict_for_runtime(
    state_dict: dict[str, torch.Tensor],
    *,
    strip_prefixes: Iterable[str] = ("model.", "pipe.dit."),
    add_prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Convert exported LoRA keys back to the local LoRALinear state format.

    Training checkpoints store runtime keys such as
    ``model.blocks.0.q.lora_A.weight``.  The exported safetensors file strips
    ``model.`` and uses DiffSynth/ComfyUI-style ``.default.weight`` keys.  This
    helper accepts either form and returns keys loadable into the module that
    actually owns the patched Linear layers, normally ``pipe.generator.model``.
    """

    imported = {}
    for name, tensor in state_dict.items():
        name = normalize_state_key(name)
        name = _strip_prefix(name, strip_prefixes)
        if name.endswith(".lora_A.default.weight"):
            name = name.replace(".lora_A.default.weight", ".lora_A.weight")
        elif name.endswith(".lora_B.default.weight"):
            name = name.replace(".lora_B.default.weight", ".lora_B.weight")
        elif name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight") or name.endswith(".alpha"):
            pass
        else:
            continue
        if add_prefix and not name.startswith(add_prefix):
            name = f"{add_prefix}{name}"
        imported[name] = tensor.detach().cpu().contiguous()
    return dict(sorted(imported.items()))


def load_generator_lora_checkpoint(path: str | Path, *, source: str = "auto") -> dict[str, torch.Tensor]:
    """Load a DMD generator LoRA state from safetensors or a trainer model.pt."""

    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported LoRA checkpoint payload: {path}")

    if source not in {"auto", "ema", "generator"}:
        raise ValueError("lora source must be one of: auto, ema, generator")
    if source in {"auto", "ema"} and "generator_ema_lora" in checkpoint:
        return checkpoint["generator_ema_lora"]
    if source in {"auto", "generator"} and "generator_lora" in checkpoint:
        return checkpoint["generator_lora"]
    if all(is_lora_state_key(key) for key in checkpoint):
        return checkpoint
    raise RuntimeError(f"No generator LoRA state found in checkpoint: {path}")


def load_lora_state_dict(module: nn.Module, state_dict: dict[str, torch.Tensor], *, strict_lora: bool = True):
    result = module.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA checkpoint keys: {result.unexpected_keys}")
    if strict_lora:
        expected = {name for name in module.state_dict() if is_lora_state_key(name)}
        provided = {name for name in state_dict if is_lora_state_key(name)}
        missing = sorted(expected - provided)
        if missing:
            raise RuntimeError(f"Missing LoRA checkpoint keys: {missing[:20]}")
    return result


def lora_enabled(config) -> bool:
    lora = getattr(config, "lora", None)
    if lora is None:
        return False
    if isinstance(lora, dict):
        return bool(lora.get("enabled", False))
    return bool(lora.get("enabled", False))


def apply_configured_lora(wrapper: nn.Module, config, *, role: str):
    if not lora_enabled(config):
        return None
    lora = config.lora
    targets = tuple(lora.get("target_modules", lora.get("targets", DEFAULT_TARGET_MODULES)))
    report = apply_lora_to_model(
        wrapper.model,
        rank=int(lora.get("rank", 64)),
        alpha=float(lora.get("alpha", lora.get("rank", 64))),
        target_modules=targets,
        freeze_base=True,
    )
    wrapper.lora_report = {"role": role, **report}
    return wrapper.lora_report
