import torch
from torch import nn

from dmd.wan22_lora import (
    LoRALinear,
    apply_lora_to_model,
    export_lora_state_dict,
)


class TinyWanBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)
        self.ffn = nn.Sequential(nn.Linear(4, 8, bias=False), nn.GELU(), nn.Linear(8, 4, bias=False))
        self.keep = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.keep(self.q(x) + self.k(x) + self.ffn(x))


def test_lora_linear_is_zero_delta_at_initialization():
    torch.manual_seed(0)
    base = nn.Linear(4, 3, bias=True)
    wrapped = LoRALinear(base, rank=2, alpha=2)

    x = torch.randn(5, 4)

    assert torch.allclose(wrapped(x), base(x))
    assert wrapped.weight.requires_grad is False
    assert wrapped.lora_A.weight.requires_grad is True
    assert wrapped.lora_B.weight.requires_grad is True
    assert torch.count_nonzero(wrapped.lora_B.weight) == 0


def test_apply_lora_to_model_wraps_only_requested_leaf_modules_and_freezes_base():
    model = nn.Module()
    model.blocks = nn.ModuleList([TinyWanBlock()])

    report = apply_lora_to_model(model, rank=2, alpha=2, target_modules=("q", "k", "ffn.0", "ffn.2"))

    assert report["patched_count"] == 4
    assert isinstance(model.blocks[0].q, LoRALinear)
    assert isinstance(model.blocks[0].k, LoRALinear)
    assert isinstance(model.blocks[0].ffn[0], LoRALinear)
    assert isinstance(model.blocks[0].ffn[2], LoRALinear)
    assert not isinstance(model.blocks[0].keep, LoRALinear)

    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {
        "blocks.0.q.lora_A.weight",
        "blocks.0.q.lora_B.weight",
        "blocks.0.k.lora_A.weight",
        "blocks.0.k.lora_B.weight",
        "blocks.0.ffn.0.lora_A.weight",
        "blocks.0.ffn.0.lora_B.weight",
        "blocks.0.ffn.2.lora_A.weight",
        "blocks.0.ffn.2.lora_B.weight",
    }


def test_export_lora_state_dict_strips_turbo_wrapper_prefix_and_adds_alpha():
    model = nn.Module()
    model.model = TinyWanBlock()
    apply_lora_to_model(model, rank=2, alpha=2, target_modules=("q",))

    exported = export_lora_state_dict(model.state_dict(), strip_prefixes=("model.",), include_alpha=True)

    assert sorted(exported) == [
        "q.alpha",
        "q.lora_A.default.weight",
        "q.lora_B.default.weight",
    ]
    assert exported["q.lora_A.default.weight"].shape == (2, 4)
    assert exported["q.lora_B.default.weight"].shape == (4, 2)
    assert exported["q.alpha"].item() == 2
