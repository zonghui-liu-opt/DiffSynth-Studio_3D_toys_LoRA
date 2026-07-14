import torch

from diffsynth.diffusion.base_pipeline import BasePipeline


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(2, 2, bias=False)


def _lora_state(target="q"):
    return {
        f"{target}.lora_A.weight": torch.ones(1, 2),
        f"{target}.lora_B.weight": torch.ones(2, 1),
    }


def test_base_pipeline_load_lora_returns_fused_match_count():
    pipe = BasePipeline(device="cpu", torch_dtype=torch.float32)
    model = _ToyModel()
    original = model.q.weight.detach().clone()
    matched = pipe.load_lora(
        model,
        state_dict=_lora_state(),
        hotload=False,
        verbose=0,
    )
    assert matched == 1
    assert torch.equal(model.q.weight, original + torch.ones_like(original))


def test_base_pipeline_load_lora_reports_zero_for_incompatible_adapter():
    pipe = BasePipeline(device="cpu", torch_dtype=torch.float32)
    model = _ToyModel()
    matched = pipe.load_lora(
        model,
        state_dict=_lora_state("does_not_exist"),
        hotload=False,
        verbose=0,
    )
    assert matched == 0
