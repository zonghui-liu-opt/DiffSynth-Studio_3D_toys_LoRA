import json
import hashlib

import pytest
import torch

from diffsynth.models.wan_video_bsa import (
    WanBSAConfig,
    bsa_config_manifest,
    resolve_bsa_checkpoint,
    save_composite_bsa_checkpoint,
    validate_bsa_manifest,
    validate_wan_bsa_provenance,
)
from diffsynth.diffusion.bsa_schedule import compile_bsa_schedule
from diffsynth.diffusion.runner import (
    advance_completed_optimizer_steps,
    build_optimizer_param_groups,
)


def _state_dict():
    return {
        "blocks.0.self_attn.q.lora_A.default.weight": torch.randn(2, 4),
        "blocks.0.self_attn.q.lora_B.default.weight": torch.randn(4, 2),
        "blocks.0.self_attn.bsa_gate_down.weight": torch.randn(2, 4),
        "blocks.0.self_attn.bsa_gate_up.weight": torch.randn(4, 2),
    }


def test_composite_checkpoint_is_atomic_complete_and_explicit_weight_continuation(tmp_path):
    config = WanBSAConfig(backend="eager_math")
    manifest = bsa_config_manifest(
        config,
        completed_optimizer_steps=200,
        max_optimizer_steps=3000,
    )
    checkpoint = tmp_path / "checkpoint-step-0000200"
    save_composite_bsa_checkpoint(_state_dict(), str(checkpoint), manifest)

    assert (checkpoint / "checkpoint_complete").is_file()
    assert (checkpoint / "direct_distill_lora.safetensors").is_file()
    assert (checkpoint / "bsa_adapter.safetensors").is_file()
    assert not (tmp_path / "checkpoint-step-0000200.tmp").exists()
    loaded = resolve_bsa_checkpoint(str(checkpoint), config)
    assert loaded["manifest"]["completed_optimizer_steps"] == 200
    assert loaded["manifest"]["resume_semantics"] == "weight_continuation"
    assert loaded["manifest"]["optimizer_state_saved"] is False

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_composite_bsa_checkpoint(_state_dict(), str(checkpoint), manifest)


def test_incomplete_or_wrong_granularity_checkpoint_fails(tmp_path):
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError, match="Incomplete BSA checkpoint"):
        resolve_bsa_checkpoint(str(incomplete), WanBSAConfig())

    checkpoint = tmp_path / "bad"
    manifest = bsa_config_manifest(WanBSAConfig())
    save_composite_bsa_checkpoint(_state_dict(), str(checkpoint), manifest)
    config_path = checkpoint / "bsa_config.json"
    payload = json.loads(config_path.read_text())
    payload["gate_granularity"] = "token"
    config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="block-gate"):
        resolve_bsa_checkpoint(str(checkpoint), WanBSAConfig())


def test_checkpoint_provenance_requires_exact_figurine_and_dense_warmstart(tmp_path):
    figurine = tmp_path / "figurine.safetensors"
    dense = tmp_path / "dense.safetensors"
    figurine.write_bytes(b"figurine")
    dense.write_bytes(b"dense-19600")

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "figurine_lora_sha256": sha256(figurine),
        "direct_distill_warmstart_sha256": sha256(dense),
    }
    actual = validate_wan_bsa_provenance(
        manifest,
        figurine_lora_path=str(figurine),
        direct_distill_warmstart_path=str(dense),
    )
    assert actual == manifest

    dense.write_bytes(b"wrong-dense")
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_wan_bsa_provenance(
            manifest,
            figurine_lora_path=str(figurine),
            direct_distill_warmstart_path=str(dense),
        )

    with pytest.raises(ValueError, match="missing required provenance"):
        validate_wan_bsa_provenance(
            {"figurine_lora_sha256": manifest["figurine_lora_sha256"]},
            figurine_lora_path=str(figurine),
            direct_distill_warmstart_path=str(dense),
        )


def test_schema_v2_checkpoint_round_trip_and_schedule_validation(tmp_path):
    config = WanBSAConfig(backend="eager_math")
    schedule = compile_bsa_schedule(
        "conservative_epoch_v1",
        steps_per_epoch=121,
        total_epochs=60,
        target_sparsity=config.target_sparsity,
    )
    manifest = bsa_config_manifest(
        config,
        schema_version=2,
        completed_optimizer_steps=200,
        max_optimizer_steps=schedule.total_optimizer_steps,
        **schedule.to_dict(),
    )
    checkpoint = tmp_path / "checkpoint-step-0000200"
    save_composite_bsa_checkpoint(_state_dict(), str(checkpoint), manifest)

    loaded = resolve_bsa_checkpoint(str(checkpoint), config)["manifest"]
    assert loaded["schema_version"] == 2
    assert loaded["schedule_sha256"] == schedule.spec_sha256
    assert loaded["schedule_runtime"]["transition_steps"] == list(
        schedule.transition_steps
    )

    invalid = dict(manifest, completed_optimizer_steps=schedule.total_optimizer_steps + 1)
    with pytest.raises(ValueError, match="outside the saved schedule"):
        validate_bsa_manifest(invalid, config)

    invalid = dict(manifest)
    invalid.pop("schedule_sha256")
    with pytest.raises(ValueError, match="missing schedule fields"):
        validate_bsa_manifest(invalid, config)


class _GroupedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = torch.nn.Parameter(torch.ones(1))
        self.gate = torch.nn.Parameter(torch.ones(1))

    def trainable_modules(self):
        return self.parameters()

    def optimizer_param_groups(self, learning_rate, weight_decay, args=None):
        return [
            {"name": "direct_distill_lora", "params": [self.lora], "lr": learning_rate},
            {"name": "bsa_gate", "params": [self.gate], "lr": 2e-5},
        ]


def test_optimizer_groups_cover_trainables_once_and_step_counter_only_advances_on_success():
    model = _GroupedModel()
    groups = build_optimizer_param_groups(model, 2e-6, 0.0)
    assert [group["name"] for group in groups] == ["direct_distill_lora", "bsa_gate"]
    assert advance_completed_optimizer_steps(7, sync_gradients=False, step_was_skipped=False) == 7
    assert advance_completed_optimizer_steps(7, sync_gradients=True, step_was_skipped=True) == 7
    assert advance_completed_optimizer_steps(7, sync_gradients=True, step_was_skipped=False) == 8

    model.optimizer_param_groups = lambda *args, **kwargs: [
        {"name": "bad", "params": [model.lora, model.lora]}
    ]
    with pytest.raises(RuntimeError, match="exactly once"):
        build_optimizer_param_groups(model, 2e-6, 0.0)
