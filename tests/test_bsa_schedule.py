import json

import pytest

from diffsynth.diffusion.bsa_schedule import (
    available_bsa_schedule_presets,
    canonical_schedule_spec,
    compile_bsa_schedule,
    compiled_bsa_schedule_from_dict,
    load_bsa_schedule,
    schedule_spec_sha256,
)


@pytest.mark.parametrize(
    ("steps_per_epoch", "expected"),
    [
        (121, (363, 484, 605, 726, 847, 968, 1089, 1210, 1331)),
        (61, (183, 244, 305, 366, 427, 488, 549, 610, 671)),
    ],
)
def test_conservative_epoch_schedule_boundaries(steps_per_epoch, expected):
    schedule = compile_bsa_schedule(
        "conservative_epoch_v1",
        steps_per_epoch=steps_per_epoch,
        total_epochs=60,
        target_sparsity=0.8,
    )

    assert schedule.transition_steps == expected
    assert schedule.total_optimizer_steps == steps_per_epoch * 60
    assert schedule.sparsity_at(expected[0] - 1) == 0.0
    assert schedule.sparsity_at(expected[0]) == 0.1
    assert schedule.sparsity_at(expected[-1] - 1) == 0.75
    assert schedule.sparsity_at(expected[-1]) == 0.8
    assert schedule.sparsity_at(schedule.total_optimizer_steps) == 0.8


def test_legacy_progress_schedule_exactly_reproduces_boundaries():
    schedule = compile_bsa_schedule(
        "legacy_progress_v1", total_optimizer_steps=1000, target_sparsity="0.8"
    )

    assert schedule.transition_steps == (50, 100, 150, 200, 250, 300, 350, 400)
    assert schedule.sparsities == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    assert schedule.sparsity_at(49) == 0.0
    assert schedule.sparsity_at(50) == 0.1
    assert schedule.sparsity_at(400) == 0.8


def test_legacy_progress_schedule_collapses_tiny_run_boundaries():
    schedule = compile_bsa_schedule(
        "legacy_progress_v1", total_optimizer_steps=2, target_sparsity=0.8
    )

    assert schedule.transition_steps == (1,)
    assert schedule.sparsities == (0.0, 0.8)
    assert [schedule.sparsity_at(step) for step in range(3)] == [0.0, 0.8, 0.8]

    one_step = compile_bsa_schedule(
        "legacy_progress_v1", total_optimizer_steps=1, target_sparsity=0.8
    )
    assert one_step.transition_steps == (1,)
    assert [one_step.sparsity_at(step) for step in range(2)] == [0.0, 0.8]


def test_loads_inline_and_at_file_json_with_stable_canonical_hash(tmp_path):
    payload = {
        "schema_version": 1,
        "basis": "optimizer_step",
        "stages": [
            {"name": "dense", "sparsity": 0, "duration": 2},
            {"sparsity": "0.80", "duration": "remainder"},
        ],
    }
    inline = json.dumps(payload)
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    inline_spec = load_bsa_schedule(inline)
    file_spec = load_bsa_schedule(f"@{path}")
    assert canonical_schedule_spec(inline_spec) == canonical_schedule_spec(file_spec)
    assert schedule_spec_sha256(inline_spec) == schedule_spec_sha256(file_spec)
    assert compile_bsa_schedule(file_spec, total_optimizer_steps=10).transition_steps == (2,)


def test_runtime_manifest_contains_canonical_schedule_identity():
    schedule = compile_bsa_schedule(
        "mature_same_shape_v1",
        steps_per_epoch=10,
        total_epochs=20,
        target_sparsity=0.8,
    )
    manifest = schedule.to_dict()

    assert manifest["schedule_sha256"] == schedule_spec_sha256(manifest["schedule_spec"])
    assert manifest["schedule_runtime"]["transition_steps"] == list(schedule.transition_steps)
    assert compiled_bsa_schedule_from_dict(manifest, target_sparsity=0.8) == schedule

    manifest["schedule_runtime"]["transition_steps"][0] += 1
    with pytest.raises(ValueError, match="schedule_runtime"):
        compiled_bsa_schedule_from_dict(manifest, target_sparsity=0.8)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"schema_version": 2, "basis": "epoch", "stages": []}, "schema_version"),
        ({"schema_version": 1, "basis": "batch", "stages": []}, "basis"),
        (
            {
                "schema_version": 1,
                "basis": "progress",
                "stages": [
                    {"sparsity": 0.5, "duration": 0.5},
                    {"sparsity": 0.4, "duration": "remainder"},
                ],
            },
            "non-decreasing",
        ),
        (
            {
                "schema_version": 1,
                "basis": "epoch",
                "stages": [
                    {"sparsity": 0, "duration": "remainder"},
                    {"sparsity": 0.8, "duration": 1},
                ],
            },
            "final",
        ),
    ],
)
def test_rejects_invalid_schedule_specs(payload, error):
    with pytest.raises(ValueError, match=error):
        load_bsa_schedule(payload)


def test_rejects_duration_that_does_not_cover_run():
    payload = {
        "schema_version": 1,
        "basis": "optimizer_step",
        "stages": [
            {"sparsity": 0, "duration": 2},
            {"sparsity": 0.8, "duration": 7},
        ],
    }
    with pytest.raises(ValueError, match="must cover 10"):
        compile_bsa_schedule(payload, total_optimizer_steps=10)


def test_rejects_target_sparsity_mismatch():
    payload = {
        "schema_version": 1,
        "basis": "progress",
        "stages": [
            {"sparsity": 0, "duration": 0.1},
            {"sparsity": 0.8, "duration": "remainder"},
        ],
    }
    with pytest.raises(ValueError, match="does not match target_sparsity"):
        compile_bsa_schedule(
            payload,
            total_optimizer_steps=100,
            target_sparsity=0.75,
        )


def test_preset_resolves_runtime_target_into_manifest_identity():
    schedule = compile_bsa_schedule(
        "conservative_epoch_v1",
        steps_per_epoch=121,
        total_epochs=60,
        target_sparsity=0.85,
    )

    assert schedule.sparsities[-1] == 0.85
    assert schedule.spec.stages[-1].sparsity != "target"
    assert schedule.to_dict()["schedule_spec"]["stages"][-1]["sparsity"] == "0.85"


def test_target_placeholder_requires_runtime_target():
    with pytest.raises(ValueError, match="provide target_sparsity"):
        compile_bsa_schedule("legacy_progress_v1", total_optimizer_steps=1000)


def test_presets_are_versioned_and_available():
    assert available_bsa_schedule_presets() == (
        "conservative_epoch_v1",
        "legacy_progress_v1",
        "mature_same_shape_v1",
    )
