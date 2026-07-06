import os

import pytest

from plot_metrics import build_series


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def load_metrics_builder():
    try:
        from dmd.training_metrics import build_dmd_metrics_record
    except ModuleNotFoundError:
        pytest.fail("DMD offline metrics helper is missing")
    return build_dmd_metrics_record


def test_dmd_metrics_record_is_plot_metrics_compatible_for_generator_step():
    build_dmd_metrics_record = load_metrics_builder()
    record = build_dmd_metrics_record(
        step=5,
        critic_loss=0.75,
        generator_loss=0.25,
        step_time_sec=12.0,
        num_frames=121,
        height=480,
        width=832,
        batch_size=1,
        world_size=4,
        train_generator=True,
        lr_generator=5e-5,
        lr_critic=4e-5,
        dfake_gen_update_ratio=5,
    )

    assert record["step"] == 5
    assert record["epoch"] == 0
    assert record["loss"] == 0.75
    assert record["critic_loss"] == 0.75
    assert record["generator_loss"] == 0.25
    assert record["step_time_sec"] == 12.0
    assert record["tokens_per_sample"] == 12090
    assert record["samples_per_step"] == 8
    assert record["lr"] == 4e-5
    assert record["lr_generator"] == 5e-5
    assert record["lr_critic"] == 4e-5
    assert record["train_generator"] is True
    assert record["dfake_gen_update_ratio"] == 5


def test_dmd_metrics_record_omits_generator_loss_on_critic_only_step():
    build_dmd_metrics_record = load_metrics_builder()
    record = build_dmd_metrics_record(
        step=6,
        critic_loss=0.5,
        generator_loss=None,
        step_time_sec=10.0,
        num_frames=121,
        height=480,
        width=832,
        batch_size=1,
        world_size=4,
        train_generator=False,
        lr_generator=5e-5,
        lr_critic=4e-5,
        dfake_gen_update_ratio=5,
    )

    assert record["loss"] == 0.5
    assert "generator_loss" not in record
    assert record["samples_per_step"] == 4
    assert record["train_generator"] is False


def test_dmd_train_cli_and_launcher_expose_metrics_path():
    train_py = open(
        os.path.join(REPO_ROOT, "third_party", "wan22_turbo", "train.py"),
        encoding="utf-8",
    ).read()
    script = open(os.path.join(REPO_ROOT, "train_figurine360_dmd_lora.sh"), encoding="utf-8").read()
    assert 'parser.add_argument("--metrics_path"' in train_py
    assert "config.metrics_path = args.metrics_path" in train_py
    assert "METRICS_PATH=${METRICS_PATH:-$OUTPUT_ROOT/metrics.jsonl}" in script
    assert "--metrics_path \"$METRICS_PATH\"" in script


def test_plot_metrics_builds_dmd_loss_series_when_fields_are_present():
    rows = [
        {
            "step": 1,
            "epoch": 0,
            "loss": 0.7,
            "critic_loss": 0.7,
            "generator_loss": 0.2,
            "step_time_sec": 10.0,
            "tokens_per_sample": 12090,
            "samples_per_step": 8,
            "lr": 4e-5,
        },
        {
            "step": 2,
            "epoch": 0,
            "loss": 0.6,
            "critic_loss": 0.6,
            "step_time_sec": 8.0,
            "tokens_per_sample": 12090,
            "samples_per_step": 4,
            "lr": 4e-5,
        },
    ]

    series = build_series(rows)

    assert series["has_dmd_losses"] is True
    assert series["critic_losses"] == [0.7, 0.6]
    assert series["generator_steps"] == [1]
    assert series["generator_losses"] == [0.2]
