import json

import pytest

from metrics_utils import MetricsWriter, load_metrics, tokens_per_sample, tokens_per_step


def test_tokens_per_sample_matches_wan22_ti2v_known_value():
    assert tokens_per_sample(num_frames=121, height=480, width=832) == 12090


@pytest.mark.parametrize(
    ("num_frames", "height", "width", "message"),
    [
        (12, 480, 832, "num_frames"),
        (121, 481, 832, "height"),
        (121, 480, 833, "width"),
    ],
)
def test_tokens_per_sample_rejects_invalid_geometry(num_frames, height, width, message):
    with pytest.raises(ValueError, match=message):
        tokens_per_sample(num_frames=num_frames, height=height, width=width)


def test_metrics_writer_appends_jsonl_and_creates_parent(tmp_path):
    metrics_path = tmp_path / "nested" / "metrics.jsonl"
    writer = MetricsWriter(metrics_path)

    writer.write(
        {
            "step": 1,
            "epoch": 0,
            "loss": 0.5,
            "step_time_sec": 2.0,
            "tokens_per_sample": 12090,
            "samples_per_step": 4,
            "lr": 1e-4,
        }
    )
    writer.write(
        {
            "step": 2,
            "epoch": 0,
            "loss": 0.4,
            "step_time_sec": 1.5,
            "tokens_per_sample": 12090,
            "samples_per_step": 4,
            "lr": 1e-4,
        }
    )

    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert rows[0]["loss"] == 0.5


def test_load_metrics_deduplicates_steps_last_record_wins(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    writer = MetricsWriter(metrics_path)
    writer.write({"step": 1, "epoch": 0, "loss": 0.6, "step_time_sec": 2.0, "tokens_per_sample": 10, "samples_per_step": 2, "lr": 1e-4})
    writer.write({"step": 2, "epoch": 0, "loss": 0.5, "step_time_sec": 2.0, "tokens_per_sample": 10, "samples_per_step": 2, "lr": 1e-4})
    writer.write({"step": 2, "epoch": 0, "loss": 0.4, "step_time_sec": 1.0, "tokens_per_sample": 10, "samples_per_step": 2, "lr": 1e-4})

    rows = load_metrics(metrics_path)

    assert [row["step"] for row in rows] == [1, 2]
    assert rows[-1]["loss"] == 0.4
    assert tokens_per_step(rows[-1]) == 20
