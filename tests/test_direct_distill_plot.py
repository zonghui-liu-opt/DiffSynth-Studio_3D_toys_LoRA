import builtins
import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "examples/wanvideo/model_training/special/direct_distill/plot_direct_distill_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("plot_direct_distill_metrics", SCRIPT_PATH)
PLOT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT_MODULE)


def write_metrics(path):
    records = [
        {
            "step": 1,
            "train/loss": 3.0,
            "train/loss_ema": 3.0,
            "train/lr": 1e-5,
            "throughput/global_video_tokens_per_sec": 100.0,
            "throughput/global_model_tokens_per_sec": 400.0,
            "throughput/global_tokens_per_hour": 1_440_000.0,
            "throughput/global_videos_per_sec": 0.5,
            "throughput/global_videos_per_hour": 1800.0,
            "throughput/step_time_sec": 2.0,
            "system/max_memory_gb": 20.0,
        },
        {
            "step": 2,
            "train/loss": 2.0,
            "train/loss_ema": 2.98,
            "train/lr": 1e-5,
            "throughput/global_video_tokens_per_sec": 120.0,
            "throughput/global_model_tokens_per_sec": 480.0,
            "throughput/global_tokens_per_hour": 1_728_000.0,
            "throughput/global_videos_per_sec": 0.6,
            "throughput/global_videos_per_hour": 2160.0,
            "throughput/step_time_sec": 1.8,
            "system/max_memory_gb": 21.0,
        },
        {
            "step": 3,
            "train/loss": 1.0,
            "train/loss_ema": 2.9404,
            "train/lr": 1e-5,
            "throughput/global_video_tokens_per_sec": 110.0,
            "throughput/global_model_tokens_per_sec": 440.0,
            "throughput/global_tokens_per_hour": 1_584_000.0,
            "throughput/global_videos_per_sec": 0.55,
            "throughput/global_videos_per_hour": 1980.0,
            "throughput/step_time_sec": 1.9,
            "system/max_memory_gb": 21.5,
        },
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return records


def test_plot_script_writes_png_csv_and_summary(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    expected_records = write_metrics(metrics_path)
    output_dir = tmp_path / "plots"

    paths = PLOT_MODULE.plot_direct_distill_metrics(metrics_path, output_dir)

    assert set(paths) == {"loss", "throughput", "csv", "summary"}
    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    assert paths["loss"].read_bytes().startswith(b"\x89PNG")
    assert paths["throughput"].read_bytes().startswith(b"\x89PNG")

    with paths["csv"].open(newline="", encoding="utf-8") as file:
        csv_records = list(csv.DictReader(file))
    assert len(csv_records) == len(expected_records)
    assert csv_records[0]["step"] == "1"
    assert csv_records[-1]["train/loss"] == "1.0"

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["num_records"] == 3
    assert summary["first_step"] == 1.0
    assert summary["last_step"] == 3.0
    assert summary["metrics"]["train/loss"] == {
        "count": 3,
        "min": 1.0,
        "max": 3.0,
        "mean": 2.0,
        "last": 1.0,
    }
    assert summary["metrics"]["throughput/global_model_tokens_per_sec"]["mean"] == pytest.approx(440.0)


def test_load_metrics_reports_bad_json_line(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text('{"step": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        PLOT_MODULE.load_metrics(metrics_path)


def test_missing_matplotlib_has_actionable_error(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="requires matplotlib"):
        PLOT_MODULE._load_pyplot()
