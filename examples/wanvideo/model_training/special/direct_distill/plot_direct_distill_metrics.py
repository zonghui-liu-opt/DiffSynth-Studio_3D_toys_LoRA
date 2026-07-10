import argparse
import csv
import json
import math
import numbers
from pathlib import Path


LOSS_KEY = "train/loss"
LOSS_EMA_KEY = "train/loss_ema"
TOKEN_RATE_KEYS = (
    "throughput/global_video_tokens_per_sec",
    "throughput/global_model_tokens_per_sec",
)
TOKEN_HOUR_KEY = "throughput/global_tokens_per_hour"
VIDEO_RATE_KEY = "throughput/global_videos_per_sec"
VIDEO_HOUR_KEY = "throughput/global_videos_per_hour"
STEP_TIME_KEY = "throughput/step_time_sec"
MEMORY_KEY = "system/max_memory_gb"


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Plotting DirectDistill metrics requires matplotlib. "
            "Install it in the offline environment before running this script."
        ) from error
    return plt


def _numeric_value(record, key, line_number):
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Line {line_number}: `{key}` must be a real numeric scalar.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Line {line_number}: `{key}` must be finite.")
    return value


def load_metrics(metrics_jsonl):
    metrics_jsonl = Path(metrics_jsonl)
    if not metrics_jsonl.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_jsonl}")

    records = []
    with metrics_jsonl.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {metrics_jsonl}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} of {metrics_jsonl} must contain a JSON object.")
            if "step" not in record:
                raise ValueError(f"Line {line_number} of {metrics_jsonl} is missing `step`.")
            _numeric_value(record, "step", line_number)
            records.append(record)

    if not records:
        raise ValueError(f"Metrics file contains no records: {metrics_jsonl}")
    return records


def _metric_pairs(records, key):
    pairs = []
    for line_number, record in enumerate(records, start=1):
        if key not in record:
            continue
        step = _numeric_value(record, "step", line_number)
        value = _numeric_value(record, key, line_number)
        pairs.append((step, value))
    return pairs


def write_metrics_csv(records, output_path):
    metric_keys = sorted({key for record in records for key in record if key != "step"})
    fieldnames = ["step", *metric_keys]
    with Path(output_path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def build_summary(records):
    metric_keys = sorted({key for record in records for key in record if key != "step"})
    summary = {
        "num_records": len(records),
        "first_step": _numeric_value(records[0], "step", 1),
        "last_step": _numeric_value(records[-1], "step", len(records)),
        "metrics": {},
    }
    for key in metric_keys:
        values = [value for _, value in _metric_pairs(records, key)]
        if not values:
            continue
        summary["metrics"][key] = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": math.fsum(values) / len(values),
            "last": values[-1],
        }
    return summary


def plot_loss(records, output_path, plt):
    raw = _metric_pairs(records, LOSS_KEY)
    ema = _metric_pairs(records, LOSS_EMA_KEY)
    if not raw:
        raise ValueError(f"Metrics do not contain required key `{LOSS_KEY}`.")
    if not ema:
        raise ValueError(f"Metrics do not contain required key `{LOSS_EMA_KEY}`.")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot([step for step, _ in raw], [value for _, value in raw], label="Raw loss", alpha=0.55)
    axis.plot([step for step, _ in ema], [value for _, value in ema], label="Loss EMA", linewidth=2)
    axis.set(title="DirectDistill loss", xlabel="Optimizer step", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_throughput(records, output_path, plt):
    panel_specs = [
        ("Token throughput", "Tokens / sec", TOKEN_RATE_KEYS),
        ("Hourly token throughput", "Tokens / hour", (TOKEN_HOUR_KEY,)),
        ("Video throughput", "Videos / sec", (VIDEO_RATE_KEY,)),
        ("Hourly video throughput", "Videos / hour", (VIDEO_HOUR_KEY,)),
        ("Optimizer step time", "Seconds", (STEP_TIME_KEY,)),
    ]
    if _metric_pairs(records, MEMORY_KEY):
        panel_specs.append(("Maximum CUDA memory", "GiB", (MEMORY_KEY,)))
    if not any(_metric_pairs(records, key) for _, _, keys in panel_specs for key in keys):
        raise ValueError("Metrics do not contain any supported throughput keys.")

    columns = 2
    rows = math.ceil(len(panel_specs) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.8 * rows), sharex=True)
    axes = list(axes.flat)
    for axis, (title, ylabel, keys) in zip(axes, panel_specs):
        has_series = False
        for key in keys:
            pairs = _metric_pairs(records, key)
            if not pairs:
                continue
            has_series = True
            axis.plot(
                [step for step, _ in pairs],
                [value for _, value in pairs],
                label=key.removeprefix("throughput/").removeprefix("system/"),
            )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if has_series:
            axis.legend()
    for axis in axes[len(panel_specs):]:
        axis.set_visible(False)
    for axis in axes[-columns:]:
        if axis.get_visible():
            axis.set_xlabel("Optimizer step")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_direct_distill_metrics(metrics_jsonl, output_dir=None):
    metrics_jsonl = Path(metrics_jsonl)
    output_dir = metrics_jsonl.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_metrics(metrics_jsonl)
    plt = _load_pyplot()

    paths = {
        "loss": output_dir / "loss.png",
        "throughput": output_dir / "throughput.png",
        "csv": output_dir / "metrics.csv",
        "summary": output_dir / "summary.json",
    }
    plot_loss(records, paths["loss"], plt)
    plot_throughput(records, paths["throughput"], plt)
    write_metrics_csv(records, paths["csv"])
    summary = build_summary(records)
    with paths["summary"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Plot DirectDistill metrics from metrics.jsonl.")
    parser.add_argument("metrics_jsonl", nargs="?", default="metrics.jsonl", help="Path to metrics.jsonl.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to the JSONL directory.")
    args = parser.parse_args()
    try:
        paths = plot_direct_distill_metrics(args.metrics_jsonl, args.output_dir)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
