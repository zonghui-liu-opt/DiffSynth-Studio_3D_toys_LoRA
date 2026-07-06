import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics_utils import load_metrics, tokens_per_step


def ema(values, alpha=0.98):
    smoothed = []
    current = None
    for value in values:
        current = value if current is None else alpha * current + (1 - alpha) * value
        smoothed.append(current)
    return smoothed


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def build_series(rows, warmup_steps=3, tokens_per_video=None):
    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    step_times = [float(row["step_time_sec"]) for row in rows]
    tokens_sec = [tokens_per_step(row) / float(row["step_time_sec"]) for row in rows]
    if tokens_per_video is None:
        videos_hour = [
            3600.0 * float(row["samples_per_step"]) / float(row["step_time_sec"])
            for row in rows
        ]
    else:
        videos_hour = [3600.0 * value / float(tokens_per_video) for value in tokens_sec]
    steady_rows = rows[warmup_steps:]
    steady_tokens = tokens_sec[warmup_steps:]
    steady_videos = videos_hour[warmup_steps:]
    tail_start = max(0, int(len(losses) * 0.8))
    return {
        "steps": steps,
        "losses": losses,
        "step_times": step_times,
        "loss_ema": ema(losses),
        "tokens_sec": tokens_sec,
        "videos_hour": videos_hour,
        "steady_tokens_per_sec": mean(steady_tokens),
        "steady_videos_per_hour": mean(steady_videos),
        "last_20pct_avg_loss": mean(losses[tail_start:]),
        "total_steps": len(rows),
        "observed_time_hours": sum(step_times) / 3600.0,
        "steady_steps": len(steady_rows),
    }


def plot_loss(series, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series["steps"], series["losses"], color="#9ecae1", linewidth=1.0, label="raw loss")
    ax.plot(series["steps"], series["loss_ema"], color="#08519c", linewidth=2.0, label="EMA loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_throughput(series, output_path):
    fig, ax_left = plt.subplots(figsize=(10, 5))
    ax_right = ax_left.twinx()
    ax_left.plot(series["steps"], series["tokens_sec"], color="#238b45", linewidth=1.4, label="tokens/s")
    ax_right.plot(series["steps"], series["videos_hour"], color="#d94801", linewidth=1.4, label="videos/hour")
    ax_left.axhline(series["steady_tokens_per_sec"], color="#238b45", linestyle="--", alpha=0.5)
    ax_right.axhline(series["steady_videos_per_hour"], color="#d94801", linestyle="--", alpha=0.5)
    ax_left.set_xlabel("step")
    ax_left.set_ylabel("global tokens/s")
    ax_right.set_ylabel("videos/hour")
    ax_left.set_title("Training Throughput")
    ax_left.grid(True, alpha=0.25)
    ax_left.text(
        0.02,
        0.95,
        f"steady mean: {series['steady_tokens_per_sec']:.2f} tokens/s, "
        f"{series['steady_videos_per_hour']:.2f} videos/hour",
        transform=ax_left.transAxes,
        va="top",
    )
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(series):
    print(f"total_steps: {series['total_steps']}")
    print(f"last_20pct_avg_loss: {series['last_20pct_avg_loss']:.6f}")
    print(f"steady_tokens_per_sec: {series['steady_tokens_per_sec']:.2f}")
    print(f"steady_videos_per_hour: {series['steady_videos_per_hour']:.2f}")
    print(f"observed_time_hours: {series['observed_time_hours']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DiffSynth training metrics JSONL.")
    parser.add_argument("--metrics_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("metrics_plots"))
    parser.add_argument("--ema_alpha", type=float, default=0.98)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--tokens_per_video", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_metrics(args.metrics_path)
    if not rows:
        raise ValueError(f"No metrics rows found in {args.metrics_path}")
    series = build_series(rows, warmup_steps=args.warmup_steps, tokens_per_video=args.tokens_per_video)
    if args.ema_alpha != 0.98:
        series["loss_ema"] = ema(series["losses"], alpha=args.ema_alpha)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(series, args.output_dir / "loss.png")
    plot_throughput(series, args.output_dir / "throughput.png")
    print_summary(series)


if __name__ == "__main__":
    main()
