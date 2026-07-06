"""Offline metrics helpers for Wan2.2 DMD training."""

from __future__ import annotations

from metrics_utils import tokens_per_sample


def build_dmd_metrics_record(
    *,
    step: int,
    critic_loss: float,
    generator_loss: float | None,
    step_time_sec: float,
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
    world_size: int,
    train_generator: bool,
    lr_generator: float,
    lr_critic: float,
    dfake_gen_update_ratio: int,
) -> dict:
    """Build a plot_metrics-compatible JSONL record for one DMD update step."""

    samples_per_step = int(batch_size) * int(world_size)
    if train_generator:
        samples_per_step *= 2

    record = {
        "step": int(step),
        "epoch": 0,
        "loss": float(critic_loss),
        "critic_loss": float(critic_loss),
        "step_time_sec": float(step_time_sec),
        "tokens_per_sample": tokens_per_sample(num_frames, height, width),
        "samples_per_step": samples_per_step,
        "lr": float(lr_critic),
        "lr_critic": float(lr_critic),
        "train_generator": bool(train_generator),
        "dfake_gen_update_ratio": int(dfake_gen_update_ratio),
    }
    if generator_loss is not None:
        record["generator_loss"] = float(generator_loss)
        record["lr_generator"] = float(lr_generator)
    return record
