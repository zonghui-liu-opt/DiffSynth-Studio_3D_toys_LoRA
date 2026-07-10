from argparse import Namespace
import math

import pytest
import torch
from accelerate import Accelerator

from diffsynth.diffusion.runner import (
    aggregate_training_step_statistics,
    compute_training_workload,
    compute_video_tokens,
    launch_training_task,
    update_loss_ema,
)


def test_video_token_formula_and_loss_ema():
    assert compute_video_tokens((2, 16, 5, 7, 9), (2, 3, 4)) == 54
    assert compute_training_workload((2, 16, 5, 7, 9), (2, 3, 4), 4) == {
        "videos": 2,
        "video_tokens": 54,
        "model_tokens": 216,
    }
    assert update_loss_ema(None, 10, beta=0.98) == 10
    assert update_loss_ema(10, 20, beta=0.98) == pytest.approx(10.2)


def test_rank_statistics_use_sum_for_counts_and_max_for_time_and_memory():
    aggregated = aggregate_training_step_statistics([
        {
            "loss_sum": 4,
            "loss_weight": 2,
            "video_tokens": 100,
            "model_tokens": 400,
            "videos": 2,
            "step_time_sec": 2,
            "max_memory_gb": 3,
        },
        {
            "loss_sum": 9,
            "loss_weight": 3,
            "video_tokens": 150,
            "model_tokens": 600,
            "videos": 3,
            "step_time_sec": 3,
            "max_memory_gb": 4,
        },
    ])
    assert aggregated["loss"] == pytest.approx(13 / 5)
    assert aggregated["video_tokens"] == 250
    assert aggregated["model_tokens"] == 1000
    assert aggregated["videos"] == 5
    assert aggregated["step_time_sec"] == 3
    assert aggregated["max_memory_gb"] == 4
    assert aggregated["global_video_tokens_per_sec"] == pytest.approx(250 / 3)
    assert aggregated["global_model_tokens_per_sec"] == pytest.approx(1000 / 3)

    invalid = [{
        "loss_sum": float("nan"), "loss_weight": 1, "video_tokens": 1,
        "model_tokens": 4, "videos": 1, "step_time_sec": 1, "max_memory_gb": 0,
    }]
    with pytest.raises(ValueError, match="not finite"):
        aggregate_training_step_statistics(invalid)


class _TinyDataset(torch.utils.data.Dataset):
    load_from_cache = False

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return {"target": torch.tensor(float(index + 1))}


class _TinyTrainingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def trainable_modules(self):
        return filter(lambda parameter: parameter.requires_grad, self.parameters())

    def forward(self, data, inputs=None):
        self._last_training_step_context = {
            "latent_shape": (1, 1, 3, 4, 4),
            "patch_size": (1, 2, 2),
            "num_inference_steps": 4,
        }
        return (self.weight - data["target"]) ** 2


class _CollectingLogger:
    def __init__(self):
        self.steps = []

    def on_step_end(self, accelerator, model, save_steps=None, **kwargs):
        self.steps.append(kwargs)

    def on_epoch_end(self, accelerator, model, epoch_id):
        pass

    def on_training_end(self, accelerator, model, save_steps=None):
        pass


def test_metrics_mode_logs_only_real_optimizer_steps(tmp_path):
    accelerator = Accelerator(gradient_accumulation_steps=2, cpu=True)
    logger = _CollectingLogger()
    args = Namespace(
        learning_rate=1e-3,
        weight_decay=0.0,
        dataset_num_workers=0,
        save_steps=None,
        num_epochs=1,
        enable_model_cpu_offload=False,
        enable_optimizer_cpu_offload=False,
        cpu_offload_split_threshold=None,
        customized_optimizer=None,
        output_path=str(tmp_path),
        enable_direct_distill_metrics=True,
        direct_distill_loss_ema_beta=0.98,
    )

    launch_training_task(
        accelerator,
        _TinyDataset(),
        _TinyTrainingModel(),
        logger,
        args=args,
    )

    expected_steps = math.ceil(
        len(_TinyDataset()) / (accelerator.num_processes * accelerator.gradient_accumulation_steps)
    )
    assert len(logger.steps) == expected_steps
    for step in logger.steps:
        metrics = step["metrics"]
        assert metrics["throughput/global_model_tokens_per_sec"] > 0
        assert metrics["throughput/global_tokens_per_hour"] == pytest.approx(
            metrics["throughput/global_model_tokens_per_sec"] * 3600
        )
        assert metrics["throughput/global_videos_per_hour"] == pytest.approx(
            metrics["throughput/global_videos_per_sec"] * 3600
        )


def test_multi_process_model_cpu_offload_is_rejected_before_training(tmp_path):
    class FakeAccelerator:
        is_main_process = False
        num_processes = 2

    args = Namespace(
        learning_rate=1e-3,
        weight_decay=0.0,
        dataset_num_workers=0,
        save_steps=None,
        num_epochs=1,
        enable_model_cpu_offload=True,
        enable_optimizer_cpu_offload=False,
        cpu_offload_split_threshold=None,
        customized_optimizer=None,
        output_path=str(tmp_path),
        enable_direct_distill_metrics=True,
        direct_distill_loss_ema_beta=0.98,
    )
    with pytest.raises(ValueError, match="gradients would not synchronize"):
        launch_training_task(
            FakeAccelerator(),
            _TinyDataset(),
            _TinyTrainingModel(),
            _CollectingLogger(),
            args=args,
        )
