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
    optimizer_steps_per_epoch,
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
    assert optimizer_steps_per_epoch(121, 1) == 121
    assert optimizer_steps_per_epoch(121, 2) == 61


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


class _TinyBSAResumeModel(_TinyTrainingModel):
    enable_bsa = True

    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.bsa_completed_optimizer_steps = 3
        self.bsa_max_optimizer_steps = None
        self.bsa_loss_ema = None
        self._wan_bsa_student_info = {
            "injected_bsa_modules": 1,
            "expected_self_attention_modules": 1,
            "cross_attention_bsa_modules": 0,
            "block_size": [4, 3, 6],
            "gate_type": "low_rank_dynamic",
            "gate_granularity": "block",
            "backend": "eager_math",
        }

    def initialize_bsa_training_schedule(self, steps_per_epoch, total_epochs):
        self.bsa_max_optimizer_steps = 4

        class Schedule:
            total_optimizer_steps = 4

            @staticmethod
            def to_dict():
                return {
                    "schedule_runtime": {
                        "steps_per_epoch": steps_per_epoch,
                        "planned_total_epochs": total_epochs,
                        "planned_total_optimizer_steps": 4,
                        "transition_steps": [2],
                        "sparsities": [0.0, 0.8],
                    }
                }

        return Schedule()

    def probe_bsa_backend(self, device):
        return None

    def write_bsa_student_info(self, include_runtime=False):
        return None

    def bsa_step_metrics(self, total_grad_norm=None):
        return {}

    def set_bsa_training_progress(self, completed_optimizer_steps, max_optimizer_steps):
        self.bsa_completed_optimizer_steps = completed_optimizer_steps
        self.bsa_max_optimizer_steps = max_optimizer_steps

    def forward(self, data, inputs=None):
        self.forward_calls += 1
        return super().forward(data, inputs=inputs)


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


def test_bsa_weight_continuation_stops_at_saved_plan_total(tmp_path):
    accelerator = Accelerator(gradient_accumulation_steps=1, cpu=True)
    logger = _CollectingLogger()
    model = _TinyBSAResumeModel()
    args = Namespace(
        learning_rate=1e-3,
        weight_decay=0.0,
        dataset_num_workers=0,
        save_steps=200,
        num_epochs=1,
        enable_model_cpu_offload=False,
        enable_optimizer_cpu_offload=False,
        cpu_offload_split_threshold=None,
        customized_optimizer=None,
        output_path=str(tmp_path),
        enable_direct_distill_metrics=False,
        direct_distill_loss_ema_beta=0.98,
        max_grad_norm=0.0,
    )

    launch_training_task(
        accelerator,
        _TinyDataset(),
        model,
        logger,
        args=args,
    )

    assert model.forward_calls == 1
    assert model.bsa_completed_optimizer_steps == 4
    assert len(logger.steps) == 1
