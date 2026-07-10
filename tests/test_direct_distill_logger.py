import importlib.util
import json
from pathlib import Path

import pytest
import torch


LOGGER_PATH = Path(__file__).parents[1] / "diffsynth/diffusion/logger.py"
SPEC = importlib.util.spec_from_file_location("direct_distill_logger", LOGGER_PATH)
LOGGER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOGGER_MODULE)
ModelLogger = LOGGER_MODULE.ModelLogger


class FakeAccelerator:
    def __init__(self, is_main_process=True):
        self.is_main_process = is_main_process


class FakeBackend:
    def __init__(self):
        self.events = []
        self.closed = False

    def log(self, key, value, step):
        self.events.append((key, value, step))

    def close(self):
        self.closed = True


class FakeBatchBackend(FakeBackend):
    def log_metrics(self, metrics, step):
        self.events.append((metrics, step))


def model_logger_with_backend(tmp_path, enable_metrics_jsonl=False):
    model_logger = ModelLogger(tmp_path, enable_metrics_jsonl=enable_metrics_jsonl)
    backend = FakeBackend()
    model_logger.loggers = [backend]
    model_logger.loggers_initialized = True
    return model_logger, backend


def test_legacy_loss_still_logs_without_creating_jsonl(tmp_path):
    model_logger, backend = model_logger_with_backend(tmp_path)

    model_logger.on_step_end(FakeAccelerator(), None, loss=torch.tensor(1.25, requires_grad=True))

    assert backend.events == [("loss", 1.25, 1)]
    assert model_logger.num_steps == 1
    assert not (tmp_path / "metrics.jsonl").exists()


def test_default_logger_keeps_legacy_no_backend_noop(tmp_path):
    model_logger = ModelLogger(tmp_path)

    model_logger.on_step_end(FakeAccelerator(), None, loss=object())

    assert model_logger.num_steps == 1
    assert not (tmp_path / "metrics.jsonl").exists()


def test_all_finite_scalar_metrics_reach_backend_and_jsonl(tmp_path):
    model_logger, backend = model_logger_with_backend(tmp_path, enable_metrics_jsonl=True)
    accelerator = FakeAccelerator()

    model_logger.on_step_end(
        accelerator,
        None,
        loss=torch.tensor(3.0),
        metrics={
            "train/loss": torch.tensor(2.0, requires_grad=True),
            "train/loss_ema": 2,
            "train/lr": 1e-5,
        },
    )
    model_logger.on_step_end(
        accelerator,
        None,
        metrics={"train/loss": 1.5, "train/loss_ema": torch.tensor(1.9)},
    )

    assert backend.events == [
        ("loss", 3.0, 1),
        ("train/loss", 2.0, 1),
        ("train/loss_ema", 2.0, 1),
        ("train/lr", 1e-5, 1),
        ("train/loss", 1.5, 2),
        ("train/loss_ema", pytest.approx(1.9), 2),
    ]
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records == [
        {
            "step": 1,
            "loss": 3.0,
            "train/loss": 2.0,
            "train/loss_ema": 2.0,
            "train/lr": 1e-5,
        },
        {"step": 2, "train/loss": 1.5, "train/loss_ema": pytest.approx(1.9)},
    ]

    model_logger.on_training_end(accelerator, None)
    assert backend.closed
    assert model_logger.metrics_jsonl_file is None


def test_batch_backend_receives_all_metrics_in_one_call(tmp_path):
    model_logger = ModelLogger(tmp_path)
    backend = FakeBatchBackend()
    model_logger.loggers = [backend]
    model_logger.loggers_initialized = True

    model_logger.on_step_end(
        FakeAccelerator(),
        None,
        loss=3.0,
        metrics={"train/loss": 2.0, "train/lr": 1e-5},
    )

    assert backend.events == [
        ({"loss": 3.0, "train/loss": 2.0, "train/lr": 1e-5}, 1),
    ]


def test_non_main_process_does_not_log_or_write_jsonl(tmp_path):
    model_logger, backend = model_logger_with_backend(tmp_path, enable_metrics_jsonl=True)

    model_logger.on_step_end(
        FakeAccelerator(is_main_process=False),
        None,
        metrics={"train/loss": 1.0},
    )

    assert model_logger.num_steps == 1
    assert backend.events == []
    assert not (tmp_path / "metrics.jsonl").exists()


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("train/loss", float("nan"), ValueError),
        ("train/loss", float("inf"), ValueError),
        ("train/loss", torch.tensor([1.0, 2.0]), ValueError),
        ("train/loss", "1.0", TypeError),
        ("train/enabled", True, TypeError),
        ("step", 1.0, ValueError),
        (1, 1.0, TypeError),
    ],
)
def test_invalid_metrics_fail_before_incrementing_step(tmp_path, key, value, error):
    model_logger = ModelLogger(tmp_path, enable_metrics_jsonl=True)

    with pytest.raises(error):
        model_logger.on_step_end(FakeAccelerator(), None, metrics={key: value})

    assert model_logger.num_steps == 0
    assert not (tmp_path / "metrics.jsonl").exists()


def test_metrics_argument_must_be_a_dict(tmp_path):
    model_logger = ModelLogger(tmp_path)

    with pytest.raises(TypeError, match="must be a dict"):
        model_logger.on_step_end(FakeAccelerator(), None, metrics=[("train/loss", 1.0)])


def test_metrics_jsonl_refuses_to_append_a_reset_run(tmp_path):
    (tmp_path / "metrics.jsonl").write_text('{"step": 99}\n', encoding="utf-8")
    model_logger = ModelLogger(tmp_path, enable_metrics_jsonl=True)
    with pytest.raises(FileExistsError, match="Use a new output directory"):
        model_logger.on_step_end(
            FakeAccelerator(), None, metrics={"train/loss": 1.0}
        )
