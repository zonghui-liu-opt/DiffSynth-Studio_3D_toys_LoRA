import json
import math
import numbers
import os

import torch
from accelerate import Accelerator


class TensorBoardLogger:
    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard is enabled. Run `tensorboard --logdir={log_dir}` to visualize the training progress.")

    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)

    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.log(key, value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class SwanLabLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        import swanlab
        project_name = os.environ.get("SWANLAB_PROJECT", project_name)
        self.swanlab = swanlab
        self.swanlab.init(project=project_name, logdir=log_dir)
        print(f"SwanLab is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.swanlab.log({key: value}, step=step)

    def log_metrics(self, metrics, step):
        self.swanlab.log(metrics, step=step)

    def close(self):
        self.swanlab.finish()


class WandbLogger:
    def __init__(self, project_name="DiffSynth-Studio", log_dir=None):
        import wandb
        project_name = os.environ.get("WANDB_PROJECT", project_name)
        self.wandb = wandb
        self.run = self.wandb.init(project=project_name, dir=log_dir)
        print(f"Wandb is enabled. Project: {project_name}")

    def log(self, key, value, step):
        self.wandb.log({key: value}, step=step)

    def log_metrics(self, metrics, step):
        self.wandb.log(metrics, step=step)

    def close(self):
        self.wandb.finish()


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x,
        enable_tensorboard_log=False,
        enable_swanlab_log=False, swanlab_project="DiffSynth-Studio",
        enable_wandb_log=False, wandb_project="DiffSynth-Studio",
        enable_metrics_jsonl=False,
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        # Loggers
        self.enable_tensorboard_log = enable_tensorboard_log
        self.enable_swanlab_log = enable_swanlab_log
        self.swanlab_project = swanlab_project
        self.enable_wandb_log = enable_wandb_log
        self.wandb_project = wandb_project
        self.loggers = []
        self.loggers_initialized = False
        # JSONL metrics are opt-in so existing training jobs keep their current
        # output files and behavior unless explicitly enabled.
        self.enable_metrics_jsonl = enable_metrics_jsonl
        self.metrics_jsonl_path = os.path.join(self.output_path, "metrics.jsonl")
        self.metrics_jsonl_file = None

    def init_loggers(self):
        if self.enable_tensorboard_log:
            self.loggers.append(TensorBoardLogger(os.path.join(self.output_path, "tensorboard_log")))
        if self.enable_swanlab_log:
            self.loggers.append(SwanLabLogger(project_name=self.swanlab_project, log_dir=os.path.join(self.output_path, "swanlab_log")))
        if self.enable_wandb_log:
            self.loggers.append(WandbLogger(project_name=self.wandb_project, log_dir=os.path.join(self.output_path, "wandb_log")))
        self.loggers_initialized = True

    @staticmethod
    def _scalar_metric(key, value):
        if not isinstance(key, str) or not key:
            raise TypeError("Metric names must be non-empty strings.")
        if key == "step":
            raise ValueError("`step` is reserved for the logger step number.")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Metric `{key}` must be a scalar tensor, but has shape {tuple(value.shape)}.")
            value = value.detach().cpu().item()
        elif hasattr(value, "item") and not isinstance(value, (str, bytes)):
            # NumPy scalar values expose item(), without requiring NumPy as a
            # logging dependency.
            if getattr(value, "ndim", 0) != 0:
                raise ValueError(f"Metric `{key}` must be scalar, but received an array-like value.")
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"Metric `{key}` must be a real numeric scalar, but received {type(value).__name__}.")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Metric `{key}` must be finite, but received {value}.")
        return value

    def _prepare_metrics(self, metrics=None, loss=None):
        if metrics is not None and not isinstance(metrics, dict):
            raise TypeError(f"`metrics` must be a dict or None, but received {type(metrics).__name__}.")
        values = {}
        if loss is not None:
            values["loss"] = loss
        if metrics is not None:
            values.update(metrics)
        return {key: self._scalar_metric(key, value) for key, value in values.items()}

    def _write_metrics_jsonl(self, metrics):
        if self.metrics_jsonl_file is None:
            os.makedirs(self.output_path, exist_ok=True)
            if os.path.exists(self.metrics_jsonl_path):
                raise FileExistsError(
                    "Refusing to append a new run with reset step/EMA state to existing metrics: "
                    f"{self.metrics_jsonl_path}. Use a new output directory."
                )
            self.metrics_jsonl_file = open(self.metrics_jsonl_path, "x", encoding="utf-8")
        record = {"step": self.num_steps, **metrics}
        self.metrics_jsonl_file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.metrics_jsonl_file.flush()

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, metrics=None, **kwargs):
        scalar_metrics = {}
        has_live_logger = bool(self.loggers) or any((
            self.enable_tensorboard_log,
            self.enable_swanlab_log,
            self.enable_wandb_log,
        ))
        if accelerator.is_main_process and (metrics is not None or self.enable_metrics_jsonl or has_live_logger):
            scalar_metrics = self._prepare_metrics(metrics=metrics, loss=kwargs.get("loss"))
        self.num_steps += 1
        if accelerator.is_main_process:
            if not self.loggers_initialized:
                self.init_loggers()
            if scalar_metrics:
                for logger in self.loggers:
                    if hasattr(logger, "log_metrics"):
                        logger.log_metrics(scalar_metrics, self.num_steps)
                        continue
                    # Compatibility for custom logger implementations that only
                    # provide the original single-metric interface.
                    for key, value in scalar_metrics.items():
                        logger.log(key, value, self.num_steps)
            if self.enable_metrics_jsonl:
                self._write_metrics_jsonl(scalar_metrics)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        try:
            if save_steps is not None and self.num_steps % save_steps != 0:
                self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            for logger in self.loggers:
                logger.close()
        finally:
            if self.metrics_jsonl_file is not None:
                self.metrics_jsonl_file.close()
                self.metrics_jsonl_file = None

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            if self.enable_metrics_jsonl and os.path.exists(path):
                raise FileExistsError(
                    f"Refusing to overwrite an existing optimizer-step checkpoint: {path}"
                )
            accelerator.save(state_dict, path, safe_serialization=True)
