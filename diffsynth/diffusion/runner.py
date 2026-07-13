import os, json, torch, importlib, math, time
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from diffsynth.core import OffloadTrainingManager


def compute_video_tokens(latent_shape, patch_size):
    """Count DiT video tokens from a B,C,T,H,W latent grid."""
    if len(latent_shape) != 5:
        raise ValueError(f"Expected a B,C,T,H,W latent shape, received {tuple(latent_shape)}.")
    if len(patch_size) != 3:
        raise ValueError(f"Expected a temporal/spatial DiT patch size, received {tuple(patch_size)}.")
    batch, _, frames, height, width = (int(value) for value in latent_shape)
    patch_t, patch_h, patch_w = (int(value) for value in patch_size)
    if min(batch, frames, height, width, patch_t, patch_h, patch_w) <= 0:
        raise ValueError("Latent and patch dimensions must all be positive.")
    return batch * math.ceil(frames / patch_t) * math.ceil(height / patch_h) * math.ceil(width / patch_w)


def compute_training_workload(latent_shape, patch_size, num_inference_steps):
    num_inference_steps = int(num_inference_steps)
    if num_inference_steps <= 0:
        raise ValueError("`num_inference_steps` must be positive when computing workload metrics.")
    video_tokens = compute_video_tokens(latent_shape, patch_size)
    videos = int(latent_shape[0])
    return {
        "videos": videos,
        "video_tokens": video_tokens,
        "model_tokens": video_tokens * num_inference_steps,
    }


def aggregate_training_step_statistics(rank_statistics):
    """Aggregate per-rank optimizer-step counters using task-defined sum/max rules."""
    rank_statistics = list(rank_statistics)
    if not rank_statistics:
        raise ValueError("At least one rank statistic is required.")
    numeric_keys = (
        "loss_sum", "loss_weight", "video_tokens", "model_tokens", "videos",
        "step_time_sec", "max_memory_gb", "max_memory_reserved_gb", "device_free_memory_gb",
    )
    for rank_id, item in enumerate(rank_statistics):
        for key in numeric_keys:
            value = float(item.get(key, 0.0))
            if not math.isfinite(value):
                raise ValueError(f"Rank {rank_id} statistic `{key}` is not finite: {value}.")
    loss_sum = sum(float(item["loss_sum"]) for item in rank_statistics)
    loss_weight = sum(float(item["loss_weight"]) for item in rank_statistics)
    video_tokens = sum(float(item["video_tokens"]) for item in rank_statistics)
    model_tokens = sum(float(item["model_tokens"]) for item in rank_statistics)
    videos = sum(float(item["videos"]) for item in rank_statistics)
    step_time = max(float(item["step_time_sec"]) for item in rank_statistics)
    max_memory_gb = max(float(item.get("max_memory_gb", 0.0)) for item in rank_statistics)
    max_memory_reserved_gb = max(
        float(item.get("max_memory_reserved_gb", 0.0)) for item in rank_statistics
    )
    free_values = [
        float(item.get("device_free_memory_gb", 0.0)) for item in rank_statistics
        if float(item.get("device_free_memory_gb", 0.0)) > 0
    ]
    device_free_memory_gb = min(free_values) if free_values else 0.0
    if loss_weight <= 0 or step_time <= 0:
        raise ValueError("Loss weight and optimizer-step time must be positive.")
    return {
        "loss": loss_sum / loss_weight,
        "video_tokens": video_tokens,
        "model_tokens": model_tokens,
        "videos": videos,
        "step_time_sec": step_time,
        "max_memory_gb": max_memory_gb,
        "max_memory_reserved_gb": max_memory_reserved_gb,
        "device_free_memory_gb": device_free_memory_gb,
        "global_video_tokens_per_sec": video_tokens / step_time,
        "global_model_tokens_per_sec": model_tokens / step_time,
        "global_videos_per_sec": videos / step_time,
    }


def update_loss_ema(previous_ema, loss, beta=0.98):
    beta = float(beta)
    loss = float(loss)
    if not 0 <= beta < 1:
        raise ValueError("Loss EMA beta must be in [0, 1).")
    if previous_ema is None:
        return loss
    return beta * float(previous_ema) + (1 - beta) * loss


def _synchronize_training_device(accelerator):
    if accelerator.device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)


def _empty_training_step_statistics():
    return {
        "loss_sum": 0.0,
        "loss_weight": 0.0,
        "video_tokens": 0.0,
        "model_tokens": 0.0,
        "videos": 0.0,
        "step_time_sec": 0.0,
        "max_memory_gb": 0.0,
        "max_memory_reserved_gb": 0.0,
        "device_free_memory_gb": 0.0,
    }


def _gather_training_step_statistics(accelerator, local_statistics):
    keys = (
        "loss_sum", "loss_weight", "video_tokens", "model_tokens", "videos",
        "step_time_sec", "max_memory_gb", "max_memory_reserved_gb", "device_free_memory_gb",
    )
    packed = torch.tensor(
        [local_statistics[key] for key in keys],
        dtype=torch.float64,
        device=accelerator.device,
    )
    gathered = accelerator.gather(packed).detach().cpu().reshape(-1, len(keys))
    return [dict(zip(keys, row.tolist())) for row in gathered]


def get_optimizer_class(customized_optimizer=None):
    if customized_optimizer is None:
        return torch.optim.AdamW
    else:
        module_name, class_name = customized_optimizer.rsplit(".", 1)
        module = importlib.import_module(module_name)
        print(f"Customized opimizer `{customized_optimizer}` imported.")
        return getattr(module, class_name)


def build_optimizer_param_groups(model, learning_rate, weight_decay, args=None):
    factory = getattr(model, "optimizer_param_groups", None)
    if callable(factory):
        groups = factory(learning_rate, weight_decay, args=args)
        if groups is not None:
            parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
            trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
            if len(parameter_ids) != len(set(parameter_ids)) or set(parameter_ids) != set(trainable_ids):
                raise RuntimeError(
                    "Optimizer param groups must cover every trainable parameter exactly once."
                )
            return groups
    return model.trainable_modules()


def advance_completed_optimizer_steps(completed_steps, *, sync_gradients, step_was_skipped):
    completed_steps = int(completed_steps)
    if completed_steps < 0:
        raise ValueError("completed_steps must be non-negative.")
    return completed_steps + int(bool(sync_gradients) and not bool(step_was_skipped))


def save_training_args(args):
    output_path = getattr(args, "output_path", None) if args is not None else None
    if output_path is None:
        return
    try:
        os.makedirs(args.output_path, exist_ok=True)
        save_path = os.path.join(args.output_path, "training_args.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4, ensure_ascii=False, default=str)
        print(f"Training arguments saved to `{save_path}`.")
    except Exception as e:
        print(f"Warning: failed to save training arguments: {e}")


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args = None,
    **kwargs,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    if accelerator.is_main_process:
        save_training_args(args)

    enable_training_metrics = bool(
        args is not None and getattr(args, "enable_direct_distill_metrics", False)
    )
    loss_ema_beta = float(getattr(args, "direct_distill_loss_ema_beta", 0.98)) if args is not None else 0.98
    if enable_training_metrics and not 0 <= loss_ema_beta < 1:
        raise ValueError("`direct_distill_loss_ema_beta` must be in [0, 1).")
    if enable_training_metrics and enable_model_cpu_offload and accelerator.num_processes > 1:
        raise ValueError(
            "Multi-process training with `enable_model_cpu_offload` is not supported: "
            "the offloaded model is not DDP/FSDP-wrapped, so gradients would not synchronize."
        )

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer_parameters = build_optimizer_param_groups(
        model, learning_rate, weight_decay, args=args
    )
    optimizer = optimizer_class(optimizer_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)
    unwrapped_model = accelerator.unwrap_model(model)
    if getattr(unwrapped_model, "enable_bsa", False):
        estimated_optimizer_steps = math.ceil(
            len(dataloader) * num_epochs / accelerator.gradient_accumulation_steps
        )
        resume_manifest = getattr(unwrapped_model, "bsa_resume_manifest", None)
        schedule_steps = (
            int(resume_manifest["max_optimizer_steps"])
            if isinstance(resume_manifest, dict) and resume_manifest.get("max_optimizer_steps")
            else estimated_optimizer_steps
        )
        unwrapped_model.set_bsa_training_progress(
            unwrapped_model.bsa_completed_optimizer_steps, schedule_steps
        )
        if schedule_steps < 1500 and accelerator.is_main_process:
            print(
                f"Warning: BSA schedule has only {schedule_steps} optimizer steps; "
                "the first formal experiment should use at least 3000."
            )
    if getattr(unwrapped_model, "enable_bsa", False):
        unwrapped_model.probe_bsa_backend(accelerator.device)
    if accelerator.is_main_process and getattr(unwrapped_model, "enable_bsa", False):
        unwrapped_model.write_bsa_student_info(include_runtime=False)
        info = unwrapped_model._wan_bsa_student_info
        print(
            "Wan BSA injected: "
            f"{info['injected_bsa_modules']}/{info['expected_self_attention_modules']} self-attention layers, "
            f"cross={info['cross_attention_bsa_modules']}, block={tuple(info['block_size'])}, "
            f"gate={info['gate_type']}/{info['gate_granularity']}, backend={info['backend']}."
        )
    bsa_runtime_info_written = False
    bsa_optimizer_info_written = False
    accumulated_statistics = _empty_training_step_statistics()
    loss_ema = getattr(unwrapped_model, "bsa_loss_ema", None)
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            if enable_training_metrics:
                if accumulated_statistics["loss_weight"] == 0 and accelerator.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(accelerator.device)
                # The timer begins after data delivery. It includes forward, backward,
                # optimizer/scheduler and zero_grad, while excluding logging/checkpoints.
                _synchronize_training_device(accelerator)
                compute_started_at = time.perf_counter()
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                if (
                    accelerator.is_main_process
                    and getattr(unwrapped_model, "enable_bsa", False)
                    and not bsa_runtime_info_written
                ):
                    unwrapped_model.write_bsa_student_info(include_runtime=True)
                    bsa_runtime_info_written = True
                accelerator.backward(loss)
                total_grad_norm = None
                max_grad_norm = float(getattr(args, "max_grad_norm", 0.0)) if args is not None else 0.0
                if accelerator.sync_gradients and max_grad_norm > 0:
                    total_grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if not bool(torch.isfinite(torch.as_tensor(total_grad_norm)).all()):
                        raise RuntimeError("Non-finite gradient norm detected; stopping before optimizer.step().")
                if enable_model_cpu_offload:
                    offload_manager.after_backward()
                optimizer.step()
                optimizer_step_was_skipped = bool(
                    getattr(accelerator, "optimizer_step_was_skipped", False)
                )
                scheduler.step()
                unwrapped_model = accelerator.unwrap_model(model)
                bsa_metrics = {}
                if accelerator.sync_gradients and not optimizer_step_was_skipped:
                    bsa_metrics_fn = getattr(unwrapped_model, "bsa_step_metrics", None)
                    if callable(bsa_metrics_fn):
                        bsa_metrics = bsa_metrics_fn(total_grad_norm)
                    if getattr(unwrapped_model, "enable_bsa", False):
                        unwrapped_model.set_bsa_training_progress(
                            advance_completed_optimizer_steps(
                                unwrapped_model.bsa_completed_optimizer_steps,
                                sync_gradients=accelerator.sync_gradients,
                                step_was_skipped=optimizer_step_was_skipped,
                            ),
                            unwrapped_model.bsa_max_optimizer_steps,
                        )
                    if (
                        accelerator.is_main_process
                        and getattr(unwrapped_model, "enable_bsa", False)
                        and not bsa_optimizer_info_written
                    ):
                        optimizer_groups = []
                        for group in optimizer.param_groups:
                            optimizer_groups.append({
                                "name": group.get("name", "unnamed"),
                                "parameter_count": sum(
                                    parameter.numel() for parameter in group["params"]
                                ),
                                "lr": float(group["lr"]),
                                "weight_decay": float(group.get("weight_decay", 0.0)),
                                "parameter_dtypes": sorted({
                                    str(parameter.dtype) for parameter in group["params"]
                                }),
                            })
                        state_dtypes = sorted({
                            str(value.dtype)
                            for state in optimizer.state.values()
                            for value in state.values()
                            if isinstance(value, torch.Tensor)
                        })
                        unwrapped_model._wan_bsa_student_info["optimizer_groups"] = optimizer_groups
                        unwrapped_model._wan_bsa_student_info["optimizer_state_dtypes"] = state_dtypes
                        unwrapped_model.write_bsa_student_info(include_runtime=True)
                        bsa_optimizer_info_written = True
                optimizer.zero_grad(set_to_none=True)
                if enable_training_metrics:
                    _synchronize_training_device(accelerator)
                    accumulated_statistics["step_time_sec"] += time.perf_counter() - compute_started_at
                    context = getattr(unwrapped_model, "_last_training_step_context", None)
                    if not isinstance(context, dict):
                        raise RuntimeError(
                            "Training metrics are enabled, but the model did not expose "
                            "`_last_training_step_context`."
                        )
                    workload = compute_training_workload(
                        context["latent_shape"], context["patch_size"], context["num_inference_steps"]
                    )
                    loss_weight = float(workload["videos"])
                    accumulated_statistics["loss_sum"] += float(loss.detach()) * loss_weight
                    accumulated_statistics["loss_weight"] += loss_weight
                    for key in ("videos", "video_tokens", "model_tokens"):
                        accumulated_statistics[key] += float(workload[key])
                    if accelerator.device.type == "cuda" and torch.cuda.is_available():
                        accumulated_statistics["max_memory_gb"] = max(
                            accumulated_statistics["max_memory_gb"],
                            torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3),
                        )
                        accumulated_statistics["max_memory_reserved_gb"] = max(
                            accumulated_statistics["max_memory_reserved_gb"],
                            torch.cuda.max_memory_reserved(accelerator.device) / (1024 ** 3),
                        )
                        free_bytes, _ = torch.cuda.mem_get_info(accelerator.device)
                        free_gb = free_bytes / (1024 ** 3)
                        previous_free = accumulated_statistics["device_free_memory_gb"]
                        accumulated_statistics["device_free_memory_gb"] = (
                            free_gb if previous_free == 0 else min(previous_free, free_gb)
                        )

                    if accelerator.sync_gradients:
                        if not optimizer_step_was_skipped:
                            rank_statistics = _gather_training_step_statistics(
                                accelerator, accumulated_statistics
                            )
                            global_statistics = aggregate_training_step_statistics(rank_statistics)
                            bsa_memory_target_met = None
                            if getattr(unwrapped_model, "enable_bsa", False):
                                hard_stop = float(getattr(args, "bsa_memory_hard_stop_gib", 75.0))
                                hard_min_free = float(
                                    getattr(args, "bsa_memory_hard_min_free_gib", 4.0)
                                )
                                target_allocated = float(
                                    getattr(args, "bsa_memory_target_allocated_gib", 70.0)
                                )
                                target_reserved = float(
                                    getattr(args, "bsa_memory_target_reserved_gib", 73.0)
                                )
                                target_min_free = float(
                                    getattr(args, "bsa_min_device_free_gib", 6.0)
                                )
                                bsa_memory_target_met = (
                                    global_statistics["max_memory_gb"] <= target_allocated
                                    and global_statistics["max_memory_reserved_gb"] <= target_reserved
                                    and (
                                        global_statistics["device_free_memory_gb"] == 0
                                        or global_statistics["device_free_memory_gb"] >= target_min_free
                                    )
                                )
                                if (
                                    global_statistics["max_memory_gb"] >= hard_stop
                                    or global_statistics["max_memory_reserved_gb"] >= hard_stop
                                    or (
                                        global_statistics["device_free_memory_gb"] > 0
                                        and global_statistics["device_free_memory_gb"] < hard_min_free
                                    )
                                ):
                                    raise RuntimeError(
                                        "BSA memory hard stop triggered: "
                                        f"allocated={global_statistics['max_memory_gb']:.2f} GiB, "
                                        f"reserved={global_statistics['max_memory_reserved_gb']:.2f} GiB, "
                                        f"free={global_statistics['device_free_memory_gb']:.2f} GiB."
                                    )
                            loss_ema = update_loss_ema(
                                loss_ema, global_statistics["loss"], beta=loss_ema_beta
                            )
                            if getattr(unwrapped_model, "enable_bsa", False):
                                unwrapped_model.bsa_loss_ema = loss_ema
                            model_tokens_per_sec = global_statistics["global_model_tokens_per_sec"]
                            videos_per_sec = global_statistics["global_videos_per_sec"]
                            metrics = {
                                "train/loss": global_statistics["loss"],
                                "train/loss_ema": loss_ema,
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "throughput/global_video_tokens_per_sec": global_statistics["global_video_tokens_per_sec"],
                                "throughput/global_model_tokens_per_sec": model_tokens_per_sec,
                                "throughput/global_tokens_per_hour": model_tokens_per_sec * 3600,
                                "throughput/global_videos_per_sec": videos_per_sec,
                                "throughput/global_videos_per_hour": videos_per_sec * 3600,
                                "throughput/step_time_sec": global_statistics["step_time_sec"],
                            }
                            if global_statistics["max_memory_gb"] > 0:
                                metrics["system/max_memory_gb"] = global_statistics["max_memory_gb"]
                                metrics["system/max_memory_allocated_gb"] = global_statistics["max_memory_gb"]
                            if global_statistics["max_memory_reserved_gb"] > 0:
                                metrics["system/max_memory_reserved_gb"] = global_statistics["max_memory_reserved_gb"]
                            if global_statistics["device_free_memory_gb"] > 0:
                                metrics["system/device_free_memory_gb"] = global_statistics["device_free_memory_gb"]
                            if bsa_memory_target_met is not None:
                                metrics["system/bsa_memory_target_met"] = float(bsa_memory_target_met)
                            if bsa_metrics:
                                metrics["train/loss_total"] = global_statistics["loss"]
                            metrics.update(bsa_metrics)
                            model_logger.on_step_end(
                                accelerator, model, save_steps, metrics=metrics
                            )
                        accumulated_statistics = _empty_training_step_statistics()
                else:
                    # Preserve the upstream micro-batch logging/checkpoint cadence unless
                    # the explicit DirectDistill metrics mode is enabled.
                    model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
    **kwargs,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    if enable_model_cpu_offload:
        dataloader = accelerator.prepare(dataloader)
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
        model.pipe.device = accelerator.device
    else:
        model.to(device=accelerator.device)
        model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

def initialize_deepspeed_gradient_checkpointing(accelerator: Accelerator):
    if getattr(accelerator.state, "deepspeed_plugin", None) is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if "activation_checkpointing" in ds_config:
            import deepspeed
            act_config = ds_config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None, 
                partition_activations=act_config.get("partition_activations", False),
                checkpoint_in_cpu=act_config.get("cpu_checkpointing", False),
                contiguous_checkpointing=act_config.get("contiguous_memory_optimization", False)
            )
        else:
            print("Do not find activation_checkpointing config in deepspeed config, skip initializing deepspeed gradient checkpointing.")
