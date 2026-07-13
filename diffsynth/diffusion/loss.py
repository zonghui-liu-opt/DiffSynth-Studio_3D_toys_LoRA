from .base_pipeline import BasePipeline
from dataclasses import replace
import torch, math, numbers


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    if "lora" in inputs:
        # Image-to-LoRA models need to load lora here.
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"]) * inputs.get("noise_scale", 1.0)
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    if "first_frame_latents" in inputs:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    if "first_frame_latents" in inputs:
        noise_pred = noise_pred[:, :, 1:]
        training_target = training_target[:, :, 1:]
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def FlowMatchSFTAudioVideoLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    # video
    noise = torch.randn_like(inputs["input_latents"])
    inputs["video_latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    # audio
    if inputs.get("audio_input_latents") is not None:
        audio_noise = torch.randn_like(inputs["audio_input_latents"])
        inputs["audio_latents"] = pipe.scheduler.add_noise(inputs["audio_input_latents"], audio_noise, timestep)
        training_target_audio = pipe.scheduler.training_target(inputs["audio_input_latents"], audio_noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred, noise_pred_audio = pipe.model_fn(**models, **inputs, timestep=timestep)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    if inputs.get("audio_input_latents") is not None:
        loss_audio = torch.nn.functional.mse_loss(noise_pred_audio.float(), training_target_audio.float())
        loss_audio = loss_audio * pipe.scheduler.training_weight(timestep)
        loss = loss + loss_audio
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    strict_mode = bool(
        inputs.get("direct_distill_target_latent_key") is not None
        or inputs.get("direct_distill_preserve_first_frame", False)
        or inputs.get("direct_distill_exclude_first_frame_loss", False)
    )
    if not strict_mode:
        # Keep the upstream DirectDistill path byte-for-byte equivalent when all
        # new controls are disabled.
        pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
        pipe.scheduler.training = True
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
            inputs["latents"] = pipe.step(
                pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs
            )
        return torch.nn.functional.mse_loss(
            inputs["latents"].float(), inputs["input_latents"].float()
        )

    raw_num_inference_steps = inputs["num_inference_steps"]
    if isinstance(raw_num_inference_steps, torch.Tensor) and raw_num_inference_steps.numel() == 1:
        raw_num_inference_steps = raw_num_inference_steps.item()
    if (
        isinstance(raw_num_inference_steps, bool)
        or not isinstance(raw_num_inference_steps, numbers.Real)
        or not math.isfinite(float(raw_num_inference_steps))
        or not float(raw_num_inference_steps).is_integer()
    ):
        raise ValueError("`num_inference_steps` must be an exact positive integer for strict DirectDistill.")
    num_inference_steps = int(raw_num_inference_steps)
    if num_inference_steps <= 0:
        raise ValueError("`num_inference_steps` must be a positive integer for DirectDistillLoss.")

    cfg_scale = float(inputs.get("cfg_scale", 1.0))
    if cfg_scale != 1.0:
        raise ValueError(
            "DirectDistillLoss only supports CFG-free student rollout with `cfg_scale=1`; "
            f"received {cfg_scale}."
        )

    target_latent_key = inputs.get("direct_distill_target_latent_key") or "input_latents"
    if not isinstance(target_latent_key, str):
        raise TypeError("`direct_distill_target_latent_key` must be a string when provided.")
    if target_latent_key not in inputs:
        raise KeyError(
            f"DirectDistillLoss target latent key `{target_latent_key}` is missing from inputs."
        )
    target_latents = inputs[target_latent_key]
    if not isinstance(inputs.get("latents"), torch.Tensor) or not isinstance(target_latents, torch.Tensor):
        raise TypeError("DirectDistillLoss requires tensor `latents` and target latents.")
    if inputs["latents"].shape != target_latents.shape:
        raise ValueError(
            "DirectDistillLoss student/target latent shapes must match, but received "
            f"{tuple(inputs['latents'].shape)} and {tuple(target_latents.shape)}."
        )

    preserve_first_frame = bool(inputs.get("direct_distill_preserve_first_frame", False))
    exclude_first_frame_loss = bool(inputs.get("direct_distill_exclude_first_frame_loss", False))
    if preserve_first_frame or exclude_first_frame_loss:
        if inputs["latents"].ndim != 5:
            raise ValueError(
                "DirectDistillLoss first-frame options require B,C,T,H,W latent tensors."
            )
        if inputs["latents"].shape[2] <= 1 and exclude_first_frame_loss:
            raise ValueError(
                "Cannot exclude the first frame from DirectDistillLoss when the latent has no remaining frames."
            )

    first_frame_latents = None
    if preserve_first_frame:
        first_frame_latents = inputs.get("first_frame_latents")
        if not isinstance(first_frame_latents, torch.Tensor):
            raise KeyError(
                "`first_frame_latents` is required when `direct_distill_preserve_first_frame=True`."
            )
        expected_shape = inputs["latents"][:, :, 0:1].shape
        if first_frame_latents.shape != expected_shape:
            raise ValueError(
                "`first_frame_latents` must match the first-frame latent slice, but received "
                f"{tuple(first_frame_latents.shape)} and {tuple(expected_shape)}."
            )
        if first_frame_latents.dtype != inputs["latents"].dtype or first_frame_latents.device != inputs["latents"].device:
            raise ValueError(
                "`first_frame_latents` must have the same dtype and device as `latents` for bitwise preservation."
            )
        if not torch.equal(target_latents[:, :, 0:1], first_frame_latents):
            raise ValueError(
                "Teacher target first-frame latent is not bitwise equal to the TI2V input-image latent. "
                "Check VAE tiling/resize settings and regenerate the teacher target."
            )
        # Wan2.2-TI2V normally performs this write in its fused-image pipeline unit.
        # Repeating it here makes the DirectDistill rollout invariant explicit before step 0.
        inputs["latents"][:, :, 0:1] = first_frame_latents

    scheduler_kwargs = {}
    if inputs.get("sigma_shift") is not None:
        scheduler_kwargs["shift"] = float(inputs["sigma_shift"])
    pipe.scheduler.set_timesteps(num_inference_steps, **scheduler_kwargs)
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    bsa_context = inputs.get("bsa_context")
    dense_anchor_weight = float(inputs.get("bsa_dense_anchor_weight", 0.0))
    dense_anchor_interval = int(inputs.get("bsa_dense_anchor_interval", 1))
    if dense_anchor_weight < 0 or dense_anchor_interval <= 0:
        raise ValueError("BSA dense anchor weight/interval are invalid.")
    anchor_active = bool(
        bsa_context is not None
        and dense_anchor_weight > 0
        and bsa_context.optimizer_step % dense_anchor_interval == 0
    )
    anchor_progress = bsa_context.optimizer_step % num_inference_steps if anchor_active else None
    dense_anchor_loss = None
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        model_inputs = inputs
        if inputs.get("bsa_context") is not None:
            model_inputs = dict(inputs)
            model_inputs["bsa_context"] = replace(
                inputs["bsa_context"], denoise_progress_id=progress_id
            )
        noise_pred = pipe.model_fn(
            **models, **model_inputs, timestep=timestep, progress_id=progress_id
        )
        if anchor_active and progress_id == anchor_progress:
            dense_inputs = dict(model_inputs)
            dense_inputs["bsa_context"] = replace(
                model_inputs["bsa_context"], sparsity=0.0, top_k=None
            )
            with torch.no_grad():
                dense_prediction = pipe.model_fn(
                    **models, **dense_inputs, timestep=timestep, progress_id=progress_id
                )
            sparse_anchor = noise_pred[:, :, 1:] if exclude_first_frame_loss else noise_pred
            dense_anchor = dense_prediction[:, :, 1:] if exclude_first_frame_loss else dense_prediction
            dense_anchor_loss = torch.nn.functional.mse_loss(
                sparse_anchor.float(), dense_anchor.float()
            ) / dense_anchor.float().square().mean().clamp_min(1e-6)
        # The strict rollout never re-noises or blends in ground-truth latents between steps.
        inputs["latents"] = pipe.scheduler.step(
            noise_pred, pipe.scheduler.timesteps[progress_id], inputs["latents"]
        )
        if preserve_first_frame:
            inputs["latents"][:, :, 0:1] = first_frame_latents

    prediction = inputs["latents"]
    if exclude_first_frame_loss:
        prediction = prediction[:, :, 1:]
        target_latents = target_latents[:, :, 1:]

    scheduler_sigmas = getattr(pipe.scheduler, "sigmas", None)
    pipe.direct_distill_last_stats = {
        "num_inference_steps": len(pipe.scheduler.timesteps),
        "timesteps": tuple(float(value) for value in pipe.scheduler.timesteps.detach().cpu()),
        "sigmas": tuple(float(value) for value in scheduler_sigmas.detach().cpu())
        if isinstance(scheduler_sigmas, torch.Tensor) else tuple(),
        "latent_shape": tuple(inputs["latents"].shape),
        "model_forward_count": len(pipe.scheduler.timesteps) + int(anchor_active),
        "dense_anchor_active": anchor_active,
        "dense_anchor_loss": None if dense_anchor_loss is None else dense_anchor_loss.detach(),
    }
    endpoint_loss = torch.nn.functional.mse_loss(prediction.float(), target_latents.float())
    pipe.direct_distill_last_stats["endpoint_loss"] = endpoint_loss.detach()
    if dense_anchor_loss is None:
        return endpoint_loss
    return endpoint_loss + dense_anchor_weight * dense_anchor_loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            denom = sigma_ - sigma
            denom = torch.sign(denom) * torch.clamp(denom.abs(), min=1e-6)
            target = (latents_ - inputs_shared["latents"]) / denom
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
