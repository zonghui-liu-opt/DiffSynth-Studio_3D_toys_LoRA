import pytest
import torch

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.diffusion.loss import DirectDistillLoss


class RecordingWanScheduler(FlowMatchScheduler):
    def __init__(self):
        super().__init__("Wan")
        self.add_noise_calls = 0
        self.step_outputs = []
        self.set_timesteps_calls = []

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        self.set_timesteps_calls.append(
            {
                "num_inference_steps": num_inference_steps,
                "denoising_strength": denoising_strength,
                "training": training,
                **kwargs,
            }
        )
        return super().set_timesteps(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            training=training,
            **kwargs,
        )

    def add_noise(self, *args, **kwargs):
        self.add_noise_calls += 1
        raise AssertionError("DirectDistillLoss must not add noise between rollout steps.")

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        output = super().step(model_output, timestep, sample, to_final=to_final, **kwargs)
        self.step_outputs.append(output)
        return output


class DummyPipe:
    def __init__(self, weight=0.25, output_bias=0.0, raise_on_pipe_step=False):
        self.scheduler = RecordingWanScheduler()
        self.in_iteration_models = ("dit",)
        self.dit = torch.nn.Parameter(torch.tensor(weight, dtype=torch.float32))
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")
        self.output_bias = output_bias
        self.raise_on_pipe_step = raise_on_pipe_step
        self.legacy_step_calls = 0
        self.model_inputs = []
        self.model_timesteps = []

    def model_fn(self, dit, latents, timestep, progress_id, **kwargs):
        self.model_inputs.append(latents)
        self.model_timesteps.append(timestep.detach().clone())
        return dit * latents + self.output_bias

    def step(self, scheduler, progress_id, noise_pred, latents, **kwargs):
        self.legacy_step_calls += 1
        if self.raise_on_pipe_step:
            raise AssertionError("The strict DirectDistill rollout must call scheduler.step directly.")
        return scheduler.step(noise_pred, scheduler.timesteps[progress_id], latents)


def _manual_rollout(start, weight, output_bias, num_inference_steps=4, sigma_shift=5.0):
    scheduler = FlowMatchScheduler("Wan")
    scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
    latents = start
    for progress_id, timestep in enumerate(scheduler.timesteps):
        noise_pred = weight * latents + output_bias
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


def test_default_options_preserve_legacy_rollout_and_full_latent_loss():
    pipe = DummyPipe(weight=0.25)
    start = torch.tensor([[[[[2.0]], [[1.0]], [[-1.0]]]]])
    target = torch.zeros_like(start)
    unused_first_frame = torch.full_like(start[:, :, 0:1], 9.0)

    loss = DirectDistillLoss(
        pipe,
        latents=start.clone(),
        input_latents=target,
        first_frame_latents=unused_first_frame,
        num_inference_steps=4,
        cfg_scale=5,  # Legacy path historically ignores cfg_scale; strict checks are opt-in.
    )

    expected = _manual_rollout(start, pipe.dit.detach(), pipe.output_bias)
    expected_loss = torch.nn.functional.mse_loss(expected.float(), target.float())
    torch.testing.assert_close(loss, expected_loss)
    assert pipe.legacy_step_calls == 4
    assert pipe.scheduler.add_noise_calls == 0
    assert not torch.equal(pipe.scheduler.step_outputs[-1][:, :, 0:1], unused_first_frame)


def test_four_step_shift_five_schedule_is_continuous_and_fully_differentiable():
    pipe = DummyPipe(weight=0.25, raise_on_pipe_step=True)
    start = torch.tensor([[[[[1.0]], [[-0.5]], [[2.0]]]]])
    teacher = torch.zeros_like(start)

    loss = DirectDistillLoss(
        pipe,
        latents=start.clone(),
        teacher_latents=teacher,
        direct_distill_target_latent_key="teacher_latents",
        num_inference_steps=4,
        sigma_shift=5,
        cfg_scale=1,
    )
    loss.backward()

    reference_scheduler = FlowMatchScheduler("Wan")
    reference_scheduler.set_timesteps(4, shift=5)
    actual_timesteps = torch.cat(pipe.model_timesteps)
    torch.testing.assert_close(actual_timesteps, reference_scheduler.timesteps)
    torch.testing.assert_close(pipe.scheduler.sigmas, reference_scheduler.sigmas)
    assert pipe.scheduler.set_timesteps_calls == [{
        "num_inference_steps": 4,
        "denoising_strength": 1.0,
        "training": False,
        "shift": 5.0,
    }]
    assert len(pipe.model_inputs) == 4
    assert pipe.legacy_step_calls == 0
    assert pipe.scheduler.add_noise_calls == 0
    for progress_id in range(1, 4):
        assert pipe.model_inputs[progress_id] is pipe.scheduler.step_outputs[progress_id - 1]
        assert pipe.model_inputs[progress_id].grad_fn is not None

    reference_weight = torch.tensor(0.25, requires_grad=True)
    reference_prediction = _manual_rollout(start, reference_weight, 0.0)
    reference_loss = torch.nn.functional.mse_loss(reference_prediction, teacher)
    reference_loss.backward()
    torch.testing.assert_close(pipe.dit.grad, reference_weight.grad)

    stats = pipe.direct_distill_last_stats
    assert stats["num_inference_steps"] == 4
    assert stats["model_forward_count"] == 4
    assert stats["latent_shape"] == tuple(start.shape)
    assert stats["timesteps"] == pytest.approx(reference_scheduler.timesteps.tolist())
    assert stats["sigmas"] == pytest.approx(reference_scheduler.sigmas.tolist())


def _run_first_frame_case(first_frame_value):
    pipe = DummyPipe(weight=0.2, output_bias=1.0, raise_on_pipe_step=True)
    start = torch.zeros(1, 1, 3, 1, 1)
    first_frame = torch.tensor([[[[[first_frame_value]]]]])
    teacher = torch.tensor([[[[[first_frame_value]], [[0.5]], [[-0.25]]]]])
    loss = DirectDistillLoss(
        pipe,
        latents=start,
        teacher_latents=teacher,
        first_frame_latents=first_frame,
        direct_distill_target_latent_key="teacher_latents",
        direct_distill_preserve_first_frame=True,
        direct_distill_exclude_first_frame_loss=True,
        num_inference_steps=4,
        sigma_shift=5,
        cfg_scale=1,
    )
    return pipe, loss, first_frame


def test_first_frame_is_bitwise_preserved_each_step_and_excluded_from_loss():
    pipe, loss, first_frame = _run_first_frame_case(first_frame_value=7.0)

    assert len(pipe.scheduler.step_outputs) == 4
    for model_input in pipe.model_inputs:
        assert torch.equal(model_input[:, :, 0:1], first_frame)
    for step_output in pipe.scheduler.step_outputs:
        assert torch.equal(step_output[:, :, 0:1], first_frame)
    assert not torch.equal(
        pipe.scheduler.step_outputs[-1][:, :, 1:],
        torch.zeros_like(pipe.scheduler.step_outputs[-1][:, :, 1:]),
    )

    _, loss_with_different_target_first_frame, _ = _run_first_frame_case(first_frame_value=-999.0)
    torch.testing.assert_close(loss, loss_with_different_target_first_frame, rtol=0, atol=0)


def test_first_frame_target_mismatch_fails_before_rollout():
    pipe = DummyPipe(raise_on_pipe_step=True)
    start = torch.zeros(1, 1, 3, 1, 1)
    with pytest.raises(ValueError, match="not bitwise equal"):
        DirectDistillLoss(
            pipe,
            latents=start,
            teacher_latents=torch.zeros_like(start),
            first_frame_latents=torch.ones_like(start[:, :, :1]),
            direct_distill_target_latent_key="teacher_latents",
            direct_distill_preserve_first_frame=True,
            direct_distill_exclude_first_frame_loss=True,
            num_inference_steps=4,
            sigma_shift=5,
            cfg_scale=1,
        )
    assert pipe.model_inputs == []


def test_cfg_scale_other_than_one_fails_before_student_forward():
    pipe = DummyPipe(raise_on_pipe_step=True)
    start = torch.zeros(1, 1, 2, 1, 1)

    with pytest.raises(ValueError, match="cfg_scale=1"):
        DirectDistillLoss(
            pipe,
            latents=start,
            teacher_latents=torch.zeros_like(start),
            direct_distill_target_latent_key="teacher_latents",
            num_inference_steps=4,
            sigma_shift=5,
            cfg_scale=5,
        )

    assert pipe.model_inputs == []
    assert pipe.scheduler.step_outputs == []
    assert pipe.scheduler.add_noise_calls == 0


def test_fractional_step_count_is_rejected_without_truncation():
    pipe = DummyPipe(raise_on_pipe_step=True)
    start = torch.zeros(1, 1, 2, 1, 1)
    with pytest.raises(ValueError, match="exact positive integer"):
        DirectDistillLoss(
            pipe,
            latents=start,
            teacher_latents=torch.zeros_like(start),
            direct_distill_target_latent_key="teacher_latents",
            num_inference_steps=4.5,
            sigma_shift=5,
            cfg_scale=1,
        )
    assert pipe.model_inputs == []
