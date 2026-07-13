import torch
import pytest
from PIL import Image

from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.pipelines.wan_video import WanVideoPipeline, WanVideoUnit_ImageEmbedderFused
from diffsynth.models.wan_video_bsa import BSAContext, WanBSAConfig


class _EmptyScheduler:
    def set_timesteps(self, *args, **kwargs):
        self.timesteps = []


class _LatentInitializer(PipelineUnit):
    def __init__(self, latents):
        super().__init__(input_params=(), output_params=("latents",))
        self.latents = latents

    def process(self, pipe):
        return {"latents": self.latents.clone()}


class _PostProcessor(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents",), output_params=("latents",))

    def process(self, pipe, latents):
        return {"latents": latents + 1}


class _DummyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decode_calls = 0

    def decode(self, latents, **kwargs):
        self.decode_calls += 1
        return latents + 10


def _build_pipeline(latents):
    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
    pipe.scheduler = _EmptyScheduler()
    pipe.units = [_LatentInitializer(latents)]
    pipe.post_units = [_PostProcessor()]
    pipe.in_iteration_models = ()
    pipe.in_iteration_models_2 = ()
    pipe.vae = _DummyVAE()

    model_load_calls = []
    pipe.load_models_to_device = lambda names: model_load_calls.append(tuple(names))
    return pipe, model_load_calls


def test_return_latents_skips_vae_decode_and_runs_post_units():
    initial_latents = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 1, 3)
    pipe, model_load_calls = _build_pipeline(initial_latents)

    output = pipe(
        num_inference_steps=0,
        return_latents=True,
        progress_bar_cmd=lambda values: values,
    )

    assert torch.equal(output, initial_latents + 1)
    assert pipe.vae.decode_calls == 0
    assert ("vae",) not in model_load_calls
    assert model_load_calls[-1] == ()


def test_default_output_still_decodes_and_unloads_models():
    initial_latents = torch.arange(6, dtype=torch.float32).reshape(1, 1, 2, 1, 3)
    pipe, model_load_calls = _build_pipeline(initial_latents)

    output = pipe(
        num_inference_steps=0,
        output_type="floatpoint",
        progress_bar_cmd=lambda values: values,
    )

    assert torch.equal(output, initial_latents + 11)
    assert pipe.vae.decode_calls == 1
    assert ("vae",) in model_load_calls
    assert model_load_calls[-1] == ()


def test_ti2v_fused_image_latent_matches_rollout_dtype_and_device():
    class FakeVAE:
        def encode(self, *args, **kwargs):
            return torch.ones(1, 2, 1, 3, 4, dtype=torch.float64)

    class FakePipe:
        device = torch.device("cpu")
        dit = type("Dit", (), {"fuse_vae_embedding_in_latents": True})()
        vae = FakeVAE()

        @staticmethod
        def load_models_to_device(names):
            pass

        @staticmethod
        def preprocess_image(image):
            return torch.zeros(1, 3, image.height, image.width)

    latents = torch.zeros(1, 2, 3, 3, 4, dtype=torch.float32)
    output = WanVideoUnit_ImageEmbedderFused().process(
        FakePipe(), Image.new("RGB", (16, 16)), latents, 16, 16, True, (8, 8), (4, 4)
    )
    assert output["first_frame_latents"].dtype == latents.dtype
    assert output["first_frame_latents"].device == latents.device
    assert torch.equal(latents[:, :, :1], output["first_frame_latents"])


def test_bsa_inference_requires_four_one_five_and_freezes_context_per_progress():
    class Scheduler:
        def set_timesteps(self, steps, **kwargs):
            self.timesteps = torch.arange(steps, dtype=torch.float32)

        def step(self, noise_pred, timestep, sample):
            return sample

    initial = torch.zeros(1, 1, 2, 1, 1)
    pipe, _ = _build_pipeline(initial)
    pipe.scheduler = Scheduler()
    contexts = []

    def model_fn(latents, bsa_context, timestep, **kwargs):
        contexts.append(bsa_context)
        return torch.zeros_like(latents)

    pipe.model_fn = model_fn
    context = BSAContext.from_config(WanBSAConfig(backend="eager_math"), sparsity=0.8)
    output = pipe(
        num_inference_steps=4,
        cfg_scale=1,
        sigma_shift=5,
        bsa_context=context,
        return_latents=True,
        progress_bar_cmd=lambda values: values,
    )
    assert torch.equal(output, initial + 1)
    assert [item.denoise_progress_id for item in contexts] == [0, 1, 2, 3]
    assert all(item is not context for item in contexts)

    with pytest.raises(ValueError, match="4 steps"):
        pipe(
            num_inference_steps=4,
            cfg_scale=5,
            sigma_shift=5,
            bsa_context=context,
            return_latents=True,
            progress_bar_cmd=lambda values: values,
        )
