import hashlib
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import send_to_device
from PIL import Image
from safetensors.torch import save_file
from diffsynth.diffusion.bsa_schedule import (
    compile_bsa_schedule,
    compiled_bsa_schedule_from_dict,
    load_bsa_schedule,
)
from diffsynth.models.wan_video_bsa import WanBSAConfig

from examples.wanvideo.model_training.train import (
    DIRECT_DISTILL_LATENT_METADATA_KEY,
    DirectDistillDataset,
    LoadDirectDistillInputImage,
    LoadDirectDistillLatents,
    WanTrainingModule,
    build_wan_model_logger,
    build_wan_special_operator_map,
    resolve_direct_distill_input_image,
    wan_parser,
)


def _make_uninitialized_module(strict=True):
    module = WanTrainingModule.__new__(WanTrainingModule)
    torch.nn.Module.__init__(module)
    module.direct_distill_strict = strict
    module.direct_distill_target_latent_key = "input_latents" if strict else None
    module.direct_distill_preserve_first_frame = strict
    module.direct_distill_exclude_first_frame_loss = strict
    module.use_gradient_checkpointing = True
    module.use_gradient_checkpointing_offload = False
    module.extra_inputs = ["seed", "rand_device", "num_inference_steps", "cfg_scale", "sigma_shift"]
    module.max_timestep_boundary = 1.0
    module.min_timestep_boundary = 0.0
    module.pipe = SimpleNamespace(
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        time_division_factor=4,
        time_division_remainder=1,
        vae=SimpleNamespace(upsampling_factor=8),
    )
    return module


def test_cached_input_image_is_preferred_and_nan_falls_back_to_loaded_video():
    cached = Image.new("RGB", (16, 16), color="red")
    video_first_frame = Image.new("RGB", (16, 16), color="blue")

    assert resolve_direct_distill_input_image({
        "input_image": cached,
        "video": [video_first_frame],
    }) is cached
    assert resolve_direct_distill_input_image({
        "input_image": [cached],
        "video": [video_first_frame],
    }) is cached
    assert resolve_direct_distill_input_image({
        "input_image": float("nan"),
        "video": [video_first_frame],
    }) is video_first_frame
    assert resolve_direct_distill_input_image({
        "input_image": None,
        "video": [video_first_frame],
    }) is video_first_frame
    # An unprocessed CSV path is not a loaded image and must not override video[0].
    assert resolve_direct_distill_input_image({
        "input_image": "cached.png",
        "video": [video_first_frame],
    }) is video_first_frame


def test_optional_cached_image_loader_keeps_missing_values_for_video_fallback(tmp_path):
    loader = LoadDirectDistillInputImage(str(tmp_path), 16, 16, 16 * 16)
    assert loader(None) is None
    assert loader(float("nan")) is None


def test_dataset_fallback_loads_only_video_first_frame_when_png_missing(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text("video,prompt\nsource.mp4,turntable\n", encoding="utf-8")
    expected = Image.new("RGB", (16, 16), "purple")
    calls = []

    def fallback(path):
        calls.append(path)
        return expected

    dataset = DirectDistillDataset(
        base_path=str(tmp_path),
        metadata_path=str(metadata_path),
        data_file_keys=(),
        input_image_fallback_operator=fallback,
    )
    row = dataset[0]
    assert row["input_image"] is expected
    assert calls == ["source.mp4"]


def test_wan_parser_exposes_strict_direct_distill_options():
    args = wan_parser().parse_args([
        "--dataset_base_path", "dataset",
        "--direct_distill_preserve_first_frame",
        "--direct_distill_exclude_first_frame_loss",
        "--direct_distill_target_latent_key", "input_latents",
        "--enable_direct_distill_metrics",
        "--direct_distill_loss_ema_beta", "0.9",
        "--bsa_sparsity_schedule", "conservative_epoch_v1",
    ])
    assert args.direct_distill_preserve_first_frame is True
    assert args.direct_distill_exclude_first_frame_loss is True
    assert args.direct_distill_target_latent_key == "input_latents"
    assert args.enable_direct_distill_metrics is True
    assert args.direct_distill_loss_ema_beta == pytest.approx(0.9)
    assert args.bsa_sparsity_schedule == "conservative_epoch_v1"

    defaults = wan_parser().parse_args(["--dataset_base_path", "dataset"])
    assert defaults.enable_direct_distill_metrics is False
    assert defaults.direct_distill_loss_ema_beta == pytest.approx(0.98)
    assert defaults.bsa_sparsity_schedule is None


def _make_bsa_schedule_module(schedule=None, resume_manifest=None):
    module = WanTrainingModule.__new__(WanTrainingModule)
    torch.nn.Module.__init__(module)
    module.enable_bsa = True
    module.bsa_config = WanBSAConfig(backend="eager_math")
    module.bsa_requested_schedule_spec = (
        load_bsa_schedule(schedule) if schedule is not None else None
    )
    module.bsa_schedule_spec = (
        module.bsa_requested_schedule_spec or load_bsa_schedule("legacy_progress_v1")
    )
    module.bsa_resume_manifest = resume_manifest
    module._bsa_resume_compiled_schedule = None
    if resume_manifest is not None and resume_manifest["schema_version"] == 2:
        module._bsa_resume_compiled_schedule = compiled_bsa_schedule_from_dict(
            resume_manifest, target_sparsity=module.bsa_config.target_sparsity
        )
        module.bsa_schedule_spec = module._bsa_resume_compiled_schedule.spec
    module.bsa_compiled_schedule = None
    module.bsa_completed_optimizer_steps = int(
        (resume_manifest or {}).get("completed_optimizer_steps", 0)
    )
    module.bsa_max_optimizer_steps = None
    module.bsa_loss_ema = None
    module._wan_bsa_student_info = {}
    module.pipe = SimpleNamespace(dit=SimpleNamespace(blocks=[]))
    return module


def test_bsa_schedule_initialization_has_one_source_for_context_and_manifest():
    module = _make_bsa_schedule_module("conservative_epoch_v1")
    compiled = module.initialize_bsa_training_schedule(steps_per_epoch=121, total_epochs=60)

    assert compiled.transition_steps == (363, 484, 605, 726, 847, 968, 1089, 1210, 1331)
    module.set_bsa_training_progress(363, compiled.total_optimizer_steps)
    assert module.current_bsa_sparsity() == 0.1
    assert module.current_bsa_context().sparsity == 0.1
    manifest = module.bsa_checkpoint_manifest()
    assert manifest["schema_version"] == 2
    assert manifest["current_requested_sparsity"] == 0.1
    assert manifest["schedule_sha256"] == compiled.spec_sha256


def test_bsa_schema_v2_resume_rejects_schedule_or_geometry_changes():
    saved = compile_bsa_schedule(
        "conservative_epoch_v1",
        steps_per_epoch=121,
        total_epochs=60,
        target_sparsity=0.8,
    )
    manifest = {
        "schema_version": 2,
        "completed_optimizer_steps": 200,
        **saved.to_dict(),
    }
    resumed = _make_bsa_schedule_module("conservative_epoch_v1", manifest)
    assert resumed.initialize_bsa_training_schedule(121, 60) == saved

    changed_geometry = _make_bsa_schedule_module("conservative_epoch_v1", manifest)
    with pytest.raises(ValueError, match="training geometry conflicts"):
        changed_geometry.initialize_bsa_training_schedule(122, 60)

    changed_schedule = _make_bsa_schedule_module("mature_same_shape_v1", manifest)
    with pytest.raises(ValueError, match="schedule conflicts"):
        changed_schedule.initialize_bsa_training_schedule(121, 60)

    mature = compile_bsa_schedule(
        "mature_same_shape_v1",
        steps_per_epoch=121,
        total_epochs=60,
        target_sparsity=0.8,
    )
    mature_manifest = {
        "schema_version": 2,
        "completed_optimizer_steps": 200,
        **mature.to_dict(),
    }
    automatic = _make_bsa_schedule_module(None, mature_manifest)
    assert automatic.initialize_bsa_training_schedule(121, 60) == mature


def test_bsa_schema_v1_resume_is_pinned_to_legacy_schedule():
    manifest = {
        "schema_version": 1,
        "completed_optimizer_steps": 200,
        "max_optimizer_steps": 7260,
    }
    resumed = _make_bsa_schedule_module(None, manifest)
    compiled = resumed.initialize_bsa_training_schedule(121, 60)
    assert compiled.transition_steps == (363, 726, 1089, 1452, 1815, 2178, 2541, 2904)

    changed = _make_bsa_schedule_module("conservative_epoch_v1", manifest)
    with pytest.raises(ValueError, match="require.*legacy_progress_v1"):
        changed.initialize_bsa_training_schedule(121, 60)


def test_metrics_cli_is_wired_to_model_logger_jsonl(tmp_path):
    args = wan_parser().parse_args([
        "--dataset_base_path", "dataset",
        "--output_path", str(tmp_path),
        "--enable_direct_distill_metrics",
    ])
    model_logger = build_wan_model_logger(args)
    assert model_logger.enable_metrics_jsonl is True
    assert model_logger.metrics_jsonl_path == str(tmp_path / "metrics.jsonl")


def test_strict_target_mode_uses_cached_image_and_teacher_latent_without_video():
    module = _make_uninitialized_module(strict=True)
    cached = Image.new("RGB", (16, 16), color="red")
    teacher = torch.randn(1, 4, 3, 2, 2)
    data = {
        "input_image": cached,
        "teacher_latent": teacher,
        "prompt": "turntable",
        "height": 16,
        "width": 16,
        "num_frames": 9,
        "seed": 1,
        "rand_device": "cpu",
        "num_inference_steps": 4,
        "cfg_scale": 1,
        "sigma_shift": 5,
    }

    inputs_shared, inputs_posi, inputs_nega = module.get_pipeline_inputs(data)

    assert inputs_shared["input_video"] is None
    assert inputs_shared["input_image"] is cached
    assert inputs_shared["input_latents"] is teacher
    assert inputs_shared["height"] == 16
    assert inputs_shared["width"] == 16
    assert inputs_shared["num_frames"] == 9
    assert inputs_shared["num_inference_steps"] == 4
    assert inputs_shared["cfg_scale"] == 1
    assert inputs_shared["sigma_shift"] == 5
    assert inputs_shared["direct_distill_target_latent_key"] == "input_latents"
    assert inputs_shared["direct_distill_preserve_first_frame"] is True
    assert inputs_shared["direct_distill_exclude_first_frame_loss"] is True
    assert inputs_posi == {"prompt": "turntable"}
    assert inputs_nega == {}


def test_legacy_pipeline_inputs_remain_video_based_when_new_options_are_disabled():
    module = _make_uninitialized_module(strict=False)
    module.extra_inputs = []
    video = [Image.new("RGB", (24, 16), color="blue") for _ in range(5)]

    inputs_shared, inputs_posi, inputs_nega = module.get_pipeline_inputs({
        "video": video,
        "prompt": "legacy",
    })

    assert inputs_shared["input_video"] is video
    assert inputs_shared["height"] == 16
    assert inputs_shared["width"] == 24
    assert inputs_shared["num_frames"] == 5
    assert "direct_distill_target_latent_key" not in inputs_shared
    assert "direct_distill_preserve_first_frame" not in inputs_shared
    assert "direct_distill_exclude_first_frame_loss" not in inputs_shared
    assert inputs_posi == {"prompt": "legacy"}
    assert inputs_nega == {}


def test_safetensors_loader_and_special_dataset_operators_round_trip(tmp_path):
    image_path = tmp_path / "first.png"
    Image.new("RGB", (24, 20), color="green").save(image_path)
    latent_path = tmp_path / "teacher.safetensors"
    expected = torch.randn(1, 4, 3, 2, 2)
    prompt = "turntable"
    provenance = {
        "object_id": "object-a",
        "sample_id": "sample-a",
        "seed": "1",
        "rand_device": "cpu",
        "teacher_fingerprint": "teacher-a",
        "source_fingerprint": "source-a",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    save_file({"latents": expected}, latent_path, metadata=provenance)

    loader = LoadDirectDistillLatents()
    actual = loader(latent_path)
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)

    bad_path = tmp_path / "bad-key.safetensors"
    save_file({"wrong": expected}, bad_path)
    with pytest.raises(KeyError, match="fixed key `latents`"):
        loader(bad_path)

    metadata_path = tmp_path / "metadata.csv"
    metadata_path.write_text(
        "input_image,teacher_latent,prompt,sample_id,teacher_fingerprint,"
        "source_fingerprint,seed,rand_device\n"
        "first.png,teacher.safetensors,turntable,sample-a,teacher-a,source-a,1,cpu\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        dataset_base_path=str(tmp_path),
        data_file_keys="input_image,teacher_latent",
        num_frames=9,
        height=16,
        width=16,
        max_pixels=16 * 16,
        direct_distill_preserve_first_frame=True,
        direct_distill_exclude_first_frame_loss=True,
        direct_distill_target_latent_key="input_latents",
    )
    dataset = DirectDistillDataset(
        base_path=str(tmp_path),
        metadata_path=str(metadata_path),
        data_file_keys=("input_image", "teacher_latent"),
        main_data_operator=lambda _: pytest.fail("main video operator must not run"),
        special_operator_map=build_wan_special_operator_map(args),
    )
    row = dataset[0]
    assert row["input_image"].size == (16, 16)
    assert torch.equal(row["teacher_latent"], expected)
    assert row[DIRECT_DISTILL_LATENT_METADATA_KEY] == provenance

    moved_row = send_to_device(row, torch.device("meta"))
    assert not hasattr(moved_row["teacher_latent"], "_direct_distill_metadata")
    assert moved_row[DIRECT_DISTILL_LATENT_METADATA_KEY] == provenance
    WanTrainingModule.validate_direct_distill_latent_provenance(
        moved_row, moved_row["teacher_latent"]
    )

    mismatched_row = dict(moved_row)
    mismatched_row[DIRECT_DISTILL_LATENT_METADATA_KEY] = {
        **provenance,
        "sample_id": "wrong-sample",
    }
    with pytest.raises(ValueError, match="sample_id.*mismatch"):
        WanTrainingModule.validate_direct_distill_latent_provenance(
            mismatched_row, moved_row["teacher_latent"]
        )

    extra_key_path = tmp_path / "extra-key.safetensors"
    save_file({"latents": expected, "extra": torch.ones(1)}, extra_key_path)
    with pytest.raises(KeyError, match="only the fixed key"):
        loader(extra_key_path)
    integer_path = tmp_path / "integer.safetensors"
    save_file({"latents": torch.ones(1, 4, 3, 2, 2, dtype=torch.int64)}, integer_path)
    with pytest.raises(ValueError, match="floating-point"):
        loader(integer_path)


def test_strict_config_rejects_full_model_same_adapter_and_resume(tmp_path):
    preset = tmp_path / "figurine.safetensors"
    preset.touch()
    valid = dict(
        task="direct_distill",
        preset_lora_path=str(preset),
        preset_lora_model="dit",
        lora_base_model="dit",
        lora_checkpoint=None,
        trainable_models=None,
        resume_from_checkpoint=None,
    )
    WanTrainingModule.validate_direct_distill_training_config(**valid)

    with pytest.raises(ValueError, match="full trainable models"):
        WanTrainingModule.validate_direct_distill_training_config(
            **{**valid, "trainable_models": "dit"}
        )
    with pytest.raises(ValueError, match="must be different files"):
        WanTrainingModule.validate_direct_distill_training_config(
            **{**valid, "lora_checkpoint": str(preset)}
        )
    with pytest.raises(ValueError, match="Use `lora_checkpoint`"):
        WanTrainingModule.validate_direct_distill_training_config(
            **{**valid, "resume_from_checkpoint": "step-10.safetensors"}
        )


class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 4, bias=False)


class TinyScheduler:
    def set_timesteps(self, *args, **kwargs):
        self.last_set_timesteps = args, kwargs


class StrictInitPipe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dit = TinyDiT()
        self.scheduler = TinyScheduler()
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")
        self.units = []
        self.preset_hotload_values = []

    def freeze_except(self, model_names):
        assert model_names == []
        self.requires_grad_(False)

    def load_lora(self, module, path, hotload=None):
        assert not hasattr(module.q, "lora_A")
        self.preset_hotload_values.append(hotload)
        with torch.no_grad():
            module.q.weight.add_(1.0)


def test_strict_init_forces_preset_fusion_before_new_lora_injection(tmp_path, monkeypatch):
    preset = tmp_path / "figurine.safetensors"
    preset.touch()
    pipe = StrictInitPipe()
    monkeypatch.setattr(
        "examples.wanvideo.model_training.train.WanVideoPipeline.from_pretrained",
        lambda **kwargs: pipe,
    )

    module = WanTrainingModule(
        task="direct_distill",
        preset_lora_path=str(preset),
        preset_lora_model="dit",
        lora_base_model="dit",
        lora_target_modules="q",
        lora_rank=2,
        direct_distill_target_latent_key="input_latents",
        use_gradient_checkpointing=True,
    )

    assert pipe.preset_hotload_values == [False]
    module.assert_direct_distill_trainable_parameters("dit")
    assert module.trainable_param_names() == {
        "pipe.dit.q.lora_A.default.weight",
        "pipe.dit.q.lora_B.default.weight",
    }


def test_only_new_lora_is_trainable_and_exported():
    module = WanTrainingModule.__new__(WanTrainingModule)
    torch.nn.Module.__init__(module)
    pipe = torch.nn.Module()
    pipe.dit = TinyDiT()
    # Simulate the already-fused preset, then freeze the resulting base weight.
    with torch.no_grad():
        pipe.dit.q.weight.add_(1.0)
    pipe.requires_grad_(False)
    pipe.dit = module.add_lora_to_model(
        pipe.dit,
        target_modules=["q"],
        lora_rank=2,
        upcast_dtype=torch.float32,
    )
    module.pipe = pipe

    module.assert_direct_distill_trainable_parameters("dit")
    trainable_names = module.trainable_param_names()
    assert trainable_names
    assert all(name.startswith("pipe.dit.") for name in trainable_names)
    assert all(
        name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
        for name in trainable_names
    )
    assert pipe.dit.q.base_layer.weight.requires_grad is False

    pipe.dit.q(torch.ones(2, 4)).sum().backward()
    gradients = {
        name: parameter.grad
        for name, parameter in module.named_parameters()
    }
    assert gradients["pipe.dit.q.base_layer.weight"] is None
    assert gradients["pipe.dit.q.lora_A.default.weight"] is not None
    assert gradients["pipe.dit.q.lora_B.default.weight"] is not None

    exported = module.export_trainable_state_dict(
        module.state_dict(), remove_prefix="pipe.dit."
    )
    assert exported
    assert len(exported) == len(trainable_names)
    assert all(
        name.endswith((".lora_A.default.weight", ".lora_B.default.weight"))
        for name in exported
    )


def test_strict_student_controls_fail_before_pipeline_units():
    with pytest.raises(ValueError, match="cfg_scale=1"):
        WanTrainingModule.validate_direct_distill_student_controls({
            "cfg_scale": 5, "num_inference_steps": 4, "sigma_shift": 5,
        })
    with pytest.raises(ValueError, match="num_inference_steps=4"):
        WanTrainingModule.validate_direct_distill_student_controls({
            "cfg_scale": 1, "num_inference_steps": 8, "sigma_shift": 5,
        })
    with pytest.raises(ValueError, match="exact integer"):
        WanTrainingModule.validate_direct_distill_student_controls({
            "cfg_scale": 1, "num_inference_steps": 4.5, "sigma_shift": 5,
        })
    with pytest.raises(ValueError, match="sigma_shift=5"):
        WanTrainingModule.validate_direct_distill_student_controls({
            "cfg_scale": 1, "num_inference_steps": 4, "sigma_shift": 3,
        })


def test_checkpoint_coverage_rejects_missing_unexpected_and_wrong_shape():
    expected = {"q.lora_A.default.weight": torch.Size((2, 4))}
    valid = {"q.lora_A.default.weight": torch.zeros(2, 4)}
    WanTrainingModule.validate_checkpoint_tensor_coverage(expected, valid)
    with pytest.raises(RuntimeError, match="missing"):
        WanTrainingModule.validate_checkpoint_tensor_coverage(expected, {})
    with pytest.raises(RuntimeError, match="unexpected"):
        WanTrainingModule.validate_checkpoint_tensor_coverage(
            expected, {**valid, "extra.lora_A.default.weight": torch.zeros(1)}
        )
    with pytest.raises(RuntimeError, match="shape_mismatch"):
        WanTrainingModule.validate_checkpoint_tensor_coverage(
            expected, {"q.lora_A.default.weight": torch.zeros(1, 4)}
        )


def test_forward_records_actual_training_step_context_without_changing_result():
    module = WanTrainingModule.__new__(WanTrainingModule)
    torch.nn.Module.__init__(module)
    module.direct_distill_strict = False
    module.direct_distill_target_latent_key = None
    module.direct_distill_preserve_first_frame = False
    module.direct_distill_exclude_first_frame_loss = False
    module.pipe = SimpleNamespace(
        device=torch.device("cpu"),
        torch_dtype=torch.float32,
        units=[],
        dit=SimpleNamespace(patch_size=(1, 2, 2)),
    )
    module.task = "direct_distill:data_process"
    module.task_to_loss = {"direct_distill:data_process": lambda pipe, *args: args}
    latents = torch.randn(1, 4, 3, 2, 2)
    inputs = ({"latents": latents, "num_inference_steps": 4}, {}, {})

    result = module({}, inputs=inputs)

    assert result[0]["latents"] is latents
    assert module._last_training_step_context == {
        "latent_shape": (1, 4, 3, 2, 2),
        "patch_size": (1, 2, 2),
        "num_inference_steps": 4,
    }
