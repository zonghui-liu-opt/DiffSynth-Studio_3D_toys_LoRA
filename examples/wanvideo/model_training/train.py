import torch, os, argparse, accelerate, warnings, math, numbers, hashlib
from collections.abc import Mapping
from PIL import Image
from safetensors import safe_open
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import (
    DataProcessingOperator,
    LoadImage,
    LoadVideo,
    LoadAudio,
    ImageCropAndResize,
    ToAbsolutePath,
)
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


DIRECT_DISTILL_LATENT_METADATA_KEY = "_direct_distill_teacher_metadata"


class LoadDirectDistillLatents(DataProcessingOperator):
    """Load the fixed `latents` tensor from a teacher safetensors file on CPU."""

    tensor_key = "latents"

    def __call__(self, data):
        if not isinstance(data, (str, os.PathLike)):
            raise TypeError("Teacher latent path must be a string or path-like object.")
        path = os.fspath(data)
        if not path.lower().endswith(".safetensors"):
            raise ValueError(f"Teacher latent file must use safetensors format: {path}")
        with safe_open(path, framework="pt", device="cpu") as file:
            keys = set(file.keys())
            metadata = file.metadata() or {}
            latents = file.get_tensor(self.tensor_key) if self.tensor_key in keys else None
        if keys != {self.tensor_key}:
            raise KeyError(
                f"Teacher latent file `{path}` must contain only the fixed key `{self.tensor_key}`; "
                f"found {sorted(keys)}."
            )
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"Teacher latent key `{self.tensor_key}` in `{path}` is not a tensor.")
        if latents.ndim != 5:
            raise ValueError(
                f"Teacher latents must have B,C,T,H,W shape, but `{path}` contains {tuple(latents.shape)}."
            )
        if not torch.is_floating_point(latents):
            raise ValueError(f"Teacher latents in `{path}` must use a floating-point dtype.")
        if not torch.isfinite(latents).all():
            raise ValueError(f"Teacher latents in `{path}` contain non-finite values.")
        latents = latents.cpu()
        latents._direct_distill_metadata = metadata
        return latents


class LoadDirectDistillInputImage(DataProcessingOperator):
    """Load and resize an optional cached first-frame image without touching the source video."""

    def __init__(self, base_path, height, width, max_pixels):
        self.to_absolute_path = ToAbsolutePath(base_path)
        self.load_image = LoadImage()
        self.resize = ImageCropAndResize(height, width, max_pixels, 16, 16)

    def __call__(self, data):
        if data is None or (
            isinstance(data, numbers.Real) and not math.isfinite(float(data))
        ):
            return None
        image = self.load_image(self.to_absolute_path(data))
        return self.resize(image)


class LoadDirectDistillVideoFirstFrame(DataProcessingOperator):
    """Read and resize only video[0] for the cached-image compatibility fallback."""

    def __init__(self, base_path, height, width, max_pixels):
        self.to_absolute_path = ToAbsolutePath(base_path)
        self.resize = ImageCropAndResize(height, width, max_pixels, 16, 16)

    def __call__(self, data):
        import imageio.v2 as imageio

        path = self.to_absolute_path(data)
        reader = imageio.get_reader(path)
        try:
            image = Image.fromarray(reader.get_data(0)).convert("RGB")
        finally:
            reader.close()
        return self.resize(image)


class DirectDistillDataset(UnifiedDataset):
    def __init__(self, *args, input_image_fallback_operator=None, **kwargs):
        self.input_image_fallback_operator = input_image_fallback_operator
        super().__init__(*args, **kwargs)

    def __getitem__(self, data_id):
        data = super().__getitem__(data_id)
        teacher_latents = data.get("teacher_latent")
        latent_metadata = getattr(teacher_latents, "_direct_distill_metadata", None)
        if isinstance(latent_metadata, Mapping):
            # Accelerate moves tensors to the training device with Tensor.to(),
            # which drops arbitrary Python attributes. Keep provenance in the
            # batch mapping so it survives workers, device placement, and DDP.
            data[DIRECT_DISTILL_LATENT_METADATA_KEY] = dict(latent_metadata)
        if self.input_image_fallback_operator is None:
            return data
        if not _is_missing_metadata_value(data.get("input_image")):
            return data
        video = data.get("video")
        if isinstance(video, str):
            data["input_image"] = self.input_image_fallback_operator(video)
        elif isinstance(video, (list, tuple)) and video:
            data["input_image"] = video[0]
        return data


def direct_distill_strict_requested(args):
    return bool(
        getattr(args, "direct_distill_preserve_first_frame", False)
        or getattr(args, "direct_distill_exclude_first_frame_loss", False)
        or getattr(args, "direct_distill_target_latent_key", None) is not None
    )


def build_wan_special_operator_map(args):
    operator_map = {
        "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(
            args.num_frames,
            4,
            1,
            frame_processor=ImageCropAndResize(512, 512, None, 16, 16),
        ),
        "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
    }
    if "input_audio" in args.data_file_keys.split(","):
        operator_map["input_audio"] = ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000)
    if direct_distill_strict_requested(args):
        # Strict mode loads the cached first frame independently and never decodes the full video.
        operator_map["input_image"] = LoadDirectDistillInputImage(
            args.dataset_base_path, args.height, args.width, args.max_pixels
        )
        operator_map["teacher_latent"] = (
            ToAbsolutePath(args.dataset_base_path) >> LoadDirectDistillLatents()
        )
    return operator_map


def _is_missing_metadata_value(value):
    if value is None:
        return True
    if isinstance(value, numbers.Real):
        return not math.isfinite(float(value))
    return False


def _unwrap_first_item(value):
    if isinstance(value, (list, tuple)):
        return None if len(value) == 0 else value[0]
    return value


def _is_loaded_image(value):
    size = getattr(value, "size", None)
    return isinstance(size, tuple) and len(size) == 2


def resolve_direct_distill_input_image(data):
    cached_image = _unwrap_first_item(data.get("input_image"))
    if not _is_missing_metadata_value(cached_image) and _is_loaded_image(cached_image):
        return cached_image

    video = data.get("video")
    if isinstance(video, (list, tuple)) and len(video) > 0:
        first_frame = video[0]
        if not _is_missing_metadata_value(first_frame) and _is_loaded_image(first_frame):
            return first_frame
    raise ValueError(
        "DirectDistill requires a loaded `input_image`, or a loaded `video` list for video[0] fallback."
    )


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        resume_from_checkpoint=None, remove_prefix_in_ckpt=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        direct_distill_preserve_first_frame=False,
        direct_distill_exclude_first_frame_loss=False,
        direct_distill_target_latent_key=None,
    ):
        super().__init__()
        self.direct_distill_preserve_first_frame = direct_distill_preserve_first_frame
        self.direct_distill_exclude_first_frame_loss = direct_distill_exclude_first_frame_loss
        self.direct_distill_strict = bool(
            direct_distill_preserve_first_frame
            or direct_distill_exclude_first_frame_loss
            or direct_distill_target_latent_key is not None
        )
        self.direct_distill_target_latent_key = (
            direct_distill_target_latent_key or "input_latents"
            if self.direct_distill_strict else None
        )
        if self.direct_distill_strict:
            self.validate_direct_distill_training_config(
                task=task,
                preset_lora_path=preset_lora_path,
                preset_lora_model=preset_lora_model,
                lora_base_model=lora_base_model,
                lora_checkpoint=lora_checkpoint,
                trainable_models=trainable_models,
                resume_from_checkpoint=resume_from_checkpoint,
            )

        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True

        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = self.parse_path_or_model_id(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)

        preset_lora_path_for_switch = preset_lora_path
        preset_lora_model_for_switch = preset_lora_model
        if self.direct_distill_strict:
            # Force a true, frozen fusion before PEFT injects the independently trainable adapter.
            self.pipe.freeze_except([])
            preset_lora_match_count = self.pipe.load_lora(
                getattr(self.pipe, preset_lora_model),
                preset_lora_path,
                hotload=False,
            )
            if preset_lora_match_count is not None and preset_lora_match_count <= 0:
                raise RuntimeError(
                    "The preset figurine360 LoRA matched zero target modules; refusing to train "
                    "a DirectDistill adapter against an unfused base model."
                )
            preset_lora_path_for_switch = None
            preset_lora_model_for_switch = None
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path_for_switch, preset_lora_model_for_switch,
            task=task,
        )
        if self.direct_distill_strict and not task.endswith(":data_process"):
            self.assert_direct_distill_trainable_parameters(lora_base_model)
            if lora_checkpoint is not None:
                self.assert_direct_distill_checkpoint_coverage(
                    lora_base_model, lora_checkpoint
                )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    @staticmethod
    def validate_direct_distill_training_config(
        task,
        preset_lora_path,
        preset_lora_model,
        lora_base_model,
        lora_checkpoint,
        trainable_models,
        resume_from_checkpoint,
    ):
        if not task.startswith("direct_distill"):
            raise ValueError("Strict DirectDistill options can only be used with a direct_distill task.")
        if resume_from_checkpoint is not None:
            raise ValueError(
                "Strict DirectDistill LoRA is injected after model loading, so `resume_from_checkpoint` "
                "cannot restore it. Use `lora_checkpoint` for DirectDistill LoRA continuation."
            )
        if not preset_lora_path:
            raise ValueError("Strict DirectDistill requires `preset_lora_path` for the frozen preset LoRA.")
        if not os.path.isfile(os.path.expanduser(preset_lora_path)):
            raise FileNotFoundError(f"Preset LoRA does not exist: {preset_lora_path}")
        if not preset_lora_model:
            raise ValueError("Strict DirectDistill requires `preset_lora_model`.")
        if not lora_base_model:
            raise ValueError("Strict DirectDistill requires `lora_base_model` for the new adapter.")
        if preset_lora_model != lora_base_model:
            raise ValueError(
                "Strict DirectDistill requires preset and new LoRA to target the same model; "
                f"received `{preset_lora_model}` and `{lora_base_model}`."
            )
        if trainable_models not in (None, ""):
            raise ValueError(
                "Strict DirectDistill must not enable full trainable models. Leave `trainable_models` unset "
                "and use only `lora_base_model`."
            )
        if lora_checkpoint is not None:
            preset_path = os.path.realpath(os.path.expanduser(preset_lora_path))
            checkpoint_path = os.path.realpath(os.path.expanduser(lora_checkpoint))
            if preset_path == checkpoint_path:
                raise ValueError(
                    "`preset_lora_path` and `lora_checkpoint` must be different files: the preset LoRA "
                    "is frozen, while lora_checkpoint is the trainable DirectDistill adapter."
                )

    def assert_direct_distill_trainable_parameters(self, lora_base_model):
        trainable_names = self.trainable_param_names()
        if not trainable_names:
            raise RuntimeError("Strict DirectDistill found no trainable LoRA parameters.")
        target_prefix = f"pipe.{lora_base_model}."
        lora_suffixes = (".lora_A.default.weight", ".lora_B.default.weight")
        invalid_names = sorted(
            name for name in trainable_names
            if not name.startswith(target_prefix) or not name.endswith(lora_suffixes)
        )
        if invalid_names:
            raise RuntimeError(
                "Strict DirectDistill permits only the new target-model PEFT A/B parameters, but found: "
                + ", ".join(invalid_names[:8])
            )
        if not any(name.endswith(lora_suffixes[0]) for name in trainable_names):
            raise RuntimeError("Strict DirectDistill found no trainable PEFT LoRA A parameters.")
        if not any(name.endswith(lora_suffixes[1]) for name in trainable_names):
            raise RuntimeError("Strict DirectDistill found no trainable PEFT LoRA B parameters.")

    @staticmethod
    def validate_checkpoint_tensor_coverage(expected_shapes, checkpoint_tensors):
        expected_keys = set(expected_shapes)
        checkpoint_keys = set(checkpoint_tensors)
        missing = sorted(expected_keys - checkpoint_keys)
        unexpected = sorted(checkpoint_keys - expected_keys)
        mismatched = sorted(
            key for key in expected_keys & checkpoint_keys
            if tuple(checkpoint_tensors[key].shape) != tuple(expected_shapes[key])
        )
        if missing or unexpected or mismatched:
            raise RuntimeError(
                "DirectDistill LoRA checkpoint does not exactly cover the new adapter: "
                f"missing={missing[:4]}, unexpected={unexpected[:4]}, shape_mismatch={mismatched[:4]}"
            )

    def assert_direct_distill_checkpoint_coverage(self, lora_base_model, checkpoint_path):
        model = getattr(self.pipe, lora_base_model)
        raw = load_state_dict(checkpoint_path, device="cpu")
        loader = self.pipe.lora_loader(torch_dtype=self.pipe.torch_dtype, device="cpu")
        converted = self.mapping_lora_state_dict(loader.convert_state_dict(raw))
        expected_shapes = {
            name: parameter.shape
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.validate_checkpoint_tensor_coverage(expected_shapes, converted)

    @staticmethod
    def validate_direct_distill_student_controls(inputs_shared):
        cfg_scale = float(inputs_shared.get("cfg_scale", 1))
        raw_steps = inputs_shared.get("num_inference_steps", 0)
        if isinstance(raw_steps, torch.Tensor) and raw_steps.numel() == 1:
            raw_steps = raw_steps.item()
        if (
            isinstance(raw_steps, bool)
            or not isinstance(raw_steps, numbers.Real)
            or not math.isfinite(float(raw_steps))
            or not float(raw_steps).is_integer()
        ):
            raise ValueError(
                f"Strict DirectDistill metadata requires an exact integer num_inference_steps=4; received {raw_steps!r}."
            )
        steps = int(raw_steps)
        sigma_shift = float(inputs_shared.get("sigma_shift", 5))
        if cfg_scale != 1:
            raise ValueError(f"Strict DirectDistill metadata requires cfg_scale=1; received {cfg_scale}.")
        if steps != 4:
            raise ValueError(f"Strict DirectDistill metadata requires num_inference_steps=4; received {steps}.")
        if sigma_shift != 5:
            raise ValueError(f"Strict DirectDistill metadata requires sigma_shift=5; received {sigma_shift}.")

    @staticmethod
    def validate_direct_distill_latent_provenance(data, teacher_latents):
        if not any(
            not _is_missing_metadata_value(data.get(key)) and data.get(key) != ""
            for key in ("sample_id", "teacher_fingerprint", "source_fingerprint")
        ):
            return
        metadata = data.get(DIRECT_DISTILL_LATENT_METADATA_KEY)
        if not isinstance(metadata, Mapping):
            # Compatibility for direct calls that have not passed through the
            # strict DirectDistillDataset/Accelerate dataloader path.
            metadata = getattr(teacher_latents, "_direct_distill_metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        for key in (
            "sample_id", "teacher_fingerprint", "source_fingerprint", "seed", "rand_device"
        ):
            expected = data.get(key)
            if _is_missing_metadata_value(expected) or expected == "":
                continue
            actual = metadata.get(key)
            if str(actual) != str(expected):
                raise ValueError(
                    f"Teacher latent metadata `{key}` mismatch: file={actual!r}, csv={expected!r}."
                )
        prompt = str(data.get("prompt", "")).strip()
        expected_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        actual_prompt_hash = metadata.get("prompt_sha256")
        if actual_prompt_hash is not None and actual_prompt_hash != expected_prompt_hash:
            raise ValueError("Teacher latent prompt hash does not match the CSV prompt.")

    @staticmethod
    def _metadata_bool(value, default=False):
        if _is_missing_metadata_value(value):
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("1", "true", "yes"):
                return True
            if normalized in ("0", "false", "no"):
                return False
            raise ValueError(f"Invalid boolean metadata value: {value!r}")
        return bool(value)

    @staticmethod
    def _metadata_pair(value, default):
        if _is_missing_metadata_value(value):
            return default
        if isinstance(value, str):
            parts = value.replace("x", ",").split(",")
            value = tuple(int(part.strip()) for part in parts if part.strip())
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Expected a two-value metadata pair, received {value!r}.")
        return tuple(int(item) for item in value)

    @staticmethod
    def _positive_metadata_int(value, name):
        if _is_missing_metadata_value(value):
            return None
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"DirectDistill metadata `{name}` must be a positive integer.")
        value_float = float(value)
        value_int = int(value_float)
        if value_float != value_int or value_int <= 0:
            raise ValueError(f"DirectDistill metadata `{name}` must be a positive integer.")
        return value_int

    def resolve_direct_distill_shape(self, data, input_image, target_latents):
        if not isinstance(target_latents, torch.Tensor) or target_latents.ndim != 5:
            shape = None if not isinstance(target_latents, torch.Tensor) else tuple(target_latents.shape)
            raise ValueError(f"DirectDistill teacher latents must have B,C,T,H,W shape; received {shape}.")

        image_size = getattr(input_image, "size", None)
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise TypeError("DirectDistill input_image must be a loaded PIL image with width/height.")
        image_width, image_height = image_size

        metadata_height = self._positive_metadata_int(data.get("height"), "height")
        metadata_width = self._positive_metadata_int(data.get("width"), "width")
        metadata_num_frames = self._positive_metadata_int(data.get("num_frames"), "num_frames")
        if metadata_height is not None and metadata_height != image_height:
            raise ValueError(
                f"DirectDistill metadata height {metadata_height} does not match input_image height {image_height}."
            )
        if metadata_width is not None and metadata_width != image_width:
            raise ValueError(
                f"DirectDistill metadata width {metadata_width} does not match input_image width {image_width}."
            )

        height = metadata_height or image_height
        width = metadata_width or image_width
        time_factor = getattr(self.pipe, "time_division_factor", None) or 4
        time_remainder = getattr(self.pipe, "time_division_remainder", None)
        time_remainder = 1 if time_remainder is None else time_remainder
        inferred_num_frames = (target_latents.shape[2] - 1) * time_factor + time_remainder
        num_frames = metadata_num_frames or inferred_num_frames
        expected_latent_frames = (num_frames - time_remainder) // time_factor + 1
        if num_frames % time_factor != time_remainder or expected_latent_frames != target_latents.shape[2]:
            raise ValueError(
                f"DirectDistill num_frames {num_frames} is incompatible with teacher latent T={target_latents.shape[2]}."
            )

        vae = getattr(self.pipe, "vae", None)
        spatial_factor = getattr(vae, "upsampling_factor", None)
        if spatial_factor is not None:
            expected_latent_height = height // spatial_factor
            expected_latent_width = width // spatial_factor
            if height % spatial_factor != 0 or width % spatial_factor != 0:
                raise ValueError(
                    f"DirectDistill height/width {(height, width)} must be divisible by VAE factor {spatial_factor}."
                )
            if target_latents.shape[-2:] != (expected_latent_height, expected_latent_width):
                raise ValueError(
                    "DirectDistill teacher latent spatial shape does not match input dimensions: "
                    f"expected {(expected_latent_height, expected_latent_width)}, "
                    f"received {tuple(target_latents.shape[-2:])}."
                )
        return height, width, num_frames

    def apply_direct_distill_controls(self, inputs_shared):
        if not self.direct_distill_strict:
            return inputs_shared
        if self.direct_distill_target_latent_key not in inputs_shared:
            teacher_latents = inputs_shared.get("teacher_latent")
            if isinstance(teacher_latents, torch.Tensor):
                inputs_shared[self.direct_distill_target_latent_key] = teacher_latents
        inputs_shared["direct_distill_target_latent_key"] = self.direct_distill_target_latent_key
        inputs_shared["direct_distill_preserve_first_frame"] = self.direct_distill_preserve_first_frame
        inputs_shared["direct_distill_exclude_first_frame_loss"] = self.direct_distill_exclude_first_frame_loss
        return inputs_shared
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                if self.direct_distill_strict:
                    inputs_shared["input_image"] = resolve_direct_distill_input_image(data)
                else:
                    inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        if inputs_shared.get("framewise_decoding", False):
            # WanToDance global model
            inputs_shared["num_frames"] = 4 * (len(data["video"]) - 1) + 1
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        if self.direct_distill_strict:
            input_image = resolve_direct_distill_input_image(data)
            teacher_latents = data.get("teacher_latent")
            if not isinstance(teacher_latents, torch.Tensor):
                raise ValueError(
                    "Strict DirectDistill requires loaded tensor metadata `teacher_latent`. "
                    "Include teacher_latent in `data_file_keys`."
                )
            self.validate_direct_distill_latent_provenance(data, teacher_latents)
            height, width, num_frames = self.resolve_direct_distill_shape(
                data, input_image, teacher_latents
            )
            inputs_shared = {
                "input_video": None,
                "input_image": input_image,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "cfg_scale": 1,
                "tiled": self._metadata_bool(data.get("tiled"), False),
                "tile_size": self._metadata_pair(data.get("tile_size"), (30, 52)),
                "tile_stride": self._metadata_pair(data.get("tile_stride"), (15, 26)),
                "rand_device": self.pipe.device,
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
                "cfg_merge": False,
                "vace_scale": 1,
                "max_timestep_boundary": self.max_timestep_boundary,
                "min_timestep_boundary": self.min_timestep_boundary,
                self.direct_distill_target_latent_key: teacher_latents,
            }
            inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
            inputs_shared["input_video"] = None
            inputs_shared["input_image"] = input_image
            inputs_shared[self.direct_distill_target_latent_key] = teacher_latents
            self.validate_direct_distill_student_controls(inputs_shared)
            return self.apply_direct_distill_controls(inputs_shared), inputs_posi, inputs_nega

        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, inputs_nega = inputs
        inputs_shared = self.apply_direct_distill_controls(inputs_shared)
        latents = inputs_shared.get("latents")
        dit = getattr(self.pipe, "dit", None)
        patch_size = getattr(dit, "patch_size", None)
        num_inference_steps = inputs_shared.get("num_inference_steps")
        if isinstance(num_inference_steps, torch.Tensor) and num_inference_steps.numel() == 1:
            num_inference_steps = num_inference_steps.item()
        if num_inference_steps is not None:
            num_inference_steps = int(num_inference_steps)
        self._last_training_step_context = {
            "latent_shape": tuple(latents.shape) if isinstance(latents, torch.Tensor) else None,
            "patch_size": tuple(patch_size) if patch_size is not None else None,
            "num_inference_steps": num_inference_steps,
        }
        inputs = inputs_shared, inputs_posi, inputs_nega
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--framewise_decoding", default=False, action="store_true", help="Enable it if this model is a WanToDance global model.")
    parser.add_argument(
        "--direct_distill_preserve_first_frame",
        default=False,
        action="store_true",
        help="Preserve TI2V first-frame latents after every DirectDistill scheduler step.",
    )
    parser.add_argument(
        "--direct_distill_exclude_first_frame_loss",
        default=False,
        action="store_true",
        help="Exclude the fixed TI2V first frame from the final DirectDistill latent loss.",
    )
    parser.add_argument(
        "--direct_distill_target_latent_key",
        type=str,
        default=None,
        help="Input key receiving metadata teacher_latent; enables strict DirectDistill when set.",
    )
    parser.add_argument(
        "--enable_direct_distill_metrics",
        default=False,
        action="store_true",
        help="Log optimizer-step DirectDistill loss/throughput metrics and metrics.jsonl.",
    )
    parser.add_argument(
        "--direct_distill_loss_ema_beta",
        type=float,
        default=0.98,
        help="Scalar loss EMA beta used by DirectDistill optimizer-step metrics.",
    )
    return parser


def build_wan_model_logger(args):
    return ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
        enable_metrics_jsonl=args.enable_direct_distill_metrics,
    )


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    strict_dataset = direct_distill_strict_requested(args)
    dataset_class = DirectDistillDataset if strict_dataset else UnifiedDataset
    dataset_kwargs = {}
    if strict_dataset:
        dataset_kwargs["input_image_fallback_operator"] = LoadDirectDistillVideoFirstFrame(
            args.dataset_base_path, args.height, args.width, args.max_pixels
        )
    dataset = dataset_class(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map=build_wan_special_operator_map(args),
        **dataset_kwargs,
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        direct_distill_preserve_first_frame=args.direct_distill_preserve_first_frame,
        direct_distill_exclude_first_frame_loss=args.direct_distill_exclude_first_frame_loss,
        direct_distill_target_latent_key=args.direct_distill_target_latent_key,
    )
    model_logger = build_wan_model_logger(args)
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
