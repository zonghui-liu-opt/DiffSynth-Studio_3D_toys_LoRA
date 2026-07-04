# Wan2.2-TI2V-5B LoRA 训练/推理调用链

本文按当前仓库源码梳理猫咪 TI2V-5B LoRA 实现，供手办 360 度旋转任务复用。结论均附源码路径、类或函数名。

## 训练链

### `train_ti2v5b_lora.sh`

- 文件：`train_ti2v5b_lora.sh`
- 作用：Stage B 上机入口，收敛本地权重、数据、输出、尺寸、卡数、指标路径等环境差异。
- 顶部变量块：
  - `MODEL_ROOT`：本地 Wan2.2-TI2V-5B 权重根目录。
  - `TOKENIZER_PATH`：本地 `google/umt5-xxl` tokenizer 目录。
  - `DATA_ROOT`：数据集根目录。
  - `METADATA_PATH`：训练 metadata，猫咪链路默认使用 `metadata_fixed.csv`。
  - `OUTPUT_ROOT`：LoRA、`training_args.json`、`metrics.jsonl` 输出目录。
  - `NUM_GPUS`：传给 `accelerate launch --num_processes`。
  - `HEIGHT/WIDTH/NUM_FRAMES`：视频训练尺寸和帧数；`NUM_FRAMES` 需满足 `(num_frames - 1) % 4 == 0`。
  - `SAVE_STEPS`：为空则按 epoch 保存；非空则按 step 保存。
  - `ENABLE_ORIENTATION_BUCKETS`：开启后传 `--enable_orientation_buckets`，横竖屏按 bucket resize。
- 本地权重：脚本检查 `diffusion_pytorch_model*.safetensors`、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.2_VAE.pth` 和 tokenizer 目录，然后生成 `--model_paths` JSON 列表，避免内网机器走下载。
- 训练命令：`accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 examples/wanvideo/model_training/train.py ...`。

### argparse 与入口

- 文件：`examples/wanvideo/model_training/train.py`
- 函数：`wan_parser()`
- 作用：组合通用训练参数和 Wan 视频尺寸参数，再追加 tokenizer、audio processor、timestep 边界、CPU 初始化、framewise decoding。
- 参数来源：
  - `diffsynth/diffusion/parsers.py:add_dataset_base_config()`：`--dataset_base_path`、`--dataset_metadata_path`、`--dataset_repeat`、`--dataset_num_workers`、`--data_file_keys`。其中 `--data_file_keys` 默认是 `image,video`。
  - `diffsynth/diffusion/parsers.py:add_model_config()`：`--model_paths`、`--model_id_with_origin_paths`、`--extra_inputs`、`--fp8_models`、`--offload_models`、`--resume_from_checkpoint`。
  - `diffsynth/diffusion/parsers.py:add_training_config()`：`--learning_rate`、`--num_epochs`、`--trainable_models`、`--find_unused_parameters`、`--weight_decay`、`--task`、`--customized_optimizer`、`--metrics_path`。
  - `diffsynth/diffusion/parsers.py:add_output_config()`：`--output_path`、`--remove_prefix_in_ckpt`、`--save_steps`。
  - `diffsynth/diffusion/parsers.py:add_lora_config()`：`--lora_base_model`、`--lora_target_modules`、`--lora_rank`、`--lora_checkpoint`、`--preset_lora_path`、`--preset_lora_model`。
  - `diffsynth/diffusion/parsers.py:add_gradient_config()`：`--use_gradient_checkpointing`、`--use_gradient_checkpointing_offload`、`--gradient_accumulation_steps`。
  - `diffsynth/diffusion/parsers.py:add_video_size_config()`：`--height`、`--width`、`--max_pixels`、`--num_frames`、`--enable_orientation_buckets`。

### 数据集读取

- 文件：`examples/wanvideo/model_training/train.py`
- 代码段：`__main__` 中创建 `UnifiedDataset`
- 作用：把 metadata 行转为训练样本 dict。
- 构造参数：
  - `base_path=args.dataset_base_path`
  - `metadata_path=args.dataset_metadata_path`
  - `repeat=args.dataset_repeat`
  - `data_file_keys=args.data_file_keys.split(",")`
  - `main_data_operator=UnifiedDataset.default_video_operator(...)`
  - `special_operator_map` 仅覆盖 `animate_face_video`、`input_audio`、`wantodance_music_path`。
- 文件：`diffsynth/core/data/unified_dataset.py`
- 类：`UnifiedDataset`
- 关键行为：
  - `load_metadata()` 对 `.json/.jsonl` 走 JSON；其他后缀直接 `pandas.read_csv(metadata_path)`。
  - `__getitem__()` 只处理 `data_file_keys` 中且 metadata 行实际存在的 key；不存在的 key 会被跳过。
  - `default_video_operator()` 对 mp4/avi/mov 等视频调用 `LoadVideo`，对 jpg/png 等图片调用 `LoadImage` 后转 list。
- 文件：`diffsynth/core/data/operators.py`
- 类：`LoadVideo`
- 作用：从视频开头按 `num_frames` 读取帧，帧数不足时向下取满足时间压缩约束的帧数，返回 PIL 图像列表。

### 训练样本进入 pipeline

- 文件：`examples/wanvideo/model_training/train.py`
- 类：`WanTrainingModule`
- 函数：`get_pipeline_inputs()`
- 作用：把 dataset 样本拆为 `inputs_shared`、`inputs_posi`、`inputs_nega`。
- 关键字段：
  - `inputs_posi["prompt"] = data["prompt"]`
  - `inputs_shared["input_video"] = data["video"]`
  - `height/width/num_frames` 来自已加载视频帧。
  - `parse_extra_inputs()` 把 `--extra_inputs` 声明的字段补进 `inputs_shared`。
- 函数：`parse_extra_inputs()`
- `input_image` 机制：若 `data.get("input_image")` 是非空 list，取第 0 张；否则回退为 `data["video"][0]`，即视频第一帧。

### 模型加载

- 文件：`examples/wanvideo/model_training/train.py`
- 类：`WanTrainingModule.__init__`
- 作用：解析模型配置并创建 `WanVideoPipeline`。
- 文件：`diffsynth/diffusion/training_module.py`
- 函数：`DiffusionTrainingModule.parse_model_configs()`
- 本地 `--model_paths` 逻辑：将 JSON 解析为列表；每个元素传入 `ModelConfig(path=path, **vram_config)`。DiT 分片可以是路径列表，T5 和 VAE 是单文件路径。
- 文件：`diffsynth/pipelines/wan_video.py`
- 函数：`WanVideoPipeline.from_pretrained()`
- 作用：通过 `download_and_load_models()` 加载模型池，再取出 `text_encoder`、`dit/dit2`、`vae`、`image_encoder` 等组件；传本地 `ModelConfig(path=...)` 时不下载。

### LoRA 注入

- 文件：`diffsynth/diffusion/training_module.py`
- 函数：`DiffusionTrainingModule.switch_pipe_to_training_mode()`
- 作用：设置训练 scheduler，冻结非训练模型，再按 `lora_base_model` 注入 LoRA。
- 函数：`parse_lora_target_modules()`
- 作用：若 `--lora_target_modules` 非空，按逗号拆分；猫咪链路显式使用 `q,k,v,o,ffn.0,ffn.2`。
- 函数：`add_lora_to_model()`
- 作用：用 PEFT `LoraConfig(r=lora_rank, lora_alpha=lora_rank, target_modules=...)` 和 `inject_adapter_in_model()` 注入 LoRA。

### 训练循环、loss 和指标

- 文件：`diffsynth/diffusion/runner.py`
- 函数：`launch_training_task()`
- 作用：创建 AdamW、ConstantLR、DataLoader，`accelerator.prepare()` 后按 epoch/step 训练。
- 训练 step：
  - `loss = model(data)` 进入 `WanTrainingModule.forward()`。
  - `accelerator.backward(loss)`。
  - `optimizer.step()`、`scheduler.step()`、`optimizer.zero_grad()`。
  - `model_logger.on_step_end(...)` 处理日志和 step ckpt。
- 指标分支：
  - `metrics_path = getattr(args, "metrics_path", None)`。
  - 主进程创建 `MetricsWriter(metrics_path)`。
  - `accelerator.sync_gradients` 时写 JSONL，字段包括 `step`、`epoch`、`loss`、`step_time_sec`、`tokens_per_sample`、`samples_per_step`、`lr`。
- 文件：`diffsynth/diffusion/loss.py`
- 函数：`FlowMatchSFTLoss()`
- 作用：采样 timestep，加噪 `input_latents`，调用 `pipe.model_fn()` 预测噪声，MSE 监督训练目标。

### checkpoint 保存

- 文件：`diffsynth/diffusion/logger.py`
- 类：`ModelLogger`
- 作用：保存训练状态中的可训练参数。
- 保存策略：
  - `on_epoch_end()` 保存 `epoch-{epoch_id}.safetensors`。
  - `on_step_end()` 在 `save_steps` 整除时保存 `step-{num_steps}.safetensors`。
  - `on_training_end()` 在 step 保存模式下补最终 step。
- `remove_prefix_in_ckpt`：
  - `save_model()` 调 `export_trainable_state_dict(..., remove_prefix=self.remove_prefix_in_ckpt)`。
  - `DiffusionTrainingModule.export_trainable_state_dict()` 只导出 `requires_grad=True` 参数，并移除如 `pipe.dit.` 的前缀。

## 推理链

### pipeline 和本地权重

- 文件：`diffsynth/pipelines/wan_video.py`
- 函数：`WanVideoPipeline.from_pretrained()`
- 作用：加载 Wan pipeline、DiT、T5、VAE 等组件。
- 本地加载方式：传 `model_configs=[ModelConfig(path=DIT_PATHS), ModelConfig(path=T5_PATH), ModelConfig(path=VAE_PATH)]`，tokenizer 传 `tokenizer_config=ModelConfig(path=TOKENIZER_PATH)`。

### LoRA 加载

- 文件：`diffsynth/diffusion/base_pipeline.py`
- 函数：`BasePipeline.load_lora()`
- 作用：读取 LoRA state dict，调用 pipeline 的 `lora_loader` 转换 key，并 fuse/hotload 到目标模块。
- TI2V LoRA 推理写法：`pipe.load_lora(pipe.dit, "epoch-N.safetensors", alpha=1)`。

### `WanVideoPipeline.__call__()`

- 文件：`diffsynth/pipelines/wan_video.py`
- 函数：`WanVideoPipeline.__call__()`
- 入参：`prompt`、`negative_prompt`、`input_image`、`height`、`width`、`num_frames`、`seed`、`cfg_scale`、`num_inference_steps`、`tiled` 等。
- 数据流：
  - `inputs_posi` 放 prompt，`inputs_nega` 放 negative prompt。
  - `inputs_shared` 放 `input_image`、shape、seed、VAE tiling 等。
  - 逐个执行 `self.units`。
  - denoise 循环里调用 `self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)`。
  - VAE decode 后返回 PIL 帧列表。
- 保存视频：调用 `diffsynth.utils.data.save_video(video, path, fps=15, quality=5)`。

## 关键机制

### TI2V 的首帧条件注入

- 文件：`examples/wanvideo/model_training/train.py`
- 函数：`WanTrainingModule.parse_extra_inputs()`
- 训练阶段若 metadata 无 `input_image` 列但 `--extra_inputs input_image` 开启，`inputs_shared["input_image"] = data["video"][0]`。
- 文件：`diffsynth/pipelines/wan_video.py`
- 类：`WanVideoUnit_ImageEmbedderFused`
- 作用：Wan2.2-TI2V-5B 专用条件注入单元。
- 行为：
  - 将 `input_image` resize 到训练尺寸并 VAE encode。
  - 把得到的 `z` 写入 `latents[:, :, 0:1]`。
  - 返回 `first_frame_latents`。
- 文件：`diffsynth/diffusion/loss.py`
- 函数：`FlowMatchSFTLoss()`
- 行为：
  - 若存在 `first_frame_latents`，加噪后再次把首帧 latent 写回 `latents[:, :, 0:1]`。
  - loss 前丢弃 `noise_pred[:, :, 0:1]` 和对应 training target，只监督首帧之后的 latents。

### token 数公式

- 文件：`metrics_utils.py`
- 函数：`tokens_per_sample(num_frames, height, width)`
- 公式：
  - `latent_frames = (num_frames - 1) // 4 + 1`
  - `tokens = latent_frames * (height // 32) * (width // 32)`
- 含义：TI2V-5B 视频时间经 VAE 压缩 4 倍；空间经 VAE 与 DiT patchify 后有效压缩为 32 倍，因此总有效压缩为 `4 x 32 x 32`。
