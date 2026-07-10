# Wan2.2-TI2V-5B + Figurine360 四步 DirectDistill（内网 H100）

本文只描述内网执行。脚本不会下载模型；所有模型、LoRA、数据与输出路径都在
`examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh`
顶部变量块中配置。

## 0. 先配置路径

打开 shell 脚本，只改顶部变量块：

- `DIT_PATHS_JSON`：同一 Wan2.2-TI2V-5B DiT 的全部 shard，保持为一个 JSON 列表；不能把 shard 当成多个独立模型。
- `TEXT_ENCODER_PATH`、`VAE_PATH`：本地文本编码器和 VAE 权重文件。
- `TOKENIZER_PATH`：本地 UMT5 tokenizer 目录。
- `FIGURINE360_LORA`：已有 rank-32 figurine360 LoRA。
- `RAW_DATA_ROOT`、`RAW_METADATA`：原始视频根目录及 `metadata.csv(video,prompt)`。
- `SMOKE_ROOT`、`TEACHER_ROOT`、`TRAIN_OUTPUT`：冒烟 teacher、正式 teacher、训练输出。
- `TRAIN_LORA_CHECKPOINT`：正式/冒烟训练默认留空，从新 adapter 开始；仅做权重续训时填写。
- `SMOKE_VALIDATION_LORA_CHECKPOINT`：只指向 smoke-train 的 checkpoint，仅用于冒烟对比。
- `FORMAL_VALIDATION_LORA_CHECKPOINT`：只指向正式 train（或正式续训）的最终 checkpoint，仅用于 held-out 验证。
- `RESUME_OUTPUT_PATH`：续训时必须指定一个全新输出目录，防止 step/EMA 重置后追加旧 JSONL 或覆盖 checkpoint。
- `PLOT_RUN_OUTPUT`：默认使用 `RESUME_OUTPUT_PATH`（若非空），否则使用 `TRAIN_OUTPUT`；也可显式指定要画图的某次训练目录。

推荐直接传本地 `model_path`，不需要把统一权重目录复制或改名。只有旧脚本硬编码默认目录时才建立软链接，例如：

```bash
ln -s /统一权重目录/Wan2.2-TI2V-5B ./models/Wan2.2-TI2V-5B
```

直接路径更清晰，也不会因链接层级改变而失效。

## 1. 离线环境检查

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh doctor
```

环境至少需要项目依赖，以及 `pytest`、`matplotlib`、`Pillow`、`imageio`、`imageio-ffmpeg`；`doctor` 会逐项 fail-fast。TensorBoard/W&B/SwanLab 都是可选项；内网无服务时只写 `metrics.jsonl`，训练后离线画图。依赖需提前放入内网镜像或离线环境，不要在服务器执行任何 `modelscope download` 或 Hugging Face 下载命令。

## 2. 低分辨率 teacher 冒烟（先做）

默认使用 2～4 条、短帧、低分辨率数据：

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh prepare-smoke
```

检查：

- `SMOKE_ROOT/metadata_direct_distill.csv` 已生成。
- `SMOKE_ROOT/first_frames/*.png` 是无损首帧缓存。
- `SMOKE_ROOT/teacher_latents/*.safetensors` 只有固定 key `latents`，shape/dtype/finite 校验通过。
- 准备命令的最终 summary 中 `failed=0`；若 `failures.jsonl` 非空，历史上已隔离并恢复的记录会保留，结合记录的 `recovered` 字段判断，不要只按文件是否存在下结论。
- 重跑同一命令会校验并跳过有效产物，不覆盖它们。

teacher 与 student 使用每条 metadata 的同一 seed。smoke 默认 canonical seed 为 `1`。正式默认 `TRAIN_SEEDS="2 3 4"`，held-out `VALIDATION_SEEDS="1"`：validation 同时满足未见首帧和未见 seed，并保留既有 figurine360 的 canonical seed=1。每个对象只属于一个 split，避免跨对象 seed 扩展造成泄漏。

## 3. 2～10 个 optimizer step 冒烟训练

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh smoke-train
```

启动断言必须通过：trainable 参数全是新 PEFT LoRA 的 `lora_A.default`/`lora_B.default`；base、figurine360、VAE、text encoder 均冻结。若误传 `--trainable_models dit`、preset 与 checkpoint 是同一文件、checkpoint 缺 key/shape 不符，或 trainable key 越界，程序会立即停止。

冒烟后检查：

- loss finite；student 每样本恰 4 次 DiT forward、CFG=1。
- checkpoint 只含 DirectDistill LoRA key。
- `metrics.jsonl` 每个真实 optimizer step 一行，并包含 loss/EMA/lr/token/视频吞吐/step time/显存。
- 用 `TRAIN_LORA_CHECKPOINT` 指向最后一个 `.safetensors`，并把 `RESUME_OUTPUT_PATH` 设为新目录后，可继续 LoRA 权重训练。

注意：当前上游 checkpoint 语义是“LoRA 权重续训”，optimizer/scheduler/global-step 会重新初始化；不是逐 bit 的完整训练状态恢复。

## 4. 冒烟 teacher/student 对比

把 `SMOKE_VALIDATION_LORA_CHECKPOINT` 指向冒烟 checkpoint，然后执行（不要设置 `TRAIN_LORA_CHECKPOINT`，否则正式训练会 warm-start smoke seed=1，破坏 held-out seed 验收）：

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh validate-smoke
```

输出包括 teacher/student 独立 MP4、并排 MP4、两份 latent 和 `validation.json`。脚本强制 student 为 4 步、CFG=1、shift=5，运行时统计并断言只有 4 次 model forward；teacher 与 student 使用同 seed、同首帧。

## 5. 正式 teacher 数据与训练

冒烟全部通过后再运行：

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh prepare
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh train
```

teacher 默认 50 步、CFG=5、shift=5；student metadata 固定 4 步、CFG=1、shift=5。teacher 若过曝、旋转不完整或首帧/身份错误，应先删除对应坏 latent 并重新生成，不能依赖 student 修复 teacher 缺陷。

## 6. 未见首帧/未见 seed 验证

先把 `FORMAL_VALIDATION_LORA_CHECKPOINT` 指向第 5 节正式训练（或正式续训）的最终 checkpoint；不得指向 smoke checkpoint。再从 metadata 的 validation split 选择至少 12 个对象，逐条运行：

```bash
VALIDATION_INDEX=0 bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh validate
```

递增 `VALIDATION_INDEX`。最终人工检查：旋转接近完整 360°、身份与首帧一致、无明显周期闪烁/过曝；同时确认两份 LoRA 可独立顺序加载，student 仍只有 4 次 forward。

## 7. 离线画图

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh plot
```

默认在本次有效训练目录的 `plots/` 生成；未续训时是 `TRAIN_OUTPUT/plots/`，设置了 `RESUME_OUTPUT_PATH` 时自动改为该续训目录。需要查看其他 run 时显式设置 `PLOT_RUN_OUTPUT`。

- `loss.png`：raw loss + beta=0.98 标量 EMA。
- `throughput.png`：视频/模型 token、视频吞吐、step time、可用时最大 CUDA 显存。
- `metrics.csv`、`summary.json`。

`global_video_tokens` 每视频只计一次；`global_model_tokens=global_video_tokens×4`；`global_tokens_per_hour` 指 model tokens/hour。多卡计数求和，step time 和显存取各 rank 最大值；计时包含 forward/backward/optimizer，不含数据准备、日志、checkpoint 和验证。若要同时启用 TensorBoard，执行训练命令时设置 `ENABLE_TENSORBOARD=1`；默认只写离线 JSONL。

## 常见错误

- **程序尝试联网**：模型或 tokenizer 路径不是本地有效路径。运行 `doctor`，不要填写 model ID。
- **target/student shape 不一致**：teacher 准备和训练的 height/width/num_frames 不一致；重新生成对应 teacher latent。
- **CFG 报错**：student metadata 的 `cfg_scale` 必须为 1；CFG=5 只属于 teacher。
- **首帧报错**：缓存 PNG 与 teacher latent 不匹配，或坏产物未被清理；先运行准备脚本校验并重生成该样本。
- **Teacher latent metadata mismatch**：`.safetensors` 必须用 `safetensors.safe_open` 检查，不能用 `torch.load`。若 `safe_open(...).metadata()` 含完整 provenance，请确认已更新到包含 Accelerate metadata sidecar 修复的版本；若 metadata 本身缺字段，重跑当前 `prepare-smoke`/`prepare` 让脚本隔离并重建旧产物。
- **0 个 LoRA 层匹配**：figurine 或 DirectDistill LoRA 与 Wan2.2-TI2V-5B 层名不匹配；不要忽略启动断言。
- **没有图表**：确认 `PLOT_RUN_OUTPUT/metrics.jsonl` 非空（默认是有效的正式或续训输出目录），并在离线环境预装 matplotlib。

所有真实 5B 权重加载、H100 吞吐与视觉质量结论都必须在内网 H100 按上述顺序验证；Mac 的合成测试不能替代这些结果。
