# figurine360-DMD 内网 H100 操作说明

## 路径变量块

只改 `train_figurine360_dmd_lora.sh` 顶部变量或在命令前通过环境变量覆盖：

- `BASE_MODEL_ROOT`：原生 Wan2.2-TI2V-5B 权重目录。
- `FIG360_LORA_PATH`：Task-01 产出的 figurine360 LoRA safetensors。
- `MERGED_MODEL_ROOT`：离线 merge 后的 teacher 目录。
- `DATA_ROOT`：真实视频数据根目录。
- `METADATA_PATH`：建议使用 `check_dataset.py` 生成的 `metadata_fixed.csv`。
- `OUTPUT_ROOT`：DMD 训练输出目录。
- `HEIGHT/WIDTH/NUM_FRAMES`：训练分辨率和帧数；脚本会自动生成 Turbo latent shape。
- `NUM_GPUS/MAX_ITERS/LOG_ITERS`：卡数、训练步数和保存间隔。

## 四条命令

1. 数据体检：

```bash
DATA_ROOT=/path/to/figurine_dataset
python3 check_dataset.py \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$DATA_ROOT/metadata.csv" \
  --height 480 --width 832 --num_frames 121 | tee "$DATA_ROOT/check_dataset.log"
```

2. merge + gate-0：

```bash
python3 tools/merge_fig360_lora.py \
  --base_model_dir /path/to/local/Wan2.2-TI2V-5B \
  --lora_path /path/to/figurine360_lora.safetensors \
  --output_dir /path/to/local/Wan2.2-TI2V-5B-fig360
```

你已在 H100 验证 merged teacher 与运行时加载 figurine360 LoRA 效果一致。后续若换 LoRA 文件，需要重新跑同 seed 低分辨率 10 步和正式 50 步对齐。

3. 冒烟训练：

```bash
BASE_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B \
FIG360_LORA_PATH=/path/to/figurine360_lora.safetensors \
MERGED_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B-fig360 \
DATA_ROOT=/path/to/figurine_dataset \
METADATA_PATH=/path/to/figurine_dataset/metadata_smoke16.csv \
OUTPUT_ROOT=./models/train/figurine360_dmd_lora_smoke \
NUM_GPUS=2 HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
MAX_ITERS=50 LOG_ITERS=25 \
bash train_figurine360_dmd_lora.sh
```

4. 正式训练：

```bash
BASE_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B \
FIG360_LORA_PATH=/path/to/figurine360_lora.safetensors \
MERGED_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B-fig360 \
DATA_ROOT=/path/to/figurine_dataset \
METADATA_PATH=/path/to/figurine_dataset/metadata_fixed.csv \
OUTPUT_ROOT=./models/train/figurine360_dmd_lora \
NUM_GPUS=4 HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
MAX_ITERS=3000 LOG_ITERS=200 \
bash train_figurine360_dmd_lora.sh
```

断点恢复：重新执行同一命令；trainer 会读取 `OUTPUT_ROOT/checkpoint_model_*/model.pt` 中最新 checkpoint。

## 推理路径

- Turbo runtime：使用 `third_party/wan22_turbo/wan2.2_fewstep.py`，checkpoint 指向训练输出的 safetensors 或后续转换后的 model。
- DiffSynth：加载顺序必须是 base -> figurine360 LoRA -> DMD LoRA。DMD LoRA 是相对 merged teacher 学到的 delta，单独挂裸 base 不成立。
- ComfyUI：同样按 base + figurine360 + `figurine360_dmd_lora_rank64.safetensors` 三件套加载。

训练保存时会在 `OUTPUT_ROOT/figurine360_dmd_lora_rank64.safetensors` 写入最新 generator EMA LoRA；每个 checkpoint 子目录也会保留同名文件。

## 常见坑

- OOM：先降 `NUM_FRAMES`，再降分辨率；LoRA-only 优化器很小，峰值主要来自三模型权重和激活。
- loss 单调降到极低：检查 generator 是否真的 step，尤其是 `dfake_gen_update_ratio=5` 和 wandb/日志里的 generator loss。
- fake loss 持续上升：fake score 追不上 student，可尝试把 ratio 升到 8 或 10。
- 过曝/过饱和：确认推理 cfg=1；若训练视频仍过曝，标准 CFG 等效目标可通过降低 `real_guidance_scale` 重训末段。
- 缺 figurine360 LoRA 推理：输出退化是预期行为，不是 DMD LoRA 损坏。
