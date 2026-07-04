# 手办 360 度旋转 LoRA 任务记录

## T1 决策

结论：采用分支 A，框架原生支持两列 metadata。训练代码零改动；手办训练脚本只需让 `--data_file_keys` 包含 `video`，同时保留 `--extra_inputs "input_image"`。

依据：

- `examples/wanvideo/model_training/train.py:67-74`：`WanTrainingModule.parse_extra_inputs()` 处理 `input_image` 时，若 `data.get("input_image")` 不是非空 list，就把 `inputs_shared["input_image"]` 设为 `data["video"][0]`。
- `diffsynth/core/data/unified_dataset.py:101-107`：`UnifiedDataset.__getitem__()` 只处理 `data_file_keys` 中且当前 metadata 行实际存在的 key。两列数据若传 `--data_file_keys "video"`，不会访问不存在的 `input_image` 列。
- `diffsynth/diffusion/parsers.py:4-10`：`--data_file_keys` 默认值是 `image,video`。手办两列训练必须显式传 `video`，避免继承猫咪三列脚本的 `video,input_image`。
- 2026-07-04 用 `modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "wanvideo/Wan2.2-TI2V-5B/metadata.csv"` 只下载官方示例 metadata，文件内容为 `video,prompt` 两列，逗号分隔。

## 数据 schema

- 猫咪动作：`video,prompt,input_image` 三列，`input_image` 指向独立首帧图。
- 手办旋转：`video,prompt` 两列，`input_image` 由训练 forward 自动回退为视频第 0 帧。
- 训练前仍建议运行 `check_dataset.py` 生成 `metadata_fixed.csv`，用于统一分隔符并补 `height,width,bucket`。

## prompt 规范

建议真实数据统一包含同一 trigger 短语，例如：

`手办360度水平旋转展示`

推理时复用同一短语，并在后面补充外观、材质和画面约束，例如：

`手办360度水平旋转展示，单个精致手办，固定相机，白色背景，平滑匀速水平旋转`

## 修改文件清单

- 新增 `docs/calltrace_ti2v_lora.md`：训练、推理、TI2V 条件注入调用链。
- 新增 `make_debug_dataset.py`：生成猫咪三列或手办两列离线调试数据。
- 修改 `check_dataset.py`：自动识别两列/三列 schema，两列时校验视频第 0 帧可解码。
- 新增 `train_figurine360_lora.sh`：手办两列 metadata 的 Stage B 训练入口。
- 新增 `infer_figurine360.py`：手办 LoRA 推理入口，支持 `--dry_run`。
- 新增 `tests/test_figurine.py`：两列数据、首帧回退、三列回归测试。
- 修改 `.gitignore`：忽略 `debug_data_figurine/`。
- 修改 `README.md`、`NOTES.md`：两特性并列说明与上机 checklist。

## Stage B checklist

### 1. 检查本地权重

`MODEL_ROOT` 下必须有：

- `diffusion_pytorch_model*.safetensors`
- `models_t5_umt5-xxl-enc-bf16.pth`
- `Wan2.2_VAE.pth`
- `google/umt5-xxl/`

预期：`train_figurine360_lora.sh` 启动前检查全部通过。若缺失，先从内网已有权重目录补齐；不要让脚本在内网尝试下载。

### 2. 体检真实数据

```bash
DATA_ROOT=/path/to/figurine_dataset

python3 check_dataset.py \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$DATA_ROOT/metadata.csv" \
  --height 480 \
  --width 832 \
  --num_frames 121 | tee "$DATA_ROOT/check_dataset.log"
```

预期：输出 `schema: two_col`、`bad_samples: 0` 或仅有可修复坏样本；输出 `metadata_fixed_path`。若出现 `missing_video`，修 metadata 路径；若出现 `first_frame_decode_error`，重编码对应视频；若 `insufficient_frames` 很多，降低 `NUM_FRAMES` 后重新体检。

### 3. 修正训练尺寸

根据 `check_dataset.py` 输出确认 `HEIGHT/WIDTH/NUM_FRAMES`。横屏默认 `480x832`，若真实数据是竖屏且开启 bucket，脚本中仍填横屏基准，竖屏会自动使用 `832x480`。

预期：`tokens_per_video` 数值可接受。若 OOM，优先降低 `NUM_FRAMES`，其次降低分辨率。

### 4. 准备 16 条冒烟 metadata

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/path/to/figurine_dataset")
df = pd.read_csv(root / "metadata_fixed.csv")
if "bucket" in df:
    smoke = pd.concat([group.head(8) for _, group in df.groupby("bucket", sort=False)])
else:
    smoke = df.head(16)
smoke.head(16).to_csv(root / "metadata_smoke16.csv", index=False)
PY
```

预期：`metadata_smoke16.csv` 存在，最多 16 行。

### 5. 2 卡冒烟 20 step

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/figurine_dataset \
METADATA_PATH=/path/to/figurine_dataset/metadata_smoke16.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_figurine360_lora_smoke \
NUM_GPUS=2 \
HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
SAVE_STEPS=20 \
NUM_EPOCHS=1 DATASET_REPEAT=3 DATASET_NUM_WORKERS=4 \
bash train_figurine360_lora.sh
```

预期：

- `OUTPUT_ROOT/metrics.jsonl` 每个 optimizer step 一行。
- `OUTPUT_ROOT/training_args.json` 存在。
- `OUTPUT_ROOT/step-20.safetensors` 或最终 step ckpt 存在。

回退动作：若 DDP 报 unused parameters，设置 `FIND_UNUSED_PARAMETERS=1` 重新冒烟；若 OOM，降低 `NUM_FRAMES` 或分辨率。

### 6. 4 卡全量训练

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/figurine_dataset \
METADATA_PATH=/path/to/figurine_dataset/metadata_fixed.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_figurine360_lora \
NUM_GPUS=4 \
HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
SAVE_STEPS= \
NUM_EPOCHS=5 DATASET_REPEAT=1 DATASET_NUM_WORKERS=8 \
bash train_figurine360_lora.sh
```

预期：每个 epoch 产生 `epoch-*.safetensors`，`metrics.jsonl` 持续增长。若 GPU 利用率周期性掉底，调大 `DATASET_NUM_WORKERS` 或提前把视频重编码到统一规格。

### 7. 指标绘图

```bash
python3 plot_metrics.py \
  --metrics_path ./models/train/Wan2.2-TI2V-5B_figurine360_lora/metrics.jsonl \
  --output_dir ./models/train/Wan2.2-TI2V-5B_figurine360_lora/plots \
  --warmup_steps 3
```

预期：生成 `loss.png` 和 `throughput.png`，终端摘要中 tokens/s、videos/hour 数值自洽。若 loss 为 NaN，回看最近数据样本和学习率。

### 8. 推理出片

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
LORA_PATH=./models/train/Wan2.2-TI2V-5B_figurine360_lora/epoch-0.safetensors \
IMAGE_PATH=/path/to/figurine_input.jpg \
OUTPUT_PATH=./outputs/figurine360_validate.mp4 \
PROMPT="手办360度水平旋转展示，单个精致手办，固定相机，白色背景，平滑匀速水平旋转" \
HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
python3 infer_figurine360.py
```

预期：输出 mp4，主体保持为输入手办并产生水平旋转运动。若主体漂移，优先统一训练/推理 trigger 短语，并检查训练集中旋转方向和背景是否一致。
