# Wan2.2-TI2V-5B LoRA 内网上机说明

本文件记录通用三列 TI2V LoRA 数据链路。手办 360 度水平旋转两列数据链路见 `NOTES_figurine.md`，训练入口是 `train_figurine360_lora.sh`，推理入口是 `infer_figurine360.py`。

## 两条特性差异

| 特性 | metadata schema | 训练脚本 | LoRA 输出 | 首帧条件图来源 |
|-|-|-|-|-|
| 通用三列 TI2V | `video,prompt,input_image` | `train_ti2v5b_lora.sh` | `./models/train/Wan2.2-TI2V-5B_lora` | 独立图片列 `input_image` |
| 手办 360 度水平旋转 | `video,prompt` | `train_figurine360_lora.sh` | `./models/train/Wan2.2-TI2V-5B_figurine360_lora` | trainer 自动取视频第 0 帧 |

两条链路共用同一套底模权重、数据体检脚本、指标记录和绘图工具；LoRA 产物互相独立，不要混用输出目录。

## 权重目录

`MODEL_ROOT` 下必须有：

- `diffusion_pytorch_model*.safetensors`：Wan2.2-TI2V-5B DiT 权重分片。
- `models_t5_umt5-xxl-enc-bf16.pth`：umT5 文本编码器权重。
- `Wan2.2_VAE.pth`：Wan2.2 VAE 权重。
- `google/umt5-xxl/`：本地 tokenizer 目录。

训练必须使用 `--model_paths` 读本地文件。内网不要用 `--model_id_with_origin_paths`，否则会尝试下载。

## 数据检查

真实 `metadata.csv` 是 tab 分隔；DiffSynth 默认按逗号读 CSV，所以先生成 `metadata_fixed.csv`：

```bash
DATA_ROOT=/path/to/dataset

python3 check_dataset.py \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$DATA_ROOT/metadata.csv" \
  --height 480 \
  --width 832 \
  --num_frames 121 | tee "$DATA_ROOT/check_dataset.log"
```

看输出里的 `resolution_counts`、`bucket_counts`、帧数分布和 `tokens_per_video`。训练时使用 `metadata_fixed.csv`；该文件会额外包含 `height,width,bucket`，横屏样本为 `480,832,landscape`，竖屏样本为 `832,480,portrait`。

## 冒烟训练

先取 16 条数据跑小规模测试。若数据里有竖屏样本，优先从每个 bucket 各取一部分，避免冒烟只覆盖横屏：

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("/path/to/dataset")
df = pd.read_csv(root / "metadata_fixed.csv")
if "bucket" in df:
    smoke = pd.concat([group.head(8) for _, group in df.groupby("bucket", sort=False)])
else:
    smoke = df.head(16)
smoke.head(16).to_csv(root / "metadata_smoke16.csv", index=False)
PY
```

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/dataset \
METADATA_PATH=/path/to/dataset/metadata_smoke16.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_lora_smoke \
NUM_GPUS=2 \
HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
ENABLE_ORIENTATION_BUCKETS=1 \
SAVE_STEPS= \
NUM_EPOCHS=1 DATASET_REPEAT=3 DATASET_NUM_WORKERS=4 \
bash train_ti2v5b_lora.sh
```

预期：

- `OUTPUT_ROOT/metrics.jsonl` 持续写入。
- `OUTPUT_ROOT/training_args.json` 存在。
- `OUTPUT_ROOT/epoch-0.safetensors` 存在。

如果 OOM，先改小分辨率/帧数做冒烟；正式训练再恢复。若 DDP 报 unused parameters，再考虑给训练命令追加 `--find_unused_parameters`。

## 正式训练

```bash
MODEL_ROOT=/path/to/local/wan \
TOKENIZER_PATH=/path/to/local/wan/google/umt5-xxl \
DATA_ROOT=/path/to/dataset \
METADATA_PATH=/path/to/dataset/metadata_fixed.csv \
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_lora \
NUM_GPUS=4 \
HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
ENABLE_ORIENTATION_BUCKETS=1 \
SAVE_STEPS= \
NUM_EPOCHS=5 DATASET_REPEAT=1 DATASET_NUM_WORKERS=8 \
bash train_ti2v5b_lora.sh
```

`HEIGHT/WIDTH` 表示横屏 bucket 的目标尺寸。开启 `ENABLE_ORIENTATION_BUCKETS=1` 后，竖屏样本会按 `832x480` bucket 处理，不再被 center crop 成横屏。`train_ti2v5b_lora.sh` 会导出 `PYTHONPATH=$PWD`，请从仓库根目录执行。

## Checkpoint 保存频率

默认 `SAVE_STEPS=` 为空，此时每个 epoch 保存一次：

- `epoch-0.safetensors`
- `epoch-1.safetensors`

如果要按 step 保存，在启动命令里设置：

```bash
SAVE_STEPS=500 bash train_ti2v5b_lora.sh
```

开启后会保存 `step-500.safetensors`、`step-1000.safetensors` 等；训练结束时如果最后一步不是 `SAVE_STEPS` 的整数倍，还会保存最终的 `step-{最后步数}.safetensors`。设置 `SAVE_STEPS` 后不再保存 `epoch-*.safetensors`。

## 指标绘图

```bash
python3 plot_metrics.py \
  --metrics_path ./models/train/Wan2.2-TI2V-5B_lora/metrics.jsonl \
  --output_dir ./models/train/Wan2.2-TI2V-5B_lora/plots \
  --warmup_steps 3
```

输出：

- `loss.png`：loss 原始曲线和 EMA 曲线。
- `throughput.png`：tokens/s 和 videos/hour。
- 终端摘要：总 step、末 20% loss、稳态 tokens/s、videos/hour。

## 注意事项

- 三列训练脚本显式传 `--data_file_keys "video,input_image"`。
- 手办训练脚本显式传 `--data_file_keys "video"`，但仍传 `--extra_inputs "input_image"`；trainer 会在 metadata 无 `input_image` 列时回退为视频首帧。
- 当前 DataLoader 每个 micro step 实际只取 1 条样本；bucket sampler 控制采样顺序和尺寸路径，不代表 dense tensor batch size > 1。
- 指标里的 `samples_per_step = grad_accum * world_size`；横屏 `480x832` 和竖屏 `832x480` token 数相同。
- 调 `DATASET_NUM_WORKERS` 时看 GPU 利用率：周期性掉底通常是解码瓶颈。
- 600 条数据规模较小，可先跑完冒烟再决定是否启用缓存数据流程。

## 验证出片

训练至少产出一个 `epoch-*.safetensors` 后，用官方验证方式加载 LoRA：

```python
import glob
import torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

MODEL_ROOT = "/path/to/local/wan"
DIT_PATHS = sorted(glob.glob(f"{MODEL_ROOT}/diffusion_pytorch_model*.safetensors"))
if not DIT_PATHS:
    raise FileNotFoundError(f"No DiT safetensors found at {MODEL_ROOT}/diffusion_pytorch_model*.safetensors")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=DIT_PATHS),
        ModelConfig(path=f"{MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(path=f"{MODEL_ROOT}/Wan2.2_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(path=f"{MODEL_ROOT}/google/umt5-xxl"),
)

pipe.load_lora(pipe.dit, "models/train/Wan2.2-TI2V-5B_lora/epoch-0.safetensors", alpha=1)
input_image = Image.open("/path/to/dataset/images_480x832/example.jpg").convert("RGB").resize((832, 480))

video = pipe(
    prompt="目标主体在镜头前自然运动",
    input_image=input_image,
    height=480,
    width=832,
    num_frames=121,
    seed=1,
    tiled=True,
)
save_video(video, "validate_wan22_ti2v5b_lora.mp4", fps=15, quality=5)
```

验证竖屏 LoRA 时，把 `input_image` 换成竖屏条件图，并把 `height=832, width=480`。
