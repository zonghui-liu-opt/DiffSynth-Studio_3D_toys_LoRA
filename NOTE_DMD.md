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
- `VALIDATION_INTERVAL`：Stage B 验证节奏；默认 200，与 `LOG_ITERS` 保持一致，训练保存 checkpoint 后用独立验证脚本跑 holdout。
- `DATALOADER_NUM_WORKERS`：DMD dataloader workers；默认 8。

## 命令

1. 数据体检：

```bash
DATA_ROOT=/path/to/figurine_dataset
python3 check_dataset.py \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$DATA_ROOT/metadata.csv" \
  --height 480 --width 832 --num_frames 121 | tee "$DATA_ROOT/check_dataset.log"
```

2. merge：

```bash
python3 tools/merge_fig360_lora.py \
  --base_model_dir /path/to/local/Wan2.2-TI2V-5B \
  --lora_path /path/to/figurine360_lora.safetensors \
  --output_dir /path/to/local/Wan2.2-TI2V-5B-fig360
```

你已在 H100 验证 merged teacher 与运行时加载 figurine360 LoRA 在同 seed/prompt/首帧下无差别；本仓库后续命令默认跳过 gate-0。后续若换 LoRA 文件，需要重新跑同 seed 低分辨率 10 步和正式 50 步对齐。

3. 5.2 Turbo/DiffSynth TI2V 对齐：

准备一个 CSV，列为 `case_id,prompt,image,seed`，至少 1 条，建议直接复用 12 条 holdout 的前几条。

Turbo merged teacher 50-step：

```bash
REPO_ROOT=/path/to/DiffSynth-Studio_3D_toys_LoRA-main
MERGED_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B-fig360
DMD_RUNTIME_DIR="$REPO_ROOT/third_party/wan22_turbo"
ALIGN_ROOT="$REPO_ROOT/models/validation/stage_b_alignment"
PROMPT="手办360度水平旋转展示"
IMAGE=/path/to/holdout_first_frame.png

mkdir -p "$DMD_RUNTIME_DIR/wan_models"
ln -sfn "$MERGED_MODEL_ROOT" "$DMD_RUNTIME_DIR/wan_models/Wan2.2-TI2V-5B"

PYTHONPATH="$REPO_ROOT:$DMD_RUNTIME_DIR:${PYTHONPATH:-}" \
python3 "$DMD_RUNTIME_DIR/wan2.2_fewstep.py" \
  --config_path "$REPO_ROOT/configs/dmd/figurine360_wan22_teacher50.yaml" \
  --output_path "$ALIGN_ROOT/turbo_teacher50.mp4" \
  --prompt "$PROMPT" --image "$IMAGE" --seed 0 \
  --h 480 --w 832 --num_frames 121 \
  --timing_json "$ALIGN_ROOT/turbo_teacher50_timing.json"
```

DiffSynth merged teacher 50-step：

```bash
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python3 "$REPO_ROOT/tools/diffsynth_wan22_ti2v_infer.py" \
  --model_root "$MERGED_MODEL_ROOT" \
  --output_path "$ALIGN_ROOT/diffsynth_teacher50.mp4" \
  --prompt "$PROMPT" --image "$IMAGE" --seed 0 \
  --height 480 --width 832 --num_frames 121 \
  --num_inference_steps 50 --cfg_scale 1.0 --sigma_shift 5.0 \
  --timing_json "$ALIGN_ROOT/diffsynth_teacher50_timing.json"
```

对齐指标：

```bash
cat > "$ALIGN_ROOT/manifest.json" <<EOF
{
  "version": 1,
  "comparisons": [{
    "case_id": "alignment-001",
    "teacher_video": "$ALIGN_ROOT/turbo_teacher50.mp4",
    "student_video": "$ALIGN_ROOT/diffsynth_teacher50.mp4",
    "teacher_timing_json": "$ALIGN_ROOT/turbo_teacher50_timing.json",
    "student_timing_json": "$ALIGN_ROOT/diffsynth_teacher50_timing.json"
  }]
}
EOF
python3 "$REPO_ROOT/tools/compute_dmd_video_metrics.py" \
  --manifest "$ALIGN_ROOT/manifest.json" \
  --output_json "$ALIGN_ROOT/alignment_metrics.json"
```

4. 冒烟训练：

```bash
BASE_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B \
FIG360_LORA_PATH=/path/to/figurine360_lora.safetensors \
MERGED_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B-fig360 \
DATA_ROOT=/path/to/figurine_dataset \
METADATA_PATH=/path/to/figurine_dataset/metadata_smoke16.csv \
OUTPUT_ROOT=./models/train/figurine360_dmd_lora_smoke \
NUM_GPUS=2 HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
MAX_ITERS=50 LOG_ITERS=25 VALIDATION_INTERVAL=25 \
bash train_figurine360_dmd_lora.sh
```

5. 正式训练：

```bash
BASE_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B \
FIG360_LORA_PATH=/path/to/figurine360_lora.safetensors \
MERGED_MODEL_ROOT=/path/to/local/Wan2.2-TI2V-5B-fig360 \
DATA_ROOT=/path/to/figurine_dataset \
METADATA_PATH=/path/to/figurine_dataset/metadata_fixed.csv \
OUTPUT_ROOT=./models/train/figurine360_dmd_lora \
NUM_GPUS=4 HEIGHT=480 WIDTH=832 NUM_FRAMES=121 \
MAX_ITERS=3000 LOG_ITERS=200 VALIDATION_INTERVAL=200 \
bash train_figurine360_dmd_lora.sh
```

断点恢复：重新执行同一命令；trainer 会读取 `OUTPUT_ROOT/checkpoint_model_*/model.pt` 中最新 checkpoint。

6. 训练后 12 holdout teacher/student 验证：

`holdout12.csv` 需要列 `case_id,prompt,image,seed`。student 使用训练输出的 EMA LoRA；teacher 使用 merged teacher 50-step。默认只写 manifest；加 `--run` 才实际生成视频。

```bash
REPO_ROOT=/path/to/DiffSynth-Studio_3D_toys_LoRA-main
OUTPUT_ROOT="$REPO_ROOT/models/train/figurine360_dmd_lora"
VALID_ROOT="$REPO_ROOT/models/validation/figurine360_dmd_lora"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python3 "$REPO_ROOT/tools/validate_dmd_stage_b.py" \
  --holdout_csv /path/to/holdout12.csv \
  --output_dir "$VALID_ROOT" \
  --runtime_dir "$REPO_ROOT/third_party/wan22_turbo" \
  --teacher_config_path "$REPO_ROOT/configs/dmd/figurine360_wan22_teacher50.yaml" \
  --student_config_path "$OUTPUT_ROOT/runtime_dmd_config.yaml" \
  --dmd_lora_path "$OUTPUT_ROOT/figurine360_dmd_lora_rank64.safetensors" \
  --height 480 --width 832 --num_frames 121 \
  --run
```

7. 计算验证指标：

```bash
python3 "$REPO_ROOT/tools/compute_dmd_video_metrics.py" \
  --manifest "$VALID_ROOT/stage_b_validation_manifest.json" \
  --output_json "$VALID_ROOT/stage_b_metrics.json"
```

## 推理路径

- Turbo runtime：

```bash
PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/wan22_turbo:${PYTHONPATH:-}" \
python3 "$REPO_ROOT/third_party/wan22_turbo/wan2.2_fewstep.py" \
  --config_path "$OUTPUT_ROOT/runtime_dmd_config.yaml" \
  --lora_path "$OUTPUT_ROOT/figurine360_dmd_lora_rank64.safetensors" \
  --output_path "$OUTPUT_ROOT/sample_student4.mp4" \
  --prompt "$PROMPT" --image "$IMAGE" --seed 0 \
  --h 480 --w 832 --num_frames 121 \
  --timing_json "$OUTPUT_ROOT/sample_student4_timing.json"
```

- DiffSynth：推荐直接用 merged teacher 目录作为 `--model_root`，再加载 DMD LoRA；等价于 base -> figurine360 LoRA -> DMD LoRA。DMD LoRA 是相对 merged teacher 学到的 delta，单独挂裸 base 不成立。

```bash
PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
python3 "$REPO_ROOT/tools/diffsynth_wan22_dmd_lora_infer.py" \
  --model_root "$MERGED_MODEL_ROOT" \
  --dmd_lora_path "$OUTPUT_ROOT/figurine360_dmd_lora_rank64.safetensors" \
  --output_path "$OUTPUT_ROOT/sample_diffsynth_student4.mp4" \
  --prompt "$PROMPT" --image "$IMAGE" --seed 0 \
  --height 480 --width 832 --num_frames 121 \
  --num_inference_steps 4 --cfg_scale 1.0 \
  --timing_json "$OUTPUT_ROOT/sample_diffsynth_student4_timing.json"
```

- ComfyUI：同样按 base + figurine360 + `figurine360_dmd_lora_rank64.safetensors` 三件套加载。

训练保存时会在 `OUTPUT_ROOT/figurine360_dmd_lora_rank64.safetensors` 写入最新 generator EMA LoRA；每个 checkpoint 子目录也会保留同名文件。

`third_party/wan22_turbo/wan2.2_fewstep.py` 也可以直接加载 checkpoint：

```bash
python3 "$REPO_ROOT/third_party/wan22_turbo/wan2.2_fewstep.py" \
  --config_path "$OUTPUT_ROOT/runtime_dmd_config.yaml" \
  --checkpoint_path "$OUTPUT_ROOT/checkpoint_model_000200/model.pt" \
  --lora_source ema \
  --output_path "$OUTPUT_ROOT/sample_from_checkpoint.mp4" \
  --prompt "$PROMPT" --image "$IMAGE" --seed 0 \
  --h 480 --w 832 --num_frames 121
```

## 常见坑

- OOM：先降 `NUM_FRAMES`，再降分辨率；LoRA-only 优化器很小，峰值主要来自三模型权重和激活。
- loss 单调降到极低：检查 generator 是否真的 step，尤其是 `dfake_gen_update_ratio=5` 和 wandb/日志里的 generator loss。
- fake loss 持续上升：fake score 追不上 student，可尝试把 ratio 升到 8 或 10。
- 过曝/过饱和：确认推理 cfg=1；若训练视频仍过曝，标准 CFG 等效目标可通过降低 `real_guidance_scale` 重训末段。
- 缺 figurine360 LoRA 推理：输出退化是预期行为，不是 DMD LoRA 损坏。
