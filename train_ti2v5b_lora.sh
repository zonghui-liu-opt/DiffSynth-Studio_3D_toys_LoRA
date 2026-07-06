#!/usr/bin/env bash
set -euo pipefail

# ====== Stage B: only edit this block on the H100 machine ======
MODEL_ROOT=${MODEL_ROOT:-/path/to/local/wan}
TOKENIZER_PATH=${TOKENIZER_PATH:-$MODEL_ROOT/google/umt5-xxl}
DATA_ROOT=${DATA_ROOT:-/path/to/dataset}
METADATA_PATH=${METADATA_PATH:-$DATA_ROOT/metadata_fixed.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-./models/train/Wan2.2-TI2V-5B_lora}
NUM_GPUS=${NUM_GPUS:-4}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-121}
NUM_EPOCHS=${NUM_EPOCHS:-5}
SAVE_STEPS=${SAVE_STEPS:-}
DATASET_REPEAT=${DATASET_REPEAT:-1}
DATASET_NUM_WORKERS=${DATASET_NUM_WORKERS:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
METRICS_PATH=${METRICS_PATH:-$OUTPUT_ROOT/metrics.jsonl}
ENABLE_ORIENTATION_BUCKETS=${ENABLE_ORIENTATION_BUCKETS:-1}
# ===============================================================

DIT_PATHS=("${MODEL_ROOT}"/diffusion_pytorch_model*.safetensors)
if [ ! -e "${DIT_PATHS[0]}" ]; then
  echo "No DiT safetensors found at ${MODEL_ROOT}/diffusion_pytorch_model*.safetensors" >&2
  exit 1
fi
if [ ! -f "${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth" ]; then
  echo "Missing ${MODEL_ROOT}/models_t5_umt5-xxl-enc-bf16.pth" >&2
  exit 1
fi
if [ ! -f "${MODEL_ROOT}/Wan2.2_VAE.pth" ]; then
  echo "Missing ${MODEL_ROOT}/Wan2.2_VAE.pth" >&2
  exit 1
fi
if [ ! -d "$TOKENIZER_PATH" ]; then
  echo "Missing tokenizer directory: ${TOKENIZER_PATH}" >&2
  exit 1
fi
if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing metadata file: ${METADATA_PATH}. Run check_dataset.py first to create metadata_fixed.csv." >&2
  exit 1
fi

MODEL_PATHS_JSON=$(python3 - "$MODEL_ROOT" "${DIT_PATHS[@]}" <<'PY'
import json
import sys

model_root = sys.argv[1]
dit_paths = sys.argv[2:]
print(json.dumps([
    dit_paths,
    f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
    f"{model_root}/Wan2.2_VAE.pth",
]))
PY
)

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

BUCKET_ARGS=()
if [ "$ENABLE_ORIENTATION_BUCKETS" = "1" ]; then
  BUCKET_ARGS+=(--enable_orientation_buckets)
fi

SAVE_ARGS=()
if [ -n "$SAVE_STEPS" ]; then
  SAVE_ARGS+=(--save_steps "$SAVE_STEPS")
fi

accelerate launch --num_processes "$NUM_GPUS" --mixed_precision bf16 \
  examples/wanvideo/model_training/train.py \
  --dataset_base_path "$DATA_ROOT" \
  --dataset_metadata_path "$METADATA_PATH" \
  --data_file_keys "video,input_image" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  "${BUCKET_ARGS[@]}" \
  --dataset_repeat "$DATASET_REPEAT" \
  --dataset_num_workers "$DATASET_NUM_WORKERS" \
  --model_paths "$MODEL_PATHS_JSON" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --learning_rate 1e-4 \
  --num_epochs "$NUM_EPOCHS" \
  "${SAVE_ARGS[@]}" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "$OUTPUT_ROOT" \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image" \
  --metrics_path "$METRICS_PATH"
