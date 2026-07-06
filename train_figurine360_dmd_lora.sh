#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ====== Stage B: only edit this block on the H100 machine ======
BASE_MODEL_ROOT=${BASE_MODEL_ROOT:-/path/to/local/Wan2.2-TI2V-5B}
FIG360_LORA_PATH=${FIG360_LORA_PATH:-/path/to/figurine360_lora.safetensors}
MERGED_MODEL_ROOT=${MERGED_MODEL_ROOT:-/path/to/local/Wan2.2-TI2V-5B-fig360}
DATA_ROOT=${DATA_ROOT:-/path/to/figurine_dataset}
METADATA_PATH=${METADATA_PATH:-$DATA_ROOT/metadata_fixed.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/models/train/figurine360_dmd_lora}
DMD_RUNTIME_DIR=${DMD_RUNTIME_DIR:-$REPO_ROOT/third_party/wan22_turbo}
CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/configs/dmd/figurine360_wan22_dmd_lora.yaml}
HEIGHT=${HEIGHT:-480}
WIDTH=${WIDTH:-832}
NUM_FRAMES=${NUM_FRAMES:-121}
NUM_GPUS=${NUM_GPUS:-4}
NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
MAX_ITERS=${MAX_ITERS:-3000}
LOG_ITERS=${LOG_ITERS:-200}
VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-200}
BATCH_SIZE=${BATCH_SIZE:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
LR=${LR:-5e-5}
LR_CRITIC=${LR_CRITIC:-5e-5}
EMA_WEIGHT=${EMA_WEIGHT:-0.995}
DISABLE_WANDB=${DISABLE_WANDB:-1}
MERGE_IF_MISSING=${MERGE_IF_MISSING:-1}
EXPECTED_LAYERS_PER_TARGET=${EXPECTED_LAYERS_PER_TARGET:-}
# ===============================================================

if [ ! -d "$DMD_RUNTIME_DIR" ]; then
  echo "Missing vendored DMD runtime: $DMD_RUNTIME_DIR" >&2
  exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Missing DMD config: $CONFIG_PATH" >&2
  exit 1
fi
if [ ! -f "$METADATA_PATH" ]; then
  echo "Missing metadata: $METADATA_PATH. Run check_dataset.py first." >&2
  exit 1
fi

if [ ! -d "$MERGED_MODEL_ROOT" ]; then
  if [ "$MERGE_IF_MISSING" != "1" ]; then
    echo "Missing merged teacher model: $MERGED_MODEL_ROOT" >&2
    exit 1
  fi
  MERGE_EXPECT_ARGS=()
  if [ -n "$EXPECTED_LAYERS_PER_TARGET" ]; then
    MERGE_EXPECT_ARGS+=(--expected_layers_per_target "$EXPECTED_LAYERS_PER_TARGET")
  fi
  python3 "$REPO_ROOT/tools/merge_fig360_lora.py" \
    --base_model_dir "$BASE_MODEL_ROOT" \
    --lora_path "$FIG360_LORA_PATH" \
    --output_dir "$MERGED_MODEL_ROOT" \
    "${MERGE_EXPECT_ARGS[@]}"
fi

mkdir -p "$OUTPUT_ROOT"
DMD_DATA_CSV=${DMD_DATA_CSV:-$OUTPUT_ROOT/dmd_dataset.csv}
RUNTIME_CONFIG_PATH=${RUNTIME_CONFIG_PATH:-$OUTPUT_ROOT/runtime_dmd_config.yaml}

export PYTHONPATH="$REPO_ROOT:$DMD_RUNTIME_DIR:${PYTHONPATH:-}"

python3 "$REPO_ROOT/tools/prepare_dmd_dataset_csv.py" \
  --dataset_root "$DATA_ROOT" \
  --metadata_path "$METADATA_PATH" \
  --output_path "$DMD_DATA_CSV" \
  --default_num_frames "$NUM_FRAMES"

python3 -m dmd.wan22_config \
  --template_path "$CONFIG_PATH" \
  --output_path "$RUNTIME_CONFIG_PATH" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --max_iters "$MAX_ITERS" \
  --log_iters "$LOG_ITERS" \
  --validation_interval "$VALIDATION_INTERVAL" \
  --batch_size "$BATCH_SIZE" \
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
  --lr "$LR" \
  --lr_critic "$LR_CRITIC" \
  --ema_weight "$EMA_WEIGHT"

WAN_MODELS_DIR="$DMD_RUNTIME_DIR/wan_models"
WAN_MODEL_LINK="$DMD_RUNTIME_DIR/wan_models/Wan2.2-TI2V-5B"
mkdir -p "$WAN_MODELS_DIR"
if [ -L "$WAN_MODEL_LINK" ]; then
  rm "$WAN_MODEL_LINK"
elif [ -e "$WAN_MODEL_LINK" ]; then
  echo "$WAN_MODEL_LINK exists and is not a symlink; move it before launching DMD." >&2
  exit 1
fi
ln -s "$MERGED_MODEL_ROOT" "$WAN_MODEL_LINK"

WANDB_ARGS=()
if [ "$DISABLE_WANDB" = "1" ]; then
  WANDB_ARGS+=(--disable-wandb)
fi

pushd "$DMD_RUNTIME_DIR" >/dev/null
torchrun --nproc_per_node="$NUM_GPUS" \
  --nnodes="$NODE_COUNT" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config_path "$RUNTIME_CONFIG_PATH" \
  --logdir "$OUTPUT_ROOT" \
  --data_path "$DMD_DATA_CSV" \
  --no_visualize \
  "${WANDB_ARGS[@]}"
popd >/dev/null
