#!/usr/bin/env bash
set -euo pipefail

# ============================ 路径变量块（只改这里） ============================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DIT_PATHS_JSON="${DIT_PATHS_JSON:-[\"/path/to/Wan2.2-TI2V-5B/model-00001.safetensors\",\"/path/to/Wan2.2-TI2V-5B/model-00002.safetensors\",\"/path/to/Wan2.2-TI2V-5B/model-00003.safetensors\"]}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-/path/to/models_t5_umt5-xxl-enc-bf16.pth}"
VAE_PATH="${VAE_PATH:-/path/to/Wan2.2_VAE.pth}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/google/umt5-xxl}"
FIGURINE360_LORA="${FIGURINE360_LORA:-/path/to/figurine360.safetensors}"
DENSE_WARMSTART_LORA="${DENSE_WARMSTART_LORA:-/path/to/dense-direct-distill.safetensors}"
SMOKE_TEACHER_ROOT="${SMOKE_TEACHER_ROOT:-${REPO_ROOT}/outputs/direct_distill_smoke_teacher}"
TEACHER_ROOT="${TEACHER_ROOT:-${REPO_ROOT}/outputs/direct_distill_teacher}"
BSA_SMOKE_OUTPUT="${BSA_SMOKE_OUTPUT:-${REPO_ROOT}/outputs/bsa_direct_distill_smoke}"
BSA_TRAIN_OUTPUT="${BSA_TRAIN_OUTPUT:-${REPO_ROOT}/outputs/bsa_direct_distill_train}"
BSA_VALIDATION_OUTPUT="${BSA_VALIDATION_OUTPUT:-${REPO_ROOT}/outputs/bsa_direct_distill_validation}"
BSA_CHECKPOINT="${BSA_CHECKPOINT:-}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
# ============================================================================

BSA_BLOCK_SIZE="${BSA_BLOCK_SIZE:-4,3,6}"
BSA_TARGET_SPARSITY="${BSA_TARGET_SPARSITY:-0.8}"
BSA_BACKEND="${BSA_BACKEND:-sdpa_gather}"
BSA_QUERY_BLOCK_CHUNK="${BSA_QUERY_BLOCK_CHUNK:-4}"
BSA_QUERY_BLOCK_CHUNK_CANDIDATES="${BSA_QUERY_BLOCK_CHUNK_CANDIDATES:-4,8,16,32}"
BSA_MASK_MODE="${BSA_MASK_MODE:-additive}"
BSA_BOUNDARY_MODE="${BSA_BOUNDARY_MODE:-fixed_padded}"
BSA_COUNT_BIAS="${BSA_COUNT_BIAS:-0}"
BSA_GATE_LR="${BSA_GATE_LR:-2e-5}"
DIRECT_DISTILL_LR="${DIRECT_DISTILL_LR:-2e-6}"
DENSE_ANCHOR_WEIGHT="${DENSE_ANCHOR_WEIGHT:-0.0}"
DENSE_ANCHOR_INTERVAL="${DENSE_ANCHOR_INTERVAL:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
# 4卡、单seed=1、约721条源数据时约产生3.9k-4.3k个真实optimizer steps。
NUM_EPOCHS="${NUM_EPOCHS:-24}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
FORMAL_HEIGHT="${FORMAL_HEIGHT:-832}"
FORMAL_WIDTH="${FORMAL_WIDTH:-480}"
FORMAL_NUM_FRAMES="${FORMAL_NUM_FRAMES:-81}"
SMOKE_HEIGHT="${SMOKE_HEIGHT:-256}"
SMOKE_WIDTH="${SMOKE_WIDTH:-448}"
SMOKE_NUM_FRAMES="${SMOKE_NUM_FRAMES:-17}"
VALIDATION_INDEX="${VALIDATION_INDEX:-0}"
DEVICE="${DEVICE:-cuda}"
# BSA训练从teacher metadata读取seed；训练与验证入口会强制检查全部为1。
SEED=1

DD_DIR="${REPO_ROOT}/examples/wanvideo/model_training/special/direct_distill"
TRAIN_PY="${REPO_ROOT}/examples/wanvideo/model_training/train.py"
VALIDATE_PY="${DD_DIR}/validate_wan22_ti2v_figurine360_bsa.py"
BENCHMARK_PY="${DD_DIR}/benchmark_wan_bsa.py"
PLOT_PY="${DD_DIR}/plot_direct_distill_metrics.py"

die() { echo "错误: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "$2不存在: $1"; }
require_dir() { [[ -d "$1" ]] || die "$2不存在: $1"; }

require_seed_one_metadata() {
  local metadata="$1"
  python3 - "${metadata}" <<'PY'
import csv
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
with path.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    if "seed" not in (reader.fieldnames or []):
        raise SystemExit(f"metadata缺少seed列: {path}")
    rows = list(reader)

if not rows:
    raise SystemExit(f"metadata没有数据行: {path}")
invalid = sorted({(row.get("seed") or "").strip() for row in rows if (row.get("seed") or "").strip() != "1"})
if invalid:
    raise SystemExit(f"BSA训练/验证只允许seed=1，发现: {invalid[:8]}")
print(f"seed=1 metadata检查通过: {len(rows)}条")
PY
}

model_paths_json() {
  python3 -c '
import json, sys
shards = json.loads(sys.argv[1])
if not isinstance(shards, list) or not shards: raise SystemExit("DIT_PATHS_JSON必须是非空JSON列表")
print(json.dumps([shards, sys.argv[2], sys.argv[3]]))
' "${DIT_PATHS_JSON}" "${TEXT_ENCODER_PATH}" "${VAE_PATH}"
}

model_path_args() {
  printf '%s\0' --model_path "${DIT_PATHS_JSON}" --model_path "${TEXT_ENCODER_PATH}" --model_path "${VAE_PATH}"
}

accelerate_prefix() {
  if [[ -n "${ACCELERATE_CONFIG}" ]]; then
    printf '%s\0' accelerate launch --config_file "${ACCELERATE_CONFIG}"
  else
    printf '%s\0' accelerate launch
  fi
}

doctor() {
  command -v accelerate >/dev/null || die "未找到accelerate"
  python3 -c 'import json,pathlib,sys; p=json.loads(sys.argv[1]); assert p and all(pathlib.Path(x).is_file() for x in p)' "${DIT_PATHS_JSON}"
  require_file "${TEXT_ENCODER_PATH}" "文本编码器"
  require_file "${VAE_PATH}" "VAE"
  require_dir "${TOKENIZER_PATH}" "tokenizer"
  require_file "${FIGURINE360_LORA}" "figurine360 LoRA"
  require_file "${DENSE_WARMSTART_LORA}" "dense DirectDistill warm-start"
  require_file "${TRAIN_PY}" "训练脚本"
  require_file "${VALIDATE_PY}" "BSA验证脚本"
  require_file "${BENCHMARK_PY}" "BSA benchmark脚本"
  [[ -z "${ACCELERATE_CONFIG}" ]] || require_file "${ACCELERATE_CONFIG}" "Accelerate配置"
  python3 -c 'import accelerate,diffsynth,imageio,imageio_ffmpeg,matplotlib,peft,PIL,safetensors,torch; print("依赖检查通过")'
  echo "本地路径与依赖检查通过；不会下载模型。"
}

train_bsa() {
  local teacher_root="$1" output="$2" height="$3" width="$4" frames="$5" epochs="$6"
  local expected_grid="${7:-}"
  doctor
  local metadata="${teacher_root}/metadata_direct_distill_train.csv"
  require_file "${metadata}" "DirectDistill训练metadata"
  require_seed_one_metadata "${metadata}"
  [[ ! -e "${output}" ]] || die "输出路径已存在，请使用新目录: ${output}"
  local launch=()
  while IFS= read -r -d '' item; do launch+=("${item}"); done < <(accelerate_prefix)
  local command=(
    "${launch[@]}" "${TRAIN_PY}"
    --dataset_base_path "${teacher_root}"
    --dataset_metadata_path "${metadata}"
    --data_file_keys input_image,teacher_latent
    --height "${height}" --width "${width}" --num_frames "${frames}"
    --dataset_repeat "${DATASET_REPEAT}"
    --model_paths "$(model_paths_json)" --tokenizer_path "${TOKENIZER_PATH}"
    --learning_rate "${DIRECT_DISTILL_LR}" --bsa_gate_learning_rate "${BSA_GATE_LR}"
    --num_epochs "${epochs}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --save_steps "${SAVE_STEPS}" --max_grad_norm 1.0
    --remove_prefix_in_ckpt pipe.dit. --output_path "${output}"
    --preset_lora_path "${FIGURINE360_LORA}" --preset_lora_model dit
    --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 --lora_rank 32
    --task direct_distill --extra_inputs seed,rand_device,num_inference_steps,cfg_scale,sigma_shift
    --direct_distill_target_latent_key input_latents
    --direct_distill_preserve_first_frame --direct_distill_exclude_first_frame_loss
    --enable_direct_distill_metrics --use_gradient_checkpointing
    --enable_bsa --bsa_block_size "${BSA_BLOCK_SIZE}"
    --bsa_target_sparsity "${BSA_TARGET_SPARSITY}" --bsa_backend "${BSA_BACKEND}"
    --bsa_query_block_chunk "${BSA_QUERY_BLOCK_CHUNK}" --bsa_mask_mode "${BSA_MASK_MODE}"
    --bsa_boundary_mode "${BSA_BOUNDARY_MODE}" --bsa_gate_granularity block
    --bsa_gate_rank 32 --bsa_gate_alpha 32 --bsa_trainable_dtype fp32
    --bsa_dense_anchor_weight "${DENSE_ANCHOR_WEIGHT}" --bsa_dense_anchor_interval "${DENSE_ANCHOR_INTERVAL}"
  )
  [[ -z "${expected_grid}" ]] || command+=(--bsa_expected_runtime_grid "${expected_grid}")
  [[ "${BSA_COUNT_BIAS}" != 1 ]] || command+=(--bsa_ragged_count_bias)
  if [[ -n "${BSA_CHECKPOINT}" ]]; then
    require_file "${BSA_CHECKPOINT}/checkpoint_complete" "组合checkpoint完成标记"
    command+=(--resume_bsa_checkpoint "${BSA_CHECKPOINT}")
  else
    command+=(--direct_distill_warmstart_lora "${DENSE_WARMSTART_LORA}")
  fi
  "${command[@]}"
}

validate_bsa() {
  doctor
  [[ -n "${BSA_CHECKPOINT}" ]] || die "验证前设置BSA_CHECKPOINT"
  require_file "${BSA_CHECKPOINT}/checkpoint_complete" "组合checkpoint完成标记"
  local metadata="${TEACHER_ROOT}/metadata_direct_distill_validation.csv"
  require_file "${metadata}" "验证metadata"
  require_seed_one_metadata "${metadata}"
  local model_args=()
  while IFS= read -r -d '' item; do model_args+=("${item}"); done < <(model_path_args)
  python3 "${VALIDATE_PY}" "${model_args[@]}" \
    --tokenizer_path "${TOKENIZER_PATH}" --figurine360_lora "${FIGURINE360_LORA}" \
    --dense_warmstart_lora "${DENSE_WARMSTART_LORA}" --bsa_checkpoint "${BSA_CHECKPOINT}" \
    --metadata_path "${metadata}" \
    --dataset_base_path "${TEACHER_ROOT}" --sample_index "${VALIDATION_INDEX}" \
    --seed "${SEED}" \
    --device "${DEVICE}" --output_dir "${BSA_VALIDATION_OUTPUT}/sample-${VALIDATION_INDEX}"
}

benchmark_bsa() {
  python3 "${BENCHMARK_PY}" --device "${DEVICE}" --chunks "${BSA_QUERY_BLOCK_CHUNK_CANDIDATES}" \
    --backend "${BSA_BACKEND}" --mask-mode "${BSA_MASK_MODE}" --sparsity "${BSA_TARGET_SPARSITY}" \
    --boundary-mode "${BSA_BOUNDARY_MODE}" \
    --output "${BSA_TRAIN_OUTPUT}-h100-single-layer-prefilter.json"
}

usage() {
  cat <<'EOF'
用法: Wan2.2-TI2V-5B-Figurine360-BSA.sh COMMAND

  doctor           检查离线路径与依赖
  bsa-smoke-train  使用已有smoke teacher latent做短训练
  bsa-train        使用正式teacher latent做联合训练
  bsa-validate     输出teacher/A dense warm-start/B joint-dense/C joint-sparse
  bsa-benchmark    独立子进程单层预筛；不能替代完整4-step optimizer benchmark
  plot             绘制BSA训练metrics.jsonl
EOF
}

case "${1:-}" in
  doctor) doctor ;;
  bsa-smoke-train) train_bsa "${SMOKE_TEACHER_ROOT}" "${BSA_SMOKE_OUTPUT}" "${SMOKE_HEIGHT}" "${SMOKE_WIDTH}" "${SMOKE_NUM_FRAMES}" "${SMOKE_EPOCHS}" ;;
  bsa-train) train_bsa "${TEACHER_ROOT}" "${BSA_TRAIN_OUTPUT}" "${FORMAL_HEIGHT}" "${FORMAL_WIDTH}" "${FORMAL_NUM_FRAMES}" "${NUM_EPOCHS}" 41,15,26 ;;
  bsa-validate) validate_bsa ;;
  bsa-benchmark) benchmark_bsa ;;
  plot) require_file "${BSA_TRAIN_OUTPUT}/metrics.jsonl" "BSA指标"; python3 "${PLOT_PY}" "${BSA_TRAIN_OUTPUT}/metrics.jsonl" --output-dir "${BSA_TRAIN_OUTPUT}/plots" ;;
  *) usage; [[ -z "${1:-}" || "${1:-}" =~ ^(-h|--help|help)$ ]] || exit 1 ;;
esac
