#!/usr/bin/env bash
set -euo pipefail

# ============================ 路径变量块（只改这里） ============================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DIT_PATHS_JSON="${DIT_PATHS_JSON:-[\"/path/to/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors\",\"/path/to/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors\",\"/path/to/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors\"]}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-/path/to/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.safetensors}"
VAE_PATH="${VAE_PATH:-/path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.safetensors}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/Wan2.2-TI2V-5B/google/umt5-xxl}"
FIGURINE360_LORA="${FIGURINE360_LORA:-/path/to/figurine360-rank32.safetensors}"
RAW_DATA_ROOT="${RAW_DATA_ROOT:-/path/to/figurine360-data}"
RAW_METADATA="${RAW_METADATA:-${RAW_DATA_ROOT}/metadata.csv}"
SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/outputs/direct_distill_smoke_teacher}"
TEACHER_ROOT="${TEACHER_ROOT:-${REPO_ROOT}/outputs/direct_distill_teacher}"
SMOKE_TRAIN_OUTPUT="${SMOKE_TRAIN_OUTPUT:-${REPO_ROOT}/outputs/direct_distill_smoke_train}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${REPO_ROOT}/outputs/direct_distill_train}"
VALIDATION_OUTPUT="${VALIDATION_OUTPUT:-${REPO_ROOT}/outputs/direct_distill_validation}"
TRAIN_LORA_CHECKPOINT="${TRAIN_LORA_CHECKPOINT:-}"
SMOKE_VALIDATION_LORA_CHECKPOINT="${SMOKE_VALIDATION_LORA_CHECKPOINT:-}"
FORMAL_VALIDATION_LORA_CHECKPOINT="${FORMAL_VALIDATION_LORA_CHECKPOINT:-}"
RESUME_OUTPUT_PATH="${RESUME_OUTPUT_PATH:-}"
PLOT_RUN_OUTPUT="${PLOT_RUN_OUTPUT:-${RESUME_OUTPUT_PATH:-${TRAIN_OUTPUT}}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
# ============================================================================

# 固定算法/运行参数。正式 student 的 4/1/5 不允许通过环境变量改写。
TRAIN_SEEDS="${TRAIN_SEEDS:-2 3 4}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-1}"
SMOKE_SEEDS="${SMOKE_SEEDS:-1}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-overexposed, flicker, incomplete rotation, deformation, identity drift}"
FORMAL_HEIGHT="${FORMAL_HEIGHT:-480}"
FORMAL_WIDTH="${FORMAL_WIDTH:-832}"
FORMAL_NUM_FRAMES="${FORMAL_NUM_FRAMES:-49}"
SMOKE_HEIGHT="${SMOKE_HEIGHT:-256}"
SMOKE_WIDTH="${SMOKE_WIDTH:-448}"
SMOKE_NUM_FRAMES="${SMOKE_NUM_FRAMES:-17}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-4}"
TEACHER_STEPS="${TEACHER_STEPS:-50}"
TEACHER_CFG="${TEACHER_CFG:-5}"
TEACHER_SHIFT="${TEACHER_SHIFT:-5}"
VALIDATION_FRACTION="${VALIDATION_FRACTION:-0.1}"
SMOKE_VALIDATION_FRACTION="${SMOKE_VALIDATION_FRACTION:-0.0}"
DATASET_REPEAT="${DATASET_REPEAT:-1}"
SMOKE_DATASET_REPEAT="${SMOKE_DATASET_REPEAT:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-100}"
VALIDATION_INDEX="${VALIDATION_INDEX:-0}"
DEVICE="${DEVICE:-cuda}"
ENABLE_TENSORBOARD="${ENABLE_TENSORBOARD:-0}"
VAE_TILED="${VAE_TILED:-1}"

DIRECT_DISTILL_DIR="${REPO_ROOT}/examples/wanvideo/model_training/special/direct_distill"
PREPARE_PY="${DIRECT_DISTILL_DIR}/prepare_wan22_ti2v_figurine360.py"
VALIDATE_PY="${DIRECT_DISTILL_DIR}/validate_wan22_ti2v_figurine360.py"
PLOT_PY="${DIRECT_DISTILL_DIR}/plot_direct_distill_metrics.py"
TRAIN_PY="${REPO_ROOT}/examples/wanvideo/model_training/train.py"

MODEL_PATH_ARGS=(
  --model_path "${DIT_PATHS_JSON}"
  --model_path "${TEXT_ENCODER_PATH}"
  --model_path "${VAE_PATH}"
)

die() {
  echo "错误: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "$2 不存在: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "$2 不存在: $1"
}

doctor_runtime() {
  command -v accelerate >/dev/null 2>&1 || die "未找到 accelerate 可执行文件"
  python3 -c '
import json, pathlib, sys
paths = json.loads(sys.argv[1])
if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
    raise SystemExit("DIT_PATHS_JSON 必须是非空 JSON 字符串列表")
missing = [path for path in paths if not pathlib.Path(path).is_file()]
if missing:
    raise SystemExit(f"DiT shard 不存在: {missing[0]}")
' "${DIT_PATHS_JSON}"
  require_file "${TEXT_ENCODER_PATH}" "文本编码器权重"
  require_file "${VAE_PATH}" "VAE 权重"
  require_dir "${TOKENIZER_PATH}" "tokenizer 目录"
  require_file "${FIGURINE360_LORA}" "figurine360 LoRA"
  require_file "${PREPARE_PY}" "teacher 准备脚本"
  require_file "${TRAIN_PY}" "训练脚本"
  require_file "${VALIDATE_PY}" "验证脚本"
  require_file "${PLOT_PY}" "绘图脚本"
  [[ -z "${ACCELERATE_CONFIG}" ]] || require_file "${ACCELERATE_CONFIG}" "Accelerate 配置"
  [[ -z "${TRAIN_LORA_CHECKPOINT}" ]] || require_file "${TRAIN_LORA_CHECKPOINT}" "训练续训 checkpoint"
  [[ -z "${SMOKE_VALIDATION_LORA_CHECKPOINT}" ]] || require_file "${SMOKE_VALIDATION_LORA_CHECKPOINT}" "冒烟验证 checkpoint"
  [[ -z "${FORMAL_VALIDATION_LORA_CHECKPOINT}" ]] || require_file "${FORMAL_VALIDATION_LORA_CHECKPOINT}" "正式验证 checkpoint"
  python3 -c 'import accelerate, diffsynth, imageio, imageio_ffmpeg, matplotlib, pandas, peft, PIL, pytest, safetensors, torch; print("Python 依赖检查通过")'
  [[ "${ENABLE_TENSORBOARD}" != "1" ]] || python3 -c 'import tensorboard'
  echo "本地路径检查通过；脚本不会自动下载模型。"
}

doctor_source() {
  doctor_runtime
  require_dir "${RAW_DATA_ROOT}" "原始数据目录"
  require_file "${RAW_METADATA}" "原始 metadata"
}

prepare_teacher() {
  local output_path="$1"
  local height="$2"
  local width="$3"
  local num_frames="$4"
  local max_samples="${5:-}"
  local determinism="${6:-0}"
  local validation_fraction="${7:-${VALIDATION_FRACTION}}"
  local train_seed_values="${8:-${TRAIN_SEEDS}}"
  local validation_seed_values="${9-${VALIDATION_SEEDS}}"
  doctor_source
  local seed_args=()
  for seed in ${train_seed_values}; do seed_args+=(--seed "${seed}"); done
  if [[ -n "${validation_seed_values}" ]]; then
    for seed in ${validation_seed_values}; do
      seed_args+=(--validation_seed "${seed}")
    done
  fi
  local command=(
    python3 "${PREPARE_PY}"
    --metadata_path "${RAW_METADATA}"
    --base_path "${RAW_DATA_ROOT}"
    --output_path "${output_path}"
    "${MODEL_PATH_ARGS[@]}"
    --tokenizer_path "${TOKENIZER_PATH}"
    --figurine_lora_path "${FIGURINE360_LORA}"
    "${seed_args[@]}"
    --height "${height}"
    --width "${width}"
    --num_frames "${num_frames}"
    --rand_device cpu
    --teacher_num_inference_steps "${TEACHER_STEPS}"
    --teacher_cfg_scale "${TEACHER_CFG}"
    --teacher_sigma_shift "${TEACHER_SHIFT}"
    --negative_prompt "${NEGATIVE_PROMPT}"
    --validation_fraction "${validation_fraction}"
    --device "${DEVICE}"
  )
  [[ "${VAE_TILED}" != "1" ]] || command+=(--tiled)
  [[ "${VAE_TILED}" == "1" ]] || command+=(--no-tiled)
  [[ -z "${max_samples}" ]] || command+=(--max_samples "${max_samples}")
  [[ "${determinism}" != "1" ]] || command+=(--verify_determinism)
  "${command[@]}"
}

model_paths_json() {
  python3 -c '
import json, sys
dit_shards = json.loads(sys.argv[1])
if not isinstance(dit_shards, list) or not dit_shards:
    raise SystemExit("DIT_PATHS_JSON 必须是非空列表")
print(json.dumps([dit_shards, sys.argv[2], sys.argv[3]]))
' "${DIT_PATHS_JSON}" "${TEXT_ENCODER_PATH}" "${VAE_PATH}"
}

accelerate_prefix() {
  if [[ -n "${ACCELERATE_CONFIG}" ]]; then
    require_file "${ACCELERATE_CONFIG}" "Accelerate 配置"
    printf '%s\0' accelerate launch --config_file "${ACCELERATE_CONFIG}"
  else
    printf '%s\0' accelerate launch
  fi
}

train_student() {
  local teacher_root="$1"
  local output_path="$2"
  local height="$3"
  local width="$4"
  local num_frames="$5"
  local epochs="$6"
  local dataset_repeat="${7:-${DATASET_REPEAT}}"
  doctor_runtime
  local effective_output="${output_path}"
  if [[ -n "${TRAIN_LORA_CHECKPOINT}" ]]; then
    require_file "${TRAIN_LORA_CHECKPOINT}" "DirectDistill LoRA checkpoint"
    [[ -n "${RESUME_OUTPUT_PATH}" ]] || die "续训时必须设置新的 RESUME_OUTPUT_PATH，避免覆盖旧 checkpoint/metrics"
    effective_output="${RESUME_OUTPUT_PATH}"
  fi
  if [[ -d "${effective_output}" ]] && find "${effective_output}" -mindepth 1 -print -quit | grep -q .; then
    die "训练输出目录非空，请使用新的路径: ${effective_output}"
  fi
  local metadata="${teacher_root}/metadata_direct_distill_train.csv"
  require_file "${metadata}" "训练 metadata"
  local launch=()
  while IFS= read -r -d '' item; do launch+=("${item}"); done < <(accelerate_prefix)
  local command=(
    "${launch[@]}" "${TRAIN_PY}"
    --dataset_base_path "${teacher_root}"
    --dataset_metadata_path "${metadata}"
    --data_file_keys "input_image,teacher_latent"
    --height "${height}"
    --width "${width}"
    --num_frames "${num_frames}"
    --dataset_repeat "${dataset_repeat}"
    --model_paths "$(model_paths_json)"
    --tokenizer_path "${TOKENIZER_PATH}"
    --learning_rate "${LEARNING_RATE}"
    --num_epochs "${epochs}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --save_steps "${SAVE_STEPS}"
    --remove_prefix_in_ckpt "pipe.dit."
    --output_path "${effective_output}"
    --preset_lora_path "${FIGURINE360_LORA}"
    --preset_lora_model dit
    --lora_base_model dit
    --lora_target_modules "q,k,v,o,ffn.0,ffn.2"
    --lora_rank 32
    --task direct_distill
    --extra_inputs "seed,rand_device,num_inference_steps,cfg_scale,sigma_shift"
    --direct_distill_target_latent_key input_latents
    --direct_distill_preserve_first_frame
    --direct_distill_exclude_first_frame_loss
    --enable_direct_distill_metrics
    --direct_distill_loss_ema_beta 0.98
    --use_gradient_checkpointing
  )
  [[ "${ENABLE_TENSORBOARD}" != "1" ]] || command+=(--enable_tensorboard_log)
  [[ -z "${TRAIN_LORA_CHECKPOINT}" ]] || {
    command+=(--lora_checkpoint "${TRAIN_LORA_CHECKPOINT}")
  }
  "${command[@]}"
}

validate_pair() {
  local teacher_root="$1"
  local metadata_name="$2"
  local output_path="$3"
  local direct_distill_lora="$4"
  doctor_runtime
  [[ -n "${direct_distill_lora}" ]] || die "验证前必须设置对应的 DirectDistill LoRA checkpoint"
  require_file "${direct_distill_lora}" "DirectDistill LoRA checkpoint"
  local metadata="${teacher_root}/${metadata_name}"
  require_file "${metadata}" "验证 metadata"
  local tile_flag="--no-tiled"
  [[ "${VAE_TILED}" != "1" ]] || tile_flag="--tiled"
  python3 "${VALIDATE_PY}" \
    "${MODEL_PATH_ARGS[@]}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --figurine360_lora "${FIGURINE360_LORA}" \
    --direct_distill_lora "${direct_distill_lora}" \
    --metadata_path "${metadata}" \
    --dataset_base_path "${teacher_root}" \
    --sample_index "${VALIDATION_INDEX}" \
    --student_num_inference_steps 4 \
    --student_cfg_scale 1 \
    --student_sigma_shift 5 \
    --device "${DEVICE}" \
    "${tile_flag}" \
    --output_dir "${output_path}/sample-${VALIDATION_INDEX}"
}

usage() {
  cat <<'EOF'
用法: Wan2.2-TI2V-5B-Figurine360.sh COMMAND

COMMAND:
  doctor          检查基础本地路径、已配置可选路径与离线依赖
  prepare-smoke   生成最多4个对象的低分辨率 teacher latent，并确定性复跑
  smoke-train     在 smoke teacher train split 上跑约2-10个 optimizer step
  validate-smoke  对 smoke 总表的指定样本做 teacher/student 对比
  prepare         生成正式 teacher latent 与互斥 train/validation metadata
  train           只读取正式 train split 训练新 DirectDistill LoRA
  validate        只读取正式 validation split 做同seed并排对比
  plot            从正式训练 metrics.jsonl 生成离线图表与摘要
EOF
}

command="${1:-}"
case "${command}" in
  doctor)
    doctor_source
    ;;
  prepare-smoke)
    prepare_teacher "${SMOKE_ROOT}" "${SMOKE_HEIGHT}" "${SMOKE_WIDTH}" "${SMOKE_NUM_FRAMES}" "${SMOKE_MAX_SAMPLES}" 1 "${SMOKE_VALIDATION_FRACTION}" "${SMOKE_SEEDS}" ""
    ;;
  smoke-train)
    train_student "${SMOKE_ROOT}" "${SMOKE_TRAIN_OUTPUT}" "${SMOKE_HEIGHT}" "${SMOKE_WIDTH}" "${SMOKE_NUM_FRAMES}" "${SMOKE_EPOCHS}" "${SMOKE_DATASET_REPEAT}"
    ;;
  validate-smoke)
    validate_pair "${SMOKE_ROOT}" metadata_direct_distill.csv "${VALIDATION_OUTPUT}/smoke" "${SMOKE_VALIDATION_LORA_CHECKPOINT}"
    ;;
  prepare)
    prepare_teacher "${TEACHER_ROOT}" "${FORMAL_HEIGHT}" "${FORMAL_WIDTH}" "${FORMAL_NUM_FRAMES}" "" 0 "${VALIDATION_FRACTION}" "${TRAIN_SEEDS}" "${VALIDATION_SEEDS}"
    ;;
  train)
    train_student "${TEACHER_ROOT}" "${TRAIN_OUTPUT}" "${FORMAL_HEIGHT}" "${FORMAL_WIDTH}" "${FORMAL_NUM_FRAMES}" "${NUM_EPOCHS}" "${DATASET_REPEAT}"
    ;;
  validate)
    validate_pair "${TEACHER_ROOT}" metadata_direct_distill_validation.csv "${VALIDATION_OUTPUT}/formal" "${FORMAL_VALIDATION_LORA_CHECKPOINT}"
    ;;
  plot)
    require_file "${PLOT_RUN_OUTPUT}/metrics.jsonl" "训练指标"
    python3 "${PLOT_PY}" "${PLOT_RUN_OUTPUT}/metrics.jsonl" --output-dir "${PLOT_RUN_OUTPUT}/plots"
    ;;
  *)
    usage
    [[ -z "${command}" || "${command}" == "help" || "${command}" == "--help" || "${command}" == "-h" ]] || exit 1
    ;;
esac
