#!/usr/bin/env bash
set -euo pipefail

# 对 input_image,prompt 测试集逐条执行同 seed 的 teacher/student 对比。
# input_image 必须是首帧图片的绝对路径；模型与 LoRA 均只从本地读取。

# ============================ 路径变量块（只改这里） ============================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DIT_PATHS_JSON="${DIT_PATHS_JSON:-[\"/path/to/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors\",\"/path/to/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors\",\"/path/to/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors\"]}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-/path/to/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.safetensors}"
VAE_PATH="${VAE_PATH:-/path/to/Wan2.2-TI2V-5B/Wan2.2_VAE.safetensors}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/path/to/Wan2.2-TI2V-5B/google/umt5-xxl}"
FIGURINE360_LORA="${FIGURINE360_LORA:-/path/to/figurine360-rank32.safetensors}"
DIRECT_DISTILL_LORA="${DIRECT_DISTILL_LORA:-/path/to/step-200.safetensors}"
TEST_METADATA="${TEST_METADATA:-/path/to/test/metadata.csv}"
TEST_OUTPUT="${TEST_OUTPUT:-${REPO_ROOT}/outputs/direct_distill_test}"
# ============================================================================

# 测试集只有 input_image,prompt 时，下列值会作为所有样本的共同推理参数。
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-832}"
NUM_FRAMES="${NUM_FRAMES:-81}"
SEED="${SEED:-1}"
RAND_DEVICE="${RAND_DEVICE:-cpu}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-overexposed, flicker, incomplete rotation, deformation, identity drift}"
TEACHER_STEPS="${TEACHER_STEPS:-50}"
TEACHER_CFG="${TEACHER_CFG:-5}"
TEACHER_SHIFT="${TEACHER_SHIFT:-5}"
FPS="${FPS:-15}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
VAE_TILED="${VAE_TILED:-1}"

# one 使用 SAMPLE_INDEX；all 使用 [START_INDEX, END_INDEX)，END_INDEX 留空表示表尾。
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

VALIDATE_PY="${REPO_ROOT}/examples/wanvideo/model_training/special/direct_distill/validate_wan22_ti2v_figurine360.py"
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

# 校验必需表头、非空字段、绝对首帧路径，并输出真实 CSV 数据行数。
# 使用 Python csv，而不是 wc -l，避免 prompt 中的逗号或换行导致计数错误。
metadata_count() {
  python3 - "${TEST_METADATA}" <<'PY'
import csv
import pathlib
import sys

metadata = pathlib.Path(sys.argv[1]).expanduser().resolve()
if not metadata.is_file():
    raise SystemExit(f"测试 metadata 不存在: {metadata}")

with metadata.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    fields = set(reader.fieldnames or [])
    missing = {"input_image", "prompt"} - fields
    if missing:
        raise SystemExit(f"metadata 缺少表头: {', '.join(sorted(missing))}")
    rows = list(reader)

if not rows:
    raise SystemExit(f"metadata 没有数据行: {metadata}")

for index, row in enumerate(rows):
    raw_image = (row.get("input_image") or "").strip()
    prompt = (row.get("prompt") or "").strip()
    if not raw_image:
        raise SystemExit(f"metadata 第 {index + 2} 行 input_image 为空")
    image = pathlib.Path(raw_image).expanduser()
    if not image.is_absolute():
        raise SystemExit(f"metadata 第 {index + 2} 行 input_image 不是绝对路径: {raw_image}")
    if not image.is_file():
        raise SystemExit(f"metadata 第 {index + 2} 行首帧不存在: {image}")
    if not prompt:
        raise SystemExit(f"metadata 第 {index + 2} 行 prompt 为空")

print(len(rows))
PY
}

doctor() {
  command -v python3 >/dev/null 2>&1 || die "未找到 python3"
  python3 - "${DIT_PATHS_JSON}" <<'PY'
import json
import pathlib
import sys

paths = json.loads(sys.argv[1])
if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
    raise SystemExit("DIT_PATHS_JSON 必须是非空 JSON 字符串列表")
missing = [path for path in paths if not pathlib.Path(path).is_file()]
if missing:
    raise SystemExit(f"DiT shard 不存在: {missing[0]}")
PY
  require_file "${TEXT_ENCODER_PATH}" "文本编码器权重"
  require_file "${VAE_PATH}" "VAE 权重"
  require_dir "${TOKENIZER_PATH}" "tokenizer 目录"
  require_file "${FIGURINE360_LORA}" "figurine360 LoRA"
  require_file "${DIRECT_DISTILL_LORA}" "DirectDistill LoRA"
  require_file "${VALIDATE_PY}" "验证脚本"
  [[ "$(realpath "${FIGURINE360_LORA}")" != "$(realpath "${DIRECT_DISTILL_LORA}")" ]] \
    || die "FIGURINE360_LORA 与 DIRECT_DISTILL_LORA 不能是同一文件"
  python3 -c 'import imageio, imageio_ffmpeg, PIL, safetensors, torch; print("Python 依赖检查通过")'
  local count
  count="$(metadata_count)"
  echo "测试集检查通过: ${count} 条样本；脚本不会下载任何模型。"
}

require_nonnegative_integer() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 必须是非负整数，收到: $2"
}

run_sample() {
  local index="$1"
  local sample_dir="${TEST_OUTPUT}/sample-${index}"
  local tile_flag="--no-tiled"
  [[ "${VAE_TILED}" != "1" ]] || tile_flag="--tiled"

  echo "开始测试样本 ${index}，输出: ${sample_dir}"
  python3 "${VALIDATE_PY}" \
    "${MODEL_PATH_ARGS[@]}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --figurine360_lora "${FIGURINE360_LORA}" \
    --direct_distill_lora "${DIRECT_DISTILL_LORA}" \
    --metadata_path "${TEST_METADATA}" \
    --dataset_base_path "$(dirname "${TEST_METADATA}")" \
    --sample_index "${index}" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --seed "${SEED}" \
    --rand_device "${RAND_DEVICE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --teacher_num_inference_steps "${TEACHER_STEPS}" \
    --teacher_cfg_scale "${TEACHER_CFG}" \
    --teacher_sigma_shift "${TEACHER_SHIFT}" \
    --student_num_inference_steps 4 \
    --student_cfg_scale 1 \
    --student_sigma_shift 5 \
    --fps "${FPS}" \
    --device "${DEVICE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    "${tile_flag}" \
    --output_dir "${sample_dir}"
}

run_one() {
  doctor
  require_nonnegative_integer "SAMPLE_INDEX" "${SAMPLE_INDEX}"
  local count
  count="$(metadata_count)"
  (( SAMPLE_INDEX < count )) || die "SAMPLE_INDEX=${SAMPLE_INDEX} 越界，测试集共有 ${count} 条"
  run_sample "${SAMPLE_INDEX}"
}

run_all() {
  doctor
  require_nonnegative_integer "START_INDEX" "${START_INDEX}"
  local count end index
  count="$(metadata_count)"
  end="${END_INDEX:-${count}}"
  require_nonnegative_integer "END_INDEX" "${end}"
  (( START_INDEX < end )) || die "需要满足 START_INDEX < END_INDEX"
  (( end <= count )) || die "END_INDEX=${end} 越界，测试集共有 ${count} 条"

  echo "将串行运行 [${START_INDEX}, ${end})；每条样本都会重新加载 5B 模型。"
  for ((index = START_INDEX; index < end; index++)); do
    if [[ "${SKIP_EXISTING}" == "1" && -f "${TEST_OUTPUT}/sample-${index}/validation.json" ]]; then
      echo "跳过已完成样本 ${index}"
      continue
    fi
    run_sample "${index}"
  done
}

usage() {
  cat <<'EOF'
用法: Wan2.2-TI2V-5B-Figurine360-Test.sh COMMAND

COMMAND:
  doctor  检查本地模型、两份 LoRA、依赖和 input_image,prompt 测试集
  one     推理 SAMPLE_INDEX 指定的一条，并生成 teacher/student 对比
  all     串行推理 [START_INDEX, END_INDEX)，默认从第 0 条到表尾

必要路径变量:
  DIT_PATHS_JSON TEXT_ENCODER_PATH VAE_PATH TOKENIZER_PATH
  FIGURINE360_LORA DIRECT_DISTILL_LORA TEST_METADATA TEST_OUTPUT

常用示例:
  SAMPLE_INDEX=0 bash Wan2.2-TI2V-5B-Figurine360-Test.sh one
  START_INDEX=0 END_INDEX=10 bash Wan2.2-TI2V-5B-Figurine360-Test.sh all

测试集只有 input_image,prompt 两列时，默认参数为 480x832@81、seed=1；
teacher=50 steps/CFG 5/shift 5，student 固定为 4 steps/CFG 1/shift 5。
EOF
}

command="${1:-}"
case "${command}" in
  doctor)
    doctor
    ;;
  one)
    run_one
    ;;
  all)
    run_all
    ;;
  *)
    usage
    [[ -z "${command}" || "${command}" == "help" || "${command}" == "--help" || "${command}" == "-h" ]] || exit 1
    ;;
esac
