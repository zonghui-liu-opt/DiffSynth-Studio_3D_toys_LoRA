#!/usr/bin/env bash
set -euo pipefail

# 测试集输入只需 input_image,prompt；不读取训练用 teacher_latent。
# 公共路径、metadata 门禁和基础参数直接复用同目录 DirectDistill Test 入口。

requested_command="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIRECT_TEST_SH="${SCRIPT_DIR}/Wan2.2-TI2V-5B-Figurine360-Test.sh"

REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../../.." && pwd)}"
DENSE_WARMSTART_LORA="${DENSE_WARMSTART_LORA:-/path/to/direct-distill-step-19600.safetensors}"
BSA_CHECKPOINT="${BSA_CHECKPOINT:-/path/to/checkpoint-step-0000200}"
TEST_OUTPUT="${TEST_OUTPUT:-${REPO_ROOT}/outputs/bsa_direct_distill_test}"

# DirectDistill Test 的 doctor 将这一路径作为第二份 LoRA 校验；BSA 入口中它
# 就是未经过 BSA 训练的 dense warm-start。
DIRECT_DISTILL_LORA="${DENSE_WARMSTART_LORA}"

# 该脚本当前是可执行入口而非 shell library；以 help 加载函数且静默其 usage。
set -- help
# shellcheck source=Wan2.2-TI2V-5B-Figurine360-Test.sh
source "${DIRECT_TEST_SH}" >/dev/null
set -- "${requested_command}"

VALIDATE_PY="${SCRIPT_DIR}/validate_wan22_ti2v_figurine360_bsa.py"

doctor_bsa() {
  doctor
  require_dir "${BSA_CHECKPOINT}" "BSA checkpoint 目录"
  require_file "${BSA_CHECKPOINT}/checkpoint_complete" "BSA checkpoint 完成标记"
  require_file "${BSA_CHECKPOINT}/direct_distill_lora.safetensors" "BSA 联合训练 DirectDistill LoRA"
  require_file "${BSA_CHECKPOINT}/bsa_adapter.safetensors" "BSA gate adapter"
  require_file "${BSA_CHECKPOINT}/bsa_config.json" "BSA checkpoint manifest"
  echo "BSA 测试入口检查通过；测试集不需要 teacher_latent，也不会下载模型。"
}

run_bsa_validator() {
  local tile_flag="--no-tiled"
  [[ "${VAE_TILED}" != "1" ]] || tile_flag="--tiled"

  python3 "${VALIDATE_PY}" \
    "${MODEL_PATH_ARGS[@]}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --figurine360_lora "${FIGURINE360_LORA}" \
    --dense_warmstart_lora "${DENSE_WARMSTART_LORA}" \
    --bsa_checkpoint "${BSA_CHECKPOINT}" \
    --metadata_path "${TEST_METADATA}" \
    --dataset_base_path "$(dirname "${TEST_METADATA}")" \
    --negative_prompt "${NEGATIVE_PROMPT}" \
    --seed "${SEED}" \
    --rand_device "${RAND_DEVICE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num_frames "${NUM_FRAMES}" \
    --teacher_num_inference_steps "${TEACHER_STEPS}" \
    --teacher_cfg_scale "${TEACHER_CFG}" \
    --teacher_sigma_shift "${TEACHER_SHIFT}" \
    --fps "${FPS}" \
    --device "${DEVICE}" \
    --torch_dtype "${TORCH_DTYPE}" \
    "${tile_flag}" \
    "$@"
}

run_sample_bsa() {
  local index="$1"
  local sample_dir="${TEST_OUTPUT}/sample-${index}"

  echo "开始 BSA 对比样本 ${index}，输出: ${sample_dir}"
  run_bsa_validator \
    --sample_index "${index}" \
    --output_dir "${sample_dir}"
}

run_one_bsa() {
  doctor_bsa
  require_nonnegative_integer "SAMPLE_INDEX" "${SAMPLE_INDEX}"
  local count
  count="$(metadata_count)"
  (( SAMPLE_INDEX < count )) || die "SAMPLE_INDEX=${SAMPLE_INDEX} 越界，测试集共有 ${count} 条"
  run_sample_bsa "${SAMPLE_INDEX}"
}

run_all_bsa() {
  doctor_bsa
  require_nonnegative_integer "START_INDEX" "${START_INDEX}"
  local count end skip_flag
  count="$(metadata_count)"
  end="${END_INDEX:-${count}}"
  require_nonnegative_integer "END_INDEX" "${end}"
  (( START_INDEX < end )) || die "需要满足 START_INDEX < END_INDEX"
  (( end <= count )) || die "END_INDEX=${end} 越界，测试集共有 ${count} 条"

  skip_flag="--skip-existing"
  [[ "${SKIP_EXISTING}" == "1" ]] || skip_flag="--no-skip-existing"
  echo "单进程 BSA 批处理 [${START_INDEX}, ${end})，输出: ${TEST_OUTPUT}"
  run_bsa_validator \
    --batch_start "${START_INDEX}" \
    --batch_end "${end}" \
    "${skip_flag}" \
    --output_dir "${TEST_OUTPUT}"
}

usage_bsa_test() {
  cat <<'EOF'
用法: Wan2.2-TI2V-5B-Figurine360-BSA-Test.sh COMMAND

COMMAND:
  doctor  检查本地模型、dense warm-start、BSA checkpoint 和 input_image,prompt 测试集
  one     推理 SAMPLE_INDEX 指定的一条，并生成 teacher/student/student_bsa 三联对比
  all     单进程批量推理 [START_INDEX, END_INDEX)，默认从第 0 条到表尾

必要路径变量:
  DIT_PATHS_JSON TEXT_ENCODER_PATH VAE_PATH TOKENIZER_PATH FIGURINE360_LORA
  DENSE_WARMSTART_LORA BSA_CHECKPOINT TEST_METADATA TEST_OUTPUT

常用示例:
  SAMPLE_INDEX=0 bash Wan2.2-TI2V-5B-Figurine360-BSA-Test.sh one
  START_INDEX=0 END_INDEX=10 bash Wan2.2-TI2V-5B-Figurine360-BSA-Test.sh all

测试集只需 input_image,prompt，默认 480x832@81、seed=1、15fps；无需 teacher_latent。
teacher 是 50-step base+Figurine360；student 是 4-step dense warm-start；
student_bsa 是 BSA checkpoint 的 4-step target-sparse student。
每条样本的三联视频精确命名为 teacher_vs_student_vs_student_bsa.mp4；
原有 joint-dense 等诊断产物仍由 validator 保留。
EOF
}

case "${requested_command}" in
  doctor)
    doctor_bsa
    ;;
  one)
    run_one_bsa
    ;;
  all)
    run_all_bsa
    ;;
  *)
    usage_bsa_test
    [[ -z "${requested_command}" || "${requested_command}" =~ ^(-h|--help|help)$ ]] || exit 1
    ;;
esac
