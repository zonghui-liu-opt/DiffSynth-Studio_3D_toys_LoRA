import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DIRECT_DISTILL = ROOT / "examples/wanvideo/model_training/special/direct_distill"
SHELL = DIRECT_DISTILL / "Wan2.2-TI2V-5B-Figurine360-BSA.sh"


def _env():
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    return environment


def run_sourced_shell(body: str) -> subprocess.CompletedProcess[str]:
    command = f'''
shell_path="$1"
set -- help
source "$shell_path" >/dev/null
{body}
'''
    return subprocess.run(
        ["bash", "-c", command, "_", str(SHELL)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_env(),
    )


def test_bsa_shell_syntax_help_and_required_commands():
    subprocess.run(["bash", "-n", str(SHELL)], check=True)
    completed = subprocess.run(
        ["bash", str(SHELL), "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )
    for command in ("doctor", "bsa-smoke-train", "train", "bsa-train", "bsa-validate", "bsa-benchmark", "plot"):
        assert command in completed.stdout
    assert "train的兼容别名" in completed.stdout


def test_bsa_shell_train_alias_uses_the_same_formal_training_dispatch():
    source = SHELL.read_text(encoding="utf-8")

    assert (
        'train|bsa-train) train_bsa "${TEACHER_ROOT}" "${BSA_TRAIN_OUTPUT}" '
        '"${FORMAL_HEIGHT}" "${FORMAL_WIDTH}" "${FORMAL_NUM_FRAMES}" '
        '"${NUM_EPOCHS}" "${BSA_SPARSITY_SCHEDULE}" ;;'
    ) in source


def test_bsa_python_entrypoints_expose_offline_help():
    for script in (
        "validate_wan22_ti2v_figurine360_bsa.py",
        "benchmark_wan_bsa.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(DIRECT_DISTILL / script), "--help"],
            check=True,
            text=True,
            capture_output=True,
            env=_env(),
        )
        assert "usage:" in completed.stdout.lower()


def test_bsa_shell_keeps_frozen_block_gate_and_student_controls():
    source = SHELL.read_text(encoding="utf-8")
    for text in (
        'BSA_BLOCK_SIZE="${BSA_BLOCK_SIZE:-4,3,6}"',
        "--bsa_gate_granularity block",
        "--direct_distill_preserve_first_frame",
        "--direct_distill_exclude_first_frame_loss",
        "--direct_distill_warmstart_lora",
        "--resume_bsa_checkpoint",
        "--bsa_sparsity_schedule",
        "SEED=1",
        'NUM_EPOCHS="${NUM_EPOCHS:-60}"',
        'BSA_SPARSITY_SCHEDULE="${BSA_SPARSITY_SCHEDULE:-conservative_epoch_v1}"',
        'BSA_SMOKE_SPARSITY_SCHEDULE="${BSA_SMOKE_SPARSITY_SCHEDULE:-legacy_progress_v1}"',
        'NUM_PROCESSES="${NUM_PROCESSES:-}"',
        'FORMAL_HEIGHT="${FORMAL_HEIGHT:-480}"',
        'FORMAL_WIDTH="${FORMAL_WIDTH:-832}"',
        'FORMAL_NUM_FRAMES="${FORMAL_NUM_FRAMES:-81}"',
        '--bsa_expected_runtime_grid "${expected_grid}"',
        'require_seed_one_metadata "${metadata}"',
    ):
        assert text in source


def test_bsa_seed_one_metadata_guard(tmp_path):
    accepted = tmp_path / "accepted.csv"
    accepted.write_text("seed,prompt\n1,a\n1,b\n", encoding="utf-8")
    rejected = tmp_path / "rejected.csv"
    rejected.write_text("seed,prompt\n1,a\n3,b\n", encoding="utf-8")

    good = run_sourced_shell(f'require_seed_one_metadata "{accepted}"')
    bad = run_sourced_shell(f'require_seed_one_metadata "{rejected}"')

    assert good.returncode == 0
    assert "seed=1 metadata检查通过: 2条" in good.stdout
    assert bad.returncode != 0
    assert "只允许seed=1" in bad.stderr


def test_bsa_metadata_preflight_rejects_swapped_formal_geometry(tmp_path):
    (tmp_path / "first.png").touch()
    (tmp_path / "teacher.safetensors").touch()
    valid = tmp_path / "valid.csv"
    valid.write_text(
        "seed,height,width,num_frames,input_image,teacher_latent\n"
        "1,480,832,81,first.png,teacher.safetensors\n",
        encoding="utf-8",
    )
    swapped = tmp_path / "swapped.csv"
    swapped.write_text(
        "seed,height,width,num_frames,input_image,teacher_latent\n"
        "1,832,480,81,first.png,teacher.safetensors\n",
        encoding="utf-8",
    )

    good = run_sourced_shell(
        f'require_seed_one_metadata "{valid}" 480 832 81 "{tmp_path}"'
    )
    bad = run_sourced_shell(
        f'require_seed_one_metadata "{swapped}" 480 832 81 "{tmp_path}"'
    )

    assert good.returncode == 0
    assert bad.returncode != 0
    assert "height,width,num_frames=(480, 832, 81)" in bad.stderr


def test_bsa_shell_can_launch_six_gpus_without_ambient_accelerate_config():
    completed = run_sourced_shell(
        'ACCELERATE_CONFIG=""; NUM_PROCESSES=6; '
        'while IFS= read -r -d "" item; do printf "[%s]" "$item"; done < <(accelerate_prefix)'
    )

    assert completed.returncode == 0
    assert completed.stdout == "[accelerate][launch][--multi_gpu][--num_processes][6]"


def test_bsa_shell_derives_runtime_grid_from_video_geometry():
    formal = run_sourced_shell('wan22_expected_grid 480 832 81')
    smoke = run_sourced_shell('wan22_expected_grid 256 448 17')
    invalid = run_sourced_shell('wan22_expected_grid 832 480 80')

    assert formal.returncode == 0 and formal.stdout == "21,15,26"
    assert smoke.returncode == 0 and smoke.stdout == "5,8,14"
    assert invalid.returncode != 0
