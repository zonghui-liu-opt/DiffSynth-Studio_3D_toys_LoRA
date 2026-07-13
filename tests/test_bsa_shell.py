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


def test_bsa_shell_syntax_help_and_required_commands():
    subprocess.run(["bash", "-n", str(SHELL)], check=True)
    completed = subprocess.run(
        ["bash", str(SHELL), "--help"],
        check=True,
        text=True,
        capture_output=True,
        env=_env(),
    )
    for command in ("doctor", "bsa-smoke-train", "bsa-train", "bsa-validate", "bsa-benchmark", "plot"):
        assert command in completed.stdout


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
    ):
        assert text in source
