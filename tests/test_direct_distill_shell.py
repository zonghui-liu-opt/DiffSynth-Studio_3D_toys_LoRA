from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SHELL = (
    ROOT
    / "examples"
    / "wanvideo"
    / "model_training"
    / "special"
    / "direct_distill"
    / "Wan2.2-TI2V-5B-Figurine360.sh"
)


def run_sourced_shell(body: str) -> subprocess.CompletedProcess[str]:
    command = f"""
shell_path="$1"
set -- help
source "$shell_path" >/dev/null
{body}
"""
    return subprocess.run(
        ["bash", "-c", command, "_", str(SHELL)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_shell_help_accepts_standard_flag():
    result = subprocess.run(
        ["bash", str(SHELL), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "prepare-smoke" in result.stdout


def test_shell_prepends_repo_root_to_pythonpath():
    result = run_sourced_shell("printf '%s' \"$PYTHONPATH\"")

    assert result.stdout.split(":", 1)[0] == str(ROOT)


def test_prepare_smoke_does_not_reintroduce_held_out_seed():
    result = run_sourced_shell(
        """
doctor_source(){ :; }
python3(){ printf '%s\\n' "$@"; }
prepare_teacher /tmp/direct-distill-smoke 256 448 17 4 1 0 1 ""
"""
    )

    arguments = result.stdout.splitlines()
    assert "--seed" in arguments
    assert arguments[arguments.index("--seed") + 1] == "1"
    assert "--validation_seed" not in arguments


def test_shell_has_separate_smoke_and_formal_validation_checkpoints():
    script = SHELL.read_text(encoding="utf-8")

    assert '"${SMOKE_VALIDATION_LORA_CHECKPOINT}"' in script
    assert '"${FORMAL_VALIDATION_LORA_CHECKPOINT}"' in script
    assert "VALIDATION_LORA_CHECKPOINT=" not in script.replace(
        "SMOKE_VALIDATION_LORA_CHECKPOINT=", ""
    ).replace("FORMAL_VALIDATION_LORA_CHECKPOINT=", "")
