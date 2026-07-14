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
TEST_SHELL = SHELL.with_name("Wan2.2-TI2V-5B-Figurine360-Test.sh")


def run_sourced_shell(body: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = f"""
shell_path="$1"
set -- help
source "$shell_path" >/dev/null
{body}
"""
    return subprocess.run(
        ["bash", "-c", command, "_", str(SHELL)],
        cwd=ROOT,
        check=check,
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


def test_prepare_formal_uses_only_seed_one():
    result = run_sourced_shell(
        """
doctor_source(){ :; }
python3(){ printf '%s\\n' "$@"; }
prepare_teacher /tmp/direct-distill-formal 832 480 81 "" 0 0.1 1 ""
"""
    )

    arguments = result.stdout.splitlines()
    assert arguments.count("--seed") == 1
    assert arguments[arguments.index("--seed") + 1] == "1"
    assert "--validation_seed" not in arguments


def test_seed_one_metadata_guard_rejects_other_seeds(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("seed,prompt\n1,ok\n2,bad\n", encoding="utf-8")
    result = run_sourced_shell(
        f'require_seed_one_metadata "{metadata}"', check=False
    )

    assert result.returncode != 0
    assert "只允许 seed=1" in result.stderr


def test_shell_has_separate_smoke_and_formal_validation_checkpoints():
    script = SHELL.read_text(encoding="utf-8")

    assert '"${SMOKE_VALIDATION_LORA_CHECKPOINT}"' in script
    assert '"${FORMAL_VALIDATION_LORA_CHECKPOINT}"' in script
    assert "VALIDATION_LORA_CHECKPOINT=" not in script.replace(
        "SMOKE_VALIDATION_LORA_CHECKPOINT=", ""
    ).replace("FORMAL_VALIDATION_LORA_CHECKPOINT=", "")


def test_custom_test_shell_help_and_syntax():
    subprocess.run(["bash", "-n", str(TEST_SHELL)], cwd=ROOT, check=True)
    result = subprocess.run(
        ["bash", str(TEST_SHELL), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "input_image,prompt" in result.stdout
    assert "480x832@81" in result.stdout
    assert "student 固定为 4 steps/CFG 1/shift 5" in result.stdout
    assert "模型只加载一次" in result.stdout

    script = TEST_SHELL.read_text(encoding="utf-8")
    assert "SEED=1" in script
    assert 'SEED="${SEED:-1}"' not in script
    assert '--batch_start "${START_INDEX}"' in script
    assert '--batch_end "${end}"' in script
    assert "每条样本都会重新加载" not in script


def test_custom_test_shell_validates_absolute_input_images(tmp_path):
    image = tmp_path / "first.png"
    image.touch()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        f'input_image,prompt\n"{image}","turn, 360 degrees"\n',
        encoding="utf-8",
    )
    command = f"""
shell_path="$1"
metadata_path="$2"
set -- help
source "$shell_path" >/dev/null
TEST_METADATA="$metadata_path"
metadata_count
"""
    result = subprocess.run(
        ["bash", "-c", command, "_", str(TEST_SHELL), str(metadata)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "1"


def test_custom_test_shell_rejects_relative_input_images(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "input_image,prompt\nfirst.png,turntable\n",
        encoding="utf-8",
    )
    command = f"""
shell_path="$1"
metadata_path="$2"
set -- help
source "$shell_path" >/dev/null
TEST_METADATA="$metadata_path"
metadata_count
"""
    result = subprocess.run(
        ["bash", "-c", command, "_", str(TEST_SHELL), str(metadata)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "不是绝对路径" in result.stderr


def test_custom_test_shell_rejects_non_one_metadata_seed(tmp_path):
    image = tmp_path / "first.png"
    image.touch()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        f'input_image,prompt,seed\n"{image}",turntable,2\n',
        encoding="utf-8",
    )
    command = f"""
shell_path="$1"
metadata_path="$2"
set -- help
source "$shell_path" >/dev/null
TEST_METADATA="$metadata_path"
metadata_count
"""
    result = subprocess.run(
        ["bash", "-c", command, "_", str(TEST_SHELL), str(metadata)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "只允许 seed=1" in result.stderr
