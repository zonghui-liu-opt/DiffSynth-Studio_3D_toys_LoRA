from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
TEST_SHELL = (
    ROOT
    / "examples"
    / "wanvideo"
    / "model_training"
    / "special"
    / "direct_distill"
    / "Wan2.2-TI2V-5B-Figurine360-BSA-Test.sh"
)


def run_sourced_shell(body: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = f'''\
shell_path="$1"
set -- help
source "$shell_path" >/dev/null
{body}
'''
    return subprocess.run(
        ["bash", "-c", command, "_", str(TEST_SHELL)],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def test_bsa_test_shell_syntax_help_and_thin_contract():
    subprocess.run(["bash", "-n", str(TEST_SHELL)], cwd=ROOT, check=True)
    result = subprocess.run(
        ["bash", str(TEST_SHELL), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "input_image,prompt" in result.stdout
    assert "无需 teacher_latent" in result.stdout
    assert "teacher_vs_student_vs_student_bsa.mp4" in result.stdout
    assert "student 是 4-step dense warm-start" in result.stdout
    assert "student_bsa 是 BSA checkpoint" in result.stdout

    script = TEST_SHELL.read_text(encoding="utf-8")
    assert 'source "${DIRECT_TEST_SH}"' in script
    assert '--dense_warmstart_lora "${DENSE_WARMSTART_LORA}"' in script
    assert '--bsa_checkpoint "${BSA_CHECKPOINT}"' in script
    assert '--batch_start "${START_INDEX}"' in script
    assert '--batch_end "${end}"' in script
    assert '--fps "${FPS}"' in script
    assert "--teacher_latent" not in script
    assert 'SEED="${SEED:-1}"' not in script


def test_bsa_test_shell_passes_one_sample_arguments():
    result = run_sourced_shell(
        r'''
python3(){ printf '%s\n' "$@"; }
TEST_METADATA=/tmp/test-metadata.csv
TEST_OUTPUT=/tmp/bsa-results
DENSE_WARMSTART_LORA=/tmp/dense.safetensors
BSA_CHECKPOINT=/tmp/checkpoint
FPS=12
run_sample_bsa 7
'''
    )
    arguments = result.stdout.splitlines()

    assert arguments[0] == "开始 BSA 对比样本 7，输出: /tmp/bsa-results/sample-7"
    assert "--dense_warmstart_lora" in arguments
    assert arguments[arguments.index("--dense_warmstart_lora") + 1] == "/tmp/dense.safetensors"
    assert "--bsa_checkpoint" in arguments
    assert arguments[arguments.index("--bsa_checkpoint") + 1] == "/tmp/checkpoint"
    assert arguments[arguments.index("--sample_index") + 1] == "7"
    assert arguments[arguments.index("--seed") + 1] == "1"
    assert arguments[arguments.index("--fps") + 1] == "12"
    assert arguments[arguments.index("--output_dir") + 1] == "/tmp/bsa-results/sample-7"
    assert "--teacher_latent" not in arguments


def test_bsa_test_shell_passes_batch_range_and_skip_policy():
    result = run_sourced_shell(
        r'''
doctor_bsa(){ :; }
metadata_count(){ printf '5\n'; }
python3(){ printf '%s\n' "$@"; }
TEST_METADATA=/tmp/test-metadata.csv
TEST_OUTPUT=/tmp/bsa-results
START_INDEX=1
END_INDEX=4
SKIP_EXISTING=0
run_all_bsa
'''
    )
    arguments = result.stdout.splitlines()

    assert arguments[0] == "单进程 BSA 批处理 [1, 4)，输出: /tmp/bsa-results"
    assert arguments[arguments.index("--batch_start") + 1] == "1"
    assert arguments[arguments.index("--batch_end") + 1] == "4"
    assert "--no-skip-existing" in arguments
    assert "--skip-existing" not in arguments
    assert arguments[arguments.index("--output_dir") + 1] == "/tmp/bsa-results"


def test_bsa_test_shell_accepts_absolute_input_images_without_teacher_latent(tmp_path):
    image = tmp_path / "first.png"
    image.touch()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        f'input_image,prompt\n"{image}","turn, 360 degrees"\n',
        encoding="utf-8",
    )
    result = run_sourced_shell(
        f'TEST_METADATA="{metadata}"\nmetadata_count'
    )

    assert result.stdout.strip() == "1"


def test_bsa_test_shell_rejects_relative_input_images(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "input_image,prompt\nfirst.png,turntable\n",
        encoding="utf-8",
    )
    result = run_sourced_shell(
        f'TEST_METADATA="{metadata}"\nmetadata_count', check=False
    )

    assert result.returncode != 0
    assert "不是绝对路径" in result.stderr


def test_bsa_test_shell_rejects_non_one_metadata_seed(tmp_path):
    image = tmp_path / "first.png"
    image.touch()
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        f'input_image,prompt,seed\n"{image}",turntable,2\n',
        encoding="utf-8",
    )
    result = run_sourced_shell(
        f'TEST_METADATA="{metadata}"\nmetadata_count', check=False
    )

    assert result.returncode != 0
    assert "只允许 seed=1" in result.stderr


def test_bsa_test_shell_doctor_requires_complete_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    result = run_sourced_shell(
        f'''\
doctor(){{ :; }}
BSA_CHECKPOINT="{checkpoint}"
doctor_bsa
''',
        check=False,
    )

    assert result.returncode != 0
    assert "checkpoint_complete" in result.stderr
