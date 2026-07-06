from pathlib import Path


def test_train_launcher_exposes_optional_save_steps():
    script = Path("train_ti2v5b_lora.sh").read_text(encoding="utf-8")

    assert "SAVE_STEPS=${SAVE_STEPS:-}" in script
    assert "SAVE_ARGS=()" in script
    assert "SAVE_ARGS+=(--save_steps \"$SAVE_STEPS\")" in script
    assert '"${SAVE_ARGS[@]}" \\' in script
