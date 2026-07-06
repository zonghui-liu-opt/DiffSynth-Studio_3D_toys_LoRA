import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

from dmd.wan22_config import load_dmd_config_summary
from tools.prepare_dmd_dataset_csv import convert_metadata_to_turbo_csv


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dmd_config_exposes_phase1_lora_and_reserved_phase23_hooks():
    summary = load_dmd_config_summary(REPO_ROOT / "configs/dmd/figurine360_wan22_dmd_lora.yaml")

    assert summary["trainer"] == "score_distillation_wan22"
    assert summary["distribution_loss"] == "dmd"
    assert summary["generator_type"] == "bidirectional"
    assert summary["denoising_step_list"] == [1000, 750, 500, 250]
    assert summary["warp_denoising_step"] is True
    assert summary["denoising_loss_type"] == "x0"
    assert summary["dfake_gen_update_ratio"] == 5
    assert summary["real_guidance_scale"] == 4.0
    assert summary["fake_guidance_scale"] == 0.0
    assert summary["lora"] == {
        "enabled": True,
        "rank": 64,
        "alpha": 64,
        "target_modules": ["q", "k", "v", "o", "ffn.0", "ffn.2"],
    }
    assert summary["attention_backend"] == "flash"
    assert summary["gan_loss_weight"] == 0.0
    assert summary["load_video_latent"] is False


def test_prepare_dmd_dataset_csv_converts_figurine_metadata_to_turbo_schema(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "videos").mkdir()
    (dataset_root / "videos" / "a.mp4").write_bytes(b"fake")
    metadata_path = dataset_root / "metadata_fixed.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt"])
        writer.writeheader()
        writer.writerow({"video": "videos/a.mp4", "prompt": "手办360度水平旋转展示"})

    output_path = tmp_path / "dmd.csv"
    report = convert_metadata_to_turbo_csv(
        dataset_root=dataset_root,
        metadata_path=metadata_path,
        output_path=output_path,
        default_num_frames=121,
    )

    assert report == {"rows": 1, "output_path": str(output_path)}
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {
            "path": str(dataset_root / "videos" / "a.mp4"),
            "text": "手办360度水平旋转展示",
            "num_frames": "121",
        }
    ]


def test_dmd_launcher_is_current_repo_scoped_and_uses_vendored_runtime():
    script = (REPO_ROOT / "train_figurine360_dmd_lora.sh").read_text(encoding="utf-8")

    assert "/Users/zonghuiliu/Documents/Codex/Wan2.2-TI2V-5B-Turbo" not in script
    assert "DMD_RUNTIME_DIR=${DMD_RUNTIME_DIR:-$REPO_ROOT/third_party/wan22_turbo}" in script
    assert "CONFIG_PATH=${CONFIG_PATH:-$REPO_ROOT/configs/dmd/figurine360_wan22_dmd_lora.yaml}" in script
    assert "tools/prepare_dmd_dataset_csv.py" in script
    assert "wan_models/Wan2.2-TI2V-5B" in script
    assert "torchrun --nproc_per_node=\"$NUM_GPUS\"" in script
    assert "PYTHONPATH=\"$REPO_ROOT:$DMD_RUNTIME_DIR" in script


def test_dmd_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(REPO_ROOT / "train_figurine360_dmd_lora.sh")], check=True)


def test_vendored_wan22_attention_imports_without_flash_attn_and_runs_sdpa_cpu():
    attention_path = REPO_ROOT / "third_party/wan22_turbo/wan22/modules/attention.py"
    spec = importlib.util.spec_from_file_location("vendored_wan22_attention_for_test", attention_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    q = torch.randn(1, 3, 2, 4)
    k = torch.randn(1, 3, 2, 4)
    v = torch.randn(1, 3, 2, 4)

    out = module.attention(q, k, v, dtype=torch.float32)

    assert out.shape == (1, 3, 2, 4)
    assert out.dtype == torch.float32
