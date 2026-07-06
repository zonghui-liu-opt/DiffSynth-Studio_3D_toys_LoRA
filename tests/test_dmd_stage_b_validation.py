import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from dmd.wan22_lora import (
    apply_lora_to_model,
    export_lora_state_dict,
    import_lora_state_dict_for_runtime,
    load_lora_state_dict,
)
from dmd.stage_b_validation import build_validation_jobs, write_validation_manifest
from dmd.video_metrics import compute_basic_video_metrics, compute_video_diff_metrics
from dmd.wan22_config import load_dmd_config_summary


REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyWanBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.ffn = nn.Sequential(nn.Linear(4, 8, bias=False), nn.GELU(), nn.Linear(8, 4, bias=False))


def test_exported_diffsynth_lora_keys_import_into_runtime_lora_modules():
    source = nn.Module()
    source.model = TinyWanBlock()
    apply_lora_to_model(source, rank=2, alpha=2, target_modules=("q", "ffn.0"))

    exported = export_lora_state_dict(source.state_dict(), strip_prefixes=("model.",), include_alpha=True)

    target = TinyWanBlock()
    apply_lora_to_model(target, rank=2, alpha=2, target_modules=("q", "ffn.0"))
    runtime_state = import_lora_state_dict_for_runtime(exported)
    load_lora_state_dict(target, runtime_state, strict_lora=True)

    assert sorted(runtime_state) == [
        "ffn.0.alpha",
        "ffn.0.lora_A.weight",
        "ffn.0.lora_B.weight",
        "q.alpha",
        "q.lora_A.weight",
        "q.lora_B.weight",
    ]


def test_wan22_fewstep_entrypoint_exposes_dmd_lora_validation_flags():
    script = (REPO_ROOT / "third_party/wan22_turbo/wan2.2_fewstep.py").read_text(encoding="utf-8")

    assert "--lora_path" in script
    assert "--checkpoint_path" in script
    assert "--lora_source" in script
    assert "--timing_json" in script
    assert "import_lora_state_dict_for_runtime" in script


def test_trainer_resume_loads_generator_ema_lora_key():
    source = (REPO_ROOT / "third_party/wan22_turbo/trainer/wan22_distillation.py").read_text(encoding="utf-8")
    load_block = source.split("# Load EMA state dict if available in checkpoint", 1)[1]
    load_block = load_block.split("##############################################################################################################", 1)[0]

    assert '"generator_ema_lora"' in load_block


def test_dmd_config_and_launcher_expose_stage_b_runtime_controls():
    summary = load_dmd_config_summary(REPO_ROOT / "configs/dmd/figurine360_wan22_dmd_lora.yaml")
    script = (REPO_ROOT / "train_figurine360_dmd_lora.sh").read_text(encoding="utf-8")
    trainer = (REPO_ROOT / "third_party/wan22_turbo/trainer/wan22_distillation.py").read_text(encoding="utf-8")

    assert summary["validation_interval"] == 200
    assert summary["dataloader_num_workers"] == 8
    assert "VALIDATION_INTERVAL=${VALIDATION_INTERVAL:-200}" in script
    assert "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}" in script
    assert "--validation_interval" in script
    assert "--dataloader_num_workers" in script
    assert "num_workers=config.get(\"dataloader_num_workers\", 8)" in trainer


def test_teacher50_inference_config_is_available_for_stage_b_alignment():
    config_path = REPO_ROOT / "configs/dmd/figurine360_wan22_teacher50.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert cfg["generator_name"] == "Wan2.2-TI2V-5B"
    assert cfg["model_name"] == "Wan2.2-TI2V-5B"
    assert len(cfg["denoising_step_list"]) == 50
    assert cfg["denoising_step_list"][0] == 1000
    assert cfg["denoising_step_list"][-1] == 20
    assert cfg["warp_denoising_step"] is True
    assert cfg["lora"]["enabled"] is False


def test_stage_b_validation_manifest_builds_teacher_and_student_jobs(tmp_path):
    holdout_csv = tmp_path / "holdout.csv"
    image_path = tmp_path / "first.png"
    image_path.write_bytes(b"fake image placeholder")
    with holdout_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "prompt", "image", "seed"])
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "case-001",
                "prompt": "a figurine rotating on a turntable",
                "image": str(image_path),
                "seed": "123",
            }
        )

    jobs = build_validation_jobs(
        holdout_csv=holdout_csv,
        output_dir=tmp_path / "out",
        runtime_dir=Path("/runtime"),
        teacher_config_path=Path("/configs/teacher50.yaml"),
        student_config_path=Path("/configs/student4.yaml"),
        dmd_lora_path=Path("/models/figurine360_dmd_lora_rank64.safetensors"),
        height=480,
        width=832,
        num_frames=121,
    )
    manifest_path = write_validation_manifest(jobs, tmp_path / "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [job["role"] for job in manifest["jobs"]] == ["teacher", "student"]
    assert manifest["jobs"][0]["case_id"] == "case-001"
    assert "--config_path" in manifest["jobs"][0]["command"]
    assert "/configs/teacher50.yaml" in manifest["jobs"][0]["command"]
    assert "--lora_path" not in manifest["jobs"][0]["command"]
    assert "--lora_path" in manifest["jobs"][1]["command"]
    assert "/models/figurine360_dmd_lora_rank64.safetensors" in manifest["jobs"][1]["command"]
    assert manifest["comparisons"][0]["teacher_video"].endswith("case-001_teacher.mp4")
    assert manifest["comparisons"][0]["student_video"].endswith("case-001_student.mp4")


def test_basic_video_metrics_report_brightness_stability_and_hist_shift():
    teacher = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    teacher[..., 0] = 120
    teacher[..., 1] = 100
    teacher[..., 2] = 80

    student = teacher.copy()
    student[2:] = np.clip(student[2:] + 40, 0, 255)

    metrics = compute_basic_video_metrics(teacher, student, bins=16)

    assert metrics["brightness_hist_shift"] > 0
    assert metrics["saturation_hist_shift"] >= 0
    assert metrics["student_interframe_brightness_std"] > metrics["teacher_interframe_brightness_std"]


def test_video_diff_metrics_report_pixel_max_and_mse():
    left = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    right = left.copy()
    right[0, 0, 0, 0] = 255

    metrics = compute_video_diff_metrics(left, right)

    assert metrics["max_pixel_diff"] == 1.0
    assert metrics["max_pixel_diff_255"] == 255.0
    assert metrics["pixel_mse"] > 0.0


def test_stage_b_validation_script_has_valid_cli_help():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/validate_dmd_stage_b.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--holdout_csv" in result.stdout
    assert "--dmd_lora_path" in result.stdout
    assert "--run" in result.stdout


def test_stage_b_metrics_script_has_valid_cli_help():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/compute_dmd_video_metrics.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--manifest" in result.stdout
    assert "--output_json" in result.stdout
    assert "--bins" in result.stdout


def test_diffsynth_wan22_ti2v_local_infer_script_has_valid_cli_help():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/diffsynth_wan22_ti2v_infer.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--model_root" in result.stdout
    assert "--num_inference_steps" in result.stdout
    assert "--cfg_scale" in result.stdout
    assert "--timing_json" in result.stdout


def test_diffsynth_wan22_dmd_lora_infer_script_has_valid_cli_help():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/diffsynth_wan22_dmd_lora_infer.py"), "--help"],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--model_root" in result.stdout
    assert "--dmd_lora_path" in result.stdout
    assert "--num_inference_steps" in result.stdout
    assert "--timing_json" in result.stdout
