import csv
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from diffsynth.core import UnifiedDataset

import check_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_make_debug_dataset(output_dir, *extra_args):
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "make_debug_dataset.py"),
            "--output_dir",
            str(output_dir),
            "--num_samples",
            "3",
            "--height",
            "64",
            "--width",
            "96",
            "--num_frames",
            "9",
            *extra_args,
        ],
        check=True,
        cwd=REPO_ROOT,
    )


def load_train_module():
    module_path = REPO_ROOT / "examples/wanvideo/model_training/train.py"
    spec = importlib.util.spec_from_file_location("wan_train_module_for_figurine", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_video_dataset(dataset_root, metadata_path):
    return UnifiedDataset(
        base_path=str(dataset_root),
        metadata_path=str(metadata_path),
        data_file_keys=["video"],
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=str(dataset_root),
            height=64,
            width=96,
            num_frames=9,
        ),
    )


def test_two_col_debug_dataset_validates_and_iterates_video_samples(tmp_path):
    dataset_root = tmp_path / "debug_data_figurine"
    run_make_debug_dataset(dataset_root, "--schema", "two_col")

    metadata_path = dataset_root / "metadata.csv"
    rows = check_dataset.read_metadata(metadata_path, check_dataset.detect_delimiter(metadata_path))

    assert list(rows[0].keys()) == ["video", "prompt"]
    assert not (dataset_root / "images_64x96").exists()

    summary = check_dataset.validate_dataset(
        dataset_root,
        metadata_path,
        height=64,
        width=96,
        num_frames=9,
    )

    assert summary["schema"] == "two_col"
    assert summary["bad_samples"] == 0

    dataset = make_video_dataset(dataset_root, summary["fixed_path"])
    for index in range(2):
        sample = dataset[index]
        assert len(sample["video"]) == 9
        assert sample["video"][0].size == (96, 64)
        assert isinstance(sample["prompt"], str)
        assert "input_image" not in sample


def test_two_col_bad_samples_report_missing_video_and_short_video(tmp_path):
    dataset_root = tmp_path / "debug_data_figurine_bad"
    run_make_debug_dataset(dataset_root, "--schema", "two_col", "--with_bad_samples")

    summary = check_dataset.validate_dataset(
        dataset_root,
        dataset_root / "metadata.csv",
        height=64,
        width=96,
        num_frames=9,
    )
    reasons = [reason for row in summary["bad_rows"] for reason in row["reasons"]]

    assert summary["schema"] == "two_col"
    assert "missing_video" in reasons
    assert "insufficient_frames" in reasons


def test_two_col_training_input_image_falls_back_to_video_first_frame(tmp_path):
    dataset_root = tmp_path / "debug_data_figurine"
    run_make_debug_dataset(dataset_root, "--schema", "two_col")
    summary = check_dataset.validate_dataset(
        dataset_root,
        dataset_root / "metadata.csv",
        height=64,
        width=96,
        num_frames=9,
    )
    sample = make_video_dataset(dataset_root, summary["fixed_path"])[0]

    train_module = load_train_module()
    module = train_module.WanTrainingModule.__new__(train_module.WanTrainingModule)
    inputs = module.parse_extra_inputs(sample, ["input_image"], {})

    assert inputs["input_image"] is sample["video"][0]


def test_three_col_debug_dataset_keeps_input_images_for_three_col_regression(tmp_path):
    dataset_root = tmp_path / "debug_data_three_col"
    run_make_debug_dataset(dataset_root, "--schema", "three_col")

    with (dataset_root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    assert list(rows[0].keys()) == ["video", "prompt", "input_image"]
    assert (dataset_root / "images_64x96").is_dir()
    assert all((dataset_root / row["input_image"]).is_file() for row in rows)

    summary = check_dataset.validate_dataset(
        dataset_root,
        dataset_root / "metadata.csv",
        height=64,
        width=96,
        num_frames=9,
    )

    assert summary["schema"] == "three_col"
    assert summary["bad_samples"] == 0


def test_figurine_train_script_uses_two_col_video_keys_with_input_image_fallback():
    script = (REPO_ROOT / "train_figurine360_lora.sh").read_text(encoding="utf-8")

    assert "OUTPUT_ROOT=${OUTPUT_ROOT:-./models/train/Wan2.2-TI2V-5B_figurine360_lora}" in script
    assert '--data_file_keys "video"' in script
    assert '--extra_inputs "input_image"' in script
    assert "FIND_UNUSED_PARAMETERS=${FIND_UNUSED_PARAMETERS:-0}" in script
    assert "METRICS_PATH=${METRICS_PATH:-$OUTPUT_ROOT/metrics.jsonl}" in script


def test_figurine_infer_build_model_configs_uses_local_wan_layout(tmp_path):
    from infer_figurine360 import build_model_configs

    model_root = tmp_path / "Wan2.2-TI2V-5B"
    model_root.mkdir()
    (model_root / "diffusion_pytorch_model-00001-of-00002.safetensors").touch()
    (model_root / "diffusion_pytorch_model-00002-of-00002.safetensors").touch()
    (model_root / "models_t5_umt5-xxl-enc-bf16.pth").touch()
    (model_root / "Wan2.2_VAE.pth").touch()
    (model_root / "google" / "umt5-xxl").mkdir(parents=True)

    model_configs, tokenizer_config = build_model_configs(model_root, model_root / "google" / "umt5-xxl")

    assert model_configs[0].path == [
        str(model_root / "diffusion_pytorch_model-00001-of-00002.safetensors"),
        str(model_root / "diffusion_pytorch_model-00002-of-00002.safetensors"),
    ]
    assert model_configs[1].path == str(model_root / "models_t5_umt5-xxl-enc-bf16.pth")
    assert model_configs[2].path == str(model_root / "Wan2.2_VAE.pth")
    assert tokenizer_config.path == str(model_root / "google" / "umt5-xxl")


def test_figurine_infer_dry_run_prints_config_without_loading_models(tmp_path):
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"placeholder")
    env = os.environ.copy()
    env.update(
        {
            "MODEL_ROOT": str(tmp_path / "missing_model_root"),
            "TOKENIZER_PATH": str(tmp_path / "missing_model_root" / "google" / "umt5-xxl"),
            "LORA_PATH": str(tmp_path / "missing_lora.safetensors"),
            "IMAGE_PATH": str(image_path),
            "OUTPUT_PATH": str(tmp_path / "out.mp4"),
        }
    )
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "infer_figurine360.py"), "--dry_run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "dry_run: true" in result.stdout
    assert "Wan2.2-TI2V-5B_figurine360" in result.stdout
