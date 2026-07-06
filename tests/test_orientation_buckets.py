import argparse
import csv
import importlib.util
from pathlib import Path

from PIL import Image

import check_dataset
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.bucket_sampler import OrientationBucketSampler
from diffsynth.core.data.operators import ImageResizeToBucketResolution


def load_parsers_module():
    module_path = Path("diffsynth/diffusion/parsers.py").resolve()
    spec = importlib.util.spec_from_file_location("diffsynth_diffusion_parsers", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_dataset_writes_landscape_and_portrait_bucket_metadata(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    metadata_path = dataset_root / "metadata.csv"
    rows = [
        {"video": "landscape.mp4", "prompt": "wide cat", "input_image": "landscape.jpg"},
        {"video": "portrait.mp4", "prompt": "tall cat", "input_image": "portrait.jpg"},
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video", "prompt", "input_image"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        (dataset_root / row["video"]).touch()
        (dataset_root / row["input_image"]).touch()

    def fake_inspect_video(video_path):
        if "landscape" in str(video_path):
            return {"width": 832, "height": 480, "frames": 121, "fps": 24.0}
        return {"width": 480, "height": 832, "frames": 121, "fps": 24.0}

    monkeypatch.setattr(check_dataset, "inspect_video", fake_inspect_video)

    summary = check_dataset.validate_dataset(dataset_root, metadata_path, height=480, width=832, num_frames=121)

    fixed_path = Path(summary["fixed_path"])
    fixed_rows = list(csv.DictReader(fixed_path.open(newline="", encoding="utf-8")))
    assert summary["bucket_counts"] == {"landscape": 1, "portrait": 1}
    assert fixed_rows[0]["bucket"] == "landscape"
    assert fixed_rows[0]["height"] == "480"
    assert fixed_rows[0]["width"] == "832"
    assert fixed_rows[1]["bucket"] == "portrait"
    assert fixed_rows[1]["height"] == "832"
    assert fixed_rows[1]["width"] == "480"


def test_image_resize_to_bucket_resolution_preserves_orientation_without_crop():
    operator = ImageResizeToBucketResolution(landscape_height=480, landscape_width=832)
    landscape = Image.new("RGB", (1664, 960), "red")
    portrait = Image.new("RGB", (960, 1664), "blue")

    assert operator(landscape).size == (832, 480)
    assert operator(portrait).size == (480, 832)


def test_unified_dataset_bucket_video_operator_resizes_image_rows_by_orientation(tmp_path):
    landscape = Image.new("RGB", (1664, 960), "red")
    portrait = Image.new("RGB", (960, 1664), "blue")
    landscape.save(tmp_path / "landscape.jpg")
    portrait.save(tmp_path / "portrait.jpg")

    operator = UnifiedDataset.default_video_operator(
        base_path=str(tmp_path),
        height=480,
        width=832,
        resize_mode="bucket",
    )

    assert operator("landscape.jpg")[0].size == (832, 480)
    assert operator("portrait.jpg")[0].size == (480, 832)


def test_orientation_bucket_sampler_groups_indices_by_metadata_bucket():
    dataset = type(
        "DatasetWithBuckets",
        (),
        {
            "data": [
                {"bucket": "landscape"},
                {"bucket": "portrait"},
                {"bucket": "landscape"},
                {"bucket": "portrait"},
            ],
            "__len__": lambda self: 4,
        },
    )()

    indices = list(OrientationBucketSampler(dataset, shuffle=False))

    assert indices == [0, 2, 1, 3]


def test_orientation_bucket_sampler_respects_dataset_repeat_length():
    dataset = type(
        "RepeatedDatasetWithBuckets",
        (),
        {
            "data": [
                {"bucket": "landscape"},
                {"bucket": "portrait"},
            ],
            "__len__": lambda self: 4,
        },
    )()

    indices = list(OrientationBucketSampler(dataset, shuffle=False))

    assert indices == [0, 2, 1, 3]


def test_video_parser_exposes_orientation_bucket_switch():
    parsers = load_parsers_module()
    parser = parsers.add_video_size_config(argparse.ArgumentParser())

    args = parser.parse_args(
        [
            "--height",
            "480",
            "--width",
            "832",
            "--enable_orientation_buckets",
        ]
    )

    assert args.enable_orientation_buckets is True
