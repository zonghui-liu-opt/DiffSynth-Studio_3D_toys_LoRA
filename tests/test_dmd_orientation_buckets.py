import csv
import importlib
import sys
import types
from pathlib import Path

from PIL import Image

from tools.prepare_dmd_dataset_csv import convert_metadata_to_turbo_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
TURBO_ROOT = REPO_ROOT / "third_party/wan22_turbo"


def _load_turbo_dataset(monkeypatch):
    monkeypatch.syspath_prepend(str(TURBO_ROOT))
    monkeypatch.setitem(sys.modules, "lmdb", types.SimpleNamespace(open=lambda *args, **kwargs: None))
    monkeypatch.setitem(sys.modules, "decord", types.SimpleNamespace(VideoReader=None))

    def fake_imread(path, flags=None):
        return __import__("numpy").array(Image.open(path).convert("RGB"))

    fake_cv2 = types.SimpleNamespace(
        IMREAD_COLOR=1,
        COLOR_BGR2RGB=0,
        imread=fake_imread,
        cvtColor=lambda image, code: image,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    sys.modules.pop("utils.dataset", None)
    return importlib.import_module("utils.dataset")


def _write_frame_dir(path: Path, size: tuple[int, int], frames: int = 1):
    path.mkdir()
    for index in range(frames):
        Image.new("RGB", size, "red").save(path / f"{index:04d}.png")


def test_dmd_csv_preserves_height_width_and_bucket_metadata(tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "landscape.mp4").write_bytes(b"fake")
    (dataset_root / "portrait.mp4").write_bytes(b"fake")
    metadata_path = dataset_root / "metadata_fixed.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video", "prompt", "num_frames", "height", "width", "bucket"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video": "landscape.mp4",
                "prompt": "wide figurine",
                "num_frames": "121",
                "height": "480",
                "width": "832",
                "bucket": "landscape",
            }
        )
        writer.writerow(
            {
                "video": "portrait.mp4",
                "prompt": "tall figurine",
                "num_frames": "121",
                "height": "832",
                "width": "480",
                "bucket": "portrait",
            }
        )

    output_path = tmp_path / "dmd.csv"
    convert_metadata_to_turbo_csv(
        dataset_root=dataset_root,
        metadata_path=metadata_path,
        output_path=output_path,
        default_num_frames=121,
    )

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))

    assert rows[0]["height"] == "480"
    assert rows[0]["width"] == "832"
    assert rows[0]["bucket"] == "landscape"
    assert rows[1]["height"] == "832"
    assert rows[1]["width"] == "480"
    assert rows[1]["bucket"] == "portrait"


def test_dmd_dataset_resizes_each_sample_to_bucket_resolution(tmp_path, monkeypatch):
    dataset_mod = _load_turbo_dataset(monkeypatch)
    landscape_dir = tmp_path / "landscape"
    portrait_dir = tmp_path / "portrait"
    _write_frame_dir(landscape_dir, size=(16, 8))
    _write_frame_dir(portrait_dir, size=(8, 16))
    csv_path = tmp_path / "dmd.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "text", "num_frames", "height", "width", "bucket"])
        writer.writeheader()
        writer.writerow(
            {
                "path": str(landscape_dir),
                "text": "wide",
                "num_frames": "1",
                "height": "4",
                "width": "8",
                "bucket": "landscape",
            }
        )
        writer.writerow(
            {
                "path": str(portrait_dir),
                "text": "tall",
                "num_frames": "1",
                "height": "8",
                "width": "4",
                "bucket": "portrait",
            }
        )

    dataset = dataset_mod.ODERegressionCSVDataset(
        csv_path,
        num_frames=1,
        h=4,
        w=8,
        enable_orientation_buckets=True,
    )

    landscape = dataset[0]
    portrait = dataset[1]

    assert tuple(landscape["video"].shape) == (3, 1, 4, 8)
    assert landscape["height"] == 4
    assert landscape["width"] == 8
    assert landscape["bucket"] == "landscape"
    assert tuple(portrait["video"].shape) == (3, 1, 8, 4)
    assert portrait["height"] == 8
    assert portrait["width"] == 4
    assert portrait["bucket"] == "portrait"


def test_dmd_bucket_sampler_groups_each_rank_by_bucket(monkeypatch):
    dataset_mod = _load_turbo_dataset(monkeypatch)
    dataset = type(
        "DmdDataset",
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

    rank0 = list(dataset_mod.BucketOffsetDistributedSampler(dataset, gpu_num=2, rank=0, shuffle=False))
    rank1 = list(dataset_mod.BucketOffsetDistributedSampler(dataset, gpu_num=2, rank=1, shuffle=False))

    assert rank0 == [0, 1]
    assert rank1 == [2, 3]
    assert dataset.data[rank0[0]]["bucket"] == dataset.data[rank1[0]]["bucket"] == "landscape"
    assert dataset.data[rank0[1]]["bucket"] == dataset.data[rank1[1]]["bucket"] == "portrait"


def test_dmd_bucket_sampler_keeps_local_batches_in_one_bucket(monkeypatch):
    dataset_mod = _load_turbo_dataset(monkeypatch)
    dataset = type(
        "DmdDataset",
        (),
        {
            "data": [
                {"bucket": "landscape"},
                {"bucket": "portrait"},
                {"bucket": "landscape"},
                {"bucket": "portrait"},
                {"bucket": "landscape"},
                {"bucket": "portrait"},
                {"bucket": "landscape"},
                {"bucket": "portrait"},
            ],
            "__len__": lambda self: 8,
        },
    )()

    rank0 = list(dataset_mod.BucketOffsetDistributedSampler(dataset, gpu_num=2, rank=0, batch_size=2, shuffle=False))
    rank1 = list(dataset_mod.BucketOffsetDistributedSampler(dataset, gpu_num=2, rank=1, batch_size=2, shuffle=False))

    for rank_indices in (rank0, rank1):
        for offset in range(0, len(rank_indices), 2):
            buckets = {dataset.data[index]["bucket"] for index in rank_indices[offset : offset + 2]}
            assert len(buckets) == 1


def test_dmd_trainer_and_launcher_expose_orientation_bucket_controls():
    trainer = (TURBO_ROOT / "trainer/wan22_distillation.py").read_text(encoding="utf-8")
    launcher = (REPO_ROOT / "train_figurine360_dmd_lora.sh").read_text(encoding="utf-8")

    assert "enable_orientation_buckets" in trainer
    assert "BucketOffsetDistributedSampler" in trainer
    assert "_batch_int(batch, \"height\"" in trainer
    assert "_batch_int(batch, \"width\"" in trainer
    assert "ENABLE_ORIENTATION_BUCKETS=${ENABLE_ORIENTATION_BUCKETS:-1}" in launcher
    assert "--enable_orientation_buckets" in launcher
