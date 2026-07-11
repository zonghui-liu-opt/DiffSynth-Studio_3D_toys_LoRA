import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image


SCRIPT = (
    Path(__file__).parents[1]
    / "examples/wanvideo/model_training/special/direct_distill/validate_wan22_ti2v_figurine360.py"
)
SPEC = importlib.util.spec_from_file_location("validate_wan22_ti2v_figurine360", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compose_side_by_side_labels_equal_frames():
    teacher = [Image.new("RGB", (12, 8), "red") for _ in range(2)]
    student = [Image.new("RGB", (12, 8), "blue") for _ in range(2)]
    paired = MODULE.compose_side_by_side(teacher, student)
    assert len(paired) == 2
    assert paired[0].size == (24, 8)


def test_compose_side_by_side_rejects_mismatch():
    with pytest.raises(ValueError, match="same non-zero"):
        MODULE.compose_side_by_side([Image.new("RGB", (4, 4))], [])


def test_metadata_prefers_cached_input_image(tmp_path):
    image_path = tmp_path / "first.png"
    Image.new("RGB", (9, 7), "green").save(image_path)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "video,input_image,prompt,seed,height,width,num_frames\n"
        "missing.mp4,first.png,test,1,480,832,49\n",
        encoding="utf-8",
    )
    row, image, source = MODULE.load_sample_from_metadata(metadata)
    assert row["prompt"] == "test"
    assert image.size == (9, 7)
    assert source == image_path.resolve()


def test_model_path_groups_keep_dit_shards_together(tmp_path):
    shards = [tmp_path / f"dit-{index}.safetensors" for index in range(3)]
    text = tmp_path / "text.safetensors"
    for path in [*shards, text]:
        path.touch()
    groups = MODULE.parse_model_path_groups([
        __import__("json").dumps([str(path) for path in shards]),
        str(text),
    ])
    assert groups[0] == tuple(path.resolve() for path in shards)
    assert groups[1] == text.resolve()


def test_latent_validation_rejects_nonfinite_and_mismatch():
    teacher = torch.zeros(1, 2, 3, 4, 5, dtype=torch.float32)
    MODULE.validate_latent_tensor(teacher, "Teacher")
    with pytest.raises(ValueError, match="shape"):
        MODULE.validate_latent_tensor(
            torch.zeros(1, 2, 3, 4, 4), "Student", expected_shape=teacher.shape
        )
    teacher[..., -1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        MODULE.validate_latent_tensor(teacher, "Teacher")


def test_batch_loads_model_once_and_runs_all_teachers_before_students(tmp_path, monkeypatch):
    image_paths = []
    for index in range(2):
        image_path = tmp_path / f"first-{index}.png"
        Image.new("RGB", (9, 7), "green").save(image_path)
        image_paths.append(image_path)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "input_image,prompt\n"
        + "".join(f"{path},prompt-{index}\n" for index, path in enumerate(image_paths)),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"
    args = SimpleNamespace(
        metadata_path=str(metadata),
        output_dir=str(output_dir),
        batch_start=0,
        batch_end=2,
        skip_existing=True,
        dataset_base_path=None,
        student_num_inference_steps=4,
        student_cfg_scale=1,
        student_sigma_shift=5,
        prompt="fallback",
        negative_prompt="negative",
        seed=1,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        teacher_num_inference_steps=50,
        teacher_cfg_scale=5.0,
        teacher_sigma_shift=5.0,
        tiled=True,
    )
    events = []
    pipe = object()

    def fake_load_teacher_pipeline(_args):
        events.append("load")
        return pipe, tmp_path / "direct.safetensors"

    def fake_run_teacher(_pipe, sample):
        events.append(f"teacher:{Path(sample['input_source']).stem}")
        return torch.zeros(1, 2, 3, 4, 5)

    def fake_fuse(_pipe, _path):
        events.append("fuse")

    def fake_finish(_pipe, _args, sample, teacher_latents, sample_dir):
        assert tuple(teacher_latents.shape) == (1, 2, 3, 4, 5)
        events.append(f"student:{Path(sample['input_source']).stem}")
        (sample_dir / "validation.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(MODULE, "load_teacher_pipeline", fake_load_teacher_pipeline)
    monkeypatch.setattr(MODULE, "run_teacher", fake_run_teacher)
    monkeypatch.setattr(MODULE, "fuse_direct_distill_lora", fake_fuse)
    monkeypatch.setattr(MODULE, "finish_student_validation", fake_finish)

    summary = MODULE.validate_batch(args)

    assert events == [
        "load",
        "teacher:first-0",
        "teacher:first-1",
        "fuse",
        "student:first-0",
        "student:first-1",
    ]
    assert summary["model_load_count"] == 1
    assert summary["figurine_lora_fusion_count"] == 1
    assert summary["direct_distill_lora_fusion_count"] == 1
    assert summary["completed"] == [0, 1]


def test_batch_skip_existing_can_avoid_model_loading(tmp_path, monkeypatch):
    image_path = tmp_path / "first.png"
    Image.new("RGB", (9, 7), "green").save(image_path)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        f"input_image,prompt\n{image_path},prompt\n", encoding="utf-8"
    )
    output_dir = tmp_path / "output"
    sample_dir = output_dir / "sample-0"
    sample_dir.mkdir(parents=True)
    (sample_dir / "validation.json").write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        metadata_path=str(metadata),
        output_dir=str(output_dir),
        batch_start=0,
        batch_end=1,
        skip_existing=True,
        student_num_inference_steps=4,
        student_cfg_scale=1,
        student_sigma_shift=5,
    )

    monkeypatch.setattr(
        MODULE,
        "load_teacher_pipeline",
        lambda _args: pytest.fail("model must not load when every sample is skipped"),
    )

    summary = MODULE.validate_batch(args)

    assert summary["selected"] == 0
    assert summary["skipped"] == [0]
    assert summary["model_load_count"] == 0
