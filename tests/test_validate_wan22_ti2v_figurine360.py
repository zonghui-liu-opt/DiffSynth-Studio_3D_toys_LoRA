import importlib.util
from pathlib import Path

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
