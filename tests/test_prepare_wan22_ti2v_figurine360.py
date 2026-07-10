import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from examples.wanvideo.model_training.special.direct_distill import (
    prepare_wan22_ti2v_figurine360 as prepare,
)


class _FakePipe:
    def __init__(self):
        self.torch_dtype = torch.float32
        self.device = "cpu"
        self.vae = SimpleNamespace(z_dim=2, upsampling_factor=16)
        self.model_load_calls = []

    def load_models_to_device(self, names):
        self.model_load_calls.append(tuple(names))


class _FakeTeacherGenerator:
    def __init__(self, first_frame):
        self.first_frame = first_frame
        self.calls = []

    def __call__(self, pipe, image, prompt, seed, config):
        self.calls.append((prompt, seed))
        latents = torch.full((1, 2, 2, 1, 1), float(seed), dtype=pipe.torch_dtype)
        latents[:, :, :1] = self.first_frame
        return latents


def _first_frame_encoder(first_frame):
    def encode(pipe, image, config):
        return first_frame.clone()

    return encode


def _fixture(tmp_path, seeds=(1, 2), verify_determinism=False):
    base_path = tmp_path / "source"
    output_path = tmp_path / "prepared"
    base_path.mkdir()
    (base_path / "object.mp4").write_bytes(b"test video placeholder")
    records = [{"video": "./object.mp4", "prompt": "a figurine rotates on a turntable"}]
    config = prepare.PreparationConfig(
        base_path=base_path,
        output_path=output_path,
        seeds=tuple(seeds),
        height=16,
        width=16,
        num_frames=5,
        validation_fraction=0.5,
        verify_determinism=verify_determinism,
    )
    pipe = _FakePipe()
    first_frame = torch.tensor([1.0, 2.0], dtype=torch.float32).reshape(1, 2, 1, 1, 1)
    reader_calls = []

    def frame_reader(path):
        reader_calls.append(path)
        return Image.new("RGB", (4, 4), color=(10, 20, 30))

    generator = _FakeTeacherGenerator(first_frame)
    return records, config, pipe, first_frame, frame_reader, reader_calls, generator


def _run_fixture(records, config, pipe, first_frame, frame_reader, generator):
    return prepare.prepare_records(
        records,
        pipe,
        config,
        frame_reader=frame_reader,
        first_frame_encoder=_first_frame_encoder(first_frame),
        teacher_generator=generator,
    )


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_safetensors_round_trip_and_first_frame_validation(tmp_path):
    path = tmp_path / "latents.safetensors"
    first_frame = torch.tensor([3.0, 4.0]).reshape(1, 2, 1, 1, 1)
    latents = torch.zeros((1, 2, 2, 1, 1), dtype=torch.float32)
    latents[:, :, :1] = first_frame

    prepare.save_latents_atomic(
        path,
        latents,
        expected_shape=tuple(latents.shape),
        expected_dtype=torch.float32,
        expected_first_frame=first_frame,
        metadata={"sample_id": "sample-a", "seed": "1"},
    )
    result = prepare.validate_latent_file(
        path,
        expected_shape=tuple(latents.shape),
        expected_dtype=torch.float32,
        expected_first_frame=first_frame,
        expected_metadata={"sample_id": "sample-a", "seed": "1"},
    )

    assert result.valid
    assert torch.equal(result.latents, latents)
    wrong_first_frame = first_frame + 1
    assert not prepare.validate_latent_file(
        path, expected_first_frame=wrong_first_frame
    ).valid
    mismatch = prepare.validate_latent_file(
        path, expected_metadata={"sample_id": "sample-b", "seed": "1"}
    )
    assert not mismatch.valid
    assert "metadata mismatch" in mismatch.reason


def test_metadata_generation_and_resume_skip_valid_outputs(tmp_path):
    fixture = _fixture(tmp_path)
    records, config, pipe, first_frame, frame_reader, reader_calls, generator = fixture

    first_summary = _run_fixture(records, config, pipe, first_frame, frame_reader, generator)
    latent_files = sorted((config.output_path / "teacher_latents").glob("*.safetensors"))
    original_bytes = {path.name: path.read_bytes() for path in latent_files}
    first_call_count = len(generator.calls)

    second_summary = _run_fixture(records, config, pipe, first_frame, frame_reader, generator)

    assert first_summary == {
        "generated": 2,
        "skipped": 0,
        "invalid_detected": 0,
        "failed": 0,
        "metadata_rows": 2,
    }
    assert second_summary["generated"] == 0
    assert second_summary["skipped"] == 2
    assert len(generator.calls) == first_call_count
    assert len(reader_calls) == 1
    assert {path.name: path.read_bytes() for path in latent_files} == original_bytes

    rows = _read_csv(config.output_path / "metadata_direct_distill.csv")
    assert len(rows) == 2
    assert {int(row["seed"]) for row in rows} == {1, 2}
    assert {row["num_inference_steps"] for row in rows} == {"4"}
    assert {row["cfg_scale"] for row in rows} == {"1.0"}
    assert {row["sigma_shift"] for row in rows} == {"5.0"}
    assert {row["teacher_num_inference_steps"] for row in rows} == {"50"}
    assert {row["teacher_cfg_scale"] for row in rows} == {"5.0"}
    assert {row["teacher_sigma_shift"] for row in rows} == {"5.0"}
    assert all(row["input_image"].endswith(".png") for row in rows)
    assert all(row["teacher_latent"].endswith(".safetensors") for row in rows)
    with Image.open(config.output_path / rows[0]["input_image"]) as cached_first_frame:
        assert cached_first_frame.size == (config.width, config.height)


def test_corrupt_latent_is_detected_quarantined_and_regenerated(tmp_path):
    fixture = _fixture(tmp_path)
    records, config, pipe, first_frame, frame_reader, _, generator = fixture
    _run_fixture(records, config, pipe, first_frame, frame_reader, generator)

    latent_files = sorted((config.output_path / "teacher_latents").glob("*.safetensors"))
    corrupt_path = latent_files[0]
    corrupt_path.write_bytes(b"not a safetensors file")
    calls_before_repair = len(generator.calls)

    summary = _run_fixture(records, config, pipe, first_frame, frame_reader, generator)

    assert summary["invalid_detected"] == 1
    assert summary["generated"] == 1
    assert summary["skipped"] == 1
    assert len(generator.calls) == calls_before_repair + 1
    assert prepare.validate_latent_file(corrupt_path).valid
    assert list(corrupt_path.parent.glob(f"{corrupt_path.stem}.corrupt-*.safetensors"))
    failure_records = [
        json.loads(line)
        for line in (config.output_path / "failures.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["stage"] == "existing_latent_validation" for record in failure_records)


def test_object_split_never_leaks_across_seeds(tmp_path):
    fixture = _fixture(tmp_path, seeds=(1, 7, 42))
    records, config, pipe, first_frame, frame_reader, _, generator = fixture
    summary = _run_fixture(records, config, pipe, first_frame, frame_reader, generator)

    rows = _read_csv(config.output_path / "metadata_direct_distill.csv")
    train_rows = _read_csv(config.output_path / "metadata_direct_distill_train.csv")
    validation_rows = _read_csv(
        config.output_path / "metadata_direct_distill_validation.csv"
    )
    assert summary["metadata_rows"] == 3
    assert len({row["object_id"] for row in rows}) == 1
    assert len({row["split"] for row in rows}) == 1
    assert len({row["sample_id"] for row in rows}) == 3
    assert {row["sample_id"] for row in train_rows}.isdisjoint(
        {row["sample_id"] for row in validation_rows}
    )
    assert len(train_rows) + len(validation_rows) == len(rows)


def test_stable_ids_and_repeatable_or_json_model_paths():
    assert prepare.stable_object_id("./folder\\video.mp4") == prepare.stable_object_id(
        "folder/video.mp4"
    )
    assert prepare.stable_sample_id("./video.mp4", "prompt", 1) == prepare.stable_sample_id(
        "video.mp4", "prompt", 1
    )
    assert prepare.stable_sample_id("video.mp4", "prompt", 1) != prepare.stable_sample_id(
        "video.mp4", "prompt", 2
    )
    assert prepare.stable_sample_id(
        "video.mp4", "prompt", 1, "teacher-a"
    ) != prepare.stable_sample_id("video.mp4", "prompt", 1, "teacher-b")
    assert prepare.parse_model_path_values(
        ["/models/dit.safetensors", '["/models/text.safetensors", "/models/vae.safetensors"]']
    ) == [
        Path("/models/dit.safetensors"),
        (
            Path("/models/text.safetensors"),
            Path("/models/vae.safetensors"),
        ),
    ]


def test_teacher_fingerprint_changes_with_generation_settings(tmp_path):
    base = prepare.PreparationConfig(base_path=tmp_path, output_path=tmp_path / "a")
    changed = prepare.PreparationConfig(
        base_path=tmp_path,
        output_path=tmp_path / "b",
        teacher_cfg_scale=6.0,
    )
    assert prepare.teacher_generation_fingerprint(base) != prepare.teacher_generation_fingerprint(changed)


def test_held_out_validation_seeds_are_disjoint(tmp_path):
    config = prepare.PreparationConfig(
        base_path=tmp_path,
        output_path=tmp_path / "out",
        seeds=(2, 3, 4),
        validation_seeds=(1,),
    )
    assert prepare.seeds_for_object_split(config, "train") == (2, 3, 4)
    assert prepare.seeds_for_object_split(config, "val") == (1,)
    with pytest.raises(ValueError, match="disjoint"):
        prepare.PreparationConfig(
            base_path=tmp_path,
            output_path=tmp_path / "bad",
            seeds=(1, 2),
            validation_seeds=(1,),
        )


def test_optional_determinism_rerun_calls_teacher_twice(tmp_path):
    fixture = _fixture(tmp_path, seeds=(1,), verify_determinism=True)
    records, config, pipe, first_frame, frame_reader, _, generator = fixture

    summary = _run_fixture(records, config, pipe, first_frame, frame_reader, generator)

    assert summary["generated"] == 1
    assert len(generator.calls) == 2


def test_teacher_generation_requests_latents_with_fixed_teacher_defaults(tmp_path):
    class RecordingPipe:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return torch.zeros((1, 2, 2, 1, 1), dtype=torch.float32)

    config = prepare.PreparationConfig(
        base_path=tmp_path,
        output_path=tmp_path / "out",
        height=16,
        width=16,
        num_frames=5,
    )
    pipe = RecordingPipe()
    output = prepare.generate_teacher_latents(
        pipe,
        Image.new("RGB", (16, 16)),
        "prompt",
        1,
        config,
    )

    assert isinstance(output, torch.Tensor)
    assert pipe.kwargs["return_latents"] is True
    assert pipe.kwargs["seed"] == 1
    assert pipe.kwargs["rand_device"] == "cpu"
    assert pipe.kwargs["num_inference_steps"] == 50
    assert pipe.kwargs["cfg_scale"] == 5.0
    assert pipe.kwargs["sigma_shift"] == 5.0


def test_max_samples_limits_source_rows_before_seed_expansion(tmp_path):
    fixture = _fixture(tmp_path)
    records, config, pipe, first_frame, frame_reader, _, generator = fixture
    (config.base_path / "object-2.mp4").write_bytes(b"second video placeholder")
    records.append({"video": "object-2.mp4", "prompt": "second object"})
    config = replace(config, max_samples=1)

    summary = _run_fixture(records, config, pipe, first_frame, frame_reader, generator)

    assert summary["metadata_rows"] == 2
    assert len(generator.calls) == 2
    rows = _read_csv(config.output_path / "metadata_direct_distill.csv")
    assert {row["prompt"] for row in rows} == {records[0]["prompt"]}


def test_train_and_validation_csvs_are_object_disjoint(tmp_path):
    base_path = tmp_path / "source"
    output_path = tmp_path / "prepared"
    base_path.mkdir()
    selected = {}
    candidate_id = 0
    while set(selected) != {"train", "val"}:
        video = f"object-{candidate_id}.mp4"
        video_path = base_path / video
        video_path.write_bytes(b"video placeholder")
        object_id = prepare.stable_object_id(
            video, prepare.source_file_fingerprint(video_path)
        )
        split = prepare.assign_object_split(object_id, validation_fraction=0.5)
        selected.setdefault(split, video)
        candidate_id += 1
    records = []
    for split, video in selected.items():
        records.append({"video": video, "prompt": f"{split} object"})

    config = prepare.PreparationConfig(
        base_path=base_path,
        output_path=output_path,
        seeds=(1, 2),
        height=16,
        width=16,
        num_frames=5,
        validation_fraction=0.5,
    )
    pipe = _FakePipe()
    first_frame = torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1, 1)
    generator = _FakeTeacherGenerator(first_frame)
    summary = prepare.prepare_records(
        records,
        pipe,
        config,
        frame_reader=lambda path: Image.new("RGB", (16, 16)),
        first_frame_encoder=_first_frame_encoder(first_frame),
        teacher_generator=generator,
    )

    train_rows = _read_csv(output_path / "metadata_direct_distill_train.csv")
    validation_rows = _read_csv(output_path / "metadata_direct_distill_validation.csv")
    train_objects = {row["object_id"] for row in train_rows}
    validation_objects = {row["object_id"] for row in validation_rows}
    assert summary["metadata_rows"] == 4
    assert train_objects
    assert validation_objects
    assert train_objects.isdisjoint(validation_objects)
    assert {row["sample_id"] for row in train_rows}.isdisjoint(
        {row["sample_id"] for row in validation_rows}
    )
    for rows in (train_rows, validation_rows):
        seeds_by_object = {}
        for row in rows:
            seeds_by_object.setdefault(row["object_id"], set()).add(int(row["seed"]))
        assert all(seeds == {1, 2} for seeds in seeds_by_object.values())
