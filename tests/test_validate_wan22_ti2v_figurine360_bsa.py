import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from PIL import Image


DIRECT_DISTILL = (
    Path(__file__).parents[1]
    / "examples/wanvideo/model_training/special/direct_distill"
)
SCRIPT = DIRECT_DISTILL / "validate_wan22_ti2v_figurine360_bsa.py"
sys.path.insert(0, str(DIRECT_DISTILL))
SPEC = importlib.util.spec_from_file_location(
    "validate_wan22_ti2v_figurine360_bsa", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _latents(value):
    latents = torch.zeros(1, 1, 2, 1, 1, dtype=torch.float32)
    latents[:, :, 1:] = value
    return latents


def _sample(prompt="sample"):
    return {
        "prompt": prompt,
        "seed": 1,
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "teacher_steps": 50,
        "teacher_cfg": 5.0,
        "teacher_shift": 5.0,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
    }


def test_three_way_output_uses_dense_student_and_target_sparse_bsa(
    tmp_path, monkeypatch
):
    events = []

    def fake_render(_pipe, _sample, _output_dir, name, _latents, fps):
        events.append(("render", name, fps))
        return [Image.new("RGB", (12, 8)) for _ in range(2)]

    def fake_save_video(frames, path, fps, quality):
        events.append(("video", Path(path).name, fps, quality, frames[0].size))

    monkeypatch.setattr(MODULE, "render_result", fake_render)
    monkeypatch.setattr(MODULE, "save_video", fake_save_video)
    args = SimpleNamespace(
        fps=12,
        bsa_checkpoint=str(tmp_path / "checkpoint-step-200"),
    )

    report = MODULE._write_sample_outputs(
        object(),
        args,
        _sample(),
        tmp_path,
        _latents(0),
        _latents(1),
        _latents(2),
        _latents(3),
        4,
        4,
        4,
        SimpleNamespace(target_sparsity=0.8),
        "manifest-sha256",
        {"injected_layers": 30},
        {"effective_sparsity": 0.8},
    )

    assert [event[1] for event in events[:4]] == [
        "teacher",
        "dense_warmstart",
        "joint_dense",
        "joint_sparse",
    ]
    assert events[-1] == (
        "video",
        "teacher_vs_student_vs_student_bsa.mp4",
        12,
        5,
        (36, 8),
    )
    assert report["comparison"] == {
        "teacher": "teacher",
        "student": "dense_warmstart",
        "student_bsa": "joint_sparse",
        "student_bsa_sparsity": 0.8,
        "video": "teacher_vs_student_vs_student_bsa.mp4",
    }
    assert report["forward_counts"] == {
        "dense_warmstart": 4,
        "joint_dense": 4,
        "joint_sparse": 4,
    }
    assert report["teacher"] == {
        "steps": 50,
        "cfg_scale": 5.0,
        "sigma_shift": 5.0,
    }
    assert report["bsa_checkpoint_manifest_sha256"] == "manifest-sha256"
    assert json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))[
        "comparison"
    ] == report["comparison"]
    assert not (tmp_path / "validation.json.tmp").exists()


def test_validation_jobs_load_two_pipelines_and_reuse_each_across_samples(
    tmp_path, monkeypatch
):
    events = []
    load_count = 0

    class Pipe:
        def __init__(self, name):
            self.name = name
            self.dit = object()

    def fake_load(_args):
        nonlocal load_count
        load_count += 1
        pipe = Pipe("dense" if load_count == 1 else "joint")
        events.append(f"load:{pipe.name}")
        return pipe

    def fake_teacher(pipe, sample):
        assert pipe.name == "dense"
        events.append(f"teacher:{sample['prompt']}")
        return _latents(0)

    def fake_fuse(pipe, _path, _label):
        events.append(f"fuse:{pipe.name}")

    def fake_four_step(pipe, sample, bsa_context=None):
        if pipe.name == "dense":
            assert bsa_context is None
            branch, value = "student", 1
        elif bsa_context.sparsity == 0:
            branch, value = "joint_dense", 2
        else:
            assert bsa_context.sparsity == pytest.approx(0.8)
            branch, value = "joint_sparse", 3
        events.append(f"{branch}:{sample['prompt']}")
        return _latents(value), 4

    def fake_write(
        _pipe,
        _args,
        sample,
        output_dir,
        teacher,
        dense,
        joint_dense,
        joint_sparse,
        *rest,
    ):
        assert [tensor[:, :, 1:].item() for tensor in (teacher, dense, joint_dense, joint_sparse)] == [
            0,
            1,
            2,
            3,
        ]
        events.append(f"write:{sample['prompt']}")
        (output_dir / "validation.json").write_text("{}\n", encoding="utf-8")
        return {"prompt": sample["prompt"]}

    checkpoint = {
        "manifest": {
            "block_size": [4, 3, 6],
            "target_sparsity": 0.8,
            "gate_type": "low_rank_dynamic",
            "gate_granularity": "block",
            "gate_rank": 32,
            "gate_alpha": 32,
        },
        "direct_distill_lora": tmp_path / "joint.safetensors",
        "bsa_adapter": tmp_path / "bsa.safetensors",
    }
    monkeypatch.setattr(MODULE, "resolve_bsa_checkpoint", lambda _path: checkpoint)
    monkeypatch.setattr(
        MODULE,
        "validate_wan_bsa_provenance",
        lambda *_args, **_kwargs: events.append("provenance"),
    )
    monkeypatch.setattr(MODULE, "load_base_and_figurine", fake_load)
    monkeypatch.setattr(MODULE, "run_teacher", fake_teacher)
    monkeypatch.setattr(MODULE, "fuse_lora", fake_fuse)
    monkeypatch.setattr(MODULE, "run_four_step", fake_four_step)
    monkeypatch.setattr(
        MODULE,
        "inject_wan_bsa",
        lambda *_args, **_kwargs: events.append("inject") or {"injected_layers": 30},
    )
    monkeypatch.setattr(
        MODULE,
        "load_wan_bsa_adapter",
        lambda *_args, **_kwargs: events.append("adapter"),
    )
    monkeypatch.setattr(MODULE, "collect_wan_bsa_runtime_info", lambda _dit: {})
    monkeypatch.setattr(MODULE, "_write_sample_outputs", fake_write)
    monkeypatch.setattr(
        MODULE, "release_device_memory", lambda: events.append("release")
    )
    args = SimpleNamespace(
        bsa_checkpoint=str(tmp_path / "checkpoint"),
        dense_warmstart_lora=str(tmp_path / "dense.safetensors"),
        figurine360_lora=str(tmp_path / "figurine.safetensors"),
    )

    def jobs():
        for index in range(2):
            yield index, _sample(f"sample-{index}"), tmp_path / f"sample-{index}"

    reports = MODULE._validate_jobs(args, jobs)

    assert list(reports) == [0, 1]
    assert events == [
        "provenance",
        "load:dense",
        "teacher:sample-0",
        "teacher:sample-1",
        "fuse:dense",
        "student:sample-0",
        "student:sample-1",
        "release",
        "load:joint",
        "fuse:joint",
        "inject",
        "adapter",
        "joint_dense:sample-0",
        "joint_sparse:sample-0",
        "write:sample-0",
        "joint_dense:sample-1",
        "joint_sparse:sample-1",
        "write:sample-1",
        "release",
    ]


def test_batch_skip_existing_avoids_model_loading(tmp_path, monkeypatch):
    image = tmp_path / "first.png"
    Image.new("RGB", (4, 4)).save(image)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(f"input_image,prompt\n{image},test\n", encoding="utf-8")
    output = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint"
    sample_dir = output / "sample-0"
    sample_dir.mkdir(parents=True)
    (sample_dir / MODULE.COMPARISON_VIDEO_NAME).touch()
    (sample_dir / "validation.json").write_text(
        json.dumps(
            {
                "comparison": {"video": MODULE.COMPARISON_VIDEO_NAME},
                "bsa_checkpoint": str(checkpoint.resolve()),
                "bsa_checkpoint_manifest_sha256": "manifest-sha256",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        metadata_path=str(metadata),
        output_dir=str(output),
        batch_start=0,
        batch_end=1,
        skip_existing=True,
        bsa_checkpoint=str(checkpoint),
    )
    preflight = []
    monkeypatch.setattr(
        MODULE,
        "_resolve_validation_checkpoint",
        lambda _args: (
            preflight.append("validated")
            or ({"manifest_sha256": "manifest-sha256"}, object())
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_jobs",
        lambda *_args, **_kwargs: pytest.fail("completed samples must not load models"),
    )

    summary = MODULE.validate_batch(args)

    assert summary["selected"] == 0
    assert summary["skipped"] == [0]
    assert summary["model_load_count"] == 0
    assert preflight == ["validated"]


@pytest.mark.parametrize(
    "report, create_video",
    [
        ("{truncated", True),
        (json.dumps({"comparison": {"video": "wrong.mp4"}}), True),
        (
            json.dumps(
                {"comparison": {"video": "teacher_vs_student_vs_student_bsa.mp4"}}
            ),
            False,
        ),
    ],
)
def test_batch_completion_requires_valid_report_and_comparison_video(
    tmp_path, report, create_video
):
    sample_dir = tmp_path / "sample-0"
    sample_dir.mkdir()
    (sample_dir / "validation.json").write_text(report, encoding="utf-8")
    if create_video:
        (sample_dir / MODULE.COMPARISON_VIDEO_NAME).touch()

    assert not MODULE._is_complete_bsa_sample(
        tmp_path, 0, tmp_path / "checkpoint", "manifest-sha256"
    )


def test_batch_completion_rejects_a_different_checkpoint(tmp_path):
    sample_dir = tmp_path / "sample-0"
    sample_dir.mkdir()
    (sample_dir / MODULE.COMPARISON_VIDEO_NAME).touch()
    (sample_dir / "validation.json").write_text(
        json.dumps(
            {
                "comparison": {"video": MODULE.COMPARISON_VIDEO_NAME},
                "bsa_checkpoint": str((tmp_path / "checkpoint-old").resolve()),
                "bsa_checkpoint_manifest_sha256": "manifest-sha256",
            }
        ),
        encoding="utf-8",
    )

    assert not MODULE._is_complete_bsa_sample(
        tmp_path, 0, tmp_path / "checkpoint-new", "manifest-sha256"
    )


def test_forced_batch_rerun_removes_stale_completion_marker(tmp_path, monkeypatch):
    image = tmp_path / "first.png"
    Image.new("RGB", (4, 4)).save(image)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(f"input_image,prompt\n{image},test\n", encoding="utf-8")
    output = tmp_path / "output"
    marker = output / "sample-0" / "validation.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"old": true}\n', encoding="utf-8")
    args = SimpleNamespace(
        metadata_path=str(metadata),
        output_dir=str(output),
        batch_start=0,
        batch_end=1,
        skip_existing=False,
        bsa_checkpoint=str(tmp_path / "checkpoint"),
        dataset_base_path=None,
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
    monkeypatch.setattr(
        MODULE,
        "_resolve_validation_checkpoint",
        lambda _args: ({"manifest_sha256": "manifest-sha256"}, object()),
    )

    def fake_validate(_args, jobs, on_complete, **_kwargs):
        assert not marker.exists()
        index, _sample_data, _sample_dir = next(jobs())
        marker.write_text("{}\n", encoding="utf-8")
        on_complete(index, {})

    monkeypatch.setattr(MODULE, "_validate_jobs", fake_validate)

    summary = MODULE.validate_batch(args)

    assert summary["completed"] == [0]
    assert marker.read_text(encoding="utf-8") == "{}\n"


def test_single_rerun_removes_stale_completion_marker(tmp_path, monkeypatch):
    output = tmp_path / "sample-0"
    output.mkdir()
    marker = output / "validation.json"
    marker.write_text('{"old": true}\n', encoding="utf-8")
    args = SimpleNamespace(output_dir=str(output))
    monkeypatch.setattr(MODULE, "load_single_sample", lambda _args: _sample())

    def fake_validate(_args, jobs):
        assert not marker.exists()
        key, _sample_data, sample_dir = next(jobs())
        assert key == 0 and sample_dir == output
        marker.write_text("{}\n", encoding="utf-8")
        return {0: {"complete": True}}

    monkeypatch.setattr(MODULE, "_validate_jobs", fake_validate)

    assert MODULE.validate(args) == {"complete": True}
    assert marker.read_text(encoding="utf-8") == "{}\n"
