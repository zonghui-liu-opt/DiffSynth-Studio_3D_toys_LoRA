"""Stage B validation manifest helpers for figurine360 DMD.

The helpers are intentionally lightweight so they can be unit-tested without
Wan weights or CUDA.  The generated commands are executed on the H100 machine
by ``tools/validate_dmd_stage_b.py --run``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


IMAGE_COLUMNS = ("image", "input_image", "first_frame", "first_frame_path")


def _read_holdout_cases(holdout_csv: Path, limit: int | None = None) -> list[dict]:
    with Path(holdout_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        rows = rows[:limit]

    cases = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt") or row.get("text")
        if not prompt:
            raise ValueError(f"Holdout row {index} is missing prompt/text.")
        image = next((row[column] for column in IMAGE_COLUMNS if row.get(column)), "")
        case_id = row.get("case_id") or row.get("id") or f"case-{index + 1:03d}"
        seed = int(row["seed"]) if row.get("seed") else 1000 + index
        cases.append({"case_id": case_id, "prompt": prompt, "image": image, "seed": seed})
    return cases


def _fewstep_command(
    *,
    runtime_dir: Path,
    config_path: Path,
    output_path: Path,
    prompt: str,
    image: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    timing_json: Path,
    dmd_lora_path: Path | None = None,
    python_bin: str = "python3",
) -> list[str]:
    command = [
        python_bin,
        str(Path(runtime_dir) / "wan2.2_fewstep.py"),
        "--config_path",
        str(config_path),
        "--output_path",
        str(output_path),
        "--prompt",
        prompt,
        "--seed",
        str(seed),
        "--h",
        str(height),
        "--w",
        str(width),
        "--num_frames",
        str(num_frames),
        "--timing_json",
        str(timing_json),
    ]
    if image:
        command.extend(["--image", image])
    if dmd_lora_path is not None:
        command.extend(["--lora_path", str(dmd_lora_path), "--lora_source", "auto"])
    return command


def build_validation_jobs(
    *,
    holdout_csv: str | Path,
    output_dir: str | Path,
    runtime_dir: str | Path,
    teacher_config_path: str | Path,
    student_config_path: str | Path,
    dmd_lora_path: str | Path,
    height: int,
    width: int,
    num_frames: int,
    limit: int | None = None,
    python_bin: str = "python3",
) -> list[dict]:
    output_dir = Path(output_dir)
    video_dir = output_dir / "videos"
    timing_dir = output_dir / "timing"
    cases = _read_holdout_cases(Path(holdout_csv), limit=limit)
    jobs = []

    for case in cases:
        for role, config_path, lora_path in (
            ("teacher", Path(teacher_config_path), None),
            ("student", Path(student_config_path), Path(dmd_lora_path)),
        ):
            output_path = video_dir / f"{case['case_id']}_{role}.mp4"
            timing_json = timing_dir / f"{case['case_id']}_{role}.json"
            jobs.append(
                {
                    "case_id": case["case_id"],
                    "role": role,
                    "prompt": case["prompt"],
                    "image": case["image"],
                    "seed": case["seed"],
                    "output_path": str(output_path),
                    "timing_json": str(timing_json),
                    "command": _fewstep_command(
                        runtime_dir=Path(runtime_dir),
                        config_path=config_path,
                        output_path=output_path,
                        prompt=case["prompt"],
                        image=case["image"],
                        seed=case["seed"],
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        timing_json=timing_json,
                        dmd_lora_path=lora_path,
                        python_bin=python_bin,
                    ),
                }
            )
    return jobs


def build_comparisons(jobs: Iterable[dict]) -> list[dict]:
    by_case = {}
    for job in jobs:
        by_case.setdefault(job["case_id"], {})[job["role"]] = job

    comparisons = []
    for case_id, roles in sorted(by_case.items()):
        if "teacher" not in roles or "student" not in roles:
            continue
        comparisons.append(
            {
                "case_id": case_id,
                "teacher_video": roles["teacher"]["output_path"],
                "student_video": roles["student"]["output_path"],
                "teacher_timing_json": roles["teacher"]["timing_json"],
                "student_timing_json": roles["student"]["timing_json"],
            }
        )
    return comparisons


def write_validation_manifest(jobs: list[dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "jobs": jobs,
        "comparisons": build_comparisons(jobs),
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
