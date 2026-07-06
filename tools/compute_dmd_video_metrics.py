#!/usr/bin/env python3
"""Compute Stage B teacher/student validation metrics from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmd.video_metrics import compute_basic_video_metrics, compute_video_diff_metrics, read_video, write_metrics_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=64)
    return parser.parse_args()


def _read_optional_json(path: str | Path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = []
    for comparison in manifest.get("comparisons", []):
        teacher_video = read_video(comparison["teacher_video"])
        student_video = read_video(comparison["student_video"])
        metrics = compute_basic_video_metrics(teacher_video, student_video, bins=args.bins)
        metrics.update(compute_video_diff_metrics(teacher_video, student_video))
        rows.append(
            {
                "case_id": comparison["case_id"],
                **metrics,
                "teacher_video": comparison["teacher_video"],
                "student_video": comparison["student_video"],
                "teacher_timing": _read_optional_json(comparison["teacher_timing_json"]),
                "student_timing": _read_optional_json(comparison["student_timing_json"]),
            }
        )
    output = {"version": 1, "metrics": rows}
    write_metrics_json(output, args.output_json)
    print(f"Wrote Stage B metrics: {args.output_json}")


if __name__ == "__main__":
    main()
