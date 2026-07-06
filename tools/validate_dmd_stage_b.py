#!/usr/bin/env python3
"""Build and optionally run Stage B teacher/student validation jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmd.stage_b_validation import build_validation_jobs, write_validation_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout_csv", type=Path, required=True, help="CSV with case_id,prompt,image,seed columns.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--runtime_dir", type=Path, default=Path("third_party/wan22_turbo"))
    parser.add_argument("--teacher_config_path", type=Path, required=True)
    parser.add_argument("--student_config_path", type=Path, required=True)
    parser.add_argument("--dmd_lora_path", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--python_bin", default="python3")
    parser.add_argument("--run", action="store_true", help="Execute generated jobs. Default only writes manifest.")
    return parser.parse_args()


def _run_job(job: dict, *, repo_root: Path, runtime_dir: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}:{runtime_dir}:{env.get('PYTHONPATH', '')}"
    Path(job["output_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(job["timing_json"]).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(job["command"], check=True, cwd=runtime_dir, env=env)


def main():
    args = parse_args()
    repo_root = REPO_ROOT
    runtime_dir = args.runtime_dir
    if not runtime_dir.is_absolute():
        runtime_dir = repo_root / runtime_dir

    jobs = build_validation_jobs(
        holdout_csv=args.holdout_csv,
        output_dir=args.output_dir,
        runtime_dir=runtime_dir,
        teacher_config_path=args.teacher_config_path,
        student_config_path=args.student_config_path,
        dmd_lora_path=args.dmd_lora_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        limit=args.limit,
        python_bin=args.python_bin,
    )
    manifest_path = write_validation_manifest(jobs, args.output_dir / "stage_b_validation_manifest.json")
    print(f"Wrote validation manifest: {manifest_path}")

    if args.run:
        for job in jobs:
            print(f"Running {job['case_id']} {job['role']} -> {job['output_path']}")
            _run_job(job, repo_root=repo_root, runtime_dir=runtime_dir)


if __name__ == "__main__":
    main()
