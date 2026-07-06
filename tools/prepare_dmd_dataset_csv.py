#!/usr/bin/env python3
"""Convert figurine metadata into the CSV schema expected by the Turbo DMD trainer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        return "\t" if "\t" in sample.splitlines()[0] else ","


def _resolve_data_path(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def _has_value(value) -> bool:
    return value is not None and str(value) != ""


def _bucket_from_height_width(height: str, width: str) -> str:
    return "landscape" if int(float(width)) >= int(float(height)) else "portrait"


def convert_metadata_to_turbo_csv(
    *,
    dataset_root: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
    default_num_frames: int,
) -> dict:
    dataset_root = Path(dataset_root).resolve()
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    delimiter = detect_delimiter(metadata_path)
    rows_out = []
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            video = row.get("video") or row.get("path")
            text = row.get("prompt") or row.get("text") or ""
            if not video:
                raise ValueError("Metadata must contain `video` or `path`.")
            num_frames = row.get("num_frames") or str(default_num_frames)
            output_row = {
                "path": str(_resolve_data_path(dataset_root, video)),
                "text": text,
                "num_frames": str(int(float(num_frames))),
            }
            height = row.get("height")
            width = row.get("width")
            bucket = row.get("bucket")
            if _has_value(height) and _has_value(width):
                output_row["height"] = str(int(float(height)))
                output_row["width"] = str(int(float(width)))
                output_row["bucket"] = bucket if _has_value(bucket) else _bucket_from_height_width(height, width)
            elif _has_value(bucket):
                output_row["height"] = ""
                output_row["width"] = ""
                output_row["bucket"] = str(bucket)
            rows_out.append(output_row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "text", "num_frames"]
    if any(any(key in row for key in ("height", "width", "bucket")) for row in rows_out):
        fieldnames += ["height", "width", "bucket"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    return {"rows": len(rows_out), "output_path": str(output_path)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--default_num_frames", type=int, default=121)
    return parser.parse_args()


def main():
    args = parse_args()
    report = convert_metadata_to_turbo_csv(**vars(args))
    print(report)


if __name__ == "__main__":
    main()
