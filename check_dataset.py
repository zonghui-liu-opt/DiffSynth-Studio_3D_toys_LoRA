import argparse
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio

from metrics_utils import tokens_per_sample


BASE_COLUMNS = ("video", "prompt")
THREE_COL_COLUMNS = ("video", "prompt", "input_image")
SCHEMA_COLUMNS = {
    "two_col": BASE_COLUMNS,
    "three_col": THREE_COL_COLUMNS,
}


def detect_delimiter(metadata_path):
    sample = Path(metadata_path).read_text(encoding="utf-8-sig")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        return "\t" if "\t" in first_line else ","


def detect_schema(fieldnames):
    fieldnames = list(fieldnames or [])
    missing = [column for column in BASE_COLUMNS if column not in fieldnames]
    if missing:
        raise ValueError(f"metadata missing required columns: {', '.join(missing)}")
    return "three_col" if "input_image" in fieldnames else "two_col"


def read_metadata_with_schema(metadata_path, delimiter):
    with Path(metadata_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    schema = detect_schema(fieldnames)
    return rows, schema


def read_metadata(metadata_path, delimiter):
    rows, _ = read_metadata_with_schema(metadata_path, delimiter)
    return rows


def orientation_bucket(height, width):
    return "landscape" if width >= height else "portrait"


def bucket_resolution(bucket, landscape_height, landscape_width):
    if bucket == "portrait":
        return landscape_width, landscape_height
    return landscape_height, landscape_width


def fixed_columns_for_schema(schema):
    return SCHEMA_COLUMNS[schema] + ("height", "width", "bucket")


def write_fixed_metadata(metadata_path, rows, schema):
    fixed_path = Path(metadata_path).with_name("metadata_fixed.csv")
    with fixed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fixed_columns_for_schema(schema)))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fixed_columns_for_schema(schema)})
    return fixed_path


def inspect_video_imageio(video_path):
    reader = imageio.get_reader(video_path)
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 0.0) or 0.0)
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = sum(1 for _ in reader)
            reader.close()
            reader = imageio.get_reader(video_path)
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
        return {"width": width, "height": height, "frames": frame_count, "fps": fps}
    finally:
        reader.close()


def inspect_video_cv2(video_path):
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("cv2 is not available") from exc
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cv2 failed to open {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return {"width": width, "height": height, "frames": frames, "fps": fps}
    finally:
        capture.release()


def inspect_video_ffprobe(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den or 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream.get("nb_frames") or 0),
        "fps": fps,
    }


def inspect_video(video_path):
    errors = []
    for inspector in (inspect_video_imageio, inspect_video_cv2, inspect_video_ffprobe):
        try:
            return inspector(video_path)
        except Exception as exc:
            errors.append(f"{inspector.__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def decode_first_frame(video_path):
    reader = imageio.get_reader(video_path)
    try:
        reader.get_data(0)
    finally:
        reader.close()


def validate_dataset(dataset_root, metadata_path, height, width, num_frames):
    dataset_root = Path(dataset_root)
    metadata_path = Path(metadata_path)
    delimiter = detect_delimiter(metadata_path)
    rows, schema = read_metadata_with_schema(metadata_path, delimiter)

    bad_rows = []
    stats = []
    fixed_rows = []
    for row_id, row in enumerate(rows):
        reasons = []
        fixed_row = {column: row[column] for column in SCHEMA_COLUMNS[schema]}
        video_path = dataset_root / row["video"]
        video_stats = None
        if not video_path.exists():
            reasons.append("missing_video")
        else:
            try:
                video_stats = inspect_video(video_path)
                stats.append(video_stats)
                if video_stats["frames"] < num_frames:
                    reasons.append("insufficient_frames")
            except Exception as exc:
                reasons.append(f"video_read_error:{exc}")
            if schema == "two_col" and video_stats is not None:
                try:
                    decode_first_frame(video_path)
                except Exception as exc:
                    reasons.append(f"first_frame_decode_error:{exc}")
        if schema == "three_col":
            image_path = dataset_root / row["input_image"]
            if not image_path.exists():
                reasons.append("missing_input_image")
        if video_stats is not None:
            bucket = orientation_bucket(video_stats["height"], video_stats["width"])
            target_height, target_width = bucket_resolution(bucket, height, width)
            fixed_row.update({"height": target_height, "width": target_width, "bucket": bucket})
        fixed_rows.append(fixed_row)
        if reasons:
            bad_rows.append({"row": row_id, "video": row["video"], "reasons": reasons})

    fixed_path = write_fixed_metadata(metadata_path, fixed_rows, schema)
    resolution_counts = Counter((item["height"], item["width"]) for item in stats)
    frame_counts = Counter(item["frames"] for item in stats)
    fps_counts = Counter(round(item["fps"], 3) for item in stats)
    bucket_counts = Counter(orientation_bucket(item["height"], item["width"]) for item in stats)
    token_count = tokens_per_sample(num_frames=num_frames, height=height, width=width)
    return {
        "schema": schema,
        "delimiter": "tab" if delimiter == "\t" else "comma",
        "fixed_path": str(fixed_path),
        "total_rows": len(rows),
        "good_samples": len(rows) - len(bad_rows),
        "bad_samples": len(bad_rows),
        "bad_rows": bad_rows,
        "resolution_counts": dict(resolution_counts),
        "bucket_counts": dict(bucket_counts),
        "frame_counts": dict(frame_counts),
        "fps_counts": dict(fps_counts),
        "recommended_height": height,
        "recommended_width": width,
        "tokens_per_video": token_count,
        "tokens_total": token_count * max(0, len(rows) - len(bad_rows)),
    }


def print_summary(summary):
    reason_counts = Counter(reason for row in summary["bad_rows"] for reason in row["reasons"])
    print(f"schema: {summary['schema']}")
    print(f"delimiter: {summary['delimiter']}")
    print(f"metadata_fixed_path: {summary['fixed_path']}")
    print(f"total_rows: {summary['total_rows']}")
    print(f"good_samples: {summary['good_samples']}")
    print(f"bad_samples: {summary['bad_samples']}")
    for reason, count in sorted(reason_counts.items()):
        print(f"{reason}: {count}")
    print(f"resolution_counts: {summary['resolution_counts']}")
    print(f"bucket_counts: {summary['bucket_counts']}")
    print(f"frame_counts: {summary['frame_counts']}")
    print(f"fps_counts: {summary['fps_counts']}")
    print(f"recommended_height: {summary['recommended_height']}")
    print(f"recommended_width: {summary['recommended_width']}")
    print(f"tokens_per_video: {summary['tokens_per_video']}")
    print(f"tokens_total: {summary['tokens_total']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Check Wan TI2V dataset metadata and media files.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = validate_dataset(args.dataset_root, args.metadata_path, args.height, args.width, args.num_frames)
    print_summary(summary)


if __name__ == "__main__":
    main()
