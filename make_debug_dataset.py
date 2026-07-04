import argparse
import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def color_for_index(index):
    palette = [
        (230, 79, 70),
        (55, 133, 230),
        (74, 170, 104),
        (238, 184, 73),
        (169, 103, 214),
    ]
    return palette[index % len(palette)]


def render_frame(width, height, frame_id, num_frames, sample_id, schema):
    image = Image.new("RGB", (width, height), (238, 240, 242))
    draw = ImageDraw.Draw(image)
    cx, cy = width // 2, height // 2
    angle = 2 * math.pi * frame_id / max(1, num_frames - 1)
    base_color = color_for_index(sample_id)

    if schema == "two_col":
        apparent_width = int(width * (0.18 + 0.28 * abs(math.cos(angle))))
        object_height = int(height * 0.62)
        x0 = cx - apparent_width // 2
        x1 = cx + apparent_width // 2
        y0 = cy - object_height // 2
        y1 = cy + object_height // 2
        shadow = (max(0, base_color[0] - 45), max(0, base_color[1] - 45), max(0, base_color[2] - 45))
        highlight = tuple(min(255, value + 38) for value in base_color)
        draw.ellipse((x0, y0, x1, y1), fill=base_color, outline=(40, 44, 52), width=2)
        if math.cos(angle) >= 0:
            draw.rectangle((cx, y0 + 3, x1 - 3, y1 - 3), fill=highlight)
        else:
            draw.rectangle((x0 + 3, y0 + 3, cx, y1 - 3), fill=shadow)
        foot_w = max(6, int(width * 0.18))
        draw.rounded_rectangle(
            (cx - foot_w, y1 - 4, cx + foot_w, y1 + max(4, height // 18)),
            radius=3,
            fill=(74, 78, 86),
        )
    else:
        radius = max(5, min(width, height) // 8)
        offset = int(math.sin(angle) * width * 0.22)
        draw.ellipse(
            (cx + offset - radius, cy - radius, cx + offset + radius, cy + radius),
            fill=base_color,
            outline=(40, 44, 52),
            width=2,
        )
        draw.polygon(
            [
                (cx + offset - radius // 2, cy - radius),
                (cx + offset, cy - radius - max(5, height // 10)),
                (cx + offset + radius // 2, cy - radius),
            ],
            fill=(255, 255, 255),
            outline=(40, 44, 52),
        )
    return image


def write_video(path, frames, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(frame) for frame in frames]
    imageio.mimsave(path, arrays, fps=fps, macro_block_size=None)


def write_metadata(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(output_dir, schema, num_samples, height, width, num_frames, fps, with_bad_samples):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    images_dir = output_dir / f"images_{height}x{width}"

    if schema == "three_col":
        images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample_id in range(num_samples):
        frames = [render_frame(width, height, frame_id, num_frames, sample_id, schema) for frame_id in range(num_frames)]
        video_rel = f"videos/sample_{sample_id:03d}.mp4"
        write_video(output_dir / video_rel, frames, fps)
        prompt = (
            f"手办360度水平旋转展示，调试色块样本 {sample_id:03d}"
            if schema == "two_col"
            else f"debug cat action sample {sample_id:03d}"
        )
        row = {"video": video_rel, "prompt": prompt}
        if schema == "three_col":
            image_rel = f"images_{height}x{width}/sample_{sample_id:03d}.jpg"
            frames[0].save(output_dir / image_rel, quality=95)
            row["input_image"] = image_rel
        rows.append(row)

    if with_bad_samples:
        short_frames = [
            render_frame(width, height, frame_id, max(1, min(3, num_frames - 1)), 99, schema)
            for frame_id in range(max(1, min(3, num_frames - 1)))
        ]
        short_rel = "videos/bad_short.mp4"
        write_video(output_dir / short_rel, short_frames, fps)
        short_row = {"video": short_rel, "prompt": "bad sample: insufficient frames"}
        missing_row = {"video": "videos/missing.mp4", "prompt": "bad sample: missing video"}
        if schema == "three_col":
            short_image_rel = f"images_{height}x{width}/bad_short.jpg"
            short_frames[0].save(output_dir / short_image_rel, quality=95)
            short_row["input_image"] = short_image_rel
            missing_row["input_image"] = short_image_rel
        rows.extend([short_row, missing_row])

    fieldnames = ["video", "prompt"] if schema == "two_col" else ["video", "prompt", "input_image"]
    write_metadata(output_dir / "metadata.csv", fieldnames, rows)
    return output_dir / "metadata.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a tiny Wan TI2V debug dataset.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--schema", choices=("three_col", "two_col"), default="three_col")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--with_bad_samples", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_path = generate_dataset(
        output_dir=args.output_dir,
        schema=args.schema,
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fps=args.fps,
        with_bad_samples=args.with_bad_samples,
    )
    print(f"metadata_path: {metadata_path}")


if __name__ == "__main__":
    main()
