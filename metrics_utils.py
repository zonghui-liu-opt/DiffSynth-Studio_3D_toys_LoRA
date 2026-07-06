import json
from pathlib import Path


REQUIRED_METRIC_FIELDS = (
    "step",
    "epoch",
    "loss",
    "step_time_sec",
    "tokens_per_sample",
    "samples_per_step",
    "lr",
)


def tokens_per_sample(num_frames, height, width):
    """Return DiT tokens/video for Wan2.2-TI2V-5B.

    Wan2.2-TI2V-5B uses VAE compression 4x16x16, then DiT patchify makes the
    effective compression 4x32x32:
    latent_frames = (num_frames - 1) // 4 + 1
    tokens = latent_frames * (height // 32) * (width // 32)
    """
    if num_frames < 1 or (num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must satisfy (num_frames - 1) % 4 == 0")
    if height < 32 or height % 32 != 0:
        raise ValueError("height must be a positive multiple of 32")
    if width < 32 or width % 32 != 0:
        raise ValueError("width must be a positive multiple of 32")
    latent_frames = (num_frames - 1) // 4 + 1
    return latent_frames * (height // 32) * (width // 32)


def tokens_per_step(record):
    return int(record["tokens_per_sample"]) * int(record["samples_per_step"])


class MetricsWriter:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        missing_fields = [field for field in REQUIRED_METRIC_FIELDS if field not in record]
        if missing_fields:
            raise ValueError(f"Missing metrics fields: {', '.join(missing_fields)}")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_metrics(path):
    rows_by_step = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_step[int(row["step"])] = row
    return [rows_by_step[step] for step in sorted(rows_by_step)]
