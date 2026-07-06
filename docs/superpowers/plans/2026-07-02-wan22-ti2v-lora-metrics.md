# Wan2.2-TI2V-5B LoRA Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU-testable Stage A tools and a minimal training metrics hook for Wan2.2-TI2V-5B LoRA SFT in DiffSynth-Studio.

**Architecture:** Keep the official Wan training loop intact and add optional metrics only when `--metrics_path` is provided. Put token math, JSONL I/O, metrics loading/deduplication, and summaries into pure-Python modules/scripts so Mac Stage A can validate them without CUDA or model weights. Use standalone dataset generation/checking scripts to reproduce the real tab-separated metadata shape and catch bad samples before H100 training.

**Tech Stack:** Python 3.10+, pytest, pandas, imageio, matplotlib, accelerate/DiffSynth existing training code.

---

### Task 1: Metrics Utilities and Tests

**Files:**
- Create: `metrics_utils.py`
- Create: `tests/test_metrics_utils.py`

- [ ] **Step 1: Write failing tests**

Cover:
- `tokens_per_sample(121, 480, 832) == 12090`
- invalid frame/height/width divisibility raises `ValueError`
- `MetricsWriter` appends JSONL records and creates parent dirs
- loading records with duplicate `step` keeps the last record

Run: `pytest tests/test_metrics_utils.py -v`
Expected: FAIL because `metrics_utils.py` does not exist.

- [ ] **Step 2: Implement utilities**

Implement:
- `tokens_per_sample(num_frames, height, width)`
- `MetricsWriter(metrics_path).write(record)`
- `load_metrics(path)` with duplicate-step last-wins behavior
- `tokens_per_step(record)` helper for plotting/tests

Run: `pytest tests/test_metrics_utils.py -v`
Expected: PASS.

### Task 2: Trainer Metrics Hook

**Files:**
- Modify: `diffsynth/diffusion/parsers.py`
- Modify: `diffsynth/diffusion/runner.py`
- Create: `tests/test_train_metrics_argparse.py`

- [ ] **Step 1: Write failing argparse/import tests**

Assert `examples.wanvideo.model_training.train.wan_parser()` exposes `--metrics_path` with default `None`.

Run: `pytest tests/test_train_metrics_argparse.py -v`
Expected: FAIL because `--metrics_path` is missing.

- [ ] **Step 2: Add parser and runner hook**

Add `--metrics_path` to training config or logger config with default `None`.
In `launch_training_task`, when `args.metrics_path` is set and `accelerator.is_main_process`, initialize `MetricsWriter`.
After an actual optimizer step, write:
- `step`
- `epoch`
- `loss`
- `step_time_sec`
- `tokens_per_sample`
- `samples_per_step`
- `lr`

Use `accelerator.sync_gradients` so gradient accumulation writes once per optimizer step. Compute `samples_per_step = 1 * accelerator.gradient_accumulation_steps * accelerator.num_processes` because the current dataloader collate returns one metadata item per micro-batch.

Run: `pytest tests/test_train_metrics_argparse.py tests/test_metrics_utils.py -v`
Expected: PASS.

### Task 3: Debug Dataset Generator and Checker

**Files:**
- Create: `make_debug_dataset.py`
- Create: `check_dataset.py`
- Create: `tests/test_dataset_tools.py`

- [ ] **Step 1: Write failing tests**

Use a temporary directory to generate 4 good samples plus bad samples. Assert:
- metadata is tab-separated with `video`, `prompt`, `input_image`
- video and image directories use `videos_HxW` and `images_HxW`
- checker reports exactly two bad rows for insufficient frames and missing input image
- checker writes `metadata_fixed.csv` for tab-separated metadata
- checker uses `metrics_utils.tokens_per_sample()`

Run: `pytest tests/test_dataset_tools.py -v`
Expected: FAIL because scripts do not exist.

- [ ] **Step 2: Implement generator**

Implement CLI args:
- `--num_videos`
- `--height`
- `--width`
- `--num_frames`
- `--output_dir`
- `--with_bad_samples`

Validate height/width multiples of 32 and `(num_frames - 1) % 4 == 0`. Generate moving color-block MP4s with imageio and first-frame JPGs with PIL/imageio. Write tab-separated `metadata.csv`.

- [ ] **Step 3: Implement checker**

Detect delimiter with `csv.Sniffer` or header inspection. Validate columns and file existence. Read video width/height/frame/fps with imageio first, cv2 fallback if present, ffprobe optional fallback. Output JSON/text summary and return nonzero only for hard CLI errors; bad samples are reported in summary for Stage A inspection. Write comma-separated `metadata_fixed.csv` when source delimiter is tab.

Run: `pytest tests/test_dataset_tools.py -v`
Expected: PASS.

### Task 4: Plotting and Fake Metrics

**Files:**
- Create: `plot_metrics.py`
- Create: `tests/gen_fake_metrics.py`
- Create: `tests/test_plot_metrics.py`

- [ ] **Step 1: Write failing tests**

Generate fake metrics with warmup and duplicate steps. Assert:
- `plot_metrics.py` writes loss and throughput PNGs
- summary contains total steps, last-20% loss, steady tokens/s, videos/hour
- duplicate steps are deduped last-wins

Run: `pytest tests/test_plot_metrics.py -v`
Expected: FAIL because plotting script does not exist.

- [ ] **Step 2: Implement plotting**

Use matplotlib non-interactive backend. Plot raw+EMA loss and throughput with dual y axes. Discard warmup steps for steady summary. Support `--tokens_per_video` override. Print summary.

Run: `pytest tests/test_plot_metrics.py -v`
Expected: PASS.

### Task 5: Dataset Smoke Test

**Files:**
- Create: `tests/test_dataset_smoke.py`
- Modify: `NOTES.md` later if smoke must degrade

- [ ] **Step 1: Write smoke test**

Generate small debug dataset without bad samples. Instantiate `diffsynth.core.UnifiedDataset` with the same operator settings as `train.py`, using `data_file_keys="video,input_image"`. Iterate two samples and assert:
- `video` is a list of PIL images with expected length and image size
- `prompt` is `str`
- `input_image` loads as list/PIL-compatible data

Run: `pytest tests/test_dataset_smoke.py -v`
Expected: PASS if imports work on Mac; otherwise document the exact import blocker in `NOTES.md` and test metadata parsing directly.

### Task 6: Training Launcher, Ignore Rules, and Notes

**Files:**
- Create: `train_ti2v5b_lora.sh`
- Modify: `.gitignore`
- Create: `NOTES.md`

- [ ] **Step 1: Add launcher**

Use a top variable block:
- `MODEL_ROOT`
- `DATA_ROOT`
- `OUTPUT_ROOT`
- `NUM_GPUS`
- `HEIGHT`
- `WIDTH`
- `NUM_FRAMES`
- `METRICS_PATH`

Launch official `train.py` with official TI2V LoRA flags, local `--model_paths`, explicit `--data_file_keys "video,input_image"`, and `--metrics_path`.

- [ ] **Step 2: Add NOTES**

Include:
- exact `MODEL_ROOT` file list
- Stage B command order
- mini metadata smoke command
- H100 tuning sequence
- validation LoRA script
- expected observations and rollback actions
- note that current TI2V trainer uses first video frame for `input_image`

- [ ] **Step 3: Update ignores**

Add:
- `debug_data/`
- `*.jsonl`
- `*.png`
- optional local output folders already covered by `/models`

Run: `bash -n train_ti2v5b_lora.sh`
Expected: PASS.

### Task 7: Full Stage A Verification

**Files:**
- Update: `task_plan.md`
- Update: `findings.md`
- Update: `progress.md`

- [ ] **Step 1: Run tests**

Run: `pytest tests/ -v`
Expected: PASS.

- [ ] **Step 2: Run manual chain**

Run:
- `python make_debug_dataset.py --with_bad_samples`
- `python check_dataset.py --dataset_root debug_data --metadata_path debug_data/metadata.csv --height 96 --width 160 --num_frames 13`
- `python tests/gen_fake_metrics.py --output /tmp/fake_metrics.jsonl`
- `python plot_metrics.py --metrics_path /tmp/fake_metrics.jsonl --output_dir /tmp/diffsynth_metrics_plots`
- `python examples/wanvideo/model_training/train.py --help`
- `bash -n train_ti2v5b_lora.sh`

Expected: all commands complete; checker reports exactly two bad samples; plot creates two PNGs; train help includes `--metrics_path`.

- [ ] **Step 3: Review diff**

Run:
- `git diff --stat`
- `git diff --check`
- targeted `rg` for `/path/to/local/wan` and hard-coded data paths

Expected: no whitespace errors, only intended files changed, all Stage B paths in launcher variable block or NOTES examples.
