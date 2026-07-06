# Figurine360 DMD Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build current-repo Stage A support for Wan2.2-TI2V-5B figurine360 DMD LoRA training.

**Architecture:** Vendor the Turbo DMD runtime under `third_party/wan22_turbo`, patch it minimally for LoRA-only training/export, and keep CPU-safe orchestration helpers in `dmd/` plus root scripts/docs. Real H100 training remains a Stage B operation driven by `train_figurine360_dmd_lora.sh`.

**Tech Stack:** Python, PyTorch, safetensors, PyYAML, pytest, bash, vendored Wan2.2 Turbo runtime.

---

### Task 1: CPU-Safe Stage A Contracts

**Files:**
- Create: `tests/test_dmd_lora_utils.py`
- Create: `tests/test_dmd_stage_a_contract.py`

- [x] **Step 1: Write failing tests**

Tests cover LoRA zero-delta init, target module wrapping, DiffSynth-compatible export keys, DMD config fields, metadata CSV conversion, launcher scope, bash syntax, and attention fallback import.

- [x] **Step 2: Verify red**

Run: `pytest tests/test_dmd_lora_utils.py tests/test_dmd_stage_a_contract.py -q`

Expected: FAIL because `dmd` package does not exist.

### Task 2: Current-Repo DMD Helpers

**Files:**
- Create: `dmd/__init__.py`
- Create: `dmd/wan22_lora.py`
- Create: `dmd/wan22_config.py`
- Create: `tools/prepare_dmd_dataset_csv.py`

- [x] **Step 1: Implement LoRA and config helpers**

Implement import-safe LoRA wrapping/export and runtime config generation.

- [x] **Step 2: Run tests**

Run: `pytest tests/test_dmd_lora_utils.py tests/test_dmd_stage_a_contract.py -q`

Expected: PASS after runtime/script files are present.

### Task 3: Vendored Turbo Runtime

**Files:**
- Create: `third_party/wan22_turbo/`
- Modify: `third_party/wan22_turbo/model/base.py`
- Modify: `third_party/wan22_turbo/trainer/wan22_distillation.py`
- Modify: `third_party/wan22_turbo/utils/distributed.py`

- [x] **Step 1: Vendor Turbo code**

Copy reference runtime without `.git`, cache, demos, or mp4 outputs.

- [x] **Step 2: Patch LoRA-only training**

Inject LoRA into generator/fake score, keep real score frozen, save LoRA-only checkpoints, export EMA LoRA safetensors, and honor `max_iters`.

### Task 4: Stage B Launcher and Docs

**Files:**
- Create: `configs/dmd/figurine360_wan22_dmd_lora.yaml`
- Create: `train_figurine360_dmd_lora.sh`
- Create: `docs/task03_baseline.md`
- Create: `NOTE_DMD.md`
- Create: `docs/task03_phase23_roadmap.md`

- [x] **Step 1: Add config and launcher**

Launcher centralizes all Stage B variables and runs the vendored runtime from current repo.

- [x] **Step 2: Add Stage A/Stage B documentation**

Document source-code findings, commands, recovery, and Phase 2/3 hooks.

### Task 5: Verification

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `log.txt`

- [x] **Step 1: Run focused and regression tests**

Run focused DMD tests, py_compile, bash syntax, and current relevant existing tests.

- [x] **Step 2: Update progress records**

Record Stage A completion and stop for user inspection.
