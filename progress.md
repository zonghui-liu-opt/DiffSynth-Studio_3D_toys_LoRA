# 进度日志

## 会话：2026-07-06

### Task-03：figurine360-DMD Stage B 剩余验证
- **状态：** local implementation complete；H100 真权重视频生成/指标执行待内网运行
- 用户确认：
  - gate-0 merged teacher 与 runtime load figurine360 LoRA 的同 seed/prompt/首帧对比已在内网实现并验证无差别。
  - 本次跳过 gate-0，其余 Stage B 验证正常推进。
- 当前执行计划：
  - 已写 Stage B 本地红灯测试并确认缺接口失败。
  - 已补 LoRA 推理加载、EMA resume、validation runner、视频指标、DiffSynth 本地推理脚本和 NOTE_DMD 命令。
  - 已运行 pytest/py_compile/bash 语法检查并记录结果。
- 创建/修改的文件：
  - `dmd/wan22_lora.py`
  - `dmd/wan22_config.py`
  - `dmd/stage_b_validation.py`
  - `dmd/video_metrics.py`
  - `tools/validate_dmd_stage_b.py`
  - `tools/compute_dmd_video_metrics.py`
  - `tools/diffsynth_wan22_ti2v_infer.py`
  - `tools/diffsynth_wan22_dmd_lora_infer.py`
  - `third_party/wan22_turbo/wan2.2_fewstep.py`
  - `third_party/wan22_turbo/trainer/wan22_distillation.py`
  - `configs/dmd/figurine360_wan22_dmd_lora.yaml`
  - `configs/dmd/figurine360_wan22_teacher50.yaml`
  - `train_figurine360_dmd_lora.sh`
  - `tests/test_dmd_stage_b_validation.py`
  - `NOTE_DMD.md`
  - `task_plan.md`
  - `progress.md`
- 测试：
  - `pytest tests/test_dmd_stage_b_validation.py tests/test_dmd_lora_utils.py tests/test_dmd_stage_a_contract.py -q`：19 passed（新增 DiffSynth DMD LoRA 脚本前）
  - `python3 -m py_compile dmd/wan22_lora.py dmd/wan22_config.py dmd/stage_b_validation.py dmd/video_metrics.py tools/validate_dmd_stage_b.py tools/compute_dmd_video_metrics.py tools/diffsynth_wan22_ti2v_infer.py tools/diffsynth_wan22_dmd_lora_infer.py third_party/wan22_turbo/wan2.2_fewstep.py third_party/wan22_turbo/trainer/wan22_distillation.py`：通过
  - `bash -n train_figurine360_dmd_lora.sh`：通过
  - `pytest tests/ -q`：43 passed, 2 warnings
  - `git diff --check`：未执行成功；当前目录不是 git 仓库

### Task-03：figurine360-DMD Stage A
- **状态：** complete（按用户要求，Stage A 后停止等待检查）
- 执行的操作：
  - 读取 Task-03 文档。
  - 读取 Turbo `train.py`、`configs/default_config.yaml`、`configs/self_forcing_wan22_dmd.yaml`、`model/dmd.py`、`model/base.py`、`trainer/wan22_distillation.py`、`pipeline/bidirectional_training.py`、`utils/dataset.py`、`utils/wan_wrapper.py`、`wan22/modules/attention.py`。
  - 确认 Turbo DMD 为 bidirectional generator，CFG 为附加引导参数化，数据 CSV 为 `path,text,num_frames`，5:1 为上游 step-ratio 语义。
  - 按 TDD 新增 DMD Stage A 测试并确认红灯。
  - vendor Turbo runtime 到 `third_party/wan22_turbo`。
  - 新增 `dmd/wan22_lora.py`、`dmd/wan22_config.py`、`tools/prepare_dmd_dataset_csv.py`。
  - patch vendored runtime：LoRA 注入、LoRA-only checkpoint、optimizer state 保存、EMA LoRA 导出、`max_iters` 停止。
  - 新增 `configs/dmd/figurine360_wan22_dmd_lora.yaml` 与 `train_figurine360_dmd_lora.sh`。
  - 新增 `docs/task03_baseline.md`、`NOTE_DMD.md`、`docs/task03_phase23_roadmap.md`。
- 创建/修改的文件：
  - `dmd/__init__.py`
  - `dmd/wan22_lora.py`
  - `dmd/wan22_config.py`
  - `tools/prepare_dmd_dataset_csv.py`
  - `configs/dmd/figurine360_wan22_dmd_lora.yaml`
  - `train_figurine360_dmd_lora.sh`
  - `tests/test_dmd_lora_utils.py`
  - `tests/test_dmd_stage_a_contract.py`
  - `third_party/wan22_turbo/`
  - `docs/task03_baseline.md`
  - `NOTE_DMD.md`
  - `docs/task03_phase23_roadmap.md`
  - `docs/superpowers/plans/2026-07-06-figurine360-dmd-stage-a.md`
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
- 测试：
  - `pytest tests/test_dmd_lora_utils.py tests/test_dmd_stage_a_contract.py -q`：8 passed
  - `python3 -m py_compile ...`：通过
  - `bash -n train_figurine360_dmd_lora.sh`：通过
  - `pytest tests/ -q`：31 passed, 2 warnings
  - 最终复核 `python3 -m py_compile ...`：通过
  - 最终复核 `bash -n train_figurine360_dmd_lora.sh && bash -n train_figurine360_lora.sh && bash -n train_ti2v5b_lora.sh`：通过

## 会话：2026-07-02

### 文档整理
- **状态：** complete
- 执行的操作：
  - 创建 `log.txt` 记录修改及作用。
  - 将 `NOTES.md` 改为中文简要内网上机说明。
  - 创建三列 TI2V LoRA 内网冒烟和正式训练命令记录。
  - 在 `.gitignore` 中添加 `!log.txt`，避免 `log*.txt` 规则忽略该文件。
- 创建/修改的文件：
  - `log.txt`
  - `NOTES.md`
  - `NOTES.md`
  - `.gitignore`

### 清理本地调试文件
- **状态：** complete
- 执行的操作：
  - 删除 MacBook 本地合成数据生成器和相关测试。
  - 删除本地生成目录 `debug_data/` 和 `tests/__pycache__/`。
  - 保留内网需要的 `check_dataset.py`、`metrics_utils.py`、`plot_metrics.py`、`train_ti2v5b_lora.sh`。
  - 更新 `log.txt` 记录删除范围。
- 创建/修改/删除的文件：
  - 删除 `make_debug_dataset.py`
  - 删除 `tests/test_dataset_tools.py`
  - 删除 `tests/test_dataset_smoke.py`
  - 删除 `tests/gen_fake_metrics.py`
  - 删除 `tests/test_plot_metrics.py`
  - 修改 `log.txt`
- 测试：
  - `pytest tests/ -q`：9 passed, 2 warnings

### 阶段 6：横竖屏分桶训练支持
- **状态：** complete
- 执行的操作：
  - 新增 orientation bucket 单测，覆盖 `check_dataset.py` metadata 输出、no-crop resize、bucket sampler 和 argparse 开关。
  - `check_dataset.py` 始终生成带 `height,width,bucket` 的 `metadata_fixed.csv`，并输出 `bucket_counts`。
  - 新增 `ImageResizeToBucketResolution`，bucket 模式下横屏输出 `480x832`、竖屏输出 `832x480`，不做 center crop。
  - 新增 `OrientationBucketSampler`，训练时按 metadata bucket 分组采样。
  - `examples/wanvideo/model_training/train.py` 在 `--enable_orientation_buckets` 时启用 bucket/no-crop 数据处理。
  - `train_ti2v5b_lora.sh` 新增 `ENABLE_ORIENTATION_BUCKETS`，默认启用。
  - 更新 `NOTES.md`、`log.txt`。
- 创建/修改的文件：
  - `tests/test_orientation_buckets.py`
  - `check_dataset.py`
  - `diffsynth/core/data/operators.py`
  - `diffsynth/core/data/unified_dataset.py`
  - `diffsynth/core/data/bucket_sampler.py`
  - `diffsynth/core/data/__init__.py`
  - `diffsynth/diffusion/parsers.py`
  - `diffsynth/diffusion/runner.py`
  - `examples/wanvideo/model_training/train.py`
  - `train_ti2v5b_lora.sh`
  - `NOTES.md`
  - `NOTES.md`
  - `log.txt`
- 测试：
  - `pytest tests/test_orientation_buckets.py -v`：6 passed
  - `pytest tests/test_metrics_utils.py tests/test_train_metrics_argparse.py tests/test_wan_training_inputs.py tests/test_orientation_buckets.py -v`：15 passed, 2 warnings
  - `pytest tests/ -v`：15 passed, 2 warnings
  - `python3 -m py_compile ...`：通过
  - `bash -n train_ti2v5b_lora.sh`：通过
  - `PYTHONPATH=. python3 examples/wanvideo/model_training/train.py --help`：`--enable_orientation_buckets` 可见
  - `python3 check_dataset.py --dataset_root debug_data/orientation_buckets ...`：横屏/竖屏各 1 条，`bad_samples: 0`
  - `UnifiedDataset` 加载 `debug_data/orientation_buckets/metadata_fixed.csv`：横屏输出 `(160, 96)`，竖屏输出 `(96, 160)`

### 阶段 7：Checkpoint 保存步数启动参数
- **状态：** complete
- 执行的操作：
  - 新增 launcher 静态测试，锁定 `SAVE_STEPS`、`SAVE_ARGS` 和 `--save_steps` 条件传参。
  - `train_ti2v5b_lora.sh` 顶部变量块新增 `SAVE_STEPS=${SAVE_STEPS:-}`。
  - 当 `SAVE_STEPS` 非空时，训练命令追加 `--save_steps "$SAVE_STEPS"`；为空时保持每 epoch 保存。
  - 更新 `NOTES.md`、`log.txt`。
- 创建/修改的文件：
  - `tests/test_train_launcher.py`
  - `train_ti2v5b_lora.sh`
  - `NOTES.md`
  - `NOTES.md`
  - `log.txt`
- 测试：
  - `pytest tests/test_train_launcher.py -v`：1 passed
  - `pytest tests/ -v`：16 passed, 2 warnings
  - `bash -n train_ti2v5b_lora.sh`：通过
  - `PYTHONPATH=. python3 examples/wanvideo/model_training/train.py --help | rg -- '--save_steps|--enable_orientation_buckets|--metrics_path'`：3 个参数可见
  - `git diff --check`：通过

### 阶段 1：需求与源码发现
- **状态：** complete
- **开始时间：** 2026-07-02
- 执行的操作：
  - 读取任务书。
  - 读取规划、写计划、TDD、执行计划相关技能说明。
  - 创建持久化规划文件。
  - 切换到功能分支 `feat/wan22-ti2v-lora-metrics`。
  - 读取 `train.py`、官方 TI2V-5B LoRA 脚本、训练 runner/logger/training_module、UnifiedDataset 与 loader 配置。
  - 写入实施计划 `docs/superpowers/plans/2026-07-02-wan22-ti2v-lora-metrics.md`。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `docs/superpowers/plans/2026-07-02-wan22-ti2v-lora-metrics.md`

### 阶段 2：计划与测试先行
- **状态：** complete
- 执行的操作：
  - 按 TDD 写入 metrics、parser、dataset tools、plotting、dataset smoke、TI2V input_image 行为测试。
  - 逐项确认红灯：缺 `metrics_utils.py`、缺 `--metrics_path`、缺 dataset/plot 脚本、TI2V input_image 使用视频首帧。
- 创建/修改的文件：
  - `tests/`

### 阶段 3：实现 Stage A 交付物
- **状态：** complete
- 执行的操作：
  - 实现 `metrics_utils.py`、trainer metrics hook、debug dataset、dataset checker、plotter、training launcher、NOTES 和 `.gitignore`。
  - 修改 TI2V `parse_extra_inputs`：显式加载 `input_image` 时优先用该列，否则回退官方视频首帧逻辑。
- 创建/修改的文件：
  - `metrics_utils.py`
  - `diffsynth/diffusion/parsers.py`
  - `diffsynth/diffusion/runner.py`
  - `examples/wanvideo/model_training/train.py`
  - `make_debug_dataset.py`
  - `check_dataset.py`
  - `plot_metrics.py`
  - `train_ti2v5b_lora.sh`
  - `.gitignore`
  - `NOTES.md`

### 阶段 4：测试与验证
- **状态：** complete
- 执行的操作：
  - 安装 Stage A 缺失依赖：imageio、pandas、matplotlib、einops、accelerate、peft、transformers、modelscope、safetensors 等。
  - 运行完整 pytest、py_compile、git diff 检查和手工链路验证。
- 创建/修改的文件：
  - `debug_data/`（被 `.gitignore` 忽略）
  - `/tmp/fake_metrics.jsonl`、`/tmp/diffsynth_metrics_plots/`（临时验证产物）

### 阶段 5：审查、提交与交付
- **状态：** complete
- 执行的操作：
  - 完成 `git diff --check`。
  - 扫描硬编码路径，确认只在 launcher 顶部变量块和 NOTES 示例中出现。
  - 提交 `fef9e4a Add Wan2.2 TI2V LoRA metrics tooling`。
  - 推送分支 `origin/feat/wan22-ti2v-lora-metrics`。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| pytest tests/ -v | 16 个测试 | 全部通过 | 16 passed, 2 warnings | pass |
| py_compile | 新增脚本 + 修改训练文件 | 无语法错误 | 通过 | pass |
| bash -n train_ti2v5b_lora.sh | 训练 launcher | 无语法错误 | 通过 | pass |
| make_debug_dataset + check_dataset | `--with_bad_samples` | 2 条坏样本准确报出 | bad_samples: 2；insufficient_frames: 1；missing_input_image: 1 | pass |
| fake metrics + plot_metrics | 300 行含重复 step | 2 张 PNG + 摘要 | loss.png、throughput.png 已生成 | pass |
| train.py --help | `PYTHONPATH=.` | 关键参数可见 | `--metrics_path` 等参数可见 | pass |
| orientation buckets | 横屏/竖屏 debug metadata | fixed metadata 含 bucket，no-crop resize，sampler 分组，repeat 不缩短 epoch | 6 passed | pass |
| check_dataset debug data | `debug_data/orientation_buckets` | 横屏/竖屏各 1 条，0 坏样本 | bucket_counts: {'landscape': 1, 'portrait': 1} | pass |
| UnifiedDataset debug load | `metadata_fixed.csv` + bucket resize | 横屏/竖屏按各自方向输出 | `(160,96)` 与 `(96,160)` | pass |
| git diff --check | 当前 diff | 无 whitespace error | 通过 | pass |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-07-02 | 创建新 goal 失败：线程已有活跃 goal | 1 | 使用已有活跃 goal 继续 |
| 2026-07-02 | `rg diffsynth/trainers` 失败：目录不存在 | 1 | 改按实际源码目录 `diffsynth/diffusion/` 勘查 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 1：需求与源码发现 |
| 我要去哪里？ | 已完成 Stage A 交付物、自测、NOTES、提交与推送 |
| 目标是什么？ | 完成 Wan2.2-TI2V-5B LoRA SFT 辅助脚本与训练指标监控 |
| 我学到了什么？ | 见 `findings.md` |
| 我做了什么？ | 见上方记录 |

---
*每个阶段完成后或遇到错误时更新此文件*
