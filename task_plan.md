# 任务计划：Wan2.2-TI2V-5B LoRA SFT 与训练指标监控

## Task-03：figurine360-DMD Stage A

### 目标
在当前仓库内完成 Wan2.2-TI2V-5B Turbo DMD 的 Stage A 可离线验证实现：vendored runtime、LoRA-only DMD 支撑、H100 launcher、配置、metadata 转换、NOTE_DMD 与 Phase 2/3 路线。完成后停止等待检查。

### 当前阶段
Stage A 完成，按用户要求停止等待检查。

### 阶段

- [x] 阅读 Task-03 任务书与 Turbo train/config/DMD/trainer/dataset/attention 源码
- [x] 确认双向路径、generator 初始化来源、数据格式、CFG 参数化、5:1 更新语义
- [x] 写入 DMD Stage A 红灯测试
- [x] vendor Turbo runtime 到 `third_party/wan22_turbo`
- [x] 新增 `dmd/` LoRA/config 工具
- [x] patch vendored runtime 的 LoRA 注入、LoRA-only checkpoint、EMA 导出、`max_iters`
- [x] 新增 DMD YAML、metadata 转换工具、H100 launcher
- [x] 新增 `docs/task03_baseline.md`、`NOTE_DMD.md`、`docs/task03_phase23_roadmap.md`
- [x] 跑完最终测试并记录结果

### 关键决策

| 决策 | 理由 |
|------|------|
| DMD runtime vendored 到当前仓库 | 满足“不依赖外部 `/Users/.../Wan2.2-TI2V-5B-Turbo`，所有实现都在当前文件夹” |
| LoRA 工具放 `dmd/`，保持 CPU/import-safe | Stage A 本地无 CUDA/权重也能测试 |
| `real_guidance_scale=4.0` | Turbo DMD 代码是附加引导 `cond + (cond-uncond)*scale`，等价标准 teacher CFG g=5 |
| `dfake_gen_update_ratio=5` 沿用上游语义 | 上游长期比例为 5 fake score : 1 generator，避免改已验证训练循环 |
| DMD LoRA 导出时 strip `model.` 前缀 | DiffSynth/ComfyUI 需要匹配 DiT 内部 key，如 `blocks.*.q.lora_A.default.weight` |

## 目标
在当前 DiffSynth-Studio 仓库中完成 Stage A 可离线验证的 Wan2.2-TI2V-5B LoRA 训练辅助工具、指标记录、绘图、数据集 smoke test 与 Stage B 上机 checklist。

## 当前阶段
完成

## 各阶段

### 阶段 1：需求与源码发现
- [x] 阅读任务书
- [x] 通读 `examples/wanvideo/model_training/train.py` 的 argparse
- [x] 通读实际训练循环、保存、日志、数据集入口
- [x] 确认 Wan2.2-TI2V-5B 官方 LoRA 示例参数与本地权重参数
- [x] 将关键发现记录到 `findings.md`
- **状态：** complete

### 阶段 2：计划与测试先行
- [x] 写入实施计划
- [x] 为 metrics、绘图去重、数据生成/体检、数据集 smoke 准备 pytest 覆盖
- [x] 先运行目标测试确认失败
- **状态：** complete

### 阶段 3：实现 Stage A 交付物
- [x] 新增 `metrics_utils.py`
- [x] 最小修改 trainer 增加 `--metrics_path`
- [x] 新增 `make_debug_dataset.py`
- [x] 新增 `check_dataset.py`
- [x] 新增 `plot_metrics.py`
- [x] 新增 `train_ti2v5b_lora.sh`
- [x] 新增/更新 `.gitignore`
- [x] 新增 `NOTES.md`
- **状态：** complete

### 阶段 4：测试与验证
- [x] `pytest tests/` 全绿
- [x] 生成 synthetic debug dataset，并用 `check_dataset.py` 验证坏样本报告
- [x] fake metrics 生成与 `plot_metrics.py` PNG 输出验证
- [x] `bash -n train_ti2v5b_lora.sh`
- [x] `train.py --help` 覆盖脚本中使用的参数名
- [x] 数据集迭代 smoke test 通过或写明降级原因
- **状态：** complete

### 阶段 5：审查、提交与交付
- [x] 审查 diff 确保默认关闭时上游行为不变
- [x] 确认无散落硬编码路径
- [x] 记录最终自测结果
- [x] 视远端状态提交并推送 GitHub
- **状态：** complete

### 阶段 6：横竖屏分桶训练
- [x] 增加横屏/竖屏 metadata debug 单测
- [x] `check_dataset.py` 输出 `height,width,bucket` 与 `bucket_counts`
- [x] 增加 no-crop orientation resize 算子
- [x] 增加 orientation bucket sampler
- [x] Wan 训练入口增加 `--enable_orientation_buckets`
- [x] H100 launcher 默认启用 orientation buckets
- [x] 更新内网上机文档与日志
- [x] 跑完本地相关测试
- **状态：** complete

## 关键问题
1. `train.py` 是否已有本地权重参数，参数名和传参形式是什么？答：已有 `--model_paths`，JSON 列表，本地 path/glob。
2. `input_image` 是否需要加入 `--data_file_keys`？答：建议显式加入以加载真实列，但当前 TI2V trainer 仍用 `data["video"][0]`。
3. 现有训练循环中 optimizer step 的准确位置、step/epoch 变量名、world size/batch size 来源是什么？答：见 `diffsynth/diffusion/runner.py`，需新增真实 optimizer step 计数。
4. 数据集类能否在 Mac CPU-only 环境直接 import 和迭代？答：可以，`tests/test_dataset_smoke.py` 已通过。

## 已做决策
| 决策 | 理由 |
|------|------|
| 任务书作为已批准实现规格 | 用户明确要求逐步精确完成任务，不再额外阻塞等待设计确认 |
| 所有新增工具模块保持 CPU-only/import-safe | 任务硬性约束要求 Stage A 无 GPU/权重可验证 |
| 横竖屏采用 orientation bucket，不做 dense tensor 混 batch | 当前训练框架默认每个 micro-step 单样本；bucket 方案在不重写 Wan forward 的前提下保留横屏 `480x832` 和竖屏 `832x480` |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 试图新建 goal 时发现已有活跃 goal | 1 | 沿用当前活跃 goal |

## 备注
- 每完成阶段更新本文件与 `progress.md`。
- 外部/不可信内容不写入本文件；源码发现写入 `findings.md`。
