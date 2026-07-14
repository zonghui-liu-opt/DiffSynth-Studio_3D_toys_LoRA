# Stage A 测试与交付报告

日期：2026-07-10
分支：`codex/wan22-ti2v5b-direct-distill`
基线：`origin/main@10a2c79`
状态：本地实现与无权重测试完成；Stage A 等待期内未 commit/push，用户已明确授权后续提交与推送。

## 实现摘要

- 在原 `DirectDistillLoss` 增加默认关闭的严格 TI2V 分支：4 步连续 rollout、CFG=1 断言、metadata shift、每步首帧覆盖、最终 loss 排除首帧、teacher target key。
- `WanVideoPipeline(return_latents=True)` 在 VAE decode 前直接返回最终 latent，默认返回视频的行为不变。
- `WanTrainingModule` 支持缓存首帧 PNG + safetensors teacher latent；strict 模式优先不读取整段源视频，缓存缺失时只解码 `video[0]` 回退。
- 训练顺序固定为 base → 融合且冻结 figurine360 → 注入新 DirectDistill LoRA；启动检查 preset 匹配层数、trainable key 与续训 checkpoint 的完整 key/shape 覆盖，checkpoint 只导出新 adapter。
- `ModelLogger` 支持任意 finite scalar、实时后端与 `metrics.jsonl`；runner 只在显式指标模式下按真实 optimizer step 记录 loss/EMA/lr/吞吐/显存。
- teacher 准备支持稳定 ID、训练/held-out seed 分离、无损首帧、直接 latent、带 provenance 的断点校验、坏文件隔离、确定性复跑、object-level train/validation split。
- 验证脚本按 base → figurine → teacher，再融合 DirectDistill → student 的顺序运行；强制 student 4 步/CFG=1/shift=5 并断言 4 次 forward。
- 内网 shell 支持同一 DiT 的多 shard 路径分组，不调用任何下载命令。

## 修改文件

核心代码：

- `diffsynth/diffusion/loss.py`
- `diffsynth/diffusion/logger.py`
- `diffsynth/diffusion/runner.py`
- `diffsynth/diffusion/base_pipeline.py`
- `diffsynth/pipelines/wan_video.py`
- `diffsynth/utils/lora/general.py`
- `examples/wanvideo/model_training/train.py`

新增交付：

- `examples/wanvideo/model_training/special/direct_distill/prepare_wan22_ti2v_figurine360.py`
- `examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh`
- `examples/wanvideo/model_training/special/direct_distill/validate_wan22_ti2v_figurine360.py`
- `examples/wanvideo/model_training/special/direct_distill/plot_direct_distill_metrics.py`
- `NOTE_DIRECT_DISTILL.md`
- 本报告

本地规划记录（用于本次 Codex 任务恢复，后续最小代码提交默认排除）：

- `task_plan.md`
- `findings.md`
- `progress.md`

用户原有工作区状态（保留但不擅自纳入后续代码提交）：旧 DMD 任务文档删除、新 `Task-figurine360-DirectDistill-Wan2.2-TI2V-5B.md` 新增。

测试：

- `tests/test_direct_distill_loss.py`
- `tests/test_wan_direct_distill_training.py`
- `tests/test_wan_video_return_latents.py`
- `tests/test_prepare_wan22_ti2v_figurine360.py`
- `tests/test_validate_wan22_ti2v_figurine360.py`
- `tests/test_direct_distill_logger.py`
- `tests/test_direct_distill_plot.py`
- `tests/test_direct_distill_runner_metrics.py`
- `tests/test_lora_match_count.py`
- `tests/test_direct_distill_shell.py`

`base_pipeline.py` 与 `general.py` 的额外最小修改仅把 LoRA loader 已计算的匹配数返回给调用方，使 0 层错配可在 teacher/strict 入口立即失败；未改融合公式。

## Stage A 必测项结果

| 要求 | 本地证据 | 结果 |
|---|---|---|
| 4 步恰 4 次 forward，shift=5 schedule 对齐 | dummy model + 独立 Wan scheduler逐项比较 timestep/sigma | 通过 |
| 相邻步无 add_noise/GT 注入/detach，可完整反传 | scheduler spy、对象/data_ptr 传递、解析梯度 | 通过 |
| 每步首帧 bitwise 保持、loss 排除、旧行为兼容 | 小 tensor 首帧/非首帧与 legacy 数值测试 | 通过 |
| teacher safetensors/metadata/断点/坏文件 | key/shape/dtype/finite/首帧/provenance、原子保存、隔离/重跑 | 通过 |
| 双 LoRA 隔离与只导出新 LoRA | mock 融合顺序、真实 backward 梯度、export 过滤、匹配数、续训覆盖 | 通过 |
| CFG=1 仅 positive；非 1 立即失败 | model 调用计数与前置异常测试 | 通过 |
| EMA/token/多卡聚合/JSONL/PNG | 纯函数、fake logger、Agg 图表、真实 2 进程 collective | 通过 |
| compileall、相关 pytest | 见下方命令 | 通过 |

## 测试命令与结果

```bash
python3 -m pytest -q tests
# 66 passed（最终复跑通过；耗时随机器负载变化）

python3 -m compileall -q \
  diffsynth \
  examples/wanvideo/model_training/special/direct_distill \
  tests
# 通过

git diff --check
# 通过

bash -n \
  examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh
# 通过

python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 --module pytest -q \
  tests/test_direct_distill_runner_metrics.py
# 两个 rank 各 4 passed；collective 正常结束
```

测试环境：macOS、Python 3.10.11、pytest 9.0.3。没有加载 CUDA、flash-attn、Wan2.2 5B 权重、figurine360 真实 LoRA 或真实视频数据。

另以假路径文件完成 shell 的三 shard 参数分组与 smoke seed 展开检查：`prepare-smoke` 只传训练 `seed=1`，不会重新带入 held-out validation seed；`--help`、仓库 `PYTHONPATH` 注入均已固化为测试。本机执行完整 `doctor` 时在依赖门禁处正确失败，原因是本机未安装 `imageio_ffmpeg`；这不作为 H100 环境通过证明，目标机必须预装后重新运行 `doctor`。

## Seed 结论

- smoke 与 held-out validation 的 canonical seed 为 `1`，保持既有 figurine360 推理习惯；正式 shell 默认训练 seeds=2/3/4、validation seed=1，使验证 seed 数值在训练中未出现。
- 每条 teacher latent 与 student 初始噪声都读取同一 metadata `seed` 和 `rand_device`；四步之间不重新采样。
- 可用 2～4 seeds 扩充数据，但同一 object 的所有 seed 必须留在同一 split；验证仍记录并复用对应 seed。
- 若正式数据只用 seed=1，其他 seed 的泛化结论必须留待 H100 另测。

## 待 H100 验证

以下不能由 Mac 合成测试证明：

1. Wan2.2-TI2V-5B 多 shard、text encoder、VAE 与两个真实 LoRA 的完整加载和层匹配数。
2. 2～4 条低分辨率 teacher latent 的 bitwise deterministic 复跑及首帧 VAE 一致性。
3. 2～10 optimizer steps 的 loss finite、新 LoRA 更新、checkpoint 权重续训和实际 H100 显存/吞吐。
4. 至少 12 个未见首帧/未见 seed 的 teacher 50-step CFG=5 与 student 4-step CFG=1 视觉对比。
5. 完整 teacher 数据生成与正式训练，以及 360°、身份、闪烁、过曝的人工验收。
6. TensorBoard/W&B/SwanLab 在目标内网环境的实际可用性；`metrics.jsonl` 与离线图表不依赖联网。
7. 目标 H100 环境完整 `doctor` 通过；当前 Mac 缺少 `imageio_ffmpeg`，仅验证了其 fail-fast 行为。

当前 checkpoint 续训沿用上游语义：`--lora_checkpoint` 恢复新 LoRA 权重，但 optimizer、scheduler 和 global step 重新初始化；文档未把它描述为逐 bit 完整训练状态恢复。

## Git 门禁

- Stage A 报告提交给用户后已停止等待；收到明确授权前没有 commit、push 或 PR，门禁执行正确。
- 用户原有的旧 DMD 任务文档删除与新 DirectDistill 任务文档新增均保留，未擅自还原。
- 用户现已明确确认提交与推送；成功后的 branch 与 commit SHA 由最终执行结果报告。
