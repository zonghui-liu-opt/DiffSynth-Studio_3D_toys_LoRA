# Wan2.2-TI2V-5B + Figurine360 DirectDistill BSA（内网 H100）

本说明只覆盖 BSA 增量。teacher latent 的准备与校验继续按 `NOTE_DIRECT_DISTILL.md` 执行。当前 Stage A 已完成 CPU 合成验证；真实 5B、H100 性能/质量和 Ascend 均待内网验证。

## 1. 配置与检查

只修改脚本顶部路径变量：

```bash
examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh
```

关键路径是底模 shards、text encoder、VAE、tokenizer、figurine360 LoRA、已成功的 dense DirectDistill warm-start LoRA，以及已有 teacher latent 根目录。脚本不联网下载。

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh doctor
```

正式设置固定为 block `(4,3,6)`、rank-32 block gate、student 4 steps / CFG 1 / shift 5。首次正式实验建议至少 3000 个真实 optimizer steps；程序低于 1500 steps 会警告。

## 2. 冒烟与正式训练

先确认原 DirectDistill teacher 数据和 dense warm-start 均已通过验收，再运行：

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh bsa-smoke-train
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh bsa-train
```

启动时必须看到：

- 30/30 个 `self_attn` 注入，cross-attention 为 0；
- `gate=low_rank_dynamic/block`，`gate_up_zero_initialized=true`；
- DirectDistill LoRA 与 BSA gate 是仅有的 trainable 参数，且均为 FP32；
- backend forward/backward probe 通过；
- 第一次真实 forward 的 runtime grid 为 `(41,15,26)` 时，得到 `N=275`、最终 `K=55`、padding ratio 约 19.24%；
- `student_model_info.json` 同时包含初始化摘要和第一次 forward 的实际 shape。

默认 `sdpa_gather` 先 gather selected KV，再调用 PyTorch SDPA；不会构造全局 `[15990,15990]` mask。稀疏率只按成功的 optimizer step 从 0 逐档退火到 0.8，gradient accumulation 的 micro-batch 不推进 schedule。

两个参数组默认：

```text
DirectDistill LoRA: lr=2e-6, weight_decay=0
BSA block gate:     lr=2e-5, weight_decay=0
global grad clip=1.0
```

`DENSE_ANCHOR_WEIGHT=0` 为默认主线。质量主配置通过显存门禁后，可从 `0.1`、interval `2` 或 `4` 做独立实验；anchor-active step 的峰值显存必须单独验收。

## 3. Checkpoint 与续训语义

每个 BSA checkpoint 是原子目录：

```text
checkpoint-step-0000200/
├── direct_distill_lora.safetensors
├── bsa_adapter.safetensors
├── bsa_config.json
├── trainer_state/state.json
└── checkpoint_complete
```

当前 P0 保存的是严格的组合权重 continuation：LoRA、gate、BSA config、已完成 step 和 schedule 位置均恢复；不保存 Adam、LR scheduler、RNG 或 dataloader 游标，因此 manifest 明确写入 `resume_semantics=weight_continuation`，不能称为逐 bit 精确恢复。续训必须使用新的输出目录，并设置：

```bash
BSA_CHECKPOINT=/path/to/checkpoint-step-0000200 \
BSA_TRAIN_OUTPUT=/path/to/new-output \
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh bsa-train
```

manifest 缺失 `gate_granularity=block`、block/config 冲突、gate key 缺失或目录无完成标记都会拒绝加载。

## 4. H100 性能选型

先用独立子进程做单层预筛：

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh bsa-benchmark
```

该结果只用于淘汰明显慢/OOM 的 chunk，不能宣称端到端加速。随后对每个候选使用独立输出目录，运行真实 4-step forward + backward + optimizer step；至少覆盖 K248、K165、K110、K55，并以所有 rank 的最慢 step / 最大 allocated-reserved / 最小 free memory 选型。

门禁：

```text
目标 allocated <=70 GiB，reserved <=73 GiB，free >=6 GiB
硬停止 allocated/reserved >=75 GiB，或 free <4 GiB
```

固定顺序：`Cq=4/full checkpoint/anchor off` 正确性 → 扫 `4,8,16,32` 和 additive/bool → 必要时 64/128 → 50-step 无泄漏。两个候选速度差不超过 3% 时选择显存更低者。不得由当前 dense 约 37/81G 推断可以 batch=2、关闭全部 checkpoint 或常驻 teacher。

## 5. A/B/C 质量验证

设置组合 checkpoint 后：

```bash
BSA_CHECKPOINT=/path/to/checkpoint-step-... \
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh bsa-validate
```

输出 teacher、A=dense warm-start、B=joint-trained dense、C=joint-trained 80% sparse 的 MP4、latent 和 `validation.json`。至少检查 12 个未见首帧/seed，最好 50 个以上，重点看身份、脸手配件、360°运动、轮廓、闪烁、尾帧和画面右边缘。

初始门槛：C 相对 teacher 的 endpoint MSE 不超过 A 的 1.10 倍；B 不出现系统性退化；盲评 C 相对 A 的胜出+持平比例至少 90%。

## 6. Ragged 与 Ascend

P0 `fixed_padded` 已正确 mask partial blocks；`--bsa_ragged_count_bias` 可独立改变 coarse routing。`compact_ragged` 是 P1 reference：会在每个 query chunk 中压缩 invalid selected-KV lane，并与 fixed 数学/梯度对齐；当前实现含 host 同步，仅用于消融诊断，未在 H100 证明更快，不能设为默认或宣称加速。

Ascend bring-up 顺序：CPU eager fixture → NPU BF16 SDPA forward/backward → additive/bool mask → target shape → 与 H100 同一 Q/K/V/Top-K fixture 对齐 → profiler → 再决定是否新增独立 `ascend_fused` provider。当前代码不包含 `.cuda()`/`.npu()`、torch_npu、Triton 或自定义 CUDA 依赖。
