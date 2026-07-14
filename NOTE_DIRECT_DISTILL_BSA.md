# Wan2.2-TI2V-5B + Figurine360 DirectDistill BSA（内网 H100）

本说明只覆盖 BSA 增量。teacher latent 的准备与校验继续按 `NOTE_DIRECT_DISTILL.md` 执行。当前 Stage A 已完成 CPU 合成验证；真实 5B、H100 性能/质量和 Ascend 均待内网验证。

## 1. 内网 H100 快速部署、配置与检查

把本仓库同步到内网机器后，在已有 CUDA/PyTorch 环境中使用内网 wheel 源安装；脚本本身不会下载模型：

```bash
cd /data/code/DiffSynth-Studio_3D_toys
python -m pip install -e .
python -m pip install matplotlib pytest
```

可以修改 BSA shell 顶部变量块，也可以直接导出同名环境变量。下面显式启动 6 卡，不依赖机器上已有的 Accelerate 默认配置：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NUM_PROCESSES=6
export REPO_ROOT=/data/code/DiffSynth-Studio_3D_toys
export DIT_PATHS_JSON='["/data/models/Wan2.2-TI2V-5B/model-00001.safetensors","/data/models/Wan2.2-TI2V-5B/model-00002.safetensors","/data/models/Wan2.2-TI2V-5B/model-00003.safetensors"]'
export TEXT_ENCODER_PATH=/data/models/models_t5_umt5-xxl-enc-bf16.pth
export VAE_PATH=/data/models/Wan2.2_VAE.pth
export TOKENIZER_PATH=/data/models/google/umt5-xxl
export FIGURINE360_LORA=/data/weights/figurine360.safetensors
export DENSE_WARMSTART_LORA=/data/weights/direct-distill-step-19600.safetensors
export TEACHER_ROOT=/data/teacher/figurine360-w832-h480-f81-seed1
export BSA_TRAIN_OUTPUT=/data/outputs/bsa-directdistill-conservative-v1
export NUM_EPOCHS=60
export GRADIENT_ACCUMULATION_STEPS=1
export BSA_SPARSITY_SCHEDULE=conservative_epoch_v1
unset ACCELERATE_CONFIG BSA_CHECKPOINT

SCRIPT=examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh
bash "$SCRIPT" doctor
test ! -e "$BSA_TRAIN_OUTPUT"
nohup bash "$SCRIPT" train >"${BSA_TRAIN_OUTPUT}.launch.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"${BSA_TRAIN_OUTPUT}.pid"
tail -F "${BSA_TRAIN_OUTPUT}.launch.log"
```

`BSA_TRAIN_OUTPUT` 必须完全不存在，空目录也会被拒绝。端口冲突时可加 `MAIN_PROCESS_PORT=29501`；已有审核过的 Accelerate YAML 时也可改用 `ACCELERATE_CONFIG=/abs/path/accelerate.yaml`。正式 teacher 必须以 `height=480,width=832,num_frames=81`（显示尺寸 W×H 为 832×480）、seed=1 生成。Wan VAE 将 81 帧压缩为 21 帧 latent，首次 forward 会强制核对 DiT grid `(21,15,26)`；不能复用 dense 默认的 `height=480,width=832,num_frames=49` teacher cache。

如果还没有 81 帧 teacher cache，先用 dense DirectDistill 的准备入口写到独立目录；其余模型和原始数据路径沿用该脚本的同名环境变量：

```bash
DENSE_SCRIPT=examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360.sh
FORMAL_HEIGHT=480 FORMAL_WIDTH=832 FORMAL_NUM_FRAMES=81 \
TEACHER_ROOT=/data/teacher/figurine360-w832-h480-f81-seed1 \
bash "$DENSE_SCRIPT" prepare
```

`train` 与原命令 `bsa-train` 完全等价。两者都会在加载 5B 模型前遍历 metadata，确认所有行都是上述 H/W/F、seed=1，且 `input_image`/`teacher_latent` 文件存在。

正式设置固定为 block `(4,3,6)`、rank-32 block gate、student 4 steps / CFG 1 / shift 5。首次正式实验建议至少 3000 个真实 optimizer steps；程序低于 1500 steps 会警告。

当前正式默认 `NUM_EPOCHS=60` 和 `conservative_epoch_v1`。在 6 卡、batch=1/rank、gradient accumulation=1、721 行训练 metadata 时，prepared DataLoader 为 121 batches/rank/epoch，总计 7260 个 optimizer steps。默认调度为：epochs 1–3 使用 0%；epochs 4–10 依次使用 10%–70%，每档一个 epoch；epoch 11 使用 75%；epochs 12–60 使用 80%。对应 completed-step 边界是 `363, 484, 605, 726, 847, 968, 1089, 1210, 1331`，因此第 1332 次 optimizer update 开始使用 80%，80% 档共训练 5929 次 update。

## 2. 冒烟、正式训练与快速修改 warm-up

先确认原 DirectDistill teacher 数据和 dense warm-start 均已通过验收，再运行：

```bash
bash "$SCRIPT" bsa-smoke-train
bash "$SCRIPT" train
```

两步 smoke 只验证链路，默认使用 `legacy_progress_v1`；正式训练才使用保守 epoch 调度。启动时必须看到：

- 30/30 个 `self_attn` 注入，cross-attention 为 0；
- `gate=low_rank_dynamic/block`，`gate_up_zero_initialized=true`；
- DirectDistill LoRA 与 BSA gate 是仅有的 trainable 参数，且均为 FP32；
- backend forward/backward probe 通过；
- 日志打印 `steps_per_epoch`、总步数、全部 transition steps 与 sparsities；
- 第一次真实 forward 的 runtime grid 为 `(21,15,26)` 时，得到 `N=150`、最终 `K=30`、padding ratio 约 24.17%；
- `student_model_info.json` 同时包含调度、初始化摘要和第一次 forward 的实际 shape。

默认 `sdpa_gather` 先 gather selected KV，再调用 PyTorch SDPA；不会构造全局 `[8190,8190]` mask。稀疏率只按成功的 optimizer step 变化，gradient accumulation 的 micro-batch 和被跳过的 optimizer step 都不推进 schedule。forward、metrics 和 checkpoint 共用同一个编译结果。

常用方案只改一个环境变量，不改 Python：

```bash
# 本次 19600-step dense DirectDistill 初始化的保守方案
BSA_SPARSITY_SCHEDULE=conservative_epoch_v1 bash "$SCRIPT" bsa-train

# 仅当 dense 权重已在完全相同 shape/数据域充分收敛时使用
BSA_SPARSITY_SCHEDULE=mature_same_shape_v1 bash "$SCRIPT" bsa-train

# 旧版按总进度前 40% 逐档增加的兼容方案
BSA_SPARSITY_SCHEDULE=legacy_progress_v1 bash "$SCRIPT" bsa-train
```

实验方案写入外部 JSON，例如 `/data/configs/bsa_warmup.json`：

```json
{
  "schema_version": 1,
  "basis": "epoch",
  "stages": [
    {"name": "dense", "sparsity": 0, "duration": 4},
    {"sparsity": 0.1, "duration": 1},
    {"sparsity": 0.2, "duration": 1},
    {"sparsity": 0.4, "duration": 1},
    {"sparsity": 0.6, "duration": 1},
    {"sparsity": 0.7, "duration": 1},
    {"sparsity": 0.75, "duration": 1},
    {"sparsity": "target", "duration": "remainder"}
  ]
}
```

```bash
BSA_SPARSITY_SCHEDULE=@/data/configs/bsa_warmup.json bash "$SCRIPT" bsa-train
```

`basis` 还支持 `progress` 和 `optimizer_step`。最后一档写 `target` 会绑定 `BSA_TARGET_SPARSITY`；若写数值，则必须与 target 一致。新实验改变 epochs、卡数、GA 或未来真正支持的 per-device batch 后，边界会从 prepared DataLoader 自动重算。当前数据管线尚不支持真正 per-device batch=2；直接改 DataLoader 会丢样本。现阶段要把大部分 update 的有效全局 batch 从 6 提到 12，设置 `GRADIENT_ACCUMULATION_STEPS=2`。此时 721 行/6 卡对应 61 steps/epoch，边界自动变为 `183, 244, 305, 366, 427, 488, 549, 610, 671`。

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

当前 P0 保存的是严格的组合权重 continuation：LoRA、gate、BSA config、完整 schedule spec/hash/runtime 与已完成 step 均恢复；不保存 Adam、LR scheduler、RNG 或 dataloader 游标，因此 manifest 明确写入 `resume_semantics=weight_continuation`，不能称为逐 bit 精确恢复。续训必须使用新的输出目录，并设置：

```bash
BSA_CHECKPOINT=/path/to/checkpoint-step-0000200 \
BSA_TRAIN_OUTPUT=/path/to/new-output \
bash "$SCRIPT" bsa-train
```

manifest 缺字段、block/config 冲突、gate key 缺失、schedule/hash/runtime 被修改、目录无完成标记，都会拒绝加载。schema-v2 续训由 checkpoint 中保存的 schedule spec 直接恢复，不依赖原 preset 名或外部 JSON 文件；同时要求每 epoch optimizer steps 和 `NUM_EPOCHS` 与原计划一致。这里的 epochs 表示整个实验计划，不是额外追加的 epochs。达到 manifest 总 step 后 runner 会精确停止。要改变训练几何或 warm-up，应从 dense DirectDistill 权重开启新的 BSA 实验，而不是把旧 BSA checkpoint 当作同一次续训。

旧 schema-v1 checkpoint 没有完整 schedule runtime，只能按 `legacy_progress_v1` 恢复。若旧任务使用 `GA>1` 且每 epoch batches 不能被 GA 整除，旧 runner 保存的总步数没有计入每 epoch flush，无法满足新版本的严格 geometry 校验；这种 checkpoint 应作为权重来源另开实验，不能宣称为同一计划的严格续训。

## 4. H100 性能选型

先用独立子进程做单层预筛：

```bash
bash examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh bsa-benchmark
```

该结果只用于淘汰明显慢/OOM 的 chunk，不能宣称端到端加速。随后对每个候选使用独立输出目录，运行真实 4-step forward + backward + optimizer step；至少覆盖 K135、K90、K60、K30，并以所有 rank 的最慢 step / 最大 allocated-reserved / 最小 free memory 选型。

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

输出 teacher、A=dense warm-start、B=joint-trained dense、C=joint-trained target-sparse 的 MP4、latent、三联视频 `teacher_vs_student_vs_student_bsa.mp4` 和 `validation.json`。三联中的 `student` 是 BSA 前的 dense warm-start，`student_bsa` 是组合 checkpoint 内联合训练后的 LoRA + BSA adapter，并使用 manifest 的 `target_sparsity`；B=joint-dense 继续作为单独诊断分支。训练与验证 metadata 都固定为 seed=1；至少检查 12 个未见对象/首帧，最好 50 个以上，重点看身份、脸手配件、360°运动、轮廓、闪烁、尾帧和画面右边缘。

若要直接复用原 DirectDistill Test 使用的 `input_image,prompt` 测试集（不含 `teacher_latent`），使用薄 BSA Test 入口：

```bash
TEST_SCRIPT=examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA-Test.sh
export BSA_CHECKPOINT=/data/outputs/bsa-directdistill-conservative-v1/checkpoint-step-...
export DENSE_WARMSTART_LORA=/data/weights/direct-distill-step-19600.safetensors
export TEST_METADATA=/data/test/metadata.csv
export TEST_OUTPUT=/data/outputs/bsa-directdistill-test

bash "$TEST_SCRIPT" doctor
SAMPLE_INDEX=0 bash "$TEST_SCRIPT" one
START_INDEX=0 END_INDEX=12 bash "$TEST_SCRIPT" all
```

`all` 在单个 Python 进程中执行：teacher 与 dense warm-start 共用第一套 pipeline，checkpoint 内的 joint-dense 与 joint-sparse 共用第二套 pipeline；不会按样本重复加载 5B 模型。每条完整样本最后才写 `validation.json`，因此默认 `SKIP_EXISTING=1` 可安全断点续跑；设 `SKIP_EXISTING=0` 可覆盖重跑。

初始门槛：C 相对 teacher 的 endpoint MSE 不超过 A 的 1.10 倍；B 不出现系统性退化；盲评 C 相对 A 的胜出+持平比例至少 90%。

## 6. Ragged 与 Ascend

P0 `fixed_padded` 已正确 mask partial blocks；`--bsa_ragged_count_bias` 可独立改变 coarse routing。`compact_ragged` 是 P1 reference：会在每个 query chunk 中压缩 invalid selected-KV lane，并与 fixed 数学/梯度对齐；当前实现含 host 同步，仅用于消融诊断，未在 H100 证明更快，不能设为默认或宣称加速。

Ascend bring-up 顺序：CPU eager fixture → NPU BF16 SDPA forward/backward → additive/bool mask → target shape → 与 H100 同一 Q/K/V/Top-K fixture 对齐 → profiler → 再决定是否新增独立 `ascend_fused` provider。当前代码不包含 `.cuda()`/`.npu()`、torch_npu、Triton 或自定义 CUDA 依赖。
