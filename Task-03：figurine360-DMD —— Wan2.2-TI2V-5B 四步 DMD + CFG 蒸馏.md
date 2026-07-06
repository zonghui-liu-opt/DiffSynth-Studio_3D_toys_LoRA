# Task-03：figurine360-DMD —— Wan2.2-TI2V-5B 四步 DMD + CFG 蒸馏（Phase 1 / 共三期）

> 执行者：Codex + GPT-5.5 Pro。总原则：**能复用参考仓库已有实现的一律复用，所有改动原地修改（in-place）、加配置开关、默认关闭、不做无关重构**。每个改造点独立 commit，附 2-cycle 小分辨率冒烟测试。

**（Stage A：MacBook 本地调试 → push GitHub → Stage B：内网 H100 正式训练）**

**项目分期**：Phase 1（本期）= 纯 DMD + CFG 蒸馏；Phase 2 = Block Sparse Attention 与 DMD **同阶段**联合训练+推理；Phase 3 = BSA 与 **DMD2**（含 GAN 项）同阶段联合训练+推理。本期必须为 Phase 2/3 预留接口（见 5.7 与第 10 节），但不实现。

## 0. 角色、工作流与总原则（必须遵守）

你是资深 ML infra 工程师，工作流分两阶段：

- **Stage A（当前，你全程参与）**：MacBook（Apple Silicon，CPU-only，无 CUDA / 无 flash_attn / 无真实数据 / 无模型权重）。你需要自己生成合成调试数据，把一切可离线验证的逻辑测通，然后 push GitHub。
- **Stage B（你不在场）**：内网 H100 80G × 2~4，**无外网**（不能访问 HF/ModelScope 自动下载权重），真实数据与权重均在本地磁盘。用户将手动按你写的 NOTE_DMD.md 操作。

硬性约束：

1. **不重复造轮子**：蒸馏训练基座是/Users/zonghuiliu/Documents/Codex/Wan2.2-TI2V-5B-Turbo`（详见第 2 节），其**双向模型 DMD 训练循环**、I2V 首帧条件、四步网格、EMA、FSDP、wandb 已是现成实现，一律直接适配启用。我们已有的 DiffSynth-Studio 工程（基于**原生 Wan2.2-TI2V-5B** 的 LoRA SFT 训练+推理+可视化，rank=32、`lora_target_modules "q,k,v,o,ffn.0,ffn.2"`、`--extra_inputs "input_image"`、`--remove_prefix_in_ckpt "pipe.dit."已完成`）在本任务中承担两件事：**离线 merge figurine360 LoRA 产出 teacher 权重**、**最终 LoRA 产物的推理端验证**。动手前先通读 Turbo 仓库 `train.py` 入口与 `configs/*.yaml` 全部字段，以及 DiffSynth 工程的 LoRA 保存/加载 key 约定。
2. **原地最小侵入**：新增逻辑用配置开关分支挂在现有路径上（该仓库用 OmegaConf YAML 而非 argparse，开关加在 config 字段上），默认关闭时与上游行为完全一致。禁止另起训练循环。
3. **Stage B 只改配置不改代码**：所有环境差异（路径、卡数）必须收敛到脚本顶部变量块。任何散落的硬编码路径都算 bug。
4. **新增的工具模块必须可脱离 GPU/权重独立 import 和测试**（不得在 import 时触发 CUDA 初始化或加载模型）。注意：Turbo 仓库依赖 flash_attn，Stage A 需给相关 import 加 try/except 守卫回退到 torch SDPA——这是唯一允许的前置代码改动，GPU 路径行为必须不变。
5. 逻辑精准：所有统计量定义写进注释；参数名以仓库当前源码为准，不凭记忆猜 API。每完成一个子任务，输出 diff 摘要 + 自测结果。

## 1. 背景与目标

已有资产：Wan2.2-TI2V-5B 原生底模；figurine360 LoRA（Task-01 产物，领域=单手办 360° 环绕旋转，rank=32，挂在 qkvo + 2 个 ffn 上）；约 700 条已配对的视频/caption 数据（训练 TI2V 任务）。

目标：训练一个 **DMD 蒸馏 LoRA（rank 64）**，使 `base + figurine360 + DMD-LoRA` 在 **4 步、cfg=1（单 forward）** 下逼近 `base + figurine360` 在 50 步 + CFG 下的生成质量。CFG 蒸馏内嵌于 DMD 的 real score 定义中，不是独立损失。fake score 与 generator 按 **5:1** 更新。本期（Phase 1）**不做 GAN 项、不做 BSA**，但两者接口必须预留，且交付文档中写明 Phase 2（BSA+DMD 同阶段训练）与 Phase 3（BSA+DMD2 同阶段训练）的实现路径（第 10 节）。

## 2. 代码基座与复用边界

基座：/Users/zonghuiliu/Documents/Codex/Wan2.2-TI2V-5B-Turb。**选择理由**：

- **双向（非自回归）DMD 已验证**：该仓库虽构建于 Self-Forcing 代码库之上，但仅借用其训练基础设施（DMD 三模型循环、OmegaConf 配置、EMA、分布式）；蒸馏对象与产物都是**标准双向全注意力的 Wan2.2-TI2V-5B**，不是 CausVid/Self-Forcing 主线的因果/分块自回归学生。外部证据：官方 Turbo 权重可被社区当作标准 5B 权重直接使用——转 Diffusers 后用标准管线 4 步 cfg=1 一次性生成 121 帧全序列；其相对底模的 delta 已被提取为 rank-64 LoRA 并 merge 回原版底模发布；与常规 5B 的 LoRA 生态兼容。这与我们"4 步 cfg=1 双向单 forward"的目标形态完全一致。
- 基于**原生 Wan2.2-TI2V-5B 权重格式**（官方 `wan_models/Wan2.2-TI2V-5B` 目录结构），与我们 DiffSynth SFT 工程同源同格式——merged teacher 权重可直接互通，无需任何格式转换。
- 已实现 **TI2V/I2V 首帧条件下的 4 步 DMD + CFG 蒸馏**并放出成品模型（4 步、cfg-free、121×704×1280）——原方案中工作量最大的"TI2V 条件注入"已被上游消化，降级为对齐验证项（5.2）。

**仓库结构速览**（以实际代码为准，不要信任本段写死的路径）：

- 环境安装：`pip install -r requirements.txt` + `pip install flash-attn --no-build-isolation` + `python setup.py develop`。
- 训练入口：`running_scripts/train/Wan2.2/dmd.sh` → `train.py --config_path configs/<xxx>.yaml`（OmegaConf YAML 配置）。
- 推理入口：`running_scripts/inference/Wan2.2/i2v_fewstep.sh` → `inference.py`（支持 `--use_ema`）。
- DMD 实现：`model/dmd.py`（三模型结构：generator / 可训练 fake score / 冻结 real score）。

**复用不改**：双向全序列 DMD 训练循环、DMD 损失与归一化、generator rollout 模拟前向与梯度截断、四步网格机制（`denoising_step_list` + `warp_denoising_step`，shift=5 warp 后即 1000/937.5/833.3/625）、I2V 首帧条件路径、负向 prompt 处理、`denoising_loss_type: x0`（Wan2.2-5B 专用，勿改成 Wan2.1 的 flow）、EMA、FSDP/梯度检查点、wandb 记录。

**需要改造（详见 Step 2）**：a) 三模型 LoRA 化（本期最大工作量项）；b) teacher/初始化权重指向 merged figurine360；c) 数据集适配 700 条 caption+首帧；d) 5:1 更新比确认与暴露；e) CFG scale 参数化确认；f) LoRA 单文件导出；g) Phase 2/3 接口预留。

**定位方式**：不要信任本档案写死的文件路径，用 grep 锚点定位：`denoising_step_list`、`warp_denoising_step`、`dfake_gen_update_ratio`（Self-Forcing 系惯用的 fake/gen 更新比参数名）、`real_guidance_scale`、`fake_guidance_scale`、`fake_score`、`denoising_loss_type`、`ema`、`i2v`、`causal`。凡本文档与源码不一致，以源码为准并在 diff 摘要中记录。

**Step 0 必须确认的四件事**（读代码即可，不用跑训练）：
 ① **双向路径确认**：grep `causal` / `kv_cache` / attention mask 相关代码，确认 Wan2.2 训练配置实例化的是**全注意力（双向）generator wrapper**、rollout 为全序列 4 步去噪而非分块自回归——上游 Self-Forcing 代码库同时含因果与双向变体，社区曾有人误判 Turbo 为自回归运行，必须以代码事实为准（此项与 5.2 的 50 步对齐验证互为交叉证据）；
 ② `dmd.sh` 的 generator 初始化来源（TODO: 想想应该用什么初始化, 能加速训练收敛)
 ③ 训练数据的组织格式（grep dataset 类：prompt 是 txt/csv/lmdb？首帧图像如何喂入？text embedding 在线计算还是离线缓存？）；
 ④ real score 的 CFG 数学形式（见 5.5）。

## 3. Step 0 —— 基线打通（不改一行代码）

- **Stage A**：clone 仓库，装 CPU 依赖（跳过 flash-attn），加 flash_attn import 守卫（见 0-4），确认：configs 可解析、`model/dmd.py` 可 import、DMD 损失在合成 latent 上可前向、上述"必须确认的四件事"有结论并写入文档。
- **Stage B（写入 NOTE_DMD.md checklist）**：原样运行官方 `i2v_fewstep.sh`（官方 Turbo model.pt + 原版底模）验证推理环境；原样运行 `dmd.sh` 50 iter（缩小分辨率）验证训练循环、checkpoint 保存、validation、wandb 全部工作。记录基线单 forward 耗时（480P/720P 各一次）与峰值显存，作为后续预测校准。

**产出**：`docs/task03_baseline.md`。

## 4. Step 1 —— 权重与数据准备

**方案决策（已定案，勿改道）**：figurine360 采用**离线 merge**而非训练时双 LoRA 挂载。理由：学生能否获得领域能力取决于 real score 是否包含 figurine360——若 real score 为裸 base，DMD 会把学生初始化中的领域能力当作"与真值分布的偏差"主动抹掉；merge 以**构造方式**保证三个角色（generator 初始化 / real score / fake score 初始化）携带同一份领域权重且 scale 完全一致，零代码改动。训练时双 LoRA 在数学上等价（LoRA 为线性 delta），但引入多 adapter 叠加顺序、alpha/scale 换算（DiffSynth 与 peft 约定不同）、requires_grad 隔离、peft 多 adapter × FSDP 兼容性等一串**静默错误面**，收益为零。禁止实现双 LoRA 训练路径。

### 4.1 merge 脚本与 gate-0 等价验证（**先行交付，第一个独立 commit**）

**脚本**：先 grep DiffSynth 工程有无现成 LoRA merge 工具，有则薄封装，无则写 `tools/merge_fig360_lora.py`。输入 = 原生底模目录 + figurine360 LoRA safetensors；输出 = 与 `wan_models/Wan2.2-TI2V-5B` 目录结构和 state_dict key 完全一致的 `Wan2.2-TI2V-5B-fig360/`。要求：merge 数学 `W' = W + (alpha/rank)·(B@A)` 以 **fp32 累加后 cast 回原 dtype（bf16）**，避免低精度舍入干扰后续逐像素对齐；正确处理 `--remove_prefix_in_ckpt "pipe.dit."` 的 key 映射；自动输出校验报告——LoRA 命中层清单、命中层数等于 6 类 target × 层数的预期值断言、**未命中层与原版 bit 级一致断言**。

**Stage A 单测（无权重，CPU 可跑）**：构造 tiny 随机 state_dict + 人造 LoRA（已知 A/B/alpha/rank），断言 merge 结果与手算 fp32 参考一致、未命中 key bit 级一致、前缀映射正确。

**Stage B gate-0（写入 NOTE_DMD.md 第一条命令，含真实推理）**：同一 DiffSynth pipeline、同 seed、同 prompt 与首帧，**路径 A** = base + 运行时挂载 figurine360 LoRA，**路径 B** = merged 权重不挂任何 LoRA。先跑 1 条 10 步低分辨率快速版，通过后跑 1 条 50 步正式版。判定标准：解码后逐像素 max|diff| ≤ 2/255（或 latent 域 MSE ≤ 1e-6 量级）——bf16 下运行时的 `Wx + (alpha/rank)·B(Ax)` 与预 merge 的 `(W+ΔW)x` 存在合理舍入差，**不要求 bit 级一致**。未通过时第一排查项：alpha/rank scaling 约定与 key 前缀映射。**gate-0 未通过前，禁止进入 5.2 与任何训练。**

**验证链顺序**：gate-0（DiffSynth 内部自等价，锁定 merge 正确性）→ 5.2（跨仓库对齐，锁定 Turbo 仓库条件路径正确性）→ 训练。

训练配置中 generator 初始化、real score、fake score 三者全部指向这份 merged 权重——天然实现"figurine360 在三个角色中都挂载且冻结"，且 real/fake/generator 初始化一致（fake 从 teacher 出发 = 标准做法）。

### 4.2 数据准备

按 Step 0-③ 确认的格式组织 700 条数据：caption + 首帧图像（从视频抽第 0 帧即可；视频本体的 latent 留给 Phase 3 的 GAN 项作真样本，本期数据集类只留开关不加载，见 5.7）。若仓库在线计算 text embedding 则无需离线缓存；确认 uncond 分支使用 **Wan 标准负向 prompt**（与我们 50 步推理用的完全一致）。条件扩充：对 700 条 caption 各生成 1–2 条改写变体（保留"360° 旋转/turntable"核心语义），条件库扩到约 1500–2000 条。

## 5. Step 2 —— 核心改造（逐项，全部原地修改）

### 5.1 四步网格（降级为纯确认项）

预期仓库默认即为 4 步：`denoising_step_list: [1000, 750, 500, 250]` + `warp_denoising_step: true`，经 shift=5 warp 后实际为 1000/937.5/833.3/625（即 t∈{1, 0.75, 0.5, 0.25} 的映射）确认启动脚本写shift前的还是shift后的。确认训练与 validation/推理使用同一份配置即可；若默认值不同则改配置，不改代码。**警告**：`warp_denoising_step` 与 `denoising_loss_type: x0` 是该仓库在 Wan2.2-5B 上的正确组合，不得凭 Wan2.1 经验改动。

### 5.2 TI2V 首帧条件（由"最大工作量项"降级为"对齐验证项"）

**前置条件：4.1 的 gate-0 已通过。**本节只负责验证两套代码库（Turbo 仓库 vs DiffSynth）条件路径的语义一致性；merge 本身的正确性已被 gate-0 单独锁定，本节对齐失败时不必回头怀疑 merge。

仓库已原生实现 I2V 蒸馏。逐项验证其语义与官方 Wan2.2 TI2V / 我们 DiffSynth SFT 推理完全一致：首帧图像经 VAE 编码为 clean latent 替换 latent 的第一个时间片；该帧 per-frame timestep 置 0（TI2V-5B 帧级 timestep 机制）；rollout 4 步采样全程保持首帧 latent 固定；real score 的 cond 与 uncond 两分支、fake score 分支三处均注入首帧条件；**CFG 只对文本做差分，首帧条件在正负分支中都保留**；若有 loss mask，确认首帧被排除。验证方法保持不变：用仓库条件路径 + merged teacher 权重跑 50 步推理，输出必须与 DiffSynth 工程的 ti2v 推理在同 seed 下逐像素接近（允许数值误差），**此项通过前不得进入训练**。该验证同时兼作 Step 0-① 双向路径确认的实证（50 步全序列去噪能对齐官方推理 ⇔ 注意力为双向全注意力）。

### 5.3 三模型 LoRA 化（本期最大工作量项）

generator 与 fake score 各注入一组可训练 LoRA：rank 64、alpha 64、target = attention qkvo + FFN 全线性层、B 零初始化；base 权重全冻结；teacher（real score）保持全冻结无 LoRA（figurine360 已 merge）。turbo仓库原生是**全参训练**（产物是 20GB model.pt），LoRA 支持需要新增：先 grep `lora` 确认仓库有无现成痕迹；无则优先用 peft，或手写约百行 LoRALinear 原地包裹目标层，挂 `lora:` 配置块默认关闭。与 FSDP 组合有坑则降级为单节点 DDP + 梯度检查点（LoRA-only 优化器状态很小，见第 8 节显存预测）。checkpoint 只保存两组 LoRA + 优化器状态；导出物为 `figurine360_dmd_lora_rank64.safetensors`（单文件，ComfyUI 命名规范，且必须能被 DiffSynth 工程推理加载——两条路径都要实测）。

**部署约束（必须写进导出说明与 NOTE_DMD.md）**：DMD LoRA 是**相对 merged 权重**训练的 delta，推理时必须以 `base + figurine360 + DMD-LoRA` 三件套（由 LoRA 线性性，运行时双 LoRA 与 `merged + DMD-LoRA` 严格等价，故无需分发 merged 权重）加载；**单独把 DMD LoRA 挂到裸 base 上不成立**。ComfyUI 与 DiffSynth 两条实测路径均按三件套配置验收。

### 5.4 5:1 TTUR

grep `dfake_gen_update_ratio`——Self-Forcing 系通常**已内置**该参数。若存在：确认其语义正是"每 N 次 fake/critic 更新对应 1 次 generator 更新"，直接设 5；若不存在或语义不符，加配置项默认 1 保持向后兼容。实现语义：以 cycle 为单位，每 cycle 执行 5 次"student 完整 4 步采样(no_grad) → fake score 加噪去噪损失 → 仅 fake LoRA step"，然后 1 次"随机出口步模拟前向 → DMD 梯度 → 仅 generator LoRA step"（随机出口步机制以仓库现有 rollout 实现为准）。注意三个坑：lr scheduler 与 max iter 按 **generator 步数**计数；梯度累积只作用于 generator 分支；wandb 中 fake loss 与 generator 更新数对齐记录（fake loss 取 5 次均值）。**严格用更新频率实现时间尺度差，两边 lr 相同，不做"fake lr ×5"的替代**。

### 5.5 CFG 蒸馏参数换算（已知坑，必读）

teacher CFG 在不同代码库有两种参数化：标准 Ho & Salimans 形式 `uncond + w·(cond − uncond)`，与 DMD2 参考实现的"附加引导"形式（数学上差一个常数偏移，`w_附加 = g − 1`）。本仓库大概率是标准形式，但**必须以代码为准**。执行判定：grep real score forward 中的 CFG 组合表达式——若为标准形式，直接设 **5.0**（等于 teacher 50 步推理惯用的 g=5）；若为附加参数化，设 **4.0**。判定依据（代码行号+表达式）写进 diff 摘要。uncond 分支必须喂 Wan 标准负向 prompt embedding 而非空串（grep 确认默认行为，不对则改）。student 与 fake 全程单条件 forward（确认 `fake_guidance_scale` 为 0 或等效关闭）。训练完成后推理 cfg=1——社区经验一致确认：蒸馏产物若仍用高 CFG + 多步推理会直接过曝崩坏；该仓库推理本就 cfg-free，validation 路径确认 cfg 硬编码为 1 即可。

### 5.6 EMA

仓库已有 EMA 实现（推理侧有 `--use_ema`），直接复用：decay 设 0.995，validation 与最终导出均用 EMA 权重。唯一改动点：LoRA 化后确认 EMA 只跟踪可训练（requires_grad）的 LoRA 参数，不对冻结 base 做全量影子拷贝。

### 5.7 Phase 2/3 接口预留（本期只留钩子，不实现）

- **注意力后端抽象**（为 Phase 2 BSA）：将 DiT 的注意力调用收敛到唯一 wrapper（grep 原生 Wan 的 attention 模块，Stage A 的 flash_attn 守卫可与此合并），加配置项 `attention_backend: flash`（默认，行为与上游完全一致），预留 `bsa` 分支位。
- **判别器钩子**（为 Phase 3 GAN）：fake score forward 加 `return_features: false` 开关（默认关闭，零开销），预留中间特征出口；损失聚合处加 `gan_loss_weight: 0.0` 配置字段（本期恒为 0，不产生任何计算分支）。
- **真样本数据位**（为 Phase 3 GAN）：数据集类加 `load_video_latent: false` 开关，Phase 3 判别真样本需要视频 latent。
- 每个钩子各附一个"开关打开也不崩（返回 dummy/直通）"的单测，可在 Stage A CPU 上跑通。

### 5.8 可视化

阅读当前代码仓SFT的loss曲线, 类似的生成generator和fake score model的loss曲线和吞吐等.

## 6. Step 3 —— 训练执行

两套配置。**冒烟配置**：480×832×81、单卡、200 cycle，验证不炸、fake loss 下降、validation 视频非噪声。**正式配置**（假设内网单机 8×H100-80G，H200 则耗时 ×0.8）。关键字段如下（字段名以 5.4/5.5 grep 确认后的真实名称为准，此处为语义示意）：

```yaml
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true          # shift=5 → 实际 1000/937.5/833.3/625
timestep_shift: 5.0
i2v: true                          # 仓库已有 I2V 开关，以实际字段名为准
denoising_loss_type: x0            # Wan2.2-5B 用 x0，保持仓库默认
dfake_gen_update_ratio: 5          # 5.4
real_guidance_scale: 5.0           # 5.5 判定后若为附加参数化则改 4.0
fake_guidance_scale: 0.0
lora:                              # 5.3 新增配置块，默认 enabled: false
  enabled: true
  rank: 64
  alpha: 64
  targets: [q, k, v, o, ffn]
lr: 5.0e-5                         # generator 与 fake 同值（LoRA 用 5e-5，非全参的量级）
batch_size: 1                      # 8 卡 DP → 有效 batch 8
max_iters: 3000                    # generator 步数；
ema_decay: 0.995
validation_interval: 200           # 固定 12 条留出条件，4 步 cfg=1
attention_backend: flash           # 5.7 预留，本期恒为 flash
gan_loss_weight: 0.0               # 5.7 预留，本期恒为 0
```

分辨率决策：**与 figurine360 训练数据同分辨率**训练（领域 LoRA 在什么分辨率上训的就在什么分辨率上蒸），不必上 121×704×1280 满格。checkpoint 每 200 generator 步保存，保留全部（后验挑点很重要，DMD 质量非单调）。

## 7. Step 4 —— 验证协议与验收标准

固定 12 条留出条件（覆盖手办品类/材质/背景），每 200 步生成 4 步 cfg=1 视频，四项指标按优先级：**旋转完整性**（人工 + 光流估计角速度：转满 360°±10%，角速度均匀性方差不劣于 teacher 的 1.5 倍）——这是梯度截断运动退化的探针；**身份一致性**（首帧与后续帧的 DINO 特征相似度 ≥ teacher 50 步基线的 95%）；**无过曝/过饱和**（帧均亮度、饱和度直方图对比 teacher 偏移 <10%）；**帧间亮度稳定**（无周期性震荡）。最终验收：4 步单 forward 端到端延迟 ≤ teacher(50 步×2 forward) 的 1/20；上述四项全过；LoRA 单文件在 **ComfyUI 与 DiffSynth 两条推理路径**均可加载出图（按 5.3 部署约束，两条路径均为 `base + figurine360 + DMD-LoRA` 三件套配置）。

## 8. 内网训练结果预测

以下预测基于 8×H100、480×832×81 训练分辨率、无 Turbo 初始化的假设，Step 0 基线实测后按比例修正（外部锚点：官方全量蒸馏 4000 iter / 16×A100 / <48h @ 704×1280×121；我们 LoRA-only + 半分辨率，单位 iter 成本应显著低于该值）。

**耗时**：每 cycle ≈ 44 个 forward 等价量（5 次 fake 更新各含 4 次采样 forward + 1 次 fwd/bwd；1 次 generator 更新含 ≤3 次模拟 forward + teacher 双 forward + fake forward + 带反传的 student forward）。480P 下 5B 单 forward 约 0.6–0.9 s → **25–40 s/cycle**，3000 cycle 约 **21–33 小时**；720P×121f 约 ×3–4 → 3–5 天。

**显存**（峰值/卡）：三份 5B bf16 权重 30 GB（FSDP 8 卡分片后权重项很小；若走 DDP 则 30 GB 常驻）+ LoRA 优化器 <1 GB + 全梯度检查点下激活。DDP 路径预计 480P 峰值 **45–55 GB**，720P 65–78 GB（贴边，建议 720P 走 FSDP 或激活 offload）。

**损失曲线形态**：fake score loss 前 100–200 cycle 快速下降后进入**低位小幅震荡的动态平衡**——它在追逐移动的 student 分布，持续走平是健康状态；单调降到极低说明 student 停止进化（检查 generator 是否真的在 step）；持续上升说明 5:1 追不上（升到 8:1）。DMD 梯度范数应在前 300 cycle 内降一个量级后趋稳。**绝对数值不作为验收依据，一切以 validation 视频为准**。

**质量时间线**：cycle 200–400 结构成型、旋转启动但纹理糊；800–1500 构图与身份接近 teacher，旋转角速度达标与否在此区间见分晓；2000–3000 细节收敛饱和。**预计最可能的问题排序**：①旋转角速度偏慢/转不满（概率中等，梯度截断固有风险）→ 触发 Phased DMD 分阶段备选方案；②帧间亮度周期震荡（概率低，5:1 已是针对性配置）→ 升 8:1 或 10:1；③轻微过饱和（概率中等）→ `real_guidance_scale` 降至 3.5–4.0（标准参数化）或 2.5–3.0（附加参数化）重训末段；④身份漂移（概率低，teacher 本身是领域内微调产物）→ 降 DMD LoRA 强度至 0.8–0.9 推理或提前停。

**交付物**：修改文件清单 + diff 摘要、两套运行脚本、`tools/merge_fig360_lora.py` + Stage A 单测 + gate-0 等价验证报告（两条路径的输出帧并排图 + max|diff| 数值）、`figurine360_dmd_lora_rank64.safetensors`（EMA）、validation 视频网格（teacher 50 步 vs student 4 步并排）、wandb 链接、复盘文档、**NOTE_DMD.md**（第 9 节）、**docs/task03_phase23_roadmap.md**（第 10 节）。

## 9. NOTE_DMD.md 要求（必交付，中文）

面向"不读代码直接操作"的内网用户，按以下结构写：

1. **路径变量块**：底模 / merged teacher / 数据目录 / 输出目录，全部收敛在脚本顶部，逐项说明。
2. **四条命令**：merge + gate-0 等价验证（4.1，含 10 步快速版与 50 步正式版）→ 5.2 的 50 步跨库对齐验证 → 冒烟训练 → 正式训练（含断点恢复方法）。
3. **推理三条路径**：a) 仓库 `inference.py` 4 步 cfg=1；b) DiffSynth 工程加载 base + figurine360 + DMD-LoRA——注意：DMD-LoRA 必须与 figurine360 同挂，加载顺序 base → figurine360 → DMD-LoRA；缺 figurine360 时输出会退化，属预期行为而非 bug；c) ComfyUI 加载单文件 LoRA（同样需三件套配置）。
4. **常见坑速查**：OOM（降分辨率/换 FSDP/激活 offload）、loss 形态异常对照表、恢复训练、cfg≠1 导致过曝。

## 10. Phase 2/3 路线（写入 docs/task03_phase23_roadmap.md，本期只写文档 + 留接口）

**Phase 2（BSA + DMD 同阶段训练）**：思路对齐"稀疏蒸馏"——把稀疏注意力引入的误差交给 DMD 损失在**同一训练过程**中吸收，而非"先蒸馏后稀疏"两阶段。实现路径：① 在 5.7 的 attention wrapper 中接入 BSA kernel（候选：开源 VSA（Video Sparse Attention）实现 / block-sparse flash-attn 变体），仅 generator（student）启用；real/fake score 保持全注意力——teacher 监督信号不能降质；② DMD 训练循环零改动，student 在稀疏预算内被迫匹配 teacher 分布；③ 稀疏率课程：前 20% iter 全注意力 warmup，再线性降至目标稀疏率，避免早期训练不稳；④ 推理与训练用同一 wrapper 同一配置，保证 train/infer 一致；⑤ 量级提示：480P 序列约 2 万 token，稀疏收益有限，BSA 主要为 720P×121f 满格服务，Phase 2 建议直接在满格分辨率上做。

**Phase 3（BSA + DMD2 同阶段训练）**：在 Phase 2 之上补齐 DMD2 的 GAN 项（DMD2 其余要素——TTUR、多步 generator、免回归损失——本期方案已具备）。实现路径：① 打开 5.7 的 `return_features`，在 fake score 中间特征上加轻量判别头（数层 MLP，只训判别头 + fake LoRA）；② 真样本 = Step 1 预留的真实视频 latent（打开 `load_video_latent: true`），假样本 = student rollout 输出；判别损失并入 fake score 更新分支；③ generator 侧加对抗项，`gan_loss_weight` 从 0 渐升（1e-2 量级起步）；④ TTUR 5:1 保持不变；⑤ 风险预案：GAN 与 BSA 同开会放大训练方差，预留"先 BSA+DMD 收敛 → 再开 GAN 微调"的两段式退路开关。

------

三个需要你拍板的开放项：一是正式训练分辨率——请确认 figurine360 数据的实际训练分辨率，文档第 6 节按此填入；二是若内网机器不是 8×H100，把型号/卡数告诉我，我更新第 8 节的预测数字；三是内网能否预置 flash-attn 的预编译 wheel（对应 CUDA/torch 版本）与 kijai 的 Turbo rank-64 LoRA 文件（5.3 初始化首选），若不能，NOTE_DMD.md 中需写 SDPA 回退路径的性能折损说明、5.3 走 delta-SVD 自提路径。

------

参考：

1. guandeh17/Self-Forcing（上游框架，configs 语义 / warp_denoising_step / denoising_loss_type；注意其主线是因果学生，我们只用其基础设施）：https://github.com/guandeh17/Self-Forcing
2. yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers（Turbo 可作为标准双向模型加载的证据）：https://huggingface.co/yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers
3. Kijai/WanVideo_comfy 的 Wan22-Turbo 目录（Turbo rank-64 LoRA 提取产物，5.3 初始化首选）：https://huggingface.co/Kijai/WanVideo_comfy
4. lightx2v/Wan2.2-Lightning（LoRA 初始化备选源 / Phased DMD 备选方案）：https://github.com/ModelTC/Wan2.2-Lightning
5. Wan-Video/Wan2.2（TI2V-5B 官方推理行为对齐基准）：https://github.com/Wan-Video/Wan2.2