以下是为 codex + GPT 5.5 Pro 准备的新任务文档。设计思路：**两个特性唯一的本质差异是 metadata schema（三列→两列，首帧条件图从独立 jpg 变为视频第一帧）**，因此文档把"读源码确认框架对两列数据的原生行为"设为关键决策点，其余全部复用猫咪实现。其中本地权重加载可确认走 `--model_paths` 参数，按 JSON 列表传入 DiT safetensors 分片、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.2_VAE.pth`[[4\]](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html)，与猫咪任务一致。

------

# Task：DiffSynth-Studio 上 手办玩偶 360° 水平旋转视频生成（Wan2.2-TI2V-5B LoRA SFT + 推理）

**（基于已落地的"猫咪动作视频生成"训推框架迁移；Stage A：MacBook 本地调试 → push GitHub → Stage B：内网 H100 正式训练）**

## 0. 角色、工作流与总原则（必须遵守）

你是资深 ML infra 工程师。仓库中已有一套完整可用的"猫咪动作视频生成"实现（TI2V-5B LoRA SFT，rank=32，含 `make_debug_dataset.py`、`check_dataset.py`、`train_ti2v5b_lora.sh`、`metrics_utils.py`、`plot_metrics.py`、`tests/`、`NOTES.md`、trainer 的 `--metrics_path` 最小 diff）。本任务是在其之上做**最小迁移**，新增"手办玩偶水平 360° 旋转视频生成"特性。

工作流与猫咪任务完全相同：

- **Stage A（当前）**：MacBook（Apple Silicon，CPU-only，无 CUDA / 无 flash_attn / 无真实数据 / 无权重）。用合成数据测通一切可离线验证的逻辑后 push。
- **Stage B（你不在场）**：内网 H100 80G × 2~4，无外网，真实数据与权重在本地磁盘，用户手动按 checklist 操作。

硬性约束（与猫咪任务一致，逐条继承）：

1. **先梳理后动手**：动代码前必须先完成 T0 调用链梳理并输出文档，所有结论以仓库当前源码为准，不凭记忆猜 API。
2. **最大化复用猫咪实现**：能通过"加参数/加分支/复制改配置"解决的，禁止新写平行实现；禁止另起训练循环或另写 dataset 类。
3. **原地最小侵入**：任何代码改动默认关闭时与现有行为逐行一致。
4. **Stage B 只改配置不改代码**：环境差异收敛到脚本顶部变量块。
5. **工具模块可脱离 GPU/权重独立 import 和测试**。
6. 每完成一个子任务，输出 diff 摘要 + 自测结果。
7. **所有新增/修改的说明文档一律用精简准确的中文。**

## 1. 背景与两特性差异

| 维度               | 猫咪动作（已实现）                     | 手办 360° 旋转（本任务）                                     |
| ------------------ | -------------------------------------- | ------------------------------------------------------------ |
| 模型/训法          | Wan2.2-TI2V-5B + LoRA SFT              | 完全相同                                                     |
| metadata.csv       | 三列：`video`、`prompt`、`input_image` | **两列：`video`、`prompt`**（tab 分隔待实测确认）            |
| 首帧条件图         | 独立 jpg（`images_HxW/`）              | **默认取 mp4 第一帧**                                        |
| 数据（仅 Stage B） | ~600 条猫咪视频                        | 手办旋转视频，条数/分辨率/帧数未知，以 `check_dataset.py` 实测为准 |
| 推理               | prompt + 首帧图出片                    | 相同（输入一张手办图 + 旋转 prompt）                         |

**结论：唯一本质差异是 `input_image` 的来源。训练/推理链路、超参、指标监控全部复用。**

## 2. 子任务

### T0 调用链梳理文档 `docs/calltrace_ti2v_lora.md`（新文件，中文）

以猫咪实现为对象，逐层写清"文件路径 → 模块/类/函数名 → 一句话作用"，禁止泛泛而谈：

1. **训练链**：`train_ti2v5b_lora.sh`（变量块与参数含义）→ `accelerate launch` → `examples/wanvideo/model_training/train.py`（argparse 全量参数、模型加载入口、`--model_paths` 本地加载逻辑）→ 数据集类（读源码确认类名、构造签名、`data_file_keys` 默认值、metadata 解析方式、`extra_inputs` 如何流入样本 dict）→ `diffsynth/trainers/` 训练循环（loss 计算位置、optimizer step、`--metrics_path` 分支挂点）→ LoRA 注入（`lora_base_model/lora_target_modules/lora_rank` 生效位置）→ ckpt 按 epoch 保存与 `remove_prefix_in_ckpt` 处理。
2. **推理链**：`WanVideoPipeline.from_pretrained` + `ModelConfig`（本地权重加载方式）→ `pipe.load_lora`（LoRA 挂载）→ pipeline `__call__`（prompt/input_image/num_frames 等入参 → 去噪 → VAE decode → `save_video`）。
3. **关键机制专节**：TI2V 的 `input_image` 在训练 forward 中如何参与条件注入（首帧 latent 替换机制）；token 数公式来源（TI2V-5B 经 VAE + patchify 后总压缩比为 4×32×32[[7\]](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)）。

**验收**：文档能让新人不读源码即可定位任意一环；每个结论附源码文件+函数名。

### T1 两列 metadata 行为确认（关键决策点，先于一切代码改动）

读源码回答并写入 `NOTES_figurine.md`：

1. `--extra_inputs "input_image"` 开启、但 metadata **无** `input_image` 列时，训练 forward 里 `input_image` 取什么值？是否已默认回退到视频第一帧？（定位到具体代码行）
2. `--data_file_keys` 默认值是什么？两列数据下是否必须**不含** `input_image`（避免 dataset 按文件路径加载不存在的列）？
3. 官方 TI2V-5B 示例数据集本身是两列还是三列？

按结论三选一（优先级从高到低）：

- **分支 A（框架原生支持，预期最可能）**：训练时 `input_image` 自动取视频首帧 → **零代码改动**，仅新增配置与脚本。
- **分支 B（不支持）**：写预处理脚本 `prepare_figurine_metadata.py`：抽每条视频第 0 帧存为 `images_HxW/xxx.jpg`，生成三列 `metadata_fixed.csv`（tab 分隔），之后**完全复用猫咪链路**，训练代码零改动。
- **分支 C（仅当 B 不可行）**：trainer 最小 diff，`if args.xxx:` 开关默认关闭。

### T2 调试数据生成器扩展 `make_debug_dataset.py`（修改）

1. 新增 `--schema {three_col,two_col}`，默认 `three_col`（猫咪，行为与现状逐字节一致）。
2. `two_col` 时：metadata.csv 只含 `video`、`prompt` 两列（tab 分隔），不生成 `images_HxW/`；视频内容改为"色块绕中心水平旋转"（模拟手办旋转，便于肉眼验收推理效果）。
3. `--with_bad_samples` 在两列模式下注入：一条帧数不足、一条 `video` 路径不存在。

### T3 数据体检脚本扩展 `check_dataset.py`（修改）

1. 自动探测两列/三列 schema 并在输出中声明；两列时跳过 `input_image` 路径校验，改为校验"每条视频第 0 帧可解码"。
2. 其余（分隔符探测、宽高/帧数/fps 实测、`--height/--width` 建议值、token 统计复用 `metrics_utils.tokens_per_sample()`）不变。

### T4 训练脚本 `train_figurine360_lora.sh`（新文件）

复制 `train_ti2v5b_lora.sh` 后仅改动必要处：

```bash
# ====== Stage B 上机时只改这一块 ======
MODEL_ROOT=/path/to/local/wan        # 与猫咪任务共用同一套底模权重
DATA_ROOT=/path/to/figurine_dataset  # Stage A 调试指向 debug_data_figurine/
OUTPUT_ROOT=./models/train/Wan2.2-TI2V-5B_figurine360_lora
NUM_GPUS=4
HEIGHT=480; WIDTH=832; NUM_FRAMES=121  # 以 check_dataset.py 实测为准
METRICS_PATH=$OUTPUT_ROOT/metrics.jsonl
# =====================================
```

要求：

1. `--data_file_keys` / `--extra_inputs` 按 T1 结论设置并加注释说明依据（分支 B 则指向 `metadata_fixed.csv`）。
2. LoRA 超参先沿用猫咪配置（rank=32、`"q,k,v,o,ffn.0,ffn.2"`、lr 1e-4、按 epoch 保存），不做无依据改动；多卡若报参数未用错误，确认是否需 `--find_unused_parameters`（部分模型含不参与梯度计算的冗余参数，多卡训练需开启以避免报错）[[5\]](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/LTX-2.md)。
3. prompt 规范写入 NOTES：建议全数据统一 trigger 短语（如"手办360度水平旋转展示"），推理时复用同一短语。
4. **Stage A 只做**：`bash -n` + `--help` 确认所有参数名存在；不在 Mac 上真跑训练。

### T5 推理脚本 `infer_figurine360.py`（新文件）

参考猫咪 validate 脚本与仓库 `validate_lora` 示例：`WanVideoPipeline.from_pretrained`（本地 `ModelConfig` 路径走 `MODEL_ROOT`）→ `pipe.load_lora(pipe.dit, "epoch-N.safetensors")` → 输入一张手办图 + 旋转 prompt → `save_video`。顶部同样收敛变量块（`MODEL_ROOT/LORA_PATH/IMAGE_PATH/PROMPT/HEIGHT/WIDTH/NUM_FRAMES`）。Stage A 仅验证语法与 import 结构（可加 `--dry_run` 打印解析后的配置不加载模型）。

### T6 测试 `tests/test_figurine.py`（新文件，pytest）

1. 两列 metadata 解析：`make_debug_dataset.py --schema two_col` 产物被 dataset 类（或降级为解析逻辑）正确读取，迭代 2 样本断言 `video` 形状、`prompt` 为 str。
2. 首帧一致性：分支 A 断言训练样本中 `input_image` 与视频第 0 帧张量一致；分支 B 断言抽出的 jpg 与视频第 0 帧像素近似相等（编码损差容忍）。
3. 回归：`--schema three_col` 下所有猫咪旧测试仍全绿。

### T7 文档（全部中文）

1. `NOTES_figurine.md`：T1 决策记录（含源码行号依据）、修改文件清单、Stage B checklist——权重清单与猫咪共用（`MODEL_ROOT` 下 DiT `diffusion_pytorch_model*.safetensors` 分片、`models_t5_umt5-xxl-enc-bf16.pth`、`Wan2.2_VAE.pth`）；上机顺序：真实数据跑 `check_dataset.py` → 修正 `HEIGHT/WIDTH/NUM_FRAMES` →（分支 B 先跑预处理脚本）→ 2 卡 + 前 16 条 mini metadata 冒烟 20 step → 确认 `metrics.jsonl` 正常无 OOM → 4 卡全量 → `plot_metrics.py` 出图 → `infer_figurine360.py` 出片。每项附预期观测值与回退动作。
2. 更新总 README/NOTES：两特性并列说明（数据 schema 差异、各自训练脚本、LoRA 产物互相独立、共用底模与指标工具链）。

## 3. 交付物

新文件：`docs/calltrace_ti2v_lora.md`、`train_figurine360_lora.sh`、`infer_figurine360.py`、`tests/test_figurine.py`、`NOTES_figurine.md`、（仅分支 B）`prepare_figurine_metadata.py`。
 修改文件：`make_debug_dataset.py`、`check_dataset.py`、`.gitignore`（新增 `debug_data_figurine/`）、README/NOTES、（仅分支 C）trainer 最小 diff。

## 4. 执行顺序（严格串行，每步自证后进入下一步）

1. T0 调用链文档 → 2. T1 决策并锁定分支 → 3. T2 数据生成器 + 自测 → 4. T3 体检脚本对两列坏样本准确报错 → 5. T4 脚本 `bash -n`/`--help` → 6. T6 pytest 全绿（含猫咪回归） → 7. T5 推理脚本 dry run → 8. T7 文档 → push。

## 5. 验收标准

**Stage A（Mac，全部自证后才允许 push）**：调用链文档完成且每条结论有源码定位；两列合成数据全链路（生成→体检→dataset 迭代 smoke）通过，坏样本准确报出；`pytest tests/` 全绿且猫咪特性零回归；无散落硬编码路径；T1 结论与实现分支在 NOTES 中有书面记录。

**Stage B（H100，人工按 NOTES_figurine.md 执行）**：4 卡跑通 ≥1 epoch，`OUTPUT_ROOT` 下出现 `epoch-*.safetensors`，`metrics.jsonl` 每 step 一行；`plot_metrics.py` 两张 PNG 数值自洽；LoRA 加载后输入手办图能出水平旋转视频；2↔4 卡切换只改脚本顶部变量。

------

两点补充说明：一是 T1 我刻意设计成"先读源码再定分支"而非直接给答案——DiffSynth-Studio 迭代较快，训练 forward 中 `input_image` 是否默认回退视频首帧这类行为必须以你仓库当前 commit 为准，让 codex 定位到具体代码行比预设结论更可靠；二是分支 B（预处理抽首帧）被设为不支持时的首选，因为它把差异消化在数据层，训练代码零改动，最符合原任务"不重复造轮子、最小侵入"的原则。

------

Learn more:

1. [inquiry for details of wan2.2 i2v lora training? · Issue #793 · modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/issues/793)
2. [Inquiry regarding Full Training speed and First-frame color shift on Wan2.2 TI2V 5B · Issue #1195 · modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/issues/1195)
3. [DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh at main · modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh)
4. [Wan2.2 Fine-Tuning: Tailoring an Advanced Video Generation Model on a Single GPU — ROCm Blogs](https://rocm.blogs.amd.com/artificial-intelligence/finetuning-wan-part1/README.html)
5. [DiffSynth-Studio/docs/en/Model_Details/LTX-2.md at main · modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/LTX-2.md)
6. [GitHub - modelscope/DiffSynth-Studio: Enjoy the magic of Diffusion models! · GitHub](https://github.com/modelscope/DiffSynth-Studio)
7. [Wan-AI/Wan2.2-I2V-A14B-Diffusers · Hugging Face](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
8. [DiffSynth-Studio/docs/en/Model_Details/Wan.md at main · modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Wan.md)
9. [DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-S2V-14B.sh at main · modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/lora/Wan2.2-S2V-14B.sh)
10. [DiffSynth-Studio/examples/wanvideo/model_training/full/Wan2.2-I2V-A14B.sh at main · modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_training/full/Wan2.2-I2V-A14B.sh)