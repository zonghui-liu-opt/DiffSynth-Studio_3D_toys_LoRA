# Task-03 Stage A 基线发现

## 结论

Stage A 已把 Wan2.2-TI2V-5B Turbo DMD runtime vendored 到当前仓库 `third_party/wan22_turbo/`，并在当前仓库新增 import-safe 的 `dmd/` 工具层、DMD LoRA 配置、metadata 转换脚本和 H100 launcher。Mac 本地不加载真实权重、不初始化 CUDA，已验证可离线覆盖的 LoRA/配置/脚本/attention fallback 逻辑。

用户已在内网 H100 验证：figurine360 LoRA 离线 merge 后权重与推理时再 load LoRA 的输出效果一致。因此 Stage A 文档把 gate-0 记录为“已由用户外部验证”，当前代码保留 merge 命令和 report。

## Step 0 代码确认

1. 双向路径：
   - `configs/self_forcing_wan22_dmd.yaml` 使用 `generator_type: bidirectional`。
   - `model/base.py` 在 `is_causal=False` 时创建 `BidirectionalTrainingPipeline`。
   - `utils/wan_wrapper.py` 对 Wan2.2 创建 `WanDiffusionWrapper(is_causal=False)`，训练路径不使用 `kv_cache`。

2. generator 初始化来源：
   - 原 Turbo `dmd.sh` 直接使用 `wan_models/Wan2.2-TI2V-5B`。
   - 当前 launcher 会把 `third_party/wan22_turbo/wan_models/Wan2.2-TI2V-5B` symlink 到 `MERGED_MODEL_ROOT`，所以 generator、fake score、real score 三者都从 merged figurine360 teacher 初始化。

3. 数据组织格式：
   - Turbo DMD trainer 使用 CSV：`path,text,num_frames`。
   - 当前 `tools/prepare_dmd_dataset_csv.py` 将 DiffSynth 的 `video,prompt` 或 `path,text` metadata 转成该格式。
   - 训练 forward 从视频 tensor 的第 0 帧经 Wan2.2 VAE 编成 `wan22_image_latent`。

4. CFG 数学形式：
   - Turbo `model/dmd.py` 使用 `pred_real_image_cond + (pred_real_image_cond - pred_real_image_uncond) * real_guidance_scale`。
   - 这是“附加引导”形式；若要等价 teacher 标准 CFG `g=5`，本任务配置设 `real_guidance_scale: 4.0`。
   - `fake_guidance_scale: 0.0`，student/fake score 为单条件 forward。

5. 5:1 更新语义：
   - Turbo trainer 中 `TRAIN_GENERATOR = self.step % dfake_gen_update_ratio == 0`，每个 step 都更新 fake score，只有 step 能整除 ratio 时额外更新 generator。
   - 设为 5 后，长期比例是 5 fake score step : 1 generator step。它不是“单个 cycle 先连续 5 fake 再 1 generator”的严格批内顺序，但保持上游已验证语义。

## Stage A 代码改造

- `dmd/wan22_lora.py`：LoRA Linear 包裹、目标层匹配、LoRA-only state_dict 过滤、DiffSynth 可加载导出。
- `dmd/wan22_config.py`：DMD YAML 摘要与 runtime config 生成，按 `HEIGHT/WIDTH/NUM_FRAMES` 自动写 latent shape。
- `tools/prepare_dmd_dataset_csv.py`：metadata 转 Turbo DMD CSV。
- `configs/dmd/figurine360_wan22_dmd_lora.yaml`：Phase 1 DMD + CFG 蒸馏配置，LoRA rank/alpha=64，Phase 2/3 hook 默认关闭。
- `train_figurine360_dmd_lora.sh`：H100 启动入口，所有路径与卡数集中在顶部变量块。
- `third_party/wan22_turbo/`：vendored Turbo runtime；已 patch LoRA 注入、LoRA-only checkpoint、EMA LoRA 导出、`max_iters` 停止条件。

## Stage A 测试

已通过：

```bash
pytest tests/test_dmd_lora_utils.py tests/test_dmd_stage_a_contract.py -q
python3 -m py_compile dmd/wan22_lora.py dmd/wan22_config.py tools/prepare_dmd_dataset_csv.py third_party/wan22_turbo/model/base.py third_party/wan22_turbo/trainer/wan22_distillation.py third_party/wan22_turbo/utils/distributed.py third_party/wan22_turbo/wan22/modules/attention.py
bash -n train_figurine360_dmd_lora.sh
```

未在 Mac Stage A 执行真实 DMD forward/training；原因是本机无 CUDA、无真实权重、无 flash-attn。Stage B 第一轮必须按 `NOTE_DMD.md` 先跑 50 iter 冒烟。
