# 发现与决策

## 需求
- Stage A 在 MacBook CPU-only 环境完成离线可验证开发，不加载真实权重、不跑真实训练。
- Stage B 在无外网 H100 环境只改脚本顶部变量块即可跑正式 LoRA SFT。
- 训练主体必须复用 `examples/wanvideo/model_training/train.py` 与官方 Wan2.2-TI2V-5B LoRA 示例，新增逻辑默认关闭。
- 新增指标工具必须可脱离 GPU/权重 import 和测试。

## 研究发现
- 仓库没有 `diffsynth/trainers/` 目录；实际训练通用逻辑在 `diffsynth/diffusion/runner.py`、`logger.py`、`training_module.py`。
- `examples/wanvideo/model_training/train.py` 的 `wan_parser()` 组合了 `diffsynth/diffusion/parsers.py` 里的 dataset/model/training/output/lora/gradient/offload/logger 参数。
- 官方 `examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh` 存在，参数为 height=480、width=832、num_frames=49、rank=32、`lora_target_modules "q,k,v,o,ffn.0,ffn.2"`、`--extra_inputs "input_image"`、`--remove_prefix_in_ckpt "pipe.dit."`。
- 任务书目标 `NUM_FRAMES=121` 与官方示例 `num_frames=49` 不一致；训练脚本顶部保留变量，默认按任务书 121，NOTES 要提示按 `check_dataset.py` 实测修正。
- `UnifiedDataset.load_metadata()` 对 csv 使用 `pandas.read_csv(metadata_path)`，未指定 `sep`；tab 分隔真实 metadata 会被错误读成单列，因此 `check_dataset.py` 需要生成逗号分隔 `metadata_fixed.csv`。
- `--data_file_keys` 默认是 `image,video`。如果 metadata 里有 `input_image` 列，默认不会加载该列。
- `WanTrainingModule.parse_extra_inputs()` 对 `extra_inputs == "input_image"` 使用 `data["video"][0]`，并不读取 `data["input_image"]`；这是当前官方示例行为。
- 训练循环在 `diffsynth/diffusion/runner.py`：`optimizer.step()`、`scheduler.step()`、`optimizer.zero_grad()` 后调用 `model_logger.on_step_end()`；`save_steps is None` 时每 epoch 调 `model_logger.on_epoch_end()` 保存 `epoch-{epoch_id}.safetensors`。
- `--model_paths` 接收 JSON 列表，经 `DiffusionTrainingModule.parse_model_configs()` 转为 `ModelConfig(path=...)`，不会触发下载；Stage B 无外网应使用此参数加载本地 DiT/T5/VAE。
- `--model_id_with_origin_paths` 会转为 `ModelConfig(model_id=..., origin_file_pattern=...)`，`ModelConfig.download_if_necessary()` 默认从 ModelScope/HF 下载，不适合 Stage B 无外网。
- 直接执行 `python examples/wanvideo/model_training/train.py --help` 时，当前 Python 不会自动把仓库根加入 import 路径；`PYTHONPATH=.` 可正常列出 `--metrics_path` 等参数，训练 launcher 已导出 `PYTHONPATH="$(pwd):..."`。
- 当前 Mac CPU 测试可直接 import `UnifiedDataset`，并通过 orientation bucket 单测覆盖图片路径加载、`input_image` 加载和 metadata 处理。
- 当前 DataLoader 使用 `collate_fn=lambda x: x[0]`，每个 micro-step 实际仍是单样本 forward；因此横竖屏 bucket 支持应优先保持现有训练语义，不直接改成 dense tensor batch。
- 横屏 `480x832` 与竖屏 `832x480` token 数相同；metrics 继续用横屏 `height,width` 计算 tokens 不影响吞吐统计。
- `ImageCropAndResize` 会 scale 后 center crop，不适合把竖屏视频强制转为横屏训练；新增 no-crop orientation resize 比修改旧算子更稳妥。

## 技术决策
| 决策 | 理由 |
|------|------|
| 先写 tests 再实现可测纯函数与脚本行为 | 任务包含明确 Stage A 验收，TDD 能锁定指标公式、JSONL 去重、数据格式 |
| trainer 只加 `if args.metrics_path:` 分支 | 满足默认关闭时上游行为不变 |
| Stage B 训练脚本使用 `--model_paths` 而非 `--model_id_with_origin_paths` | 前者直接本地路径加载，不触发无外网下载 |
| `check_dataset.py` 对 tab metadata 输出 `metadata_fixed.csv` | 当前 `UnifiedDataset` 的 CSV 读取不兼容 tab |
| `train_ti2v5b_lora.sh` 显式传 `--data_file_keys "video,input_image"` 并备注当前 trainer 使用视频首帧 | 真实数据列可被加载；同时保留官方 TI2V 现有条件图逻辑 |
| TI2V `input_image` 分支优先使用已加载的 `data["input_image"][0]`，否则回退 `data["video"][0]` | 让真实 metadata 的首帧条件图生效，同时保留官方示例缺少该列时的行为 |
| 横竖屏采用 `--enable_orientation_buckets` 显式开启 | 默认保留上游 center-crop 行为；内网 launcher 默认开启以匹配当前真实数据 |
| `check_dataset.py` 始终输出 `metadata_fixed.csv` | 即使原 metadata 已是逗号分隔，也需要追加 `height,width,bucket` 供 bucket sampler 使用 |
| 新增 `ImageResizeToBucketResolution` 而不是改 `ImageCropAndResize` | 避免影响其他模型/示例的既有裁剪语义 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| 当前线程已有活跃 goal | 沿用已有 goal，不重复创建 |
| 任务书提到 `diffsynth/trainers/`，实际目录不存在 | 按源码定位到 `diffsynth/diffusion/runner.py` 等实际训练模块 |

## 资源
- `Task：DiffSynth-Studio 上 Wan2.2-TI2V-5B LoRA SFT + 训练指标监控.md`

## 视觉/浏览器发现
- 暂无。

---
*每执行2次查看/浏览器/搜索操作后更新此文件*
*防止视觉信息丢失*
