# Task-03 Phase 2/3 路线

## Phase 2：BSA + DMD 同阶段训练

当前 Stage A 已在配置中预留 `attention_backend: flash`。Phase 2 不应先蒸馏后稀疏，而应在同一 DMD 训练过程中让 student 使用 BSA，real/fake score 保持全注意力。

实施路径：

1. 在 vendored Wan2.2 attention wrapper 中把 `flash` 分支扩展为 `flash|sdpa|bsa`。
2. 仅 generator/student 根据 `attention_backend=bsa` 走 block sparse kernel；real score 和 fake score 强制全注意力。
3. 加稀疏率课程：前 20% iter 全注意力 warmup，随后线性降到目标稀疏率。
4. 推理脚本使用同一 wrapper 和同一稀疏配置，避免 train/infer mismatch。

## Phase 3：BSA + DMD2 同阶段训练

当前配置中预留 `gan_loss_weight: 0.0` 与 `load_video_latent: false`。Phase 3 在 Phase 2 基础上打开 GAN 项。

实施路径：

1. 给 fake score forward 增加 `return_features`，从中间 block 输出特征。
2. 数据集打开 `load_video_latent: true`，真实视频 latent 作为 discriminator real data。
3. fake data 来自 student rollout；fake score update 分支同时训练 fake LoRA 和轻量判别头。
4. generator loss 加对抗项，`gan_loss_weight` 从 1e-2 量级 warmup。
5. 若 BSA+GAN 同开不稳，保留两段式退路：先 BSA+DMD 收敛，再打开 GAN 微调。
