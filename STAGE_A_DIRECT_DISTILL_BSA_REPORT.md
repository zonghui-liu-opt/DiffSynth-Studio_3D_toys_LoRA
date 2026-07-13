# DirectDistill BSA Stage A 报告

## 已实现

- `(4,3,6)` W-fastest 3D geometry、可逆 pack/unpack、partial-block mask 与 metadata cache；
- FP32 coarse masked mean / score / softmax / AV、稳定 tie-break Top-K、optional count bias；
- selected-KV chunked `sdpa_gather` 与 `eager_math` reference，无全局 L×L mask；
- query-block/channel rank-32 dynamic gate，zero `gate_up`，chunk 内融合；
- Wan self-attention 动态注入，复用原 q/k/v/o/norm/attn，cross-attention 保持 dense；
- `model_fn_wan_video` 与旧 `WanModel.forward` 的真实 runtime grid/context 透传，BSA+USP/unsupported 组合 fail-fast；
- dense warm-start + LoRA/gate 联合训练、FP32 两参数组、optimizer-step 离散退火、grad clip、optional dense anchor；
- 结构化 student info、backend probe、BSA metrics、原子组合权重 checkpoint 与严格 manifest；
- teacher/A/B/C 验证入口、独立子进程单层 H100 预筛 benchmark、内网 shell 与中文 NOTE。

## Stage A 验证边界

本地只证明 CPU tensor 数学、梯度、注入/加载接口、默认兼容与静态正确性。没有加载真实 5B 权重，没有执行 H100/NPU kernel，没有生成视频，也没有端到端性能或质量结论。

最终命令与结果：

```text
PYTHONPATH=. pytest -q tests                         102 passed
python -m compileall -q <BSA及接入文件>              passed
bash -n Wan2.2-TI2V-5B-Figurine360-BSA.sh           passed
validate/benchmark/shell --help                      passed
git diff --check + 新文件no-index whitespace check  passed
CPU独立子进程benchmark smoke                         passed（非H100结论）
```

## Stage B 必验

1. 真实模型严格 30/30 注入、warm-start key 全覆盖、FP32 trainable/Adam state；
2. target grid `(41,15,26)` 的 s=0 parity 与 K55 runtime 摘要；
3. 2–10 step loss/gradient/checkpoint smoke，随后 50-step 显存无泄漏；
4. 各 K bucket 的完整 4-step optimizer benchmark 和实际 SDPA kernel；
5. 至少 12 个、最好 50 个 held-out 的 teacher/A/B/C 质量评测；
6. Ascend 真机的标准 SDPA mask、forward/backward 与同 fixture 对齐。

## 已知 P0 边界

- `compact_ragged` 已提供与 fixed 对齐的 P1 reference，但含 host 同步且未证明加速；count-aware routing 已实现；
- checkpoint 是组合权重 continuation，不是 optimizer/RNG/dataloader 精确恢复；
- 单层 benchmark 仅为候选预筛，不能替代完整 DirectDistill optimizer-step wall-clock；
- portable SDPA 即使数学正确，也可能慢于当前 dense FA2/FA3，必须如实 profile。
