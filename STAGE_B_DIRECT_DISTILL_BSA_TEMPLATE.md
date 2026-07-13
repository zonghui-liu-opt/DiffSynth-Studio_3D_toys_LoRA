# DirectDistill BSA Stage B 结果模板

## 环境

```text
commit/branch:
GPU / world size:
PyTorch / CUDA / driver:
dtype / batch / gradient accumulation:
grid / block / backend / mask / boundary:
```

## Correctness

```text
30/30 injection, cross=0:
s=0 cosine / relative-L2:
runtime N/K/padding:
backend probe output/gradient error:
LoRA/gate gradient finite and nonzero:
checkpoint round-trip:
```

## 完整 optimizer-step 性能

| K | chunk | mask | checkpoint | median/P10/P90 | allocated GiB | reserved GiB | min free GiB | kernel | 结论 |
|---:|---:|---|---|---|---:|---:|---:|---|---|

50-step allocated/reserved 漂移与泄漏结论：

## 质量

| 样本数 | A vs teacher MSE | B vs teacher MSE | C vs teacher MSE | C/A ratio | C相对A胜出+持平 |
|---:|---:|---:|---:|---:|---:|

尾帧、右边缘、身份、配件、旋转、闪烁主观结论：

## 最终配置与待办

```text
query_block_chunk_by_top_k:
checkpoint policy:
dense anchor:
是否达到显存门禁:
是否达到质量门槛:
Ascend待验证项:
```
