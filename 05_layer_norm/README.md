# 练习 05：LayerNorm

先打开 `layer-norm-lecture.html`，观察一行数据依次经历求均值、中心化、求方差、归一化和 affine 变换。

## 一个 program 负责什么

当前教学版本采用“一行一个 program”：

```text
program 0 → 第 0 行的全部列
program 1 → 第 1 行的全部列
...
```

不同行由 grid 中不同 program 并行处理；一行中的所有列由 `tl.arange(0, BLOCK_SIZE)` 表达为 Triton tensor，并在单个 program 内进行向量化 reduction。

## 算法推导

对一行 `x[0:n_cols]`：

```text
mean      = sum(x) / n_cols
centered  = x - mean
variance  = sum(centered²) / n_cols
rstd      = 1 / sqrt(variance + eps)
normalized = centered * rstd
output    = normalized * weight + bias
```

这里使用 population variance，也就是除以 `n_cols`，不是样本方差的 `n_cols - 1`。

## 你的任务

只完成 `layer_norm_kernel` 的核心逻辑。外围校验、输出分配、grid、BLOCK_SIZE 和测试由 Codex 维护。

建议按顺序实现：

1. program ID、行基地址和列 offsets；
2. 列边界 mask 与加载；
3. FP32 mean reduction；
4. FP32 variance reduction；
5. `rsqrt(variance + eps)`；
6. weight/bias 与写回 mask。

## 运行测试

```bash
conda activate triton
cd /mnt/d/ai-learn/practice/05_layer_norm
pytest -q
```

最大测试只有 `32 × 1000` 个元素，对 12GB 显卡非常轻量。

## 教学实现边界

- 当前只接受二维输入，并对最后一维做 LayerNorm。
- 一行必须能由单个 program 处理，`BLOCK_SIZE = next_power_of_2(n_cols)`。
- 当前限制 `n_cols <= 65536`，实际可用上限还会受到寄存器和共享资源限制。
- 工业实现会根据列数选择 warps、分块或多阶段 reduction，并融合残差、bias、激活或量化逻辑。
- 数值稳定性、寄存器压力和 occupancy 的关系会在 Profiling 章节继续观察。
