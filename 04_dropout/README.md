# 练习 04：Dropout

先打开 `dropout-lecture.html`，观察随机 seed、丢弃概率和 program 选择如何改变逐元素 mask。

## 一个 program 负责什么

一个 program 负责扁平输入中的一段连续元素：

```text
program 0 → offsets [0, BLOCK_SIZE)
program 1 → offsets [BLOCK_SIZE, 2 × BLOCK_SIZE)
...
```

grid 中多个 program 并行处理不同的数据块；单个 program 内，`tl.arange` 表达整组元素的向量化处理，不是 Python 逐元素循环。

## 算法

对每个有效 offset：

```text
random = 基于 seed 和 offset 的确定性随机数
keep = random > p
output = x / (1 - p)  if keep else 0
```

除以 `1 - p` 称为 inverted dropout，它让输出的期望值与输入一致。尾部不足一个 block 的位置仍需使用边界 mask。

## 你的任务

只实现 `dropout.py` 中的 kernel 核心逻辑。外围参数校验、启动代码、测试与错误定位由 Codex 负责。

建议按以下顺序完成：

1. program ID 与一维 offsets；
2. 边界 mask；
3. `tl.rand(seed, offsets)`；
4. keep mask；
5. inverted scaling 与写回。

## 运行测试

```bash
conda activate triton
cd /mnt/d/ai-learn/practice/04_dropout
pytest -q
```

最大测试只有 32768 个 FP32 元素，显存占用很小。

## 教学实现边界

- 当前练习处理单个 tensor，不涉及训练/推理模式切换。
- 使用无状态、基于 seed 与 offset 的随机数，不维护全局 RNG 状态。
- 只验证统计性质与同 seed 可复现性，不要求随机 mask 与 PyTorch 完全相同。
- 工业实现还要考虑 RNG 状态管理、分布式可复现性、向量化访存、融合算子和反向传播。
