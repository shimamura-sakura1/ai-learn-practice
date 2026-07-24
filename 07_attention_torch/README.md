# 练习 07：用 PyTorch 基础算子手写 Attention

本章的目标不是调用封装，而是亲手追踪 Attention 的数学、shape、causal mask、多头 layout 和中间内存。核心实现只允许使用 PyTorch 基础操作；`torch.nn.MultiheadAttention` 和 `torch.nn.functional.scaled_dot_product_attention` 仅可出现在测试、实验或 benchmark reference 中。

## 实现边界

手写实现中可以使用 `torch.matmul`、`@`、`transpose`、`permute`、`reshape`、`view`、`contiguous`、`softmax`、`masked_fill`、`nn.Linear` 和 `torch.sqrt`。不得使用官方 Attention 封装、FlashAttention、xFormers、用一条 `einsum` 包办计算，或用 helper 隐藏核心过程。

## 分阶段完成

每次只完成当前阶段的 TODO，并运行对应测试；失败时先按 shape、transpose、head layout、scale、mask、softmax axis、matmul、output merge、parameter mapping、dtype 分类。

### 阶段 1：单头、无 mask

在 `scaled_dot_product_attention` 中追踪：

```text
Q [B, Sq, Dh]
K [B, Sk, Dh]
V [B, Sk, Dv]
scores = QK^T
probs = softmax(scores / sqrt(Dh))
out = probs V
```

验收输出 shape、每行概率和以及与独立 reference 的一致性。

### 阶段 2：causal mask

在 softmax 前加入 mask；`j > i` 的位置必须被屏蔽，概率为 0，第 `i` 个 query 只能看到 `0...i`。本练习的 causal cross-shape 语义采用矩阵左上角对齐。

### 阶段 3：多头 layout

必须在函数体中显式写出，不要封装成 helper：

```text
[B, S, D] → [B, S, H, Dh] → [B, H, S, Dh]
[B, H, S, Dh] → [B, S, H, Dh] → [B, S, D]
```

同时完成整除检查、batched score、scale、mask、softmax 和 value 聚合。

### 阶段 4：QKV projection

在完整模块中加入 `Q = XWq`、`K = XWk`、`V = XWv`，完成多头计算、head 合并与 `out_proj`。

### 阶段 5：对齐 PyTorch

测试会把四组投影参数映射到 `torch.nn.MultiheadAttention(batch_first=True, dropout=0)`，比较 forward output、每头 attention probabilities、输入梯度和 projection weight gradients。

## 推荐命令

从项目根目录执行第一阶段：

```bash
python -m pytest practice/07_attention_torch/test_attention.py -q -k single_head
python -m pytest test_attention.py -q -k single_head
```

继续执行：

```bash
python -m pytest practice/07_attention_torch/test_attention.py -q
python practice/07_attention_torch/inspect_attention.py
python practice/07_attention_torch/benchmark_attention.py --mode memory
```

实现尚未完成时，相关测试因 `NotImplementedError` 失败是预期行为。性能测试可能分配很大的 `S×S` tensor，默认参数保持保守；理论内存表本身不会分配这些 tensor。

## 最终验收

代码部分需通过单头、causal、多头 layout、完整模块、官方 forward 和 backward 对齐；口头部分需能不看代码写出所有 shape 变化，并解释 softmax 维度、scale、mask、`B/H` 独立性及 score/probs 的理论内存。最后完成 `acceptance.md`，至少答对 26/30。

