# 01 Vector Add 课程复盘

## 已掌握内容

- 理解 grid 中多个 Triton program 的并行关系。
- 使用 `tl.program_id` 确定当前 program 负责的数据块。
- 使用 `tl.arange` 表达 program 内的一组元素。
- 使用 offsets 完成 GPU 指针偏移。
- 使用 mask 防止最后一个数据块真实访问越界。
- 从 Python 包装函数分配输出、构造 grid 并启动 kernel。
- 理解 PyTorch CUDA tensor 会自动作为 Triton 指针参数传入。

## 错误原因

- 曾把 `cdiv` 错写为 `torch.cdiv`；它属于 Triton 的 Python API。
- 曾使用会返回浮点数的除法计算 program 数量。
- 输入校验最初使用了与测试契约不一致的异常类型和信息。

## 性能观察

- 当前 kernel 将输入和输出各进行一次主要显存访问，属于典型的带宽受限逐元素算子。
- 固定 `BLOCK_SIZE` 能正确运行，但没有针对不同输入规模进行调优。
- 当前阶段只验证了正确性，尚未使用 profiler 系统分析带宽利用率。

## 实现边界

- 只处理一维连续 CUDA tensor。
- 没有支持任意 stride。
- 没有 dtype 和设备组合的完整通用校验。
- 没有 autotune 或针对不同 GPU 的配置选择。

## 验收结果

```text
10 passed
```

本课已完成。
