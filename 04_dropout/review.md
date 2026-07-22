# 04 Dropout 课程总结

> 课程状态：**数值与行为测试通过；已记录一项边界写回问题**  
> 标签：`#一维分块` `#随机数` `#逐元素-mask` `#inverted-dropout` `#11项测试通过`

## 一个 program 负责什么

一个 program 负责扁平 tensor 中一段长度为 `BLOCK_SIZE` 的连续元素。grid 中多个 program 并行处理不同数据块；单个 program 内由 `tl.arange` 一次表达一组 offsets，而不是 Python 循环逐元素执行。

## 已掌握

- 用 `program_id`、`BLOCK_SIZE` 和 `tl.arange` 构造一维 offsets。
- 用 `tl.rand(seed, offsets)` 为每个逻辑位置生成可复现的随机数。
- 区分边界 mask 与随机 keep mask。
- 用 `tl.where` 表达保留或归零。
- 用 `x / (1 - p)` 实现 inverted dropout，使输出期望保持不变。

## 错误与原因

- 最初的 `tl.store` 没有传入边界 mask。最后一个 program 的无效 lane 会写到输出范围之外。
- 现有数值测试仍可能全部通过，因为少量 GPU 越界写入不一定立即报错或改变可观察结果。
- 这说明数值一致性测试不能替代对 load/store 边界的静态检查或内存检查工具。

## 性能观察

- Dropout 属于逐元素、通常受内存带宽限制的算子，计算量很小。
- 独立 kernel 需要读取输入并写回输出；与相邻算子融合通常比单独优化算术更有价值。
- 无状态 RNG 让每个 offset 可独立生成随机值，适合 program 间并行。
- 本章只验证正确性与统计行为，没有进行 benchmark。

## 教学实现边界

- 输入要求 contiguous，按一维连续内存处理。
- 只实现前向 inverted dropout，不包含反向传播。
- 没有接入 PyTorch 全局 RNG 状态，也不保证与 PyTorch 生成相同 mask。
- 没有处理多 GPU/分布式训练中的 RNG 流与可复现性。
- 工业实现通常会融合相邻算子，并严格管理 RNG offset、设备状态和前后向的一致性。

## 课程验收

- `p=0`、同 seed 可复现、不同 seed 产生不同 mask。
- 多个 p 值的保留率与 inverted scaling 通过行为测试。
- shape、dtype、device 以及非法参数校验通过。
- 最终结果：`11 passed`。
- 已知待办：核心实现中的 `tl.store` 仍需添加 `mask=mask`，因此不能把数值测试通过等同于完整内存安全验收。
