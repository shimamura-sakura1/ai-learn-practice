# 03 Matrix Multiplication 课程总结

> 课程状态：**已完成（基础正确性验收）**  
> 标签：`#二维-tile` `#stride` `#K维累加` `#tl-dot` `#已通过测试`

## 一个 program 负责什么

一个 Triton program 负责输出矩阵 `C` 的一个二维 tile。grid 中不同的 program 并行处理不同的 C tile；单个 program 内，`tl.arange` 构造的 Triton tensor 向量化表达 tile 内的多行、多列元素，不是 Python 逐元素循环。

对 `C[M, N] = A[M, K] @ B[K, N]`：

- program 根据 `program_id(0/1)` 确定当前 C tile 的 M/N 范围；
- 根据逻辑下标、stride 和基地址构造 A、B、C 的二维指针；
- 在 program 内沿 K 维分块循环；
- 每轮读取 `A[BM, BK]` 和 `B[BK, BN]`，通过 `tl.dot` 累加到 `C[BM, BN]`；
- 边缘 tile 通过逻辑下标 mask 避免越界。

## 已理解

- program 根据 C tile 划分。
- 基地址 + offsets + stride 构造二维指针。
- K 维在 program 内循环累加。
- tile 带来 A/B 片上复用。

## 仍不确定

- program 到 warp/thread 的精确映射。
- `tl.load` 后数据实际落在哪一层存储。
- `num_warps` / `num_stages` 的性能影响。
- tile size 如何影响寄存器与 occupancy。

这些问题不影响本章基础语义验收，将在后续 Profiling 与性能优化阶段通过编译结果、性能数据和 Nsight 工具继续验证，而不是只凭抽象描述判断。

## 内存与索引

二维地址的核心公式：

```text
A[m, k] = a_ptr + m * stride_am + k * stride_ak
B[k, n] = b_ptr + k * stride_bk + n * stride_bn
C[m, n] = c_ptr + m * stride_cm + n * stride_cn
```

`offsets_m[:, None]` 与 `offsets_k[None, :]` 通过广播形成 `[BM, BK]`；同理，K/N 形成 `[BK, BN]`，M/N 形成 `[BM, BN]`。mask 比较的是逻辑下标与 M/N/K 边界，而不是内存地址。

## 错误与原因

- Python 启动端误用 `tl.cdiv`：kernel 外应使用 `triton.cdiv`。
- M/N/K 对应关系写反：A 是 `[M, K]`，B 是 `[K, N]`。
- 输出 tensor 未指定设备或 dtype：导致 CPU/GPU 或 FP32/FP16 不一致。
- K 循环起点重复乘 `BLOCK_SIZE_K`：循环变量已经是当前 K 分块的起点。
- 用指针和形状比较生成 mask：mask 应由逻辑 offsets 构造。
- 覆盖原始基地址：容易让后续 K 循环在已偏移地址上再次累加。
- masked load 未明确补零：K 维越界位置参与点积时应使用 `other=0.0`。
- 测试曾逐字匹配错误消息：这属于测试过度约束，现已改为验证是否正确拒绝非法输入。

## 性能观察

- 当前只完成正确性验证，尚未进行可靠 benchmark，因此不能宣称比 `torch.matmul` 更快。
- tile 让 A 的元素可以服务同一 tile 中多个 N 输出，让 B 的元素可以服务多个 M 输出，减少重复读取的需求。
- FP16 输入、FP32 累加兼顾 Tensor Core 路径与累加精度，最终写回输入 dtype。
- tile 越大不必然越快：复用增加的同时，寄存器需求也会上升，可能降低 occupancy。

## 教学实现边界

当前版本是用于理解执行模型的基础实现：

- 使用固定的 `64 × 64 × 64` tile，未做硬件相关 autotune；
- 使用直接的二维 grid，未实现 grouped program ordering 或更复杂的数据复用调度；
- 没有比较 `num_warps`、`num_stages` 和不同 tile 组合；
- 没有分析寄存器数量、occupancy、共享内存、访存吞吐和 Tensor Core 利用率；
- 没有覆盖完整工业矩阵乘法所需的 dtype、布局、转置、批量维度和融合 epilogue。

工业实现通常会针对形状、GPU 架构和数据类型选择配置，并通过 autotune 与 profiler 数据决定 tile、warps、stages 和 program 排序。

## 课程验收

- 数值结果与 `torch.matmul` 对齐。
- 覆盖规则形状和非 tile 整数倍形状。
- 覆盖最小矩阵及多个 M/N/K 组合。
- 非二维、CPU 输入和内侧维度不匹配能够被外围校验拒绝。
- 最终结果：`9 passed`。

结论：第三章的基础语法和正确性目标已完成；性能机制仍保持“未确定”标签，留待第 07 章 Profiling 通过实测解决。
