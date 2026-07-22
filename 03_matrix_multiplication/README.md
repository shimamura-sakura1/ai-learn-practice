# 练习 03：Matrix Multiplication

先打开 `matmul-lecture.html`，观察：

- 一个 program 始终负责 C 的一个二维 tile。
- 改变 `pid_m` / `pid_n` 会选择不同的输出 tile。
- 沿 K 维前进时，A tile 与 B tile 会变化，但累加目标 C tile 不变。

然后在 `matmul.py` 中独立实现：

- `matmul_kernel`：二维 tile 索引、A/B 指针、K 维循环、`tl.dot` 和写回。
- `matmul`：外围检查、输出分配、grid 与 kernel 启动。

练习框架只包含 imports、函数签名、职责注释和未实现占位。

## 第一阶段目标

暂时不考虑 autotune、GROUP_SIZE_M 或 L2 cache 重排。先完成最直接的二维 grid 教学版本：

```text
grid = (M 方向 tile 数量, N 方向 tile 数量)
```

## 运行测试

```bash
conda activate triton
cd /mnt/d/ai-learn/practice/03_matrix_multiplication
pytest -q
```

最大测试只包含 `128 × 64` 与 `64 × 96` 的 FP16 矩阵，显存占用远低于 10 MB。

## 教学实现边界

- 第一阶段使用直接二维 grid，不做 program 重排。
- 使用固定的 `BLOCK_SIZE_M/N/K`，不做 autotune。
- 重点是正确性、stride、二维指针和 K 维累加。
- 工业实现还需要考虑 Tensor Core 配置、warp 数量、流水级数、occupancy、L2 数据复用和不同形状的配置选择。
