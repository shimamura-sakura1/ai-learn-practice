# 练习 02：Fused Softmax

先在浏览器打开 `softmax-lecture.html`，按顺序观察：

```text
输入 → 补齐与 mask → 减最大值 → 指数 → 归一化
```

然后在 `softmax.py` 中独立实现：

- `softmax_kernel`：一个 program 处理矩阵的一行。
- `softmax`：检查输入、准备输出并启动 kernel。

练习框架只提供函数签名和职责说明，不包含实现代码。

## 运行测试

```bash
conda activate triton
cd /mnt/d/ai-learn/practice/02_fused_softmax
pytest -q
```

测试最大输入为 `32 × 1024`，显存占用远低于 10 MB。
