# 练习 01：Triton 向量加法

请在 `vector_add.py` 中独立实现两个函数：

- `vector_add_kernel`：在 GPU 上执行向量加法。
- `vector_add`：准备参数、启动 kernel 并返回结果。

框架只提供函数签名和职责说明，不包含实现步骤或参考答案。

## 运行测试

在 VS Code 的 WSL 终端中执行：

```bash
conda activate triton
cd /mnt/d/ai-learn/practice/01_vector_add
pytest -q
```

最大测试向量只有 65,537 个元素，显存占用远低于 10 MB。
