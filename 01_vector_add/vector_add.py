"""Triton 向量加法练习框架。"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 在 GPU 上计算两个向量的逐元素加法。
    # 一个 Triton program 负责处理一部分元素。
    # 需要保证访问不会超出向量的有效范围。
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE
    block = offsets + tl.arange(0, BLOCK_SIZE)
    mask = block < n_elements

    x = tl.load(x_ptr+block, mask=mask)
    y = tl.load(y_ptr+block, mask=mask)
    results = x + y
    tl.store(output_ptr+block, results, mask=mask)

def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 准备向量加法所需的输入、输出和启动参数。
    # 启动上面的 Triton kernel 完成计算。
    # 返回包含计算结果的 GPU tensor。
    if x.shape != y.shape:
        raise ValueError(f"x 和 y 的形状必须相同，实际为 x.shape={x.shape}, y.shape={y.shape}")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x 和 y 必须是 CUDA tensor")
    
    n = x.numel()
    output = torch.zeros_like(x)
    block_size = 64
    m = triton.cdiv(n, block_size)
    grid = (m, )
    vector_add_kernel[grid](
        x,
        y,
        output,
        n,
        block_size,
    )
    return output
