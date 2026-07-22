"""Triton Dropout 练习框架。"""

import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # 在 GPU 上实现 inverted dropout。
    # 一个 program 负责输入 tensor 中一段连续元素。
    # 为每个逻辑 offset 生成可复现的随机数，决定保留或丢弃元素。
    # 被保留的元素需要除以 (1 - p)，并用 mask 保护尾部越界位置。
    pid = tl.program_id(0)
    block = pid * BLOCK_SIZE
    offset = block + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    x = tl.load(x_ptr + offset, mask=mask, other=0)
    seed_mask = tl.rand(seed, offset)
    keep_mask = seed_mask > p
    output = tl.where(keep_mask, x / (1-p), 0)
    tl.store(output_ptr + offset, output)


def dropout(x: torch.Tensor, p: float, seed: int = 0) -> torch.Tensor:
    # 校验 tensor、概率和 seed，并分配与输入相同形状的输出。
    # 将输入视为一维连续元素，构造 grid 后启动 dropout_kernel。
    # 外围校验与启动代码由 Codex 维护；kernel 核心逻辑由学习者实现。
    if not isinstance(x, torch.Tensor):
        raise TypeError("x 必须是 torch.Tensor")
    if not x.is_cuda:
        raise ValueError("x 必须位于 CUDA 设备上")
    if not x.is_floating_point():
        raise ValueError("x 必须是浮点 tensor")
    if not x.is_contiguous():
        raise ValueError("当前教学实现要求 x contiguous")
    if not isinstance(p, (int, float)) or isinstance(p, bool):
        raise TypeError("p 必须是数值")
    if not 0.0 <= float(p) < 1.0:
        raise ValueError("p 必须满足 0 <= p < 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是整数")

    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output

    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    dropout_kernel[grid](
        x,
        output,
        n_elements,
        float(p),
        seed,
        BLOCK_SIZE=block_size,
    )
    return output
