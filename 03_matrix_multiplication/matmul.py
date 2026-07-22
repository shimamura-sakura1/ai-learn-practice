"""Triton 分块矩阵乘法练习框架。"""

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 在 GPU 上计算 C = A @ B。
    # 一个 Triton program 负责 C 的一个二维 tile。
    # 沿 K 维分块读取 A/B，并把局部乘积累加到同一个输出 tile。
    row_id = tl.program_id(0)
    col_id = tl.program_id(1)
    offsets_m = row_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_n = col_id * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = (c_ptr + offsets_m[:, None] * stride_cm + offsets_n[None, :] * stride_cn)
    mask_c = (
        (offsets_m[:, None] < M) & (offsets_n[None, :] < N)
    )
    output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        offsets_k = k + tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = (a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak)
        b_ptrs = (b_ptr + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn)
        mask_a = (
            (offsets_m[:, None] < M) & (offsets_k[None ,:] < K)
        )
        mask_b =  (
            (offsets_k[:, None] < K) & (offsets_n[None ,:] < N)
        )
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        output += tl.dot(a, b)
    tl.store(c_ptr, output, mask=mask_c)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # 检查矩阵形状、设备和 dtype，并分配输出矩阵。
    # 根据 M/N 方向的 tile 数量构造 grid，再启动 Triton kernel。
    # 返回形状为 (M, N) 的结果 tensor。
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("输入必须是二维矩阵")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("输入必须位于 CUDA 设备上")
    if a.device != b.device:
        raise ValueError("两个输入必须位于同一 CUDA 设备上")
    if a.dtype != b.dtype:
        raise ValueError("两个输入必须具有相同 dtype")
    if a.shape[1] != b.shape[0]:
        raise ValueError("矩阵内侧维度不匹配")
    c = torch.zeros((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    stride_ak = a.stride(1)
    stride_am = a.stride(0)
    stride_bk = b.stride(0)
    stride_bn = b.stride(1)
    stride_cm = c.stride(0)
    stride_cn = c.stride(1)
    M, K, N = a.shape[0], a.shape[1], b.shape[1]
    BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_K = 64, 64, 64
    grid_x, grid_y = triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)
    grid = (grid_x, grid_y)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    return c
