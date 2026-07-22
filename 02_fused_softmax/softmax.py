"""Triton 行级 Softmax 练习框架。"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 在 GPU 上计算二维矩阵每一行的 Softmax。
    # 一个 Triton program 负责一整行，并在行内完成归约。
    # 需要处理数值稳定性以及补齐位置的越界访问。
    pid = tl.program_id(0) # 这里按照行切分，每一列最大blocksize元素，这里默认规定 ncols < blocksize
    offsets = pid * input_row_stride + tl.arange(0, BLOCK_SIZE)

    mask = offsets < pid * input_row_stride + n_cols

    x = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
    
    m = tl.max(x)
    sigma = tl.sum(tl.exp(x - m), axis=0)
    output = tl.exp(x-m) / sigma
    output_offsets = pid * output_row_stride + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptr + output_offsets, output, mask=mask)

def softmax(x: torch.Tensor) -> torch.Tensor:
    # 检查输入，并准备输出 tensor 与 kernel 启动参数。
    # 为矩阵的每一行启动一个 Triton program。
    # 返回与输入形状相同的行级 Softmax 结果。
    num = x.numel()
    if len(x.shape) == 1:
        n, m = 1, x.shape[0]
    else:
        n, m = x.shape[0], x.shape[1]
    block_size = triton.next_power_of_2(m)
    grid = (n,)
    output = torch.zeros_like(x)
    input_row_stride = x.stride(0)
    output_row_stride = output.stride(0)
    softmax_kernel[grid](
        x,
        output,
        input_row_stride,
        output_row_stride,
        m,
        block_size,
    )
    return output
