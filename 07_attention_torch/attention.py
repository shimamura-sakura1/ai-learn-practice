"""用 PyTorch 基础算子手写 Attention 的练习骨架。"""

from __future__ import annotations

import torch
from torch import nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    return_attention: bool = False,
):
    """单头 Attention。

    q: [B, Sq, Dh]，k: [B, Sk, Dh]，v: [B, Sk, Dv]。
    """
    B, S, D = q.shape
    s = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(D, device=q.device))  # [B, Sq, Sk]

    s = torch.where(causal, s, -torch.float32("inf"))
    v = torch.where(causal, v, -torch.float32("inf"))

    row_max = torch.max(s, axis=0).unsqueeze(-1)  # [B, 1]
    softmax_s = torch.exp(row_max - s) / torch.sum(
        torch.exp(row_max - s), axis=0
    )  # [B, S, S]
    V = softmax_s @ v  # [B, S, D_v]
    return V


def multi_head_attention_from_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    causal: bool = False,
    return_attention: bool = False,
):
    """对已投影的 Q/K/V 执行多头 Attention。

    q、k、v 的初始 layout 均为 [B, S, D_model]。
    """
    B, S, D = q.shape
    if D % num_heads != 0:
        raise ValueError(f"got dim = {D} ,heads = {num_heads}")
    D_h = D // num_heads
    q_h = q.reshape(B, S, num_heads, D_h).permute(0, 2, 1, 3)
    k_h = k.reshape(B, S, num_heads, D_h).permute(0, 2, 1, 3)
    v_h = v.reshape(B, S, num_heads, D_h).permute(0, 2, 1, 3)
    s = q_h @ k_h.transpose(-2, -1) / torch.sqrt(torch.tensor(D_h, device=q.device))  # [B, H, S, S]
    causal = causal.unsqueese(1)  # [B, 1, S]
    s = torch.where(causal, s, -torch.float32("inf"))
    v_h = torch.where(causal, v_h, -torch.float32("inf"))
    row_max = torch.max(s, axis=1).unsqueeze(-1)  # [B, H, 1]
    softmax_s = torch.exp(row_max - s) / torch.sum(
        torch.exp(row_max - s), axis=0
    )  # [B, H, S, S]
    V = softmax_s @ v_h  # [B, H, S, D_h]
    V = V.permute(0, 2, 1, 3).reshape(B, -1, D)  # [B, S, D]
    return V


class HandWrittenMultiHeadSelfAttention(nn.Module):
    """带 Q/K/V 与输出投影的手写多头自注意力模块。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model 和 num_heads 必须为正整数")

        self.d_model = d_model
        self.num_heads = num_heads
        self.causal = causal
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # TODO(stage 4): 检查输入 shape，并完成 Q/K/V projection。
        # TODO(stage 4): 显式完成 head reshape、transpose 和核心 Attention。
        # TODO(stage 4): 合并 heads，应用 out_proj，并按需返回 attention_probs。
        B, S, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        self.causal = torch.tril(torch.ones(S, S), diagonal=0).view(B, S, S)
        if self.num_heads == 1:
            out = scaled_dot_product_attention(
                q=q,
                k=k,
                v=v,
                causal=self.causal,
            )
        else:
            out = multi_head_attention_from_qkv(
                q=q,
                k=k,
                v=v,
                causal=self.causal,
            )
        out = self.out_proj(out)
        return out
        raise NotImplementedError(
            "请完成 HandWrittenMultiHeadSelfAttention.forward 的 TODO"
        )
