from __future__ import annotations

import math

import torch
from torch import nn

from .config import TransformerConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)  # [B, S, NH, D]
        return x.transpose(1, 2)  # [B, NH, S, D]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, NH, S, D]
        batch, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()  # [B, S, NH, D]
        return x.view(batch, seq_len, self.hidden_size)  # [B, S, H]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        q = self._split_heads(self.q_proj(x))  # [B, NH, S, D]
        k = self._split_heads(self.k_proj(x))  # [B, NH, S, D]
        v = self._split_heads(self.v_proj(x))  # [B, NH, S, D]

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, NH, S, S]

        seq_len = scores.shape[-1]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )  # [S, S]
        scores = scores.masked_fill(~causal_mask, float("-inf"))  # [B, NH, S, S]

        probs = torch.softmax(scores, dim=-1)  # [B, NH, S, S]
        context = torch.matmul(probs, v)  # [B, NH, S, D]
        context = self._merge_heads(context)  # [B, S, H]
        return self.out_proj(context)  # [B, S, H]
