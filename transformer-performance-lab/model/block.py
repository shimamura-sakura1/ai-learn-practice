from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .config import TransformerConfig
from .mlp import SwiGLUMLP
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        x = x + self.attention(self.attn_norm(x))  # [B, S, H]
        x = x + self.mlp(self.mlp_norm(x))  # [B, S, H]
        return x
