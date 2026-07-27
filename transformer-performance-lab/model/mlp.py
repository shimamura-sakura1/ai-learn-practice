from __future__ import annotations

import torch
from torch import nn

from .config import TransformerConfig


class SwiGLUMLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        gate = torch.nn.functional.silu(self.gate_proj(x))  # [B, S, I]
        up = self.up_proj(x)  # [B, S, I]
        hidden = gate * up  # [B, S, I]
        return self.down_proj(hidden)  # [B, S, H]
