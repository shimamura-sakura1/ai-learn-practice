from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H]
        variance = x.pow(2).mean(dim=-1, keepdim=True)  # [B, S, 1]
        x_norm = x * torch.rsqrt(variance + self.eps)  # [B, S, H]
        return x_norm * self.weight  # [B, S, H]
