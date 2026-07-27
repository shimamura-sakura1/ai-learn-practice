from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 32000
    batch_size: int = 2
    sequence_length: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        return self.hidden_size * self.mlp_ratio
