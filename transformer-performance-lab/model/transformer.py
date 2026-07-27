from __future__ import annotations

import torch
from torch import nn

from .block import TransformerBlock
from .config import TransformerConfig
from .rmsnorm import RMSNorm


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, S]
        x = self.token_embedding(input_ids)  # [B, S, H]
        for block in self.blocks:
            x = block(x)  # [B, S, H]
        x = self.final_norm(x)  # [B, S, H]
        return self.lm_head(x)  # [B, S, V]

    def loss(self, input_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # targets: [B, S]
        logits = self(input_ids)  # [B, S, V]
        vocab_size = logits.shape[-1]
        return torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size),  # [B*S, V]
            targets.view(-1),  # [B*S]
        )
