"""Naive attention entry point used before Triton kernels are introduced."""

from __future__ import annotations

import torch

from .torch_attention import attention as torch_attention


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    return_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute naive attention by materializing the [B, H, S, S] matrix.

    This wrapper exists so later phases can replace individual steps with
    Triton kernels without changing call sites.
    """
    return torch_attention(q, k, v, causal=causal, return_probs=return_probs)
