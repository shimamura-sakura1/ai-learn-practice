"""PyTorch reference implementation for scaled dot-product attention.

This module is intentionally explicit: it materializes the score matrix and
probability matrix so the memory cost is visible before moving to Triton and
FlashAttention-style kernels.
"""

from __future__ import annotations

import math

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    """Validate Q/K/V tensors for the lab attention API.

    Expected shapes:
        q: [B, H, Sq, D]
        k: [B, H, Sk, D]
        v: [B, H, Sk, Dv]
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors with shape [B, H, S, D]")

    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v must share batch and head dimensions")

    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must share the key/value sequence length")

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must share the head dimension")


def _causal_mask(sq: int, sk: int, device: torch.device) -> torch.Tensor:
    """Create a causal mask for score rows [Sq] and key columns [Sk].

    For self-attention Sq == Sk, row i can attend to columns <= i.
    """
    # q_positions = torch.arange(sq, device=device)[:, None]  # [Sq, 1]
    # k_positions = torch.arange(sk, device=device)[None, :]  # [1, Sk]
    causal_mask = torch.tril(
        torch.ones(sq, sk, device=device, dtype=torch.bool),
        diagonal=0
    )
    return causal_mask
    return k_positions <= q_positions  # [Sq, Sk]


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    return_probs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute reference scaled dot-product attention.

    Args:
        q: Query tensor with shape [B, H, Sq, D].
        k: Key tensor with shape [B, H, Sk, D].
        v: Value tensor with shape [B, H, Sk, Dv].
        causal: If true, apply a lower-triangular causal mask over [Sq, Sk].
        return_probs: If true, return both output and attention probabilities.

    Returns:
        output: Tensor with shape [B, H, Sq, Dv].
        probs: Optional tensor with shape [B, H, Sq, Sk].

    Math:
        scores = q @ k.transpose(-2, -1) / sqrt(D)  # [B, H, Sq, Sk]
        probs = softmax(scores, dim=-1)             # [B, H, Sq, Sk]
        output = probs @ v                          # [B, H, Sq, Dv]
    """
    _validate_qkv(q, k, v)

    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    # [B, H, Sq, D] @ [B, H, D, Sk] -> [B, H, Sq, Sk]
    with record_function("attention/qk_matmul"):
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        sq, sk = scores.shape[-2:]
        mask = _causal_mask(sq, sk, scores.device)  # [Sq, Sk]
        scores = scores.masked_fill(~mask, float("-inf"))
    with record_function("attention/softmax"):
        probs = torch.softmax(scores, dim=-1)  # [B, H, Sq, Sk]
    
    with record_function("attention/pv_matmul"):
        output = torch.matmul(probs, v)  # [B, H, Sq, Dv]

    if return_probs:
        return output, probs
    return output
