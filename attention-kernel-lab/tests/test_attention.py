"""Correctness tests for Phase 1 PyTorch reference attention."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from attention.torch_attention import attention


def _device_dtype_cases() -> list[tuple[str, torch.dtype]]:
    cases = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        cases.append(("cuda", torch.float16))
        cases.append(("cuda", torch.float32))
        if torch.cuda.is_bf16_supported():
            cases.append(("cuda", torch.bfloat16))
    return cases


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 2e-5, 2e-5


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    atol, rtol = _tolerances(actual.dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize(("device", "dtype"), _device_dtype_cases())
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("batch", "heads", "seq_len", "head_dim"),
    [
        (1, 1, 1, 8),
        (1, 8, 16, 64),
        (2, 4, 33, 32),
        (4, 16, 64, 64),
    ],
)
def test_attention_matches_torch_sdpa(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Compare explicit attention against PyTorch SDPA for [B, H, S, D]."""
    torch.manual_seed(1000 + batch + heads + seq_len + head_dim)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    actual = attention(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)

    assert actual.shape == (batch, heads, seq_len, head_dim)
    assert torch.isfinite(actual).all()
    _assert_close(actual, expected)


@pytest.mark.parametrize("causal", [False, True])
def test_attention_probabilities_are_normalized_and_causal(causal: bool) -> None:
    """Check probability shape, row sums, and future-token masking."""
    torch.manual_seed(2024)
    batch, heads, seq_len, head_dim = 2, 3, 11, 16
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)
    v = torch.randn(batch, heads, seq_len, head_dim)

    output, probs = attention(q, k, v, causal=causal, return_probs=True)

    assert output.shape == (batch, heads, seq_len, head_dim)
    assert probs.shape == (batch, heads, seq_len, seq_len)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(batch, heads, seq_len))
    if causal:
        future_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1)
        assert torch.count_nonzero(probs.masked_select(future_mask)) == 0


def test_attention_supports_value_dim_different_from_head_dim() -> None:
    """Attention output uses Dv from V, while QK scaling uses D from Q/K."""
    torch.manual_seed(3030)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 24)

    output = attention(q, k, v)

    assert output.shape == (2, 4, 8, 24)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "v_shape"),
    [
        ((2, 8, 16), (2, 8, 16), (2, 8, 16)),
        ((2, 4, 8, 16), (2, 5, 8, 16), (2, 4, 8, 16)),
        ((2, 4, 8, 16), (2, 4, 8, 32), (2, 4, 8, 16)),
        ((2, 4, 8, 16), (2, 4, 9, 16), (2, 4, 8, 16)),
    ],
)
def test_attention_rejects_invalid_shapes(
    q_shape: tuple[int, ...],
    k_shape: tuple[int, ...],
    v_shape: tuple[int, ...],
) -> None:
    """Invalid Q/K/V contracts should fail before doing matmul work."""
    q = torch.randn(*q_shape)
    k = torch.randn(*k_shape)
    v = torch.randn(*v_shape)

    with pytest.raises(ValueError):
        attention(q, k, v)
