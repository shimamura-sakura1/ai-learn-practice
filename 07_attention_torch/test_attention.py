"""Attention 练习的分阶段 correctness 测试。"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from attention import (
    HandWrittenMultiHeadSelfAttention,
    multi_head_attention_from_qkv,
    scaled_dot_product_attention,
)


SHAPES = [(1, 1, 4, 1), (1, 4, 8, 2), (2, 7, 16, 4), (3, 16, 32, 8)]


def _device_dtype_cases():
    cases = [("cpu", torch.float32)]
    if torch.cuda.is_available():
        cases.append(("cuda", torch.float16))
        if torch.cuda.is_bf16_supported():
            cases.append(("cuda", torch.bfloat16))
    return cases


DEVICE_DTYPES = _device_dtype_cases()


def _tolerances(dtype):
    if dtype == torch.float32:
        return 2e-5, 2e-5
    if dtype == torch.float16:
        return 3e-3, 3e-3
    return 2e-2, 2e-2


def _assert_close_with_context(actual, expected, *, context):
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    abs_error = (actual_f - expected_f).abs()
    rel_error = abs_error / expected_f.abs().clamp_min(1e-12)
    flat_index = int(abs_error.reshape(-1).argmax().item())
    index = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_index), abs_error.shape))
    atol, rtol = _tolerances(actual.dtype)
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        details = (
            f"{context}\n"
            f"dtype={actual.dtype}, max_abs_error={abs_error.max().item():.6g}, "
            f"max_relative_error={rel_error.max().item():.6g}\n"
            f"worst_index={index}, mine={actual_f[index].item():.6g}, "
            f"reference={expected_f[index].item():.6g}"
        )
        raise AssertionError(details) from error


def _official_single_head(q, k, v, causal):
    return F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), dropout_p=0.0, is_causal=causal
    ).squeeze(1)


def _official_mha_from_qkv(q, k, v, num_heads, causal):
    d_model = q.shape[-1]
    module = nn.MultiheadAttention(d_model, num_heads, dropout=0.0, bias=False, batch_first=True)
    eye = torch.eye(d_model, device=q.device, dtype=q.dtype)
    with torch.no_grad():
        module.in_proj_weight.copy_(torch.cat((eye, eye, eye), dim=0))
        module.out_proj.weight.copy_(eye)
    mask = None
    if causal:
        mask = torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool).triu(1)
    return module(q, k, v, attn_mask=mask, need_weights=True, average_attn_weights=False)


@pytest.mark.parametrize(("device", "dtype"), DEVICE_DTYPES)
def test_single_head_cross_shape_and_reference(device, dtype):
    torch.manual_seed(10)
    q = torch.randn(2, 3, 8, device=device, dtype=dtype)
    k = torch.randn(2, 5, 8, device=device, dtype=dtype)
    v = torch.randn(2, 5, 8, device=device, dtype=dtype)

    out, probs = scaled_dot_product_attention(q, k, v, return_attention=True)
    expected = _official_single_head(q, k, v, causal=False)

    assert out.shape == (2, 3, 8)
    assert probs.shape == (2, 3, 5)  # 这也是本 API 中 scores 的 shape 合约。
    assert torch.isfinite(out).all()
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=atol, rtol=rtol)
    _assert_close_with_context(
        out,
        expected,
        context="B=2, Sq=3, Sk=5, D=8, H=1, causal=False",
    )


@pytest.mark.parametrize(("b", "s", "d", "h"), SHAPES)
def test_single_head_shapes_probabilities_and_finite(b, s, d, h):
    torch.manual_seed(11 + s)
    q, k, v = (torch.randn(b, s, d) for _ in range(3))
    out, probs = scaled_dot_product_attention(q, k, v, return_attention=True)
    assert out.shape == (b, s, d)
    assert probs.shape == (b, s, s)
    assert torch.isfinite(out).all() and torch.isfinite(probs).all()
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(b, s), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("s", [1, 4, 7, 16])
def test_single_head_causal_mask(s):
    torch.manual_seed(20 + s)
    q, k, v = (torch.randn(2, s, 8) for _ in range(3))
    out, probs = scaled_dot_product_attention(q, k, v, causal=True, return_attention=True)
    future = torch.ones(s, s, dtype=torch.bool).triu(1)
    assert torch.count_nonzero(probs.masked_select(future)) == 0
    expected = _official_single_head(q, k, v, causal=True)
    _assert_close_with_context(out, expected, context=f"B=2, S={s}, D=8, H=1, causal=True")


@pytest.mark.parametrize(("b", "s", "d", "h"), SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_multi_head_matches_independent_reference(b, s, d, h, causal):
    torch.manual_seed(30 + s)
    q, k, v = (torch.randn(b, s, d) for _ in range(3))
    out, probs = multi_head_attention_from_qkv(q, k, v, h, causal=causal, return_attention=True)
    expected_out, expected_probs = _official_mha_from_qkv(q, k, v, h, causal)
    context = f"B={b}, S={s}, D={d}, H={h}, causal={causal}"
    assert out.shape == (b, s, d)
    assert probs.shape == (b, h, s, s)
    assert torch.isfinite(out).all()
    _assert_close_with_context(out, expected_out, context=context)
    _assert_close_with_context(probs, expected_probs, context=context)


def test_multi_head_matches_per_head_loop():
    torch.manual_seed(41)
    b, s, d, h = 2, 7, 16, 4
    q, k, v = (torch.randn(b, s, d) for _ in range(3))
    actual = multi_head_attention_from_qkv(q, k, v, h)
    width = d // h
    per_head = [
        _official_single_head(
            q[..., head * width : (head + 1) * width],
            k[..., head * width : (head + 1) * width],
            v[..., head * width : (head + 1) * width],
            causal=False,
        )
        for head in range(h)
    ]
    expected = torch.cat(per_head, dim=-1)
    _assert_close_with_context(actual, expected, context="B=2, S=7, D=16, H=4, per-head loop")


def test_rejects_non_divisible_d_model():
    q = k = v = torch.randn(2, 4, 10)
    with pytest.raises((ValueError, AssertionError), match="(?i)(divis|整除|head)"):
        multi_head_attention_from_qkv(q, k, v, num_heads=3)


def _copy_parameters_to_official(mine, official):
    with torch.no_grad():
        official.in_proj_weight.copy_(
            torch.cat((mine.q_proj.weight, mine.k_proj.weight, mine.v_proj.weight), dim=0)
        )
        if official.in_proj_bias is not None:
            official.in_proj_bias.copy_(
                torch.cat((mine.q_proj.bias, mine.k_proj.bias, mine.v_proj.bias), dim=0)
            )
        official.out_proj.weight.copy_(mine.out_proj.weight)
        if official.out_proj.bias is not None:
            official.out_proj.bias.copy_(mine.out_proj.bias)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_full_module_forward_and_backward_match_torch(bias, causal):
    torch.manual_seed(53)
    b, s, d, h = 2, 7, 16, 4
    mine = HandWrittenMultiHeadSelfAttention(d, h, bias=bias, causal=causal)
    official = nn.MultiheadAttention(d, h, dropout=0.0, bias=bias, batch_first=True)
    _copy_parameters_to_official(mine, official)
    x_mine = torch.randn(b, s, d, requires_grad=True)
    x_ref = x_mine.detach().clone().requires_grad_(True)

    out_mine, probs_mine = mine(x_mine, return_attention=True)
    mask = torch.ones(s, s, dtype=torch.bool).triu(1) if causal else None
    out_ref, probs_ref = official(
        x_ref, x_ref, x_ref, attn_mask=mask, need_weights=True, average_attn_weights=False
    )
    context = f"B={b}, S={s}, D={d}, H={h}, dtype=float32, causal={causal}, bias={bias}"
    _assert_close_with_context(out_mine, out_ref, context=context)
    _assert_close_with_context(probs_mine, probs_ref, context=context)

    out_mine.square().mean().backward()
    out_ref.square().mean().backward()
    assert torch.isfinite(x_mine.grad).all()
    _assert_close_with_context(x_mine.grad, x_ref.grad, context=context + ", input gradient")
    mappings = (
        (mine.q_proj.weight.grad, official.in_proj_weight.grad[:d], "q_proj"),
        (mine.k_proj.weight.grad, official.in_proj_weight.grad[d : 2 * d], "k_proj"),
        (mine.v_proj.weight.grad, official.in_proj_weight.grad[2 * d :], "v_proj"),
        (mine.out_proj.weight.grad, official.out_proj.weight.grad, "out_proj"),
    )
    for actual_grad, expected_grad, name in mappings:
        assert actual_grad is not None and torch.isfinite(actual_grad).all()
        _assert_close_with_context(actual_grad, expected_grad, context=context + f", {name} gradient")

