"""dropout.py 的行为正确性测试。"""

import pytest
import torch

from dropout import dropout


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="这个练习需要 CUDA GPU",
)


def test_p_zero_returns_input():
    x = torch.randn(1003, device="cuda", dtype=torch.float32)
    actual = dropout(x, p=0.0, seed=17)

    torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_same_seed_is_reproducible():
    x = torch.ones(4096, device="cuda", dtype=torch.float32)

    first = dropout(x, p=0.35, seed=123)
    second = dropout(x, p=0.35, seed=123)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_different_seeds_change_mask():
    x = torch.ones(4096, device="cuda", dtype=torch.float32)

    first = dropout(x, p=0.35, seed=123)
    second = dropout(x, p=0.35, seed=124)

    assert not torch.equal(first, second)


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
def test_keep_rate_and_inverted_scaling(p):
    x = torch.ones(32768, device="cuda", dtype=torch.float32)
    actual = dropout(x, p=p, seed=42)

    kept = actual != 0
    observed_keep_rate = kept.float().mean().item()
    assert abs(observed_keep_rate - (1.0 - p)) < 0.025

    expected_kept_value = torch.tensor(1.0 / (1.0 - p), device="cuda")
    torch.testing.assert_close(
        actual[kept],
        expected_kept_value.expand_as(actual[kept]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_preserves_shape_dtype_and_device():
    x = torch.randn(17, 19, device="cuda", dtype=torch.float32)
    actual = dropout(x, p=0.2, seed=9)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device


@pytest.mark.parametrize("p", [-0.1, 1.0, 1.2])
def test_rejects_invalid_probability(p):
    x = torch.ones(16, device="cuda")

    with pytest.raises(ValueError):
        dropout(x, p=p, seed=0)


def test_rejects_cpu_tensor():
    with pytest.raises(ValueError):
        dropout(torch.ones(16), p=0.5, seed=0)
