"""layer_norm.py 的行为正确性测试。"""

import pytest
import torch
import torch.nn.functional as F

from layer_norm import layer_norm


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="这个练习需要 CUDA GPU",
)


@pytest.mark.parametrize(
    ("n_rows", "n_cols"),
    [(1, 1), (3, 7), (16, 64), (11, 257), (32, 1000)],
)
def test_matches_torch_float32(n_rows, n_cols):
    torch.manual_seed(42)
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)
    weight = torch.randn(n_cols, device="cuda", dtype=torch.float32)
    bias = torch.randn(n_cols, device="cuda", dtype=torch.float32)

    actual = layer_norm(x, weight, bias, eps=1e-5)
    expected = F.layer_norm(x, (n_cols,), weight, bias, eps=1e-5)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


def test_matches_torch_float16():
    torch.manual_seed(7)
    x = torch.randn(8, 513, device="cuda", dtype=torch.float16)
    weight = torch.randn(513, device="cuda", dtype=torch.float16)
    bias = torch.randn(513, device="cuda", dtype=torch.float16)

    actual = layer_norm(x, weight, bias)
    expected = F.layer_norm(x, (513,), weight, bias)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_normalized_rows_have_zero_mean_and_unit_variance():
    torch.manual_seed(11)
    x = torch.randn(10, 1024, device="cuda", dtype=torch.float32)
    weight = torch.ones(1024, device="cuda")
    bias = torch.zeros(1024, device="cuda")

    actual = layer_norm(x, weight, bias, eps=1e-8)

    torch.testing.assert_close(actual.mean(dim=1), torch.zeros(10, device="cuda"), atol=2e-5, rtol=0)
    torch.testing.assert_close(actual.var(dim=1, unbiased=False), torch.ones(10, device="cuda"), atol=2e-4, rtol=0)


def test_preserves_shape_dtype_and_device():
    x = torch.randn(4, 17, device="cuda", dtype=torch.float16)
    weight = torch.ones(17, device="cuda", dtype=torch.float16)
    bias = torch.zeros(17, device="cuda", dtype=torch.float16)

    actual = layer_norm(x, weight, bias)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    assert actual.device == x.device


def test_rejects_wrong_affine_shape():
    x = torch.randn(4, 8, device="cuda")
    weight = torch.ones(7, device="cuda")
    bias = torch.zeros(8, device="cuda")

    with pytest.raises(ValueError):
        layer_norm(x, weight, bias)


def test_rejects_cpu_input():
    with pytest.raises(ValueError):
        layer_norm(torch.randn(4, 8), torch.ones(8), torch.zeros(8))
