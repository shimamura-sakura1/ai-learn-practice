"""matmul.py 的正确性测试。"""

import pytest
import torch

from matmul import matmul


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="这个练习需要 CUDA GPU",
)


def test_small_example():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float16)
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda", dtype=torch.float16)

    actual = matmul(a, b)
    expected = torch.matmul(a, b)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("M", "N", "K"),
    [
        (1, 1, 1),
        (7, 5, 3),
        (16, 16, 16),
        (33, 29, 17),
        (128, 96, 64),
    ],
)
def test_matches_torch_for_different_shapes(M, N, K):
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    actual = matmul(a, b)
    expected = torch.matmul(a, b)

    assert actual.shape == (M, N)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_rejects_incompatible_shapes():
    a = torch.randn(4, 7, device="cuda", dtype=torch.float16)
    b = torch.randn(5, 3, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError):
        matmul(a, b)


def test_rejects_non_matrix_input():
    a = torch.randn(16, device="cuda", dtype=torch.float16)
    b = torch.randn(16, 8, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError):
        matmul(a, b)


def test_rejects_cpu_tensor():
    a = torch.randn(4, 8, dtype=torch.float16)
    b = torch.randn(8, 3, dtype=torch.float16)

    with pytest.raises(ValueError):
        matmul(a, b)
