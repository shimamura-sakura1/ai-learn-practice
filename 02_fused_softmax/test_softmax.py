"""softmax.py 的正确性测试。"""

import pytest
import torch

from softmax import softmax


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="这个练习需要 CUDA GPU",
)


def test_small_example():
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]],
        device="cuda",
    )

    actual = softmax(x)
    expected = torch.softmax(x, dim=1)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    ("n_rows", "n_cols"),
    [
        (1, 1),
        (2, 7),
        (8, 64),
        (17, 257),
        (32, 1024),
    ],
)
def test_matches_torch_for_different_shapes(n_rows, n_cols):
    torch.manual_seed(42)
    x = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float32)

    actual = softmax(x)
    expected = torch.softmax(x, dim=1)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        actual.sum(dim=1),
        torch.ones(n_rows, device="cuda"),
        rtol=1e-4,
        atol=1e-5,
    )


def test_is_numerically_stable():
    x = torch.tensor(
        [[10_000.0, 9_999.0, 9_998.0], [-10_000.0, -10_001.0, -9_999.0]],
        device="cuda",
    )

    actual = softmax(x)
    expected = torch.softmax(x, dim=1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_rejects_non_matrix_input():
    x = torch.randn(16, device="cuda")

    with pytest.raises(ValueError, match="二维"):
        softmax(x)


def test_rejects_cpu_tensor():
    x = torch.randn(4, 8)

    with pytest.raises(ValueError, match="CUDA tensor"):
        softmax(x)
