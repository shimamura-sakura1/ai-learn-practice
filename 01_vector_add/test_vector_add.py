"""vector_add.py 的正确性测试。

运行：pytest -q
"""

import pytest
import torch

from vector_add import vector_add


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="这个练习需要 CUDA GPU",
)


def test_small_example():
    """先用肉眼容易理解的 4 个元素检查基本加法。"""

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    y = torch.tensor([10.0, 20.0, 30.0, 40.0], device="cuda")

    actual = vector_add(x, y)
    expected = torch.tensor([11.0, 22.0, 33.0, 44.0], device="cuda")

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "n_elements",
    [
        1,
        17,
        255,
        256,
        257,
        4096,
        65_537,
    ],
)
def test_matches_torch_for_different_sizes(n_elements):
    """重点检查小尺寸、整块尺寸及最后一块需要 mask 的尺寸。"""

    torch.manual_seed(42)
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn(n_elements, device="cuda", dtype=torch.float32)

    actual = vector_add(x, y)
    expected = x + y

    torch.testing.assert_close(actual, expected)


def test_rejects_different_shapes():
    x = torch.randn(16, device="cuda")
    y = torch.randn(17, device="cuda")

    with pytest.raises(ValueError, match="形状必须相同"):
        vector_add(x, y)


def test_rejects_cpu_tensors():
    x = torch.randn(16)
    y = torch.randn(16)

    with pytest.raises(ValueError, match="必须是 CUDA tensor"):
        vector_add(x, y)
