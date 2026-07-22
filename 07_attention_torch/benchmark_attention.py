"""普通 Attention 的理论中间内存与性能对比。"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from attention import multi_head_attention_from_qkv


SEQUENCE_LENGTHS = (128, 512, 1024, 2048, 4096, 8192)


def _human_bytes(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def theoretical_sizes(b: int, h: int, s: int, dh: int, dtype: torch.dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    qkv_each_elements = b * h * s * dh
    square_elements = b * h * s * s
    output_elements = b * h * s * dh
    return {
        "qkv_each_elements": qkv_each_elements,
        "qkv_each_bytes": qkv_each_elements * element_size,
        "qkv_total_bytes": 3 * qkv_each_elements * element_size,
        "score_bytes": square_elements * element_size,
        "probs_bytes": square_elements * element_size,
        "output_bytes": output_elements * element_size,
        "score_probs_bytes": 2 * square_elements * element_size,
    }


def print_memory_table(b: int, h: int, dh: int, dtype: torch.dtype):
    print("=== 理论 tensor 大小（不是完整运行时峰值显存）===")
    print(f"B={b}, H={h}, Dh={dh}, dtype={dtype}, bytes/element={torch.empty((), dtype=dtype).element_size()}")
    print("当前普通实现会显式物化 score 和 probs；后续 FlashAttention 章节将避免完整物化它们。")
    header = (
        f"{'S':>6} {'Q/K/V each (elements, bytes)':>34} {'QKV total':>12} "
        f"{'score':>12} {'probs':>12} {'output':>12} {'score+probs':>14} {'vs previous':>12}"
    )
    print(header)
    previous = None
    for s in SEQUENCE_LENGTHS:
        sizes = theoretical_sizes(b, h, s, dh, dtype)
        ratio = "-" if previous is None else f"{sizes['score_probs_bytes'] / previous:.2f}x"
        each = f"{sizes['qkv_each_elements']:,}, {_human_bytes(sizes['qkv_each_bytes'])}"
        print(
            f"{s:6d} {each:>34} {_human_bytes(sizes['qkv_total_bytes']):>12} "
            f"{_human_bytes(sizes['score_bytes']):>12} {_human_bytes(sizes['probs_bytes']):>12} "
            f"{_human_bytes(sizes['output_bytes']):>12} {_human_bytes(sizes['score_probs_bytes']):>14} {ratio:>12}"
        )
        previous = sizes["score_probs_bytes"]
    print("注意：表中的间隔不全是 S 翻倍；只有 S 恰好翻倍时，score+probs 才应显示约 4.00x。")


def _synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _median_runtime_ms(fn, warmup: int, iterations: int, device: torch.device):
    for _ in range(warmup):
        fn()
    _synchronize(device)
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        _synchronize(device)
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def run_performance(b: int, h: int, s: int, dh: int, dtype: torch.dtype, device: torch.device, iterations: int):
    d_model = h * dh
    q_flat = torch.randn(b, s, d_model, device=device, dtype=dtype)
    k_flat = torch.randn_like(q_flat)
    v_flat = torch.randn_like(q_flat)
    q_heads = q_flat.reshape(b, s, h, dh).transpose(1, 2)
    k_heads = k_flat.reshape(b, s, h, dh).transpose(1, 2)
    v_heads = v_flat.reshape(b, s, h, dh).transpose(1, 2)

    handwritten = lambda: multi_head_attention_from_qkv(q_flat, k_flat, v_flat, h)
    official = lambda: F.scaled_dot_product_attention(q_heads, k_heads, v_heads, dropout_p=0.0)
    print(f"\n=== 性能对比：B={b}, H={h}, S={s}, Dh={dh}, {dtype}, {device} ===")
    try:
        hand_ms = _median_runtime_ms(handwritten, 3, iterations, device)
        print(f"手写普通 Attention: {hand_ms:.3f} ms (median)")
    except NotImplementedError as error:
        print(f"手写普通 Attention: 尚未实现，跳过（{error}）")
    official_ms = _median_runtime_ms(official, 3, iterations, device)
    print(f"PyTorch SDPA reference: {official_ms:.3f} ms (median)")
    print("两条路径的输入 layout 不同，且官方 SDPA 可能选择融合 kernel；结果用于观察，不是严格 kernel 微基准。")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("memory", "performance", "all"), default="all")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--sequence", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main():
    args = _parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if args.mode in ("memory", "all"):
        print_memory_table(args.batch, args.heads, args.head_dim, dtype)
    if args.mode in ("performance", "all"):
        run_performance(
            args.batch, args.heads, args.sequence, args.head_dim, dtype, device, args.iterations
        )


if __name__ == "__main__":
    main()

