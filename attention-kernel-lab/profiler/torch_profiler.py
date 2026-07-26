"""Profile the PyTorch reference attention baseline.

This script answers three Phase 3 questions:
    1. How many CUDA events/kernels does the explicit baseline launch?
    2. Which operations take most CUDA time?
    3. How much CUDA memory is allocated at peak?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function, tensorboard_trace_handler


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from attention.torch_attention import attention


def _dtype_from_name(name: str) -> torch.dtype:
    """Convert a CLI dtype name to a torch dtype."""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _sync(device: torch.device) -> None:
    """Synchronize CUDA work when profiling on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_inputs(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q/K/V tensors with shape [B, H, S, D]."""
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v


def _cuda_event_count(prof) -> int:
    """Count low-level CUDA events observed by PyTorch Profiler."""
    return sum(1 for event in prof.events() if "cuda" in str(getattr(event, "device_type", "")).lower())


def _cuda_time_total_ms(prof) -> float:
    """Return total CUDA event time in milliseconds."""
    total_us = 0.0
    for event in prof.events():
        if "cuda" in str(getattr(event, "device_type", "")).lower():
            total_us += float(getattr(event, "self_cuda_time_total", 0.0) or 0.0)
    return total_us / 1e3


def run_profile(args: argparse.Namespace) -> None:
    """Run PyTorch profiler for one attention shape."""
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dtype_name = args.dtype
    if dtype_name is None:
        dtype_name = "float16" if device.type == "cuda" else "float32"
    dtype = _dtype_from_name(dtype_name)
    if dtype in (torch.float16, torch.bfloat16) and device.type == "cpu":
        raise RuntimeError(f"{dtype} profiling is intended for CUDA; use --dtype float32 on CPU")

    q, k, v = _make_inputs(
        args.batch,
        args.heads,
        args.seq_len,
        args.head_dim,
        dtype=dtype,
        device=device,
    )

    for _ in range(args.warmup):
        attention(q, k, v, causal=args.causal)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    on_trace_ready = None
    if args.tensorboard_logdir is not None:
        args.tensorboard_logdir.mkdir(parents=True, exist_ok=True)
        on_trace_ready = tensorboard_trace_handler(str(args.tensorboard_logdir))

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=args.with_stack,
        on_trace_ready=on_trace_ready,
    ) as prof:
        for _ in range(args.iterations):
            with record_function("explicit_attention_forward"):
                attention(q, k, v, causal=args.causal)
            prof.step()

    _sync(device)

    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(
        f"shape: B={args.batch}, H={args.heads}, S={args.seq_len}, D={args.head_dim}, "
        f"dtype={dtype_name}, causal={args.causal}, device={device}"
    )
    print(f"iterations: {args.iterations}, warmup: {args.warmup}")
    if device.type == "cuda":
        print(f"cuda_event_count: {_cuda_event_count(prof)}")
        print(f"cuda_event_time_ms_total: {_cuda_time_total_ms(prof):.3f}")
        print(f"cuda_peak_memory_MiB: {torch.cuda.max_memory_allocated(device) / (1024**2):.1f}")
    else:
        print("cuda_event_count: n/a")
        print("cuda_event_time_ms_total: n/a")
        print("cuda_peak_memory_MiB: n/a")

    print()
    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit))

    if args.trace is not None:
        prof.export_chrome_trace(str(args.trace))
        print(f"\nchrome_trace: {args.trace}")

    if args.tensorboard_logdir is not None:
        print(f"\ntensorboard_logdir: {args.tensorboard_logdir}")
        print(f"view_with: tensorboard --logdir {args.tensorboard_logdir}")


def parse_args() -> argparse.Namespace:
    """Parse profiler CLI arguments."""
    parser = argparse.ArgumentParser(description="Profile explicit PyTorch attention.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        help="default: float16 on CUDA, float32 on CPU",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=20)
    parser.add_argument("--with-stack", action="store_true", help="record Python stacks; slower")
    parser.add_argument("--trace", type=Path, help="optional Chrome trace output path")
    parser.add_argument("--tensorboard-logdir", type=Path, help="optional TensorBoard profiler logdir")
    return parser.parse_args()


def main() -> None:
    """Profile the PyTorch attention baseline."""
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be >= 0 and --iterations must be > 0")
    run_profile(args)


if __name__ == "__main__":
    main()
