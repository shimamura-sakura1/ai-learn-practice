"""Measure peak CUDA memory for explicit PyTorch attention.

The important Phase 3 observation is that the baseline materializes:
    scores: [B, H, S, S]
    probs:  [B, H, S, S]

This script prints both measured peak memory and the theoretical size of those
two quadratic tensors.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from attention.torch_attention import attention


@dataclass(frozen=True)
class MemoryResult:
    """Memory result for one shape."""

    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: str
    causal: bool
    peak_memory_mib: float | None
    qkv_mib: float
    scores_mib: float
    probs_mib: float
    scores_plus_probs_mib: float


def _parse_int_list(raw: str) -> tuple[int, ...]:
    """Parse comma-separated positive integers from CLI arguments."""
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _dtype_from_name(name: str) -> torch.dtype:
    """Convert a CLI dtype name to a torch dtype."""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _sync(device: torch.device) -> None:
    """Synchronize CUDA work when measuring GPU memory."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mib(elements: int, dtype: torch.dtype) -> float:
    """Convert tensor elements to MiB for a dtype."""
    element_size = torch.empty((), dtype=dtype).element_size()
    return elements * element_size / (1024**2)


def _theoretical_sizes(
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[float, float, float, float]:
    """Return theoretical QKV, scores, probs, and scores+probs sizes in MiB."""
    qkv_elements = 3 * batch * heads * seq_len * head_dim
    square_elements = batch * heads * seq_len * seq_len
    scores_mib = _mib(square_elements, dtype)
    probs_mib = _mib(square_elements, dtype)
    return _mib(qkv_elements, dtype), scores_mib, probs_mib, scores_mib + probs_mib


def run_case(
    *,
    batch: int,
    heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool,
    iterations: int,
) -> MemoryResult:
    """Measure peak memory for one self-attention shape."""
    shape = (batch, heads, seq_len, head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(iterations):
        output = attention(q, k, v, causal=causal)
        _sync(device)
        del output

    peak_mib = None
    if device.type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    qkv_mib, scores_mib, probs_mib, scores_plus_probs_mib = _theoretical_sizes(
        batch, heads, seq_len, head_dim, dtype
    )
    return MemoryResult(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
        dtype=str(dtype).replace("torch.", ""),
        causal=causal,
        peak_memory_mib=peak_mib,
        qkv_mib=qkv_mib,
        scores_mib=scores_mib,
        probs_mib=probs_mib,
        scores_plus_probs_mib=scores_plus_probs_mib,
    )


def print_result(result: MemoryResult) -> None:
    """Print one compact memory result row."""
    peak = "n/a" if result.peak_memory_mib is None else f"{result.peak_memory_mib:.1f}"
    print(
        f"{result.batch:>2} {result.heads:>3} {result.seq_len:>5} {result.head_dim:>4} "
        f"{result.dtype:>8} {str(result.causal):>6} "
        f"{peak:>12} {result.qkv_mib:>10.1f} {result.scores_mib:>11.1f} "
        f"{result.probs_mib:>10.1f} {result.scores_plus_probs_mib:>17.1f}"
    )


def write_csv(path: Path, results: list[MemoryResult]) -> None:
    """Write memory results to CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MemoryResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def parse_args() -> argparse.Namespace:
    """Parse memory profiler CLI arguments."""
    parser = argparse.ArgumentParser(description="Measure explicit attention memory.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence-lengths", type=_parse_int_list, default=(128, 256, 512, 1024, 2048))
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        help="default: float16 on CUDA, float32 on CPU",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    return parser.parse_args()


def main() -> None:
    """Measure peak allocated memory."""
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dtype_name = args.dtype
    if dtype_name is None:
        dtype_name = "float16" if device.type == "cuda" else "float32"
    dtype = _dtype_from_name(dtype_name)
    if dtype in (torch.float16, torch.bfloat16) and device.type == "cpu":
        raise RuntimeError(f"{dtype} memory profiling is intended for CUDA; use --dtype float32 on CPU")

    print(
        f"{'B':>2} {'H':>3} {'S':>5} {'D':>4} {'dtype':>8} {'causal':>6} "
        f"{'peak_MiB':>12} {'qkv_MiB':>10} {'scores_MiB':>11} "
        f"{'probs_MiB':>10} {'scores+probs_MiB':>17}"
    )

    results = []
    for seq_len in args.sequence_lengths:
        result = run_case(
            batch=args.batch,
            heads=args.heads,
            seq_len=seq_len,
            head_dim=args.head_dim,
            dtype=dtype,
            device=device,
            causal=args.causal,
            iterations=args.iterations,
        )
        results.append(result)
        print_result(result)

    if args.csv is not None:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
