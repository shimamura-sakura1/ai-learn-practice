"""Benchmark PyTorch reference attention.

The Phase 2 benchmark intentionally measures the explicit PyTorch baseline from
``attention/torch_attention.py``. That baseline materializes both the score
matrix and the probability matrix, so this script reports latency, peak CUDA
memory, and approximate math throughput for the same shapes that later Triton
and FlashAttention kernels will use.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from attention.torch_attention import attention


DEFAULT_BATCHES = (1, 4)
DEFAULT_HEADS = (8, 16, 32)
DEFAULT_SEQUENCE_LENGTHS = (128, 256, 512, 1024, 2048, 4096)
DEFAULT_HEAD_DIMS = (64, 128)


@dataclass(frozen=True)
class BenchmarkCase:
    """One self-attention benchmark shape.

    Shape convention:
        q, k, v: [B, H, S, D]
    """

    batch: int
    heads: int
    seq_len: int
    head_dim: int


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured benchmark result for one shape."""

    backend: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: str
    causal: bool
    median_latency_ms: float
    peak_memory_mib: float | None
    tflops: float
    tokens_per_second: float


def _parse_int_list(raw: str) -> tuple[int, ...]:
    """Parse comma-separated positive integers from CLI arguments."""
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _dtype_from_name(name: str) -> torch.dtype:
    """Convert a CLI dtype name to a torch dtype."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _sync(device: torch.device) -> None:
    """Synchronize CUDA work when benchmarking on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mib(device: torch.device) -> float | None:
    """Return peak CUDA memory allocated in MiB, or None on CPU."""
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def _attention_flops(case: BenchmarkCase) -> int:
    """Approximate forward FLOPs for QK^T and P @ V.

    Each matmul is counted as 2 * M * N * K FLOPs.
    QK^T: [S, D] @ [D, S] -> [S, S]
    P @ V: [S, S] @ [S, D] -> [S, D]
    """
    return 4 * case.batch * case.heads * case.seq_len * case.seq_len * case.head_dim


def _make_inputs(case: BenchmarkCase, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random self-attention inputs with shape [B, H, S, D]."""
    shape = (case.batch, case.heads, case.seq_len, case.head_dim)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    return q, k, v


def _median_latency_ms(fn, *, warmup: int, iterations: int, device: torch.device) -> float:
    """Measure median wall-clock latency with CUDA synchronization."""
    for _ in range(warmup):
        fn()
    _sync(device)

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        _sync(device)
        samples_ms.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples_ms)


def run_case(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    causal: bool,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    """Run one benchmark case and return latency, memory, and throughput."""
    q, k, v = _make_inputs(case, dtype, device)

    def fn() -> torch.Tensor:
        return attention(q, k, v, causal=causal)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    median_ms = _median_latency_ms(fn, warmup=warmup, iterations=iterations, device=device)
    peak_mib = _peak_memory_mib(device)
    seconds = median_ms / 1e3
    flops = _attention_flops(case)

    return BenchmarkResult(
        backend="torch",
        batch=case.batch,
        heads=case.heads,
        seq_len=case.seq_len,
        head_dim=case.head_dim,
        dtype=str(dtype).replace("torch.", ""),
        causal=causal,
        median_latency_ms=median_ms,
        peak_memory_mib=peak_mib,
        tflops=flops / seconds / 1e12,
        tokens_per_second=(case.batch * case.heads * case.seq_len) / seconds,
    )


def iter_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    """Build the benchmark shape grid from CLI arguments."""
    if args.quick:
        return [BenchmarkCase(batch=1, heads=8, seq_len=128, head_dim=64)]
    return [
        BenchmarkCase(batch=batch, heads=heads, seq_len=seq_len, head_dim=head_dim)
        for batch in args.batches
        for heads in args.heads
        for seq_len in args.sequence_lengths
        for head_dim in args.head_dims
    ]


def _format_memory(value: float | None) -> str:
    """Format peak memory for terminal output."""
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def print_result(result: BenchmarkResult) -> None:
    """Print one compact benchmark table row."""
    print(
        f"{result.backend:>7} "
        f"{result.batch:>2} {result.heads:>3} {result.seq_len:>5} {result.head_dim:>4} "
        f"{result.dtype:>8} {str(result.causal):>6} "
        f"{result.median_latency_ms:>10.3f} "
        f"{_format_memory(result.peak_memory_mib):>12} "
        f"{result.tflops:>9.3f} "
        f"{result.tokens_per_second:>12.0f}"
    )


def write_csv(path: Path, results: list[BenchmarkResult]) -> None:
    """Write benchmark results to CSV."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BenchmarkResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark PyTorch reference attention.")
    parser.add_argument("--quick", action="store_true", help="run a single smoke-test shape")
    parser.add_argument("--batches", type=_parse_int_list, default=DEFAULT_BATCHES)
    parser.add_argument("--heads", type=_parse_int_list, default=DEFAULT_HEADS)
    parser.add_argument("--sequence-lengths", type=_parse_int_list, default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--head-dims", type=_parse_int_list, default=DEFAULT_HEAD_DIMS)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        help="default: float16 on CUDA, float32 on CPU",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--csv", type=Path, help="optional CSV output path")
    return parser.parse_args()


def main() -> None:
    """Run attention benchmarks."""
    args = parse_args()
    device = torch.device(args.device)
    dtype_name = args.dtype
    if dtype_name is None:
        dtype_name = "float16" if device.type == "cuda" else "float32"
    dtype = _dtype_from_name(dtype_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if dtype in (torch.float16, torch.bfloat16) and device.type == "cpu":
        raise RuntimeError(f"{dtype} benchmark is intended for CUDA; use --dtype float32 on CPU")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be >= 0 and --iterations must be > 0")

    print(
        f"{'backend':>7} {'B':>2} {'H':>3} {'S':>5} {'D':>4} "
        f"{'dtype':>8} {'causal':>6} {'lat_ms':>10} {'peak_MiB':>12} "
        f"{'TFLOP/s':>9} {'tokens/s':>12}"
    )

    results = []
    for case in iter_cases(args):
        result = run_case(
            case,
            dtype=dtype,
            device=device,
            causal=args.causal,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results.append(result)
        print_result(result)

    if args.csv is not None:
        write_csv(args.csv, results)


if __name__ == "__main__":
    main()
