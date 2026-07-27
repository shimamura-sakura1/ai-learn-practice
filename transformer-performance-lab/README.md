# transformer-performance-lab

Learning project for Transformer performance work on PyTorch, CUDA profiling,
Triton kernel replacement, and eventually a FlashAttention-style forward kernel.

The goal is not to hide the model behind a large framework. Each milestone keeps
the code small enough to inspect and profile directly.

## Current Status

Only Milestone 1 is implemented.

Implemented:

- Minimal decoder-only Transformer in pure PyTorch.
- Explicit causal self-attention using `QK^T`, causal mask, softmax, and `PV`.
- RMSNorm, SwiGLU MLP, residual Transformer blocks, final RMSNorm, and LM head.
- Random-token forward, cross entropy loss, and backward smoke test.

Not implemented yet:

- Benchmark harness.
- PyTorch Profiler or Nsight annotations.
- Triton kernels.
- FlashAttention.
- Optimizer, DataLoader, tokenizer, distributed training, or `torch.compile`.

## Environment

Expected runtime:

- Python 3.10 or newer.
- PyTorch with CPU support.
- Optional CUDA PyTorch build for GPU smoke testing.

The CPU smoke test always runs with FP32. If CUDA is available, the smoke test
runs one extra pass with BF16 when supported, otherwise FP16.

## Install

From the parent directory:

```bash
cd transformer-performance-lab
python -m pip install torch
```

If this machine already has PyTorch installed, no extra install step is needed
for Milestone 1.

## Smoke Test

```bash
python -m tests.smoke_test
```

Expected output on CPU-only machines:

```text
cpu fp32 smoke test passed
cuda not available; skipped cuda smoke test
```

Expected output on CUDA machines also includes a CUDA pass message.

## Model Data Flow

Input random token ids:

```text
input_ids: [B, S]
```

Decoder-only Transformer:

```text
token embedding: [B, S] -> [B, S, H]
N x Transformer block: [B, S, H] -> [B, S, H]
final RMSNorm: [B, S, H] -> [B, S, H]
LM head: [B, S, H] -> [B, S, V]
cross entropy: logits [B, S, V], targets [B, S]
```

Each Transformer block:

```text
RMSNorm -> causal multi-head self-attention -> residual
RMSNorm -> SwiGLU MLP -> residual
```

Attention is intentionally explicit:

```text
q, k, v: [B, NH, S, D]
scores = q @ k^T: [B, NH, S, S]
causal mask: [S, S]
probs = softmax(scores): [B, NH, S, S]
context = probs @ v: [B, NH, S, D]
output: [B, S, H]
```

## Next Milestone

Milestone 2 will add a benchmark harness with warmup, CUDA synchronize,
repeated measurements, and peak memory statistics. It is intentionally not
implemented in Milestone 1.
