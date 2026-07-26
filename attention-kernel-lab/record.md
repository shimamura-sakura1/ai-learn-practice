# Attention Kernel Lab - Experiment Record

This file records profiling observations and conclusions for the lab.

## Phase 3 - PyTorch Explicit Attention

### Experiment: CUDA kernel timeline

Date: 2026-07-27

Environment:

- GPU: NVIDIA GeForce RTX 4070 Ti
- PyTorch: 2.13.0+cu130
- Triton: 3.7.1
- Device: CUDA

Input configuration:

```text
B = 1
H = 8
S = 1024
D = 64
dtype = float16
causal = False
```

The measured iteration was selected from the steady-state iterations rather
than the first profiled iteration, which contained `cudaMalloc` activity.

#### Kernel timing

| Stage | Mathematical operation | CUDA kernel duration | Share of kernel time |
|---|---|---:|---:|
| QK matmul | `Q @ K.transpose(-2, -1)` | 25 us 697 ns | 28.2% |
| Scale | `scores * (1 / sqrt(D))` | 14 us 241 ns | 15.6% |
| Softmax | `softmax(scores, dim=-1)` | 23 us 553 ns | 25.9% |
| PV matmul | `probs @ V` | 27 us 553 ns | 30.3% |
| **Total** | One explicit-attention forward | **91 us 44 ns** | **100%** |

Combined matrix-multiplication time:

```text
QK matmul + PV matmul = 53 us 250 ns
Share of kernel time  = 58.5%
```

Non-matmul time:

```text
Scale + softmax       = 37 us 794 ns
Share of kernel time  = 41.5%
```

#### Trace interpretation

One explicit PyTorch attention forward launches four separate CUDA kernels:

```text
QK matmul -> scale -> softmax -> PV matmul
```

The `aten::*` events on the CPU track represent PyTorch dispatch, shape and
layout handling, and CUDA kernel launches. The long CUDA implementation names
on the GPU stream represent the actual GPU computation.

The two matrix multiplications account for most of the kernel time, but scale
and softmax still account for 41.5%. More importantly, the separate operations
read and write the full `[B, H, S, S]` score/probability data through global
memory. This intermediate-memory traffic is a key motivation for later
FlashAttention-style fusion.

### Previously measured scaling

| Sequence length | Median latency | Peak allocated memory |
|---:|---:|---:|
| 128 | 0.074 ms | 9.1 MiB |
| 256 | 0.074 ms | 11.1 MiB |
| 512 | 0.079 ms | 18.1 MiB |
| 1024 | 0.129 ms | 44.1 MiB |
| 2048 | 1.056 ms | 144.1 MiB |
| 4096 | 4.281 ms | 536.1 MiB |

For FP16 with `B=1` and `H=8`:

```text
scores size = B * H * S * S * 2 bytes
probs size  = B * H * S * S * 2 bytes
```

At `S=1024`, scores and probabilities are 16 MiB each. At `S=4096`,
they are 256 MiB each. This demonstrates the quadratic memory cost of explicit
attention.

## Next Experiment

Capture PyTorch CUDA memory snapshots for `S=1024` and `S=4096`, then inspect
them with the PyTorch memory visualizer.

Questions to answer:

1. Can the score and probability allocations be identified by size and stack?
2. When is each allocation created and released?
3. Are scores and probabilities simultaneously live at peak memory?
4. How much of the measured peak is tensor data versus allocator overhead?

