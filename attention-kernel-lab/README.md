# Attention Kernel Lab

This lab builds attention kernels in stages:

1. PyTorch baseline
2. Benchmarking
3. Profiling
4. Backend interface
5. Naive Triton attention
6. FlashAttention forward

## Phase 1: PyTorch Reference Attention

The current implementation lives in `attention/torch_attention.py`.

Input layout:

```text
q: [B, H, Sq, D]
k: [B, H, Sk, D]
v: [B, H, Sk, Dv]
```

For self-attention in this phase, `Sq == Sk == S`.

The math is intentionally explicit:

```text
scores = q @ k.transpose(-2, -1) / sqrt(D)  # [B, H, Sq, Sk]
probs = softmax(scores, dim=-1)             # [B, H, Sq, Sk]
output = probs @ v                          # [B, H, Sq, Dv]
```

With `causal=True`, a lower-triangular mask is applied before softmax, so token
row `i` cannot attend to key columns greater than `i`.

## Why Naive Attention Uses O(S^2) Memory

For `[B, H, S, D]` self-attention, the score tensor has shape `[B, H, S, S]`.
The softmax probability tensor has the same shape. Even if `D` is fixed,
doubling `S` makes each of these tensors four times larger.

This is the memory pressure that later phases will expose with profiling and
then reduce with FlashAttention-style tiling.

## Phase 2: Benchmark

The benchmark entry point is `benchmark/benchmark_attention.py`.

Quick smoke test:

```bash
python benchmark/benchmark_attention.py --quick
```

Full Phase 2 grid:

```bash
python benchmark/benchmark_attention.py
```

Default grid:

```text
B: 1, 4
H: 8, 16, 32
S: 128, 256, 512, 1024, 2048, 4096
D: 64, 128
```

Metrics:

```text
lat_ms:    median latency after warmup
peak_MiB: torch.cuda.max_memory_allocated(), CUDA only
TFLOP/s:  approximate throughput for QK^T and P @ V
tokens/s: B * H * S / latency
```

The script uses warmup iterations and `torch.cuda.synchronize()` around timed
runs. The full grid intentionally includes large `S`; some shapes can run out
of memory on smaller GPUs because the PyTorch baseline materializes `[B,H,S,S]`
scores and probabilities.

Useful options:

```bash
python benchmark/benchmark_attention.py --quick --iterations 10
python benchmark/benchmark_attention.py --sequence-lengths 128,256,512 --csv results.csv
python benchmark/benchmark_attention.py --causal --dtype float16
```

To restrict execution to physical GPU 4:

```bash
CUDA_VISIBLE_DEVICES=4 python benchmark/benchmark_attention.py --device cuda
```

Inside the process, physical GPU 4 will appear as `cuda:0`.

## Phase 3: Profiling

PyTorch Profiler entry point:

```bash
python profiler/torch_profiler.py --seq-len 512 --heads 8 --batch 1
```

On physical GPU 4:

```bash
CUDA_VISIBLE_DEVICES=4 python profiler/torch_profiler.py \
  --device cuda \
  --seq-len 512 \
  --heads 8 \
  --batch 1
```

It reports:

```text
cuda_event_count
cuda_event_time_ms_total
cuda_peak_memory_MiB
operator table sorted by CUDA self time
```

Optional Chrome trace:

```bash
CUDA_VISIBLE_DEVICES=4 python profiler/torch_profiler.py \
  --device cuda \
  --seq-len 1024 \
  --trace torch_attention_trace.json
```

TensorBoard Profiler output:

Install the PyTorch TensorBoard profiler plugin once per environment:

```bash
pip install torch-tb-profiler
```

```bash
CUDA_VISIBLE_DEVICES=4 python profiler/torch_profiler.py \
  --device cuda \
  --seq-len 1024 \
  --heads 8 \
  --batch 1 \
  --iterations 10 \
  --tensorboard-logdir runs/torch_attention_s1024
```

Open it with:

```bash
tensorboard --logdir runs
```

Then select the Profile tab. Use the trace view to inspect CUDA launch order,
operator timing, memory activity, and the separation between QK^T, softmax, and
P @ V.

Memory profiler entry point:

```bash
python profiler/memory_profile.py --sequence-lengths 128,256,512,1024
```

On physical GPU 4:

```bash
CUDA_VISIBLE_DEVICES=4 python profiler/memory_profile.py \
  --device cuda \
  --sequence-lengths 128,256,512,1024,2048 \
  --heads 8 \
  --batch 1
```

The memory table prints both measured peak memory and theoretical tensor sizes:

```text
qkv_MiB:          3 * B * H * S * D
scores_MiB:       B * H * S * S
probs_MiB:        B * H * S * S
scores+probs_MiB: 2 * B * H * S * S
```

This is the core profiling lesson: the explicit PyTorch baseline writes the
quadratic score matrix to global memory, then writes another quadratic softmax
probability matrix. FlashAttention will later avoid materializing these full
`[B,H,S,S]` tensors by computing attention in tiles and maintaining an online
softmax state.

## Run Tests

From this directory:

```bash
pytest -q
```

From the repository root:

```bash
pytest -q attention-kernel-lab/tests
```
