# Nsight Notes

Use Nsight after the PyTorch Profiler scripts have shown which operators matter.
For this phase, the goal is observation, not optimization.

## TensorBoard Profiler

Generate a visual PyTorch Profiler trace:

```bash
CUDA_VISIBLE_DEVICES=4 python profiler/torch_profiler.py \
  --device cuda \
  --seq-len 1024 \
  --heads 8 \
  --batch 1 \
  --iterations 10 \
  --tensorboard-logdir runs/torch_attention_s1024
```

Open TensorBoard:

```bash
tensorboard --logdir runs
```

Focus on:

- Trace view: CUDA kernel launch order and CPU gaps.
- Operator view: CUDA time for matmul, softmax, masking, and copies.
- Memory view: allocations from `[B,H,S,S]` scores and probabilities.

## Nsight Systems

Timeline view:

```bash
CUDA_VISIBLE_DEVICES=4 nsys profile \
  -o reports/torch_attention_nsys \
  python profiler/torch_profiler.py --seq-len 1024 --heads 8 --batch 1 --iterations 5
```

Questions to answer:

- How many CUDA kernels appear for one explicit attention forward?
- Are matmul kernels separated from softmax and mask kernels?
- Is there CPU overhead between CUDA launches?

## Nsight Compute

Kernel-level view:

```bash
CUDA_VISIBLE_DEVICES=4 ncu \
  --set full \
  -o reports/torch_attention_ncu \
  python profiler/torch_profiler.py --seq-len 1024 --heads 8 --batch 1 --iterations 1
```

Questions to answer:

- Are QK^T and P @ V compute-heavy compared with softmax?
- Is softmax more memory-bandwidth sensitive?
- How much global memory traffic comes from reading/writing `[B,H,S,S]`?

For later Triton phases, compare each custom kernel against these PyTorch
baseline observations.
