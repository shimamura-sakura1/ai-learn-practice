# transformer-performance-lab Task List

Current stop point: Milestone 1 is complete. Do not start Milestone 2 until the
Milestone 1 code and smoke test have been reviewed.

## Project Principles

- Build the smallest runnable version first.
- Advance one milestone at a time.
- Prefer code that is easy to read, benchmark, and profile.
- Use random token inputs only; no DataLoader, dataset, or tokenizer.
- Do not use `torch.compile` until there is a baseline benchmark.
- Do not use `torch.nn.functional.scaled_dot_product_attention` for the first attention baseline.
- Do not add Triton FlashAttention before the profiling and module replacement milestones.
- Every benchmark must include warmup, CUDA synchronize, repeated measurements, and peak memory stats.
- Every stage must have commands, expected outputs, and acceptance criteria.

## Milestone 1: Minimal Decoder-Only Transformer

Status: complete.

Goal:

Build a minimal pure PyTorch decoder-only Transformer that can run forward and
backward on random tokens.

Implemented files:

- `model/config.py`
- `model/rmsnorm.py`
- `model/attention.py`
- `model/mlp.py`
- `model/block.py`
- `model/transformer.py`
- `tests/smoke_test.py`
- `README.md`

Model flow:

```text
input_ids [B, S]
-> token embedding [B, S, H]
-> N x TransformerBlock [B, S, H]
-> final RMSNorm [B, S, H]
-> LM head [B, S, V]
-> cross entropy loss with random targets [B, S]
```

Transformer block:

```text
RMSNorm
-> explicit causal multi-head self-attention
-> residual
-> RMSNorm
-> SwiGLU MLP
-> residual
```

Validation command:

```bash
cd /zero/liuzeyuan/ai-learn-practice/transformer-performance-lab
python -m tests.smoke_test
```

Acceptance criteria:

- Forward output shape is `[B, S, V]`.
- Loss is finite.
- Backward produces finite gradients for key parameters.
- CPU FP32 path works.
- CUDA BF16 or FP16 path works when CUDA is available.

Observed result:

```text
cpu fp32 smoke test passed
cuda torch.bfloat16 smoke test passed
```

## Milestone 2: Benchmark Harness

Status: not started.

Goal:

Create a reliable benchmark harness for the full model forward and
forward-plus-backward pass.

Planned files:

- `benchmark/bench_model.py`
- `benchmark/sweep.py`
- `results/`

Required benchmark behavior:

- Random `input_ids` and random `targets`.
- Configurable `batch_size`, `sequence_length`, `hidden_size`, `num_layers`,
  `num_heads`, `vocab_size`, and `dtype`.
- Warmup iterations.
- Repeated timing iterations.
- `torch.cuda.synchronize()` before and after timing when using CUDA.
- Peak CUDA memory stats with `torch.cuda.reset_peak_memory_stats()` and
  `torch.cuda.max_memory_allocated()`.
- CSV output into `results/`.

Example command to implement later:

```bash
python benchmark/bench_model.py \
  --device cuda \
  --dtype bfloat16 \
  --batch-size 2 \
  --sequence-length 128 \
  --hidden-size 256 \
  --num-layers 2 \
  --num-heads 4 \
  --warmup 5 \
  --iterations 20 \
  --csv results/model_baseline.csv
```

Acceptance criteria:

- Reports median latency in milliseconds.
- Reports peak memory in MiB.
- Reports tokens per second.
- Writes a CSV row with the model config and measured metrics.
- CPU smoke benchmark works.
- CUDA benchmark works when CUDA is available.

Do not add profiler or Triton in this milestone.

## Milestone 3: Module Benchmarking

Status: not started.

Goal:

Measure the performance contribution of individual Transformer modules.

Planned file:

- `benchmark/bench_modules.py`

Modules to benchmark:

- Token embedding.
- RMSNorm.
- Causal multi-head attention.
- SwiGLU MLP.
- LM head.
- Full Transformer block.

Acceptance criteria:

- Each module has its own timing row.
- Attention and MLP can be compared at the same `[B, S, H]`.
- Memory stats are included for CUDA.
- Output clearly shows which modules dominate latency and memory.

Do not add profiler or Triton in this milestone.

## Milestone 4: PyTorch Profiler

Status: not started.

Goal:

Use PyTorch Profiler to inspect the full model and key modules.

Planned files:

- `profiler/profile_model.py`
- `profiler/nvtx_utils.py`

Required profiler behavior:

- Profile a small model config first.
- Use named regions for embedding, each block, attention, MLP, final norm, and LM head.
- Generate TensorBoard trace output under `results/profiler/`.
- Print an operator table sorted by CUDA time when CUDA is available.

Example command to implement later:

```bash
python profiler/profile_model.py \
  --device cuda \
  --dtype bfloat16 \
  --sequence-length 256 \
  --tensorboard-logdir results/profiler/model_s256
```

Acceptance criteria:

- TensorBoard trace is generated.
- Operator table identifies matmul, softmax, RMSNorm, and MLP work.
- The profile can explain where time and memory are going.

Do not add Nsight or Triton in this milestone.

## Milestone 5: Nsight Systems

Status: not started.

Goal:

Use Nsight Systems to inspect CUDA kernel launch order and CPU gaps.

Planned docs:

- Add Nsight commands to `README.md`.

Example command to validate later:

```bash
nsys profile \
  -o results/nsys/model_s256 \
  python profiler/profile_model.py --device cuda --dtype bfloat16 --sequence-length 256
```

Acceptance criteria:

- Nsight report is generated.
- CUDA kernel launch order is visible.
- CPU gaps between launches can be identified.
- Attention, MLP, and norm regions are distinguishable.

Do not add Triton in this milestone.

## Milestone 6: Triton RMSNorm Replacement

Status: not started.

Goal:

Replace RMSNorm with a simple Triton implementation and validate correctness and
end-to-end impact.

Planned file:

- `kernels/fused_residual_rmsnorm.py`

Acceptance criteria:

- Triton RMSNorm output matches PyTorch within dtype-appropriate tolerance.
- Forward and backward strategy is explicitly documented.
- Module benchmark compares PyTorch RMSNorm vs Triton RMSNorm.
- Full model benchmark compares baseline vs replacement.

Do not implement FlashAttention in this milestone.

## Milestone 7: Triton Attention Building Blocks

Status: not started.

Goal:

Prepare attention-related Triton kernels without implementing full
FlashAttention yet.

Possible work:

- Benchmark existing PyTorch explicit attention.
- Add isolated Triton softmax or matmul experiments only if they serve the
  attention learning path.
- Compare outputs and latency against PyTorch module-level baselines.

Acceptance criteria:

- Correctness tests exist for each kernel.
- Kernel benchmarks use the same warmup, synchronize, repeated timing, and
  memory conventions as earlier milestones.
- Results explain whether the replacement improves the end-to-end model.

Do not implement FlashAttention in this milestone.

## Milestone 8: FlashAttention-Style Triton Forward

Status: not started.

Goal:

Implement a FlashAttention-style Triton forward kernel after the baseline,
benchmark, profiler, and module-replacement groundwork is complete.

Planned file:

- `kernels/flash_attention.py`

Acceptance criteria:

- Forward output matches explicit PyTorch attention within tolerance.
- Causal masking is correct.
- Memory usage avoids materializing full `[B, NH, S, S]` score and probability
  tensors.
- Module benchmark compares explicit PyTorch attention vs FlashAttention-style
  Triton attention.
- Full model benchmark measures end-to-end impact.

## Immediate Next Action

Read these files before starting Milestone 2:

- `model/attention.py`
- `model/transformer.py`
- `tests/smoke_test.py`

Then run:

```bash
python -m tests.smoke_test
```

After that, start Milestone 2 only when ready to build the benchmark harness.
