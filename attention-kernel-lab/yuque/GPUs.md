**<u>Lecture Goal: Make GPU and its processing less magic</u>**

+ Understand the design philosophy of GPUs
+ Understand when and why GPUs get slow
+ Understand how to make fast algorithms on GPUs



**<u>本节内容：</u>**

+ GPUs in depth
    - 设计目的、计算架构、存储架构、计算模式、硬件 Scaling
+ GPU 性能
    - Operation intensity、性能优化方法（control divergence、low precision、operator fusion、recomputation、memory coalescing、tiling）
+ Optimizing GPU performance with FlatAttention



**<u>Further Reading：</u>**

+ What Shapes Do Matric Multiplications Like? [https://www.thonking.ai/p/what-shapes-do-matrix-multiplications#footnote-1-142904770](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications#footnote-1-142904770)
+ Making Deep Learning Go Brrr From First Principles [https://horace.io/brrr_intro.html](https://horace.io/brrr_intro.html)



# BackGround
**<u>（1）Scaling-driven LM progress</u>**

计算 Scaling 可使语言模型性能实现可预测的提升 ==> 更高效的硬件 + <font style="color:#DF2A3F;">更好的使用</font> 也是语言模型发展的一大助力

![Language modeling performance improves smoothly as we increase the model size, dataset size, and amount of compute. [Kaplan et al. Neural Scaling Laws. OpenAI, 2020.]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760920489172-530ea5f1-acb3-4eaa-b88c-33b9900634bf.png)



**<u>（2）硬件发展瓶颈</u>**

摩尔定律：集成电路上可以容纳的晶体管数目在大约每经过18个月到24个月便会增加一倍。换言之，处理器的性能大约每两年翻一倍，同时价格下降为之前的一半。

+ 根本原因：制程微缩（process scaling）

| 时代 | 制程节点 | 晶体管长度 |
| --- | --- | --- |
| 1980s | 1 µm | 1000 nm |
| 2000s | 130 nm → 65 nm | 纳米时代开始 |
| 2010s | 32 nm → 7 nm | FinFET 三维晶体管 |
| 2020s | 5 nm → 3 nm → 2 nm | 极紫外光刻（EUV）极限 |


+ 瓶颈：
    - 晶体管尺寸已接近原子级别（2-3nm ~ 10 原子），进一步缩小存在极大技术挑战
    - 经济模式失效，如 3nm 制造成本 ～ 7nm 的 3 倍
    - 频率墙（～3-5Ghz），再提频功耗按平方增长，同时散热成为问题，导致晶体管变多 ≠ 有效性能线性提升。

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760922078260-05288380-6e7b-422b-aebe-c526bb61652a.png)

（3）模式切换：Parallel Scaling

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760922225229-93ec85fc-f0e6-4367-86e4-90edbccdd388.png)

# GPUs in Depth
## GPUs vs. CPUs
> <font style="color:rgb(138, 143, 141);">Source: </font>[https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/)
>

CPUs optimize for a few, fast threads while GPUs optimize for many many threads

+ GPUs dedicate most of their transistors for data processing while CPUs also need to reserve die area for big caches, control units, and so on. 
+ CPUs optimize for latency, i.e., each thread finished quickly
+ GPUs optimize for throughput, i.e., total processed data

<!-- 这是一张图片，ocr 内容为： -->
![GPUs devote more transistors to compute data processing.](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760922336344-9e2dae8a-839d-4b2c-8d8f-1256ff9418a1.png)

<!-- 这是一张图片，ocr 内容为： -->
![Low latency or high throughput.](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760922360608-f9ce0486-6c8b-4b20-8bb8-05a95c487316.png)

## Execution Units in GPUs
> Source: [https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
>

GPU

 └── GPCs (Graphics Processing Clusters)

      └── SMs (Streaming Multiprocessors)

           └── SPs (Streaming Processors)



GPUs  have many SM and each SM independently execute blocks (jobs).

<!-- 这是一张图片，ocr 内容为： -->
![GA100 full GPU with 128 SMs (streaming multiprocessors).](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760922771891-32f0f250-0fda-46ea-a0a4-d75f36bd8c7c.png)

Each SM further contains many SPs (streaming processor) that can execute ‘threads’ in parallel

+ 64 INT32 core
+ 64 FP32 core: 19.5 peak TFLOPS with FP32
+ 32 FP64 cores: 9.7 peak TFLOPS with FP64
+ 4 Tensor cores: 156 TFLOPS with TF32

<!-- 这是一张图片，ocr 内容为： -->
![SM: 1/128 of the above GPU.](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760922795197-010af4aa-dcba-46c4-bc1f-165ecbf455b3.png)

<!-- 这是一张图片，ocr 内容为：MATMUL VS. NON-MATMUL FLOPS ACROSS GPUS 10 NON-MATMUL MATMUL TFLOP/S 102 101 K80 V100 A100 H100 M80 P100 GPU -->
![Tensor cores (introduced since V-series) are specialized hardware designed for matmuls. 硬件直接执行矩阵乘加！Matmuls are >10x faster than floating points ops.](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760932159347-e94649f3-de88-4b1b-a252-b4d3881fc58c.png)

## Memory in GPUs
分层存储结构：

+ 存储离 SM 越近，访问速度越快
+ L1 and shared memory is inside the SM
+ global memory are the memory ships next to the GPUs

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760925193549-72793a8f-a136-4a77-988b-79f91ac2f332.png)

| 存储类型 | 位置 | 容量（A100） | 访问速度（时钟周期） | 作用 |
| --- | --- | --- | --- | --- |
| **寄存器** | **on-chip** | 256 KB/SM | 1-2 | 每个线程私有，最快 |
| **共享内存/L1 Cache** | **on-chip** | 192 KB/SM 可配置 | 20-30 | 线程块共享（SM 独占），适合数据重用 |
| **L2 Cache** | **on-chip** | 48 MB | ～200 | SM共享，加速全局内存访问 |
| **全局显存 (DRAM)** | **off-chip** | 40/80GB | ～290 | 模型参数、特征图存储 |
| **NVLink** | **** |  | ~200ns，近DRAM | NVLink 4.0（H100）：**每 GPU 600GB/s带宽** |
| **PCIe** | **** |  | >1us，比NVL慢5-10倍 | PCIe 5.0 x16：**64GB/s带宽** |


+ 时钟周期=1/核心时钟频率，目前主流GPU时钟频率在1.x-2.x GHz，因此可简单认为时钟周期为<1ns。
+ 访问延迟增长关系：寄存器 --(x10)--> L1  --(x10)--> L2  --(x1.5)--> DRAM。
+ SRAM (on-chip) 的价格约为 DRAM 的 100 倍，有 近 1 个数量级的访问速度提升（L1 vs. Global）
+ 延迟决定响应时间，带宽决定吞吐。

## Execution Model of a GPU
There are 3 important layers in the execution models:

1. **Blocks**: blocks are groups of threads. Each block runs on a SM with its own shared memory
2. **Warps**: Threads are always execute in a 'warp' of 32 consecutively numbered threads
3. **Threads**: Threads do the work in parallel - all threads execute the same instructions but with different inputs
    1. SIMT (same instruction multiple threads)：线程并行，每线程独立寄存器，统一指令控制

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760929994556-609ea96e-70ff-44e9-bdea-a443ab700825.png)

SM

 ├── Warp Scheduler ×4

 │    └── 发射 warp 指令

 ├── Execution Units (ALU, Tensor, LD/ST)

 └── On-chip Memory (Registers, Shared, L1)



**<u>Warp 执行的调度与发射机制：</u>**

1. 线程通过 Warp 组织：例如某个 Block 有 256 个进程，会被自动分成 256/32=8 个 Warps
2. Warp 在 SM 上排队：每个 SM 能同时驻留数十个 warp（例如 A100 最多 64 个 warp/SM），这些 warp 在 SM 内的调度器队列中等待执行
3. Warp Scheduler 发射指令：
    1. 每个 scheduler 每个时钟周期可以选取 1 个 warp
    2. warp 发后，把指令分发给对应执行单元，例如 FP32/INT/LD/ST/Tensor

以 A100 为例，每个 SM 每个时钟周期可以同时发出 4 条 warp 指令，这些 warp 可以来自不同的线程块（Block）、不同的程序阶段，调度器会动态选取ready 状态的 warp 发射执行。

由于 A100 的 SM 有多个功能单元（FP32、FP64、Tensor、Load/Store），warp scheduler 可以在同一个周期调度多个任务执行，从而实现多 pipeline 的真正并行执行。

+ 并行条件：指令类型不同、资源无冲突

## Memory Model of a GPU
![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760931786757-58db3911-ff53-468f-bece-1537a17a550d.png)

Each thread can access its own register and shared memory within the block.

Information that goes across blocks need to be read/written to global memory (slow).

## Compute Scaling vs. Memory Scaling
![The scaling of the bandwidth of different generations of interconnections & Memory, as well as the Peak FLOPS. As can be seen, the bandwidth is increasing very slowly. [https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760932536075-9251f8b8-e5fb-495c-9f44-ba7c5fb1c3fd.png)

Data movement from DRAM to GPU SRAMs are often the bottleneck. It’s hard to keep our compute units fed with data!

## Summary
**<u>GPUs are massively parallel</u>**

+ Multiple SMs/SPs
+ SIMT programming model
+ Threads are lightweight

**<u>Compute (especially matmums) have scaled faster than memory</u>**

**<u>Respect the memory hierarchy to make thinks go fast</u>**

# GPU Performance
GPU performance can be complex, even for something as simple as a square matmul

<!-- 这是一张图片，ocr 内容为： -->
![Source: What Shapes Do Matric Multiplications Like?](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760919119632-b30e9efe-2f6a-4765-b7c3-33d1f6272b10.png)

## The Roofline Model
```python
def f(x: Tensor[N]):
    for _ in range(repeat):
        x = x * 2
    return x
```

+ Memory access = 2 * N
+ FLOPs = N * repeat * iters_per_second
+ Memory bandwidth = bytes_per_elem * 2 * N * iters_per_second

![左图纵坐标：f函数运行时间；中图纵坐标：每秒浮点操作数 FLOP/s；右图纵坐标：每秒使用带宽量。 [https://horace.io/brrr_intro.html]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760935174430-3b0bafd3-166e-4ba0-88dd-9c29cebb1cac.png)

+  Runtime doesn't increase noticeably at all until we're performing 64 multiplication
    - ==> We're mostly memory-bandwidth bound up until that point, 
+ FLOP/s increase linearly from 0.2T to 9.75T (testing with FP64?)
+ Memory bandwidth achieved starts out near the peak, and as we increase our compute intensity it starts to drop
+ Memory-bounded with repeat < 32 and compute-bounded with repeat > 64

<font style="color:rgb(51, 51, 51);"></font>

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760935676345-02464cc4-f981-41e4-8415-bda475612b1c.png)

Operational Intensity (or compute/arithmetic intensity) = #FLOPs / #memory_access

Efficiency comparison of GPU vs python on CPU:

+ An A100 has a peak 312 TeraFLOPS with bf16/tf16
+ Python can perform 32 million additions in one second
+ Python executing a single FLOP = A100 executing <font style="color:#DF2A3F;">9.75 million</font> FLOPS



**<font style="color:#DF2A3F;">Making ML workload fasts largely reudcing to avoid being memory bound: </font>**

+ control divergence (the only one that is not memory-related) 
+ low precision computation
+ operator fusion
+ recomputation
+ memory coalescing and DRAM
+ tiling

## (1) Control Divergence
GPUs operate in a SIMT model: every thread in a warp is executing the same instruction

+ 若出现分支发散，warp 会分成多个执行掩码组顺序执行（mask 掩掉未活跃线程）

<!-- 这是一张图片，ocr 内容为：INSTRUCTION DECODER AND WARP SCHEDULER GPU SIMT CUDA CON CUDA CORE CUDA CORE CUOA CORE CUOA CONE CUDA CONE 1 INSTRUCTION-MULTIPLE THREADS REGISTERS REGISTERS REGISTERS REGISTERS REGISTERS REGISTERS THREAD -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760936558708-1c51c332-3ee6-431f-8b62-c7bf2d2a6a26.png)

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760936498126-64213136-b349-4dcd-a9a3-0cfa81c5a604.png)

SIMT 模型支持条件语句，但在执行上会引入额外开销。

## (2) Low Precision Computation
Few bits to move, and usually faster to calculate.

<!-- 这是一张图片，ocr 内容为： -->
![https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf](https://cdn.nlark.com/yuque/0/2025/png/52437949/1757907235075-3be7f4a5-658b-4e91-86c4-75cc8ef5f93e.png)

**<u>Example: elementwise ReLU on a vector of size n</u>**

+ Float32 case
    - Memory access: 1 read & 1 write per element, float32 = 8n bytes
    - Operations: 1 comparison op per elemment, n FLOP
    - Intensity: 0.125 FLOP / byte
+ Float16 case
    - Memory access: 1 read & 1 write per element, float16 = 4n bytes
    - Operations: 1 comparison op per elemment, n FLOP
    - Intensity: 0.25 FLOP / byte



**<u>Lots of operations in modern GPUs are accelerated via low/mixed precision operations</u>**

+ Operations that can use 16-bit storage, e.g., fp16/bf16
    - matrix multiplications
    - most pointwise operations: rule, tanh, add, mul
+ Operations that need more precision, thus fp32/fp16
    - adding small values to large sums can lead to rounding errors
    - reduction operations: sum, softmax, normalization
+ Operations that need more range, thus fp32/bf16
    - pointwise operations where |f(x)| >> x, like exp, pow, log
    - loss functions

<!-- 这是一张图片，ocr 内容为：SIGN MANTISSA EXPONENT 8 BITS FP32 23 BITS TF32 RANGE TENSOR FLOAT 32 (TF32) 8 BITS 10 BITS TF32 PRECISION FP16 5 BITS 10 BITS BFLOAT16 (BF16) 7 BITS 8 BITS -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937341885-34174be9-7b1d-482b-9649-f298e2f15c83.png)

<!-- 这是一张图片，ocr 内容为：CONVERT TO FULL PRECISION TF32 MORE PRODUCTS PRODUCT 111 FP32 FP32 OUTPUT FP32 SUM WITH FP32 ACCUMULATOR -->
![https://nvlabs.github.io/eccv2020-mixed-precision-tutorial/files/dusan_stosic-training-neural-networks-with-tensor-cores.pdf](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937286281-0dea05f8-edb4-45b1-896b-f092a17a125e.png)

## (3) Operator Fusion
> [https://horace.io/brrr_intro.html](https://horace.io/brrr_intro.html)
>

GPU: 工厂

SRAM：仓库

Operator fusion：同一批零件在工厂完成多轮的深加工处理

+ Shipping back and forth is somewhat silly
+ Compute scale up, memory does not

<!-- 这是一张图片，ocr 内容为：口口 口口口 口口口口 口口口口 COMPUTE MEMORY . -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937413871-c7499cf5-9297-4ddc-b332-6fdc463e4bd8.png)  <!-- 这是一张图片，ocr 内容为：口口口口 口口口口 AAAAA 口口口口 COMPUTE MEMORY -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937423101-8d4fd2bb-d573-4a35-b1d9-a06dc1ad23da.png)

<!-- 这是一张图片，ocr 内容为：MEMORIY COMPUTE DED AAAAT LAAAAF -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937621698-b3f73224-2dcc-4427-96a5-9626e0dd46cc.png)   <!-- 这是一张图片，ocr 内容为：MEMORY . COMPUTE 2个7个口 -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937630557-4f1c2996-2461-456e-9a53-96e25f7dd99c.png)



**<u>Example: computing </u>**$ \sin^2x+ \cos^2 x $

![Naively launching 5 CUDA kernels for computation. [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937802224-48d2fbcb-f21e-4bdc-93d7-909308a93bd8.png)

![All 5 pointwise operations can be fused into a single CUDA kernel call (torch.compile can automatically complete the fusion.) [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760937894335-26ca13ff-449b-49cb-8050-aaf95c48e2c1.png)



**<u>Kernel launch overhead: a lot of nontrivial work happens before any GPU thread executes</u>**

+ Kernel launch overhead = the fixed host/device cost of preparing and dispatching a GPU kernel (typically 5–20 µs)
+ Dominating when you have many small ones.

## (4) Recomputation
![In backpropagation, we store the activations (yellow) and compute Jacobians (green). [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760938617411-b1c334f1-8c6c-43bd-9bdc-cd5b11c72536.png)

![The 8 mem reads/writes result in very low arithmetic intensity, teeefble for performance! [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760938687100-4a08b200-9113-4b0d-8895-a651ca7761ee.png)

![Throwing away computation can actually be optimal, saving 3/8 memory access. [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760939121844-4dca14d1-56cc-4178-bdc8-ee3db986c4f6.png)

**<u>Example: memory access and computation comparison for sigmoid on a 8192-vector in fp32</u>**

+ $ \sigma(z) = 1/(1+e^z) $

<details class="lake-collapse"><summary id="uae7c3102"><span class="ne-text">memory access and computation comparison</span></summary><p id="ud43de2ea" class="ne-p"><span class="ne-text">A100 memory bandwidth ~ 2TB/s</span></p><p id="uca0d251c" class="ne-p"><span class="ne-text">A100 FP32 peak FLOPs: 19.5 TFLOP/s</span></p><p id="ue21504eb" class="ne-p"><span class="ne-text">3 memory access costs 3*8192*4 / 2TB/s = 49.152 us</span></p><p id="u42ebc0d8" class="ne-p"><span class="ne-text">2 sigmoid computation costs 2*3*8192 / 19.5TFLOP/s = 2.52 us</span></p></details>


**<u>Transformer activation recomputation from the perspective of memory footprint reduction: </u>**

<!-- 这是一张图片，ocr 内容为： -->
![https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=activation_recomputation](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761133223482-df178580-3909-4bea-8cfb-b3bc554c201e.png)

<!-- 这是一张图片，ocr 内容为： -->
![Total memory required for the activations in mixed precision.](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761133351998-10111fc6-b70a-4443-bff2-acbe895665d8.png)

> Activation memory usage is not static for a given model; rather, it scales linearly with the batch size and quadratically with the sequence length.
>

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761133512740-8487eda4-ead0-4cc2-9e67-56608fba95ad.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761133528617-9f6c4d27-5109-46c7-b0e4-988431bcb86d.png)

> For a GPT-3 (175B) model, this means a 70% activation memory reduction at a 2.7% compute cost.
>

## (5) Memory Coalescing and DRAM
Memory Coalescing：内存访问合并



**<u>DRAM (global memory) is read in burst mode: each read gives you many bytes</u>**

+ Each address space is partitioned into burst sections
+ Whenever a location is accessed, all locations in the same section are also delivered to the processor
+ In practice, each section occupies 128-bytes or more



**<u>Memory accessed are coalesced if all the threads (32 in a warp) fall within the same burst</u>**

+ When all threads of a warp execute a load instruction, if all accessed locations fall into the same burst section (actually with consecutive addresses), only one DRAM request will be make and the access is fully coalesced.

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760940885161-76c16c45-b238-4340-a9a1-18ac1ee57dd5.png)

![Each thread correspond to a row (left) or column (right), which case is memory clalesced? [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760941402081-76d0cce2-aba6-43ce-a1d0-96ae12a9d88a.png)





## (6) Tiling (The big one)
**<font style="color:#DF2A3F;">Tiling is the idea of grouping and ordering threads to minimize global memory access.</font>**

+ Warp 级，统一将一部分数据放入 shared memory 中



**<u>Example: matrix multiplication</u>**

![Memory access is not coalesced, and repeated. [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760941608087-616aadae-3662-415a-93ba-c3ec6a15aecb.png)

(1) 考虑流式逐队加载的情况：每个线程每次从 M/N 矩阵各加载 1 个 元素，然后完成乘加计算

+ M 矩阵的加载没有内存对齐（not coalesced），且会被重复访问
+ N 矩阵的加载部分内存对齐，同时也存在重复访问



(2) Tiling: store and reuse information in shared memory

```plain
Load M0,0 and N0,0 tiles into SHM # M0,0 stands for a subblock of 1/4M
compute partial sums for P with tiles
Load M0,1 and N1,0 tiles into SHM
...
```

+ repeated reads now access shared memory (instead of global memory)
+ memory access can be coalesced

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760961087353-675ca73a-71bd-41e3-b873-e6b5a7f09b2e.png)



**<u>Tiling and non-tiling comparison</u>**

+ Non-tiled matmum: each input is read N times from global memory (~300 ns)
+ Tiled mat-mum: each input is read N/T times from glboal memory, and T times with each tile (20-30 ns)
    - a factor of T reduction in global memory access



**<u>Complexity with tiling</u>**

+ tile size not divide matrix size and lead to low utilization

<!-- 这是一张图片，ocr 内容为：TILE QUANTIZATION RESULTS IN SIX THREAD BLOCKS BEING LAUNCHED, TWO OF WHICH WASTE MOST OF THEIR WORK 256 257 128 128 256 256 128 128 (B) (A) -->
![https://www.thonking.ai/p/what-shapes-do-matrix-multiplications](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760961529374-9d387c6e-e45e-4360-beaf-876b68da9c3f.png)

+ memory layout alignment

<!-- 这是一张图片，ocr 内容为：UNALIGNED LAYOUT ALIGNED LAYOUT ONE NICE TILE TWO BAD TILES -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760961896602-44204a19-0dc1-44b2-a272-c334ba97cb21.png)

<font style="color:#DF2A3F;">Tiling needs to consider coalesced memory access, shared memory size, and divisibility of matrix dim</font>



<!-- 这是一张图片，ocr 内容为：ANDREJ KARPATHY @KARPATHY THE MOST DRAMATIC OPTIMIZATION TO NANOGPT SO FAR (~25% SPEEDUP) IS TO SIMPLY INCREASE VOCAB SIZE FROM 50257 TO 50304 (NEAREST MULTIPLE OF 64). THIS CALCULATES ADDED USELESS DIMENSIONS BUT GOES DOWN A DIFFERENT KERNEL PATH WITH MUCH HIGHER OCCUPANCY. CAREFUL WITH YOUR POWERS OF 2. 10:36 AM. FEB 3,2023 .1.2MVIEWS -->
![Effects of good tiling.](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760961986907-a94e2d51-1fb4-4fb1-b1f2-31ae0b547791.png)

## Puting Things Together
<!-- 这是一张图片，ocr 内容为：FLOPS ACHIEVED FO FOR SQUARE MATMULS (COLOR CODED BY WHETHER A SHAPE IS DIVISIBLE BY K) K2 250 K8 K16 K32 200 128 150 TF/S 100 50 1536 4096 3584 1024 3072 0 2560 512 2048 N:NXN NXN MATMUL -->
![Source: What Shapes Do Matric Multiplications Like?](https://cdn.nlark.com/yuque/0/2025/png/52437949/1760962092992-39757e33-ec76-4a5f-8832-51afaccd887b.png)

1. The general upward trend?
2. Multiple "levels" of FLOPs seems related to shape divisibility, why?
3. Periodic behaviors?

<details class="lake-collapse"><summary id="udc3d91bb"><span class="ne-text">Answers to the above questions</span></summary><ol class="ne-ol"><li id="u88cd95cd" data-lake-index-type="0"><span class="ne-text">compute intensity</span></li><li id="uea856426" data-lake-index-type="0"><span class="ne-text">tiling memory layouts, i.e., more elements to tile than necessary</span></li><li id="u5a2d55ce" data-lake-index-type="0"><span class="ne-text">Sudden #tiles increase leads to a new round of SM parallel execution</span></li></ol><p id="ufb20db95" class="ne-p"><span class="ne-text"></span></p><p id="u5b894f74" class="ne-p"><span class="ne-text">1792 vs. 1793 with a tile size of 256x128</span></p><ul class="ne-ul"><li id="u6fdef7c7" data-lake-index-type="0"><span id="Zc1JX" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c79358d7950b5b15b22bc43954330484.svg"></span><span class="ne-text">tiles</span></li><li id="u4df62800" data-lake-index-type="0"><span id="XxE3n" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/dc32c711de224d11405e49de8c259f33.svg"></span><span class="ne-text">tiles</span></li><li id="u81515e84" data-lake-index-type="0"><span class="ne-text">A100 has 108 SMs, can not execute all 120 once</span></li></ul><p id="u785a2a4b" class="ne-p"><span class="ne-text">Wave Quantization：wave means a set of blocks</span></p><p id="uef4c2843" class="ne-p"><span class="ne-text"></span></p><p id="u959e1ffe" class="ne-p" style="text-align: center"><img src="https://cdn.nlark.com/yuque/0/2025/png/52437949/1760919119632-b30e9efe-2f6a-4765-b7c3-33d1f6272b10.png" width="548" title="Source: What Shapes Do Matric Multiplications Like?" crop="0,0,1,1" id="rdHaF" class="ne-image"></p></details>
# Optimizing GPU Performance with FlatAttention
> Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS, 2022.
>

Transformer self-attention has time and memory complexity quadratic in sequence length, making it struggling in dealling with longer context

+ Sparse/linear attention reduces theoretic FLOPs, yet does not display wall-closk speedup
+ Memory access might be the more critical factor



**<u>FlashAttention's proposal</u>**: making attention algorithm IO-aware, i.e., carefully accounting for reads and writes to different levels of fast and slow memory via

1. Tilling: restructure the attention computation to incrementally perform the softmax reduction
2. Recomputation: store the softmax normalization factor from the forward pass (thus avoid materizalizing the attention matrice $ S \in \mathbb{R}^{N \times N} $to HBM) to quickly recompute attention on-chip in the backward pass
3. Kernel Fusion: fuse all the attention operations into one GPU kernel



**<u>Incremental computation of softmax:</u>** 

+ softmax(full_vec)  = [w_1 * softmax(sub_vec_1), w_2 * softmax(sub_vec_2)]

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761131264179-8c812eb7-f498-4952-9c11-d0ebe34646c7.png)



**<u>FlashAttention Algorithm</u>**

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761130678327-ce44be73-faa4-4045-917e-134ff2ee808f.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761131849363-e4f88058-f843-4a92-9e00-76124610b0e0.png)

课堂思考：why block size is $ \frac{M}{4d} $?

实际执行：

+ 外循环串行（$ K_j $/$ V_j $）执行；内循环（$ Q_i $/$ O_i $）在所有 SM 上并行
+ 一个 SM Load $ K_j $/$ V_j $后，先放 L2 cache，再放入自己的 SMem，其它 SM 直接读 L2 中的 KV
+ 这样保证了统一时候$ O_i/l_i/m_i $只有一个 SM 在执行，不会冲突



Memory access with seq len $ N $and head dimension $ d $: 

+ Standard: $ O(Nd + N^2) $
+ Flash: $ O(N^2d^2M^{-1}) $, why?
+ In practice: $ d=64/128 $, $ M=128*1024 \sim 10 d^2 $

<details class="lake-collapse"><summary id="uc5c30dad"><span class="ne-text">FlashAttention Memory Access Detail</span></summary><p id="u4f188aab" class="ne-p"><span id="UlhHp" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e3d1240d077b97bd040174ad7e4edd33.svg"></span></p></details>


**<u>Performance: </u>**

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761132136231-d94b22f6-b4a1-4f49-8212-7ecb9b4e9469.png)

+ FlashAttention with more FLOPs (due to recomputation) but much less memory access, resulting faster end-to-end wall-clock time without approximation
    - 7.6x speedup vs. PyTorch version on GPT-2
+ End-to-end wall-clock speedup for model training
    - 15% speedup on BERT-large (seq length 512)
    - 3x speedup on GPT-2 (seq length 1K)
    - 2.4x speedup on long-range arena (seq length 1-4K)

# Summary
**<u>GPU hardware</u>**

+ Stream Multiprocessors (SM): 108-128 SMs per A100
    - ALUs: 32-64 float cores, 4 tensor cores, Load/Store per A100 SM
    - Warp Schedulers: 4 per A100 SM
    - L1 cache & shared memory (on-chip SRAM): 192KB per A100 SM
    - L0 cache or registers (on-chip SRAM):  thread-owned
+ L2 cache (on-chip SRAM): 40M per A100, 20TB/s bandwidth ==> <font style="color:#DF2A3F;">small and fast (all SRAM)</font>
+ Global memory (DRAM): 40-80G for a100, 1.5-2TB/s bandwidth ==> <font style="color:#DF2A3F;">big and slow</font>

**<u>Execution model</u>**

+ Grid: collection of thread blocks，对应一次 kernel launch 的所有线程
+ Thread blocks (也称 Cooperative Thread Array, CTA)：GPU 调度基本单元，每个 block 在一个 SM 上执行，一般包含 128-1024 线程
    - 一个 SM 上通常有多个 blocks 常驻（例如 4–8 个）
    - 受 SM 容量限制，GPU 实际执行中不能一次性把所有 block 都分配到 SM 上，而是以 wave 的形式调度，<font style="color:#DF2A3F;">A scheduling wave = the set of blocks that can be resident on all SMs concurrently.</font>
    - wave quantization: # blocks divide #SMs, and should be >= 4 x #SMs
+ Warp：一组固定的 32 个 threads，硬件上实际执行的基本单位，由 warp scheduler 发射执行，共享同一指令流（SIMT 模式）
+ Thread：实际完成工作
+ 一个类比：
    - Gird：学校
    - Block/CTA：班级，是分配教室（SM）的基本单位，共享班级硬件资源（L1 cache & shared memory）
    - Warp：小组，实际干活的协同小组，例如每个小组负责一天值日，在 GPU 内每个人干的事情完全相同
    - Thread：个人，实际干活的人

**<u>Hardware scaling and arithmetic intensity</u>**

+ FLOP scaling >> memory bandwidth scaling
+ arithmetic intensity = # FLOPs / # bytes
    - high (100~): compute-bound
    - low (~1): memory-bound，需要避免

**<u>Improving GPU performance</u>**

+ control divergence
+ reduce memory access volumn: low prevision computation, operator fusion, recomputation
+ optimize memory request: memory coalescing, tiling

