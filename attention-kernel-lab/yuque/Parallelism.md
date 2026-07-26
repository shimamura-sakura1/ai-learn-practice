**<u>本节内容：</u>**

+ Background
    - GPUs in distributed environment, collective communication
+ Different forms of parallel LLM training
    - data parallel (DP): ZeRO
    - model parallel: pipeline parallel (PP), tensor parallel (TP)
    - activation parallel: sequence parallel (SP)
+ Parallelism in practice
    - Combining parallelism strategies into 3D parallelism
    - Implementation parallelism with torch.dist

# Background
## GPU-based scaling still faces limitation in both compute and memory
+ 过去20年里，单GPU FLOP/s提升了6万倍，带宽提升30-100倍。

![[https://medium.com/riselab/ai-and-memory-wall-2cb4265cb0b8]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762070483709-0f6c79d0-392e-41d0-bd8d-c5dded6c0366.png)

+ 但模型 scaling 速度远超 GPU scaling
    - FLOPs：1 年增长 4375 倍
    - 显存（DRAM）：GPT-3 全参训练时，模型部分的显存开销为 3000GB vs. 80GB per A100/H100

| 模型 / 年份 | 参数规模 | 数据规模 | FLOPs |
| --- | --- | --- | --- |
| GPT-2 / 2019 | 1.5B   | ~40G | 1 |
| GPT-3 / 2020 | 175B (<font style="color:#DF2A3F;">x116</font>) | 300B (~1.5T) | <font style="color:#DF2A3F;">4375</font> |


![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762070678384-94070cfe-8a6f-4dd5-bff8-c8b2e5a0a1c0.png)

+ 2020 年后的语言模型，均由多个 GPUs （在分布式环境下）联合训练完成

## GPUs in distributed environment
![[GPT-NeoX-20B, 2022.]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762073656881-65e573f2-0310-491f-8965-809760139f4c.png)

Different layers of Networking: 

+ GT/s: GigaTransfers per second
    - PCIe: 1b/transfers，需要考虑编码效率得到实际带宽
    - NVLink：8bit/transfers (NVLink 1.0/2.0); 4bit/transfers (NVLink >= 3.0)
+ intra-node: fully-connected via NVLink
    - in/out 各 12 lanes，即单向 50x4x12/8=<font style="color:#DF2A3F;">300 GB/s</font>
        * 对单 GPU， 所有 out flow 共享 300 GB/s，因此若 GPU0 同时和 GPU1&2 通信时，共享该带宽
        * 例外是广播，在 NVSwitch 层面实现数据复制，此时每接收端可实现 300 GB/s 的带宽
    - 总带宽上限（单向）：96 lanes = 96 * 25 GB/s = 2.34 TB/s
+ internode: via RDMA（Remote Direct Memory Access，IB 是经典的 RDMA 实现）
    - RDMA 允许一台计算机直接访问另一台计算机的内存
        * 不需要经过对方 CPU 的参与
        * 无需 memory-cache 拷贝
        * 4-lane HDR IB 对应的单向带宽约 25 GB/s（不考虑编码损耗），由 2 个 GPU 共享，故单 GPU 的跨节点带宽为 <font style="color:#DF2A3F;">12.5 GB/s</font>

| 接口类型 | 应用架构 | 传输速率 (GT/s) | 编码方式 | 每通道有效带宽 (GB/s，单向) | 最大连接数 | 双向总带宽 (GB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| PCIe 1.x |  | 2.5 | 8b/10b | 0.25 | x16 | 8 |
| PCIe 2.x |  | 5 | 8b/10b | 0.5 | x16 | 16 |
| PCIe 3.x | P/V/T | 8 | 128b/130b | 0.985 | x16 | 31.5 |
| PCIe 4.0 | Ampere | 16 | 128b/130b | 1.969 | x16 | 63 |
| PCIe 5.0 | Hopper | 32 | 128b/130b | 3.938 | x16 | 126 |
| PCIe 6.0 | Blackwell | 64 | 236b/256b | 7.563 | x16 | 242 |
| NVLink 1.0 | Pascal | 20 |  | 20 | 4 | 160 |
| NVLink 2.0 | Volta | 25 |  | 25 | 6 | 300 |
| NVLink 3.0 | Ampere | 50 |  | 25 | 12 | 600 |
| NVLink 4.0 | Hopper | 50 |  | 25 | 18 | 900 |
| NVLink 5.0 | Blackwell | 100 |  | 50 | 18 | 1800 |


## Collective Communication Primitives (CP)
以下示意图均来自 [Stanford CS336]；图例颜色代表数据拷贝，白=各颜色汇总

1. p：参与通信的进程 / GPU 数量
2. n：每个进程本地持有的数据量（字节）
3. 以下考虑整体网络通信量



+ Reduce: 

<details class="lake-collapse"><summary id="udeb7cd6c"><span class="ne-text">通信量？</span></summary><p id="ud5434c1e" class="ne-p"><span id="wR9oL" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/322f86266a85b8dd2428374aea867c37.svg"></span></p></details>
<!-- 这是一张图片，ocr 内容为： -->
![X=0/1/2/3](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762095678709-ff0d634c-b435-4027-9d60-6afa1ffbdace.png)

+ Broadcast: 

<details class="lake-collapse"><summary id="u58d37963"><span class="ne-text">通信量？</span></summary><p id="uddad49f1" class="ne-p"><span id="GHjQT" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/322f86266a85b8dd2428374aea867c37.svg"></span></p></details>
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762095791456-d859f91a-5399-436a-b850-9ae4dd672718.png)

+ All gather: 

<details class="lake-collapse"><summary id="ueaf73a66"><span class="ne-text">通信量？</span></summary><p id="u10f064d1" class="ne-p"><span id="go1o4" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/322f86266a85b8dd2428374aea867c37.svg"></span></p></details>
<!-- 这是一张图片，ocr 内容为： -->
![Y=?; count=?](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762095739750-b7e0d4aa-c50c-4704-abe6-75f5b526677d.png)

+ Reduce scatter: 

<details class="lake-collapse"><summary id="u6053d624"><span class="ne-text">通信量？</span></summary><p id="uef0499ce" class="ne-p"><span id="Rsiso" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/322f86266a85b8dd2428374aea867c37.svg"></span></p></details>
<!-- 这是一张图片，ocr 内容为： -->
![X=?; Y=?; count=?](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762095774008-5cf3cd40-e7ce-4ae9-99d9-386456c824a7.png)

+ All reduce: 
    - naively: $ np(p-1) $

<details class="lake-collapse"><summary id="u7bd092ad"><span class="ne-text">如何优化去掉其中一个 p?</span></summary><p id="u38a10bdd" class="ne-p"><span class="ne-text">reduce + broadcasting，此时通信量为</span><span id="lyKK3" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/29718a4b722279e8b269543f3dfd5feb.svg"></span></p><ul class="ne-ul"><li id="u8ac6444c" data-lake-index-type="0"><span class="ne-text">核心思想：尽可能将带宽用于传输最终结果</span></li><li id="u855149f3" data-lake-index-type="0"><span class="ne-text">此时 root 节点成为通信瓶颈</span></li></ul></details>
<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762095696456-0519e563-832a-454c-b6c1-18530e2c3c21.png)

+ <font style="color:#DF2A3F;">All-reduce = reduce-scatter + all-gather</font>
    - 核心思想：将整体网络通信均匀分布在 GPU 之间，网络整体通信量保持$ 2n(p-1) $不变，单 GPU 的通信量为$ 2*\frac{n}{p}*(p-1) \approx 2n $
        * 该方法优化约$ p $倍通信量的由来：考虑最大通信量节点
    - 带宽最优 all-reduce

![[https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762098681124-c4920a6f-d24a-4fcb-a4da-fe2d70669122.png)

<!-- 这是一张图片，ocr 内容为： -->
![第1轮ring通信实现reduce-scatter](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762134858659-69b74caa-c071-437d-b575-0ff027f4c5e9.png)<!-- 这是一张图片，ocr 内容为： -->
![第2轮ring通信实现all-gather](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762134377833-58fe0cea-44f8-46fb-8b13-9bc5599a754f.png)

<u>课堂讨论</u>：reduce 和 broadcating 在实现中可以如何优化缓解 root 节点的通信瓶颈？

## SuperPOD
+ NVIDIA DGX (Deep GPU Xceleration)

<!-- 这是一张图片，ocr 内容为： -->
![https://www.exxactcorp.com/blog/HPC/NVIDIA-H100](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762142747214-c8f0606c-7bd5-4c11-aabe-500bef5b2770.png)

+ TPU networking：环形网络 / toroidal mesh

![toroidal mesh的设计依据和优势是什么？[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762143332292-0c4da18c-aa71-43ac-98d2-d7b680b763b5.png)

# LLM Parallelism Primitives
在多机多卡分布式 LM 训练中，我们希望实现：

+ 线性的存储伸缩（linear memory scaling）：可容纳的最大模型规模随 #GPUs 线性增长
+ 线性的计算伸缩（linear compute scaling）：模型有效的训练 FLOPs 随 #GPUs 线性增长

## Data Parallel (DP)
### Naive DP
通过梯度下降优化模型：

$ \theta_{t+1} = \theta_t - \eta \sum_{i=1}^B \nabla f(x_i) $

Naive DP 将当前 batch 均匀分配给$ M $个 GPU，各 GPU 计算好 micro-batch 的梯度后同步，然后完成参数更新

<details class="lake-collapse"><summary id="u8c1778fa"><span class="ne-text">课堂讨论：对应哪个 CP，compute scaling / memory scaling / communication overhead 如何？</span></summary><ol class="ne-ol"><li id="ue5f0b234" data-lake-index-type="0"><span class="ne-text">All-reduce</span></li><li id="ud7948f77" data-lake-index-type="0"><span class="ne-text">compute scaling 优：每个 GPU 独立负责</span><span id="i2vOS" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/535d174142e5d273e1ac42fbccc68d47.svg"></span><span class="ne-text">个样本，当该 micro-batchsize 足够大时，可以用满 FLOP/s</span></li><li id="u88827f48" data-lake-index-type="0"><span class="ne-text">communication overhead 可接收：每个 GPU 每 batch 需传输 2x #params 的数据，该部分通信开销可隐藏在计算中</span></li><li id="u806e06e8" data-lake-index-type="0"><span class="ne-text">memory scaling 不可接受：瓶颈是模型可以 fits in 单 GPU，最大模型规模不随 #GPUs 增长</span></li></ol><ol class="ne-list-wrap"><ol ne-level="1" class="ne-ol"><li id="u8104cb58" data-lake-index-type="0"><span class="ne-text">每个 GPU 需加载 1 份完整的模型事实上会限制 micro-batchsize 的规模</span></li></ol></ol></details>


**<u>DP vs. MP:</u>**

+ DP 具有更高的计算粒度（computational granularity）和更低的通信量（communication volume）
+ <font style="color:#000000;">MP 降低了计算粒度，同时又增加了通信开销</font>
+ DP 的 compute scaling efficiency 较 MP 更高（特别是后者需要跨节点通信的时候）
+ MP 的 memory scaling efficiency 较 DP 更优



**<u>Naive DP 的模型部分的显存：</u>**5 copies of weights and 16 bytes per param

+ [model] 2 bytes for fp/bf 16 model parameters
+ [gradient] 2 bytes for fp/bf 16 gradients
+ [optimizer state] 4 bytes for fp32 master weights (updated in optimizer)
+ [optimizer state] 4 bytes for fp32 Adam first moment estimates
+ [optimizer state] 4 bytes for fp32 Adam second moment estimates
+ _注：上述划分仅为方便后续 ZeRO 讨论，该划分并非普遍接受的标准_

### ZeRO levels 1-3
> Rajbhandari et al. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. SC, 2020. (Microsoft)
>

ZeRO (Zero Redundancy Optimizer): eliminate memory redundancies

+ retain low communication volume and high computational granularity (of DP)
+ scale the model size proportional to the number of devices with sustained high eﬃciency (like MP)

![[ZeRO paper]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762147657175-da0b93cd-a94b-4111-adc3-5861429873a5.png)

**<u>ZeRO stage 1</u>**

+ Every GPU has the parameters + gradients
+ Splitting up the optimizer states across all GPUs
+ Each GPUs is responsible for updating a subset of params

<details class="lake-collapse"><summary id="uf4184754"><span class="ne-text" style="color: rgb(138, 143, 141); font-size: 14px">ZeRO stage 1 如何计算？</span></summary><p id="ub80f5aab" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2025/png/52437949/1762148438073-dc8ad1ca-4a51-4ad1-b4dc-08c0c704bf4b.png" width="1022" title="[Stanford CS336]" crop="0,0,1,1" id="yLRQs" class="ne-image"></p></details>
Zero stage 1 vs. native DP: 

+ CP: 1 all-reduce (gradients) vs. 1 reduce-scatter (send gradients) + 1 all-gather (gather params)
+ communication cost: 2x#params vs. 2x#params
+ model-part memory: $ (4+K)\Phi $ vs. $ (4+\frac{K}{N_d})\Phi $

<font style="color:#DF2A3F;">>> ZeRO stage 1 的显存优化是 free lunch!</font>



**<u>ZeRO stage 2</u>**

> As each data parallel process only updates its corresponding parameter partition, it only needs
>
> the reduced gradients for the corresponding parameters. 
>

+ Also keep the gradient shared across GPUs
+ Use the roughly same tricks as stage 1
+ Each GPU must compute a full gradient, but never instantiate a full graident vector（阅后即焚 ^_^）

<details class="lake-collapse"><summary id="uc1fc9f13"><span class="ne-text" style="color: rgb(138, 143, 141); font-size: 14px">ZeRO stage 2 如何计算？</span></summary><p id="u96c4a808" class="ne-p"><img src="https://cdn.nlark.com/yuque/0/2025/png/52437949/1762149692572-850010d8-9ab4-464a-ac29-0009cbf90cb9.png" width="1196" title="[Stanford CS336]" crop="0,0,1,1" id="u93cbf549" class="ne-image"></p></details>
Zero stage 2 vs. Zero stage 1:

+ gradient reduce-scatter 替换为 reduce，并被分散在 backward 过程中
+  <font style="color:#DF2A3F;">communication cost 总体保持不变，额外的系统复杂度包括梯度释放、reduce bucket 管理等</font>



课堂讨论：从 ZeRO-1/2 的计算流程看，它们是如何切分 os/gradient 的？

+ Option 1：不同 GPUs 负责不同的 weight tensors（类似 PP）
+ Option 2：weight tensor 被拆然后交由不同的 GPUs 负责（类似 TP）



**<u>ZeRO stage 3 (a.k.a Fully Sharded Data Parallel, FSDP)</u>**

+ Shard everything, including parameters
+ incremental communication / computation
+ Send and request parameters on demand

<!-- 这是一张图片，ocr 内容为： -->
![https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp2-works](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762178765487-44e32e17-3d43-447c-b31e-10b81b7bec4d.png)

+ Communication cost: 2 all-gather + 1 reduce-scatter
+ Subtle difference (yes, there exists) between ZeRO stage 3 and FSDP

| 分片粒度 | 参数张量级（每层） | 模块级（按 instance 划分参数集合） |
| --- | --- | --- |
| All-Gather 时机 | 每层 forward / backward 前 | 每个 instance forward / backward 前 |
| Free 时机 | 每层计算完即释放 | 每个 instance 计算完即释放 |
| 通信模式 | 更细粒度 | 可控制粒度，通过 instance 合并层 |
| 取舍 | 显存峰值更小 | 通信次数更少，带宽利用率高（大包） |


![[Zhao et al. PyTorch FSDP. VLDB, 2023.]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762181864196-1232d95e-6ae9-4343-9f43-37ad60ce6d5e.png)

+ $ y = W_1W_0x + W_2W_0x $
+ 增量计算和通信，用完立即释放 full weights，如 W1 和 W2
+ 通信和计算重叠，如在 FWD0 中 AG1 同步进行，从而提升 GPU 计算效率
+ 整体看，<font style="color:#DF2A3F;">ZeRO stage 3 的通信开销变为 stage 1/2 的 1.5 倍，同时产生了通信同步开销</font>

> ZeRO (stage-3) powers DP to fit models with arbitrary size as long as there are suﬃcient number of devices to share the model states.
>



**<u>Summary</u>**

+ native DP：通信开销为 2 x #params（all-reduce）
+ ZeRO stage 1：通信开销同为 2 x #params (reduce-scatter + all-gather)，免费的优化
+ ZeRO stage 2：通信开销依然为 2 x #params (reduce + all-gather)，涉及额外系统开销
+ ZeRO stage 3：通信开销增长为 3 x #params (reduce-scatter + all-gather x 2)，且有些计算需要等待数据同步
+ 实践建议：可以直接使用 ZeRO stage 2，综合考量 stage 3 的显存收益与训练时间开销
+ 课堂讨论：3 个 stage 的 memory/compute scaling 性能如何？

<!-- 这是一张图片，ocr 内容为： -->
![建议所有同学阅读ZeRO论文。](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762233762182-20d912bc-2155-4bfd-850a-d5f64735cc57.png)



**<u>Issues of ZeRO DP:</u>**

+ requiring #gpus <= batchsize, yet there’s diminishing returns to batch sizes

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762355918417-0d887da5-ca6d-490a-8be5-658b586b7a8c.png)

<!-- 这是一张图片，ocr 内容为： -->
![McCandlish et al. An Empirical Model of Large-Batch Training. 2018.](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762355943676-bf9906a2-bbf5-4688-b05c-5ff8e7e46a35.png)

> When the batch size is very small, the approximation will have very high variance, and the resulting gradient update will be mostly noise. 
>
> By contrast, when the batch size is very large, the batch gradient will almost exactly match the true gradient. As a result, doubling the batch size will barely improve the update, we will use twice as much computation for little gain. 
>
> Intuitively, the transition should occur roughly where the noise and signal of the gradient are balanced, where the variance of the gradient is at the same scale as the gradient itself. 
>
> Formalizing this heuristic observation leads to the gradient noise scale.
>

    - batch size 也是模型训练阶段的一种效率资源！
+ ZeRO stage 1&2 do not linear scale memory, and stage 3 (while nice in memory) could be slow and does not reduce activation memory

![PTD-P: Pipeline, Tensor, and Data Parallel. [Narayanan et al. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. 2021.]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762356381780-26f9833a-36fd-4376-b7cb-0b8c73ccc280.png)

## Model Parallel (MP)
> Huang et al. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. NeurIPS, 2019. (Google)
>
> Shoeybi et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. Arxiv, 2019. (NVIDIA)
>
> Narayanan et al. Efficient large-scale language model training on GPU clusters using megatron-LM. SC, 2021. (NVIDIA)
>

MP splits up the parameters across GPUs (like ZeRO-3), but communicate activations (while ZeRO-3 sends params)

+ scaling up in memory without changing batch size



**<u>ZeRO-3 vs. MP 在模型拆分方面的区别：</u>**

+ ZeRO-3 只在存储层面对模型进行拆分，计算层面没有（all-gather weights）
+ MP 在存储和计算层面同时对模型进行拆分，换言之每个 GPU 只基于其负责的权重进行计算

**<u>两类模型并行方法：</u>**

+ Pipeline parallel: split model horizontally (along depth)
+ Tensor parallel: split model vertically (along width)

### Pipeline Parallel (PP)
**<u>Layer-wise parallel</u>**

+ 在 between-layer 处切模型，并将分割后的 sub-model 顺序放置在 GPU 之上
+ Things to communicate: activations and partial gradients
+ Where communication happen: 模型切割处

![Layer-wise parallel [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762415354952-a204c3da-326f-43c4-87d3-cfd1c234358d.png)

+ 问题：large bubble，当 n 块 GPU 共同完成 PP 计算时，每个 GPU 仅在 1/n 时间内有效使用，见下图 (b)
    - 所有 GPU 在绝大部份时间里处于 idle 状态，等待前序/后续 GPU 的 forward/backward pass 结果

![[GPipe paper]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762413766586-24090e42-f79d-4d92-89ab-17cee44809cc.png)

![GPipe schudule. [Narayanan+, 2021]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762738291190-e9ab503e-a2df-4ca0-a4d2-48d57f858a7b.png)

**<u>A solution: pipeline parallel</u>**

+ 将 batch 拆分为 micro-batches, 例如上图 (c) 中一个 batch 被拆成 4 份micro-batches，并以流水线方式并行处理 micro-batches
+ 一个 micro-batch 处理完并发送至后续 GPU 后，即可开始下一个 batch 的计算
+ 课堂讨论：bubble / total time for PP with K GPUs/stages and M micro-batches?

<details class="lake-collapse"><summary id="u77937dd0"><span class="ne-text">bubble ratio for PP when neglecting communication latency</span></summary><p id="u15fac687" class="ne-p"><span class="ne-text">(K - 1) / (M + K - 1)</span></p></details>
+ <font style="color:#000000;">We need a large batch size!</font>
    - <font style="color:#000000;">The bubble overhead is to be negligible when M ≥ 4 × K [GPipe]</font>

![As we increase the number of pipeline stages, we also increase the size of the model by proportionally increasing the number of layers in the model. micro-batch size = 1. [Narayanan+, 2021]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762418404039-93f83adc-bf3b-4b39-ade9-0e51fb59d52b.png)

    - 课堂讨论：请计算在 PP=8 的情况下，两个 batch size = 8/128 对应的 bubble ratio
+ 课堂讨论：在有 bubble 的情况下，PP 依然被使用，其优势在哪里？

<details class="lake-collapse"><summary id="ue1468d05"><span class="ne-text">PP 优势与应用场景</span></summary><ol class="ne-ol"><li id="ud88352a3" data-lake-index-type="0"><span class="ne-text">Pipelines linearly scales memory</span></li><li id="ucf6367b3" data-lake-index-type="0"><span class="ne-text">Pipeline can have good communication properties: depends only on activations with size </span><span id="Rb9xd" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/43370c6cc245b967eac8d1a92ebc8378.svg"></span><span class="ne-text">and is point-to-point</span></li></ol><p id="ue03b4c25" class="ne-p"><span class="ne-text">通常会将 PP 应用于较慢的网络链路（inter-nodes）中，以实现更好的 memory-wise scaling。</span></p></details>


**<u>More advanced PP</u>**

+ Interleaved 1F1B pipeline schedules，即下图 top 部分

<details class="lake-collapse"><summary id="ubfd80b03"><span class="ne-text">课堂讨论：Interleaved 1F1B 优势</span></summary><ol class="ne-ol"><li id="uc4085819" data-lake-index-type="0"><span class="ne-text" style="color: #000000">The time spent in the bubble is the same, </span></li><li id="ucac9a969" data-lake-index-type="0"><span class="ne-text" style="color: #000000">The number of outstanding forward passes is at most the number of pipeline stages (K) instead of the number of microbatch (M), being more memory-efficient when M &gt;&gt; K</span></li></ol></details>
+ Interleaved stage schedules: 即下图 bottom 部分
    - GPU 负责多个 chunk（对应 stage），例如 GPU0 负责 layers 1&2 + layers 9&10 （原始 layers 1-4）
    - 通信带宽换取更高的 GPU 使用率

![[Narayanan+, 2021]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762419212559-cf33a075-9029-4cec-8a22-81944c042602.png)

<details class="lake-collapse"><summary id="u6e689920"><span class="ne-text">课堂讨论：Interleaved stage scheduling 下 bubble 影响？</span></summary><p id="u0b8e77f8" class="ne-p"><span class="ne-text">降低 bubble 比例，若一个 GPU 上有 v 个 stage，不考虑通信开销下，bubble 绝对量降为原先 1/v</span></p></details>
+ Zero bubble pipelining [Qi et al. Zero Bubble (Almost) Pipeline Parallelism. ICLR, 2024.]

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762421230236-33ad16b8-c341-4704-a6c7-4f6e5fbc8044.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762421388873-b923049c-ce37-40b5-849a-812d4176af10.png)

### Tensor Parallel (TP)
Simple matrix multiplication observation: decompose into submatrices, add partial sums

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762422332539-7a08bdc5-eb4c-4dfe-a5c5-5faa4257d198.png)<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762422351344-80ef927b-52c8-4f70-97b6-2f57f4403c75.png)

**Tensor parallelism: GPUs have submatrices of parameters**

![[Megatron-LM paper]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762422442377-367eea93-e423-4017-804d-a10cdf2c9935.png)

+ For MLP: 
    - $ A = [A_1, A_2] $指 column-wise split；$ B = \left[\begin{array}c B_1 \\ B_2 \end{array}\right] $指 row-wise split
    - $ f $：identify in forward pass and all-reduce in backward pass
    - $ [Y_1, Y_2] = [\text{GeLU}(XA_1), \text{GeLU}(XA_2)] $
    - $ g $：all-reduce in forward pass and identify in backward pass
+ For attention: <font style="color:#000000;">exploit inherent parallelism in the multihead attention operation,</font>
    -  matrix multiply corresponding to each attention head is done locally on one GPU.



![TP within a node (up to 8 GPUs) introduces a 20% overhead. [Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762423169080-51d3f86c-ed2c-436b-a185-9b7a02eb5131.png)



<details class="lake-collapse"><summary id="u21f5084c"><strong><span class="ne-text" style="text-decoration: underline">TP vs. PP: pros and cons</span></strong></summary><p id="ua668bbaa" class="ne-p"><span class="ne-text">Pros: </span></p><ol class="ne-ol"><li id="u2255cc0f" data-lake-index-type="0"><span class="ne-text">no bubble, i.e., no waiting time when network is fast enough</span></li><li id="u0054d54d" data-lake-index-type="0"><span class="ne-text">low infra complexity</span></li><li id="u3635f7b0" data-lake-index-type="0"><span class="ne-text">does not need large batch sizes</span></li></ol><p id="u05ba8d3f" class="ne-p"><span class="ne-text"></span></p><p id="uc3da31d9" class="ne-p"><span class="ne-text">Cons: much larger communication</span></p><ul class="ne-ul"><li id="uc6e1cb30" data-lake-index-type="0"><span class="ne-text">PP: </span><span id="HI5qt" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e0333a3970d40646d382da4261b3d438.svg"></span><span class="ne-text">P2P communication per </span><span class="ne-text" style="color: #000000">microbatch (for either forward and backward)</span></li><li id="u480db206" data-lake-index-type="0"><span class="ne-text">TP: </span><span id="pDC40" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/c02da51f486395375eb569b3e382fb03.svg"></span><span class="ne-text">per layer, i.e.,</span><span id="BBgqB" class="ne-math"><img src="https://cdn.nlark.com/yuque/__latex/e0333a3970d40646d382da4261b3d438.svg"></span><span class="ne-text">all-reduced twice in both forward and backward</span></li></ul><p id="ud98a2367" class="ne-p" style="text-align: center"><img src="https://cdn.nlark.com/yuque/0/2025/png/52437949/1762439498428-c8e69373-4553-45d3-87a8-8869c5cb8a3c.png" width="501" title="[Megatron-LM paper]" crop="0,0,1,1" id="uf08b7e9d" class="ne-image"></p></details>
+ Use TP only when having low-latency, high-bandwidth interconnects (intra-node)

## Activation Parallel (Sequence Parallel)
> Korthikanti et al. Reducing Activation Recomputation in Large Transformer Modesl. 2022. (NVIDIA)
>

![GPU memory usage is dynamic [Standord CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762439699169-15d468f6-3681-48d1-810d-758b8ed14ab5.png)



<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762442391886-0562cc03-27bc-4bc8-acde-8b29567d957a.png)

**<u>Activations memory per layer</u>** = $ sbh(34 + 5 \frac{as}{h}) $bytes

+ Attention block: $ 11sbh + 5 as^2b $
    - Input: $ 2sbh $
    - QK^T: storing both Q and K with$ 4sbh $
    - Softmax: $ 2as^2b $
    - Softmax dropout: a mask with $ as^2b $and dropout result $ 2as^2b $
    - Attention over values: $ 2sbh $
    - Output and dropout: $ 2sbh $and $ sbh $(mask)
+ MLP: $ 19 sbh $
    - inputs to linear layers: $ 2sbh + 8 sbh = 10 sbh $
    - GeLU:  $ 8 sbh $
    - Drouput: $ sbh $(mask)
+ LayerNorm: $ 4 sbh $
    - $ 2 sbh \times 2 $

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762441226356-6d9a9300-588e-4a79-8a3e-b22febd9d17f.png)



**<u>With tensor parallel, activations memory per layer</u>** = $ sbh(10 + \frac{24}{t} + 5 \frac{as}{ht}) $bytes

+ the fixed 10 comes from LayerNorm (4), Dropout (2), and inputs to the attention and MLP (4)
+ These will continue to grow with model size



**<u>Sequence parallel: make memory truly linear</u>**

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762442999063-f85cc84d-f790-45f4-8d29-2eec2e4eccea.png)

+ Observation: all the $ 10sbh $are pointwise ops over sequence, so split up these layer along the sequence axis
+ In the forward pass:$ g $is all-gather , $ \bar{g} $is reduced-scatter; those are reversed in the backward pass

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762444230990-9930fbd7-3776-4df9-9670-09310880975f.png)

## Other Parallelism Strategies
**<u>Context parallel / ring attention: split activations across GPUS in a long sequence</u>**

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762444676004-a3456a77-6e25-4fa9-8609-4362ad94b430.png)



**<u>Expert parallel: split experts across GPUS via all-to-all communication</u>**

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762444654500-b84b9d11-125b-49a0-b116-e9a06c96cdce.png)

## Parallelism Summary
| Parallelism | Sync | Memory | Bandwidth | Batch size | Easy to use |
| --- | --- | --- | --- | --- | --- |
| DDP/ ZeRO1 | per-batch | <font style="color:#DF2A3F;">No scaling</font> | 2 x #params | <font style="color:#DF2A3F;">Linear</font> | Very |
| ZeRO3 (FSDP) | <font style="color:#DF2A3F;">3x Per-FSDP block</font> | Linear | 3 x #params | <font style="color:#DF2A3F;">Linear</font> | Very |
| Pipeline | Per-pipeline | Linear | Activation + gradient | <font style="color:#DF2A3F;">Linear</font> | <font style="color:#DF2A3F;">No</font> |
| Tensor+Seq | <font style="color:#DF2A3F;">2x per transformer block</font> | Linear | <font style="color:#DF2A3F;">8 x activatation per layer</font> | No impact | <font style="color:#DF2A3F;">No</font> |


需平衡多种资源制定并行策略：memory, bandwidth, batch size

# Parallelism in practice
## Combining multiple parallelism strategies
**<u>3D Parallelism</u>**

1. Until your model fits in memory
    1. TP up to #GPUs / node
    2. PP across nodes, or use ZeRO-3
2. Then util you run out of GPUs
    1. scale the rest with data parallel
3. Consider use gradient accumulation to trade higher batch size for better communication efficiency

![[Stanford CS336]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762445394154-2b687d89-7fe5-4ea5-8e86-648e8820de5d.png)



**<u>General Parallelism Strategy</u>**

![How about DP size? [Narayanan+, 2021]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762445599736-d94b7ea6-d0a2-495f-aba7-00415a5a840f.png)

+ TP first up to 8, then caps out as 8
+ PP then goes up to make the model fit
+ DP gradually decreases with scale: 32 (till 76B) -> 24 -> 15 -> 9 -> 6



![Careful 3D parallelism gives linear gains (flat utilization). [Narayanan+, 2021]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762446056068-75de064b-f064-487a-9de2-e3565f945ae6.png)

![TP=8 is often optimal [Narayanan+, 2021]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762446168011-67c17f39-3d3e-460c-8af9-f9429260cab7.png)

![Activation recomputation enables larger batches, improving throughput. [Narayanan+, 2021]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762446340513-af32cd0a-b64d-4fa9-8fe1-6b4601842fb6.png)

## Implementing parallelism by our own
In GPUs (single-gpus or cross-nodes), compute (arithmetic logic units) is far from inputs/outputs data: 

+ kernel level: reduce memory accesses via fusion/tiling
+ job level: reduce communication across GPUs/nodes via (smartly) replication/sharding



Generalized hierarchy (from small/fast to big/slow):

1. Single node, single GPU: L1 cache / shared memory
2. Single node, single GPU: HBM
3. Single node, multi-GPU: NVLink > PCIe
4. Multi-node, multi-GPU: NVSwitch > RDMA

### Distributed communication
**<u>Terminology:</u>**

+ World size: number of devices (e.g., 4)
+ Rank: a device (e.g., 0, 1, 2, 3)
+ Communication primitive: Better/faster abstraction than managing P2P communication yourself
    - one-to-all: broadcast, scatter
    - all-to-one: gather, reduce
    - all-to-all: reduce-scatter, all-gather, all-reduce
    - Broadcast/scatter is inverse of gather/reduce
    - Reduce: performs some associative/commutative operation (sum, min, max)
    - All: means destination is all devices

<!-- 这是一张图片，ocr 内容为： -->
![https://docs.pytorch.org/tutorials/_images/broadcast.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762739662758-8daeca50-ac3d-4b35-9fe0-efe89eed17ac.png)<!-- 这是一张图片，ocr 内容为： -->
![https://docs.pytorch.org/tutorials/_images/scatter.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762739682572-a59666ba-ffbb-459e-8edc-9718b002b23f.png)

<!-- 这是一张图片，ocr 内容为： -->
![https://docs.pytorch.org/tutorials/_images/gather.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762739775359-d6c70d9b-72f8-4487-b639-40d93377ee69.png)<!-- 这是一张图片，ocr 内容为： -->
![https://docs.pytorch.org/tutorials/_images/reduce.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762739802470-aad3696f-80e1-43a5-8b4a-1a3ff3c40ae5.png)



<!-- 这是一张图片，ocr 内容为： -->
![https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762739852299-0fbc251d-3b40-4e55-88cd-a32967b477d1.png)<!-- 这是一张图片，ocr 内容为： -->
![https://docs.pytorch.org/tutorials/_images/all_gather.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762739825212-aacec34f-cee6-4f0b-a474-4075e16619d5.png)

<!-- 这是一张图片，ocr 内容为： -->
![https://docs.pytorch.org/tutorials/_images/all_reduce.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762739873050-2c5567d2-b0cf-410a-82cd-2aa63b3f17cb.png)



**<u>NVIDIA Collective Communication Library (NCCL)</u>**

+ NCCL translates collective operations into low-level packets that are sent between GPU
+ Detects topology of hardware (e.g., number of nodes, switches, NVLink/PCIe)
+ Optimizes the path between GPUs
+ Launches CUDA kernels to send/receive data
+ We will not directly program with NCCL
+ Further reading: [https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31880/)



**<u>PyTorch distributed library (`torch.distributed`</u>**

+ Documentation: [https://pytorch.org/docs/stable/distributed.html](https://pytorch.org/docs/stable/distributed.html)
+ Provides clean interface for collective operations
+ Supports multiple backends for different hardware: gloo (CPU), nccl (GPU)



```python
def spawn(func: Callable, world_size: int, *args, **kwargs):
    # Note: assume kwargs are in the same order as what main needs
    if sys.gettrace():
        # If we're being traced, run the function directly, since we can't trace through mp.spawn
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
    else:
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)
        """
        作用：并行地为训练/测试启动多个独立的 Python 进程，每个进程会运行相同的
        目标函数（这里是 func），用于实现多进程的分布式或并行工作，例如每个进程对
        应一个分布式 rank / GPU。
        - 对每个进程 i（i 从 0 到 nprocs-1），spawn 会在新进程中调用 func(i, *args)。
        - join=True：父进程阻塞直到所有子进程退出（等待子进程完成）。
        """

def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()

############################################################

def benchmarking():
    # Let's see how fast communication happens (restrict to one node).
    world_size = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    print(f"Running benchmarking with world_size={world_size}")

    # All-reduce
    spawn(all_reduce, world_size=world_size, num_elements=100 * 1024**2)

    # Reduce-scatter
    spawn(reduce_scatter, world_size=world_size, num_elements=100 * 1024**2)


def all_reduce(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create tensor
    tensor = torch.randn(num_elements, device=get_device(rank))

    # Warmup
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform all-reduce
    start_time = time.time()
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kernels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[all_reduce] Rank {rank} PID {os.getpid()}: all_reduce(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    size_bytes = tensor.element_size() * tensor.numel()
    sent_bytes = size_bytes * 2 * (world_size - 1) / world_size  # 2x because reduce-scatter + all-gather
    bandwidth = sent_bytes / duration
    print(f"[all_reduce] Rank {rank}: all_reduce measured bandwidth = {round(bandwidth / 1024**3, 2)} GB/s", flush=True)

    cleanup()


def reduce_scatter(rank: int, world_size: int, num_elements: int):
    setup(rank, world_size)

    # Create input and outputs
    input = torch.randn(world_size, num_elements, device=get_device(rank))  # Each rank has a matrix
    output = torch.empty(num_elements, device=get_device(rank))

    # Warmup
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here

    # Perform reduce-scatter
    start_time = time.time()
    dist.reduce_scatter_tensor(output=output, input=input, op=dist.ReduceOp.SUM, async_op=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA kerels to finish
        dist.barrier()            # Wait for all the processes to get here
    end_time = time.time()

    duration = end_time - start_time
    print(f"[reduce_scatter] Rank {rank} PID {os.getpid()}: reduce_scatter(world_size={world_size}, num_elements={num_elements}) took {render_duration(duration)}", flush=True)

    # Measure the effective bandwidth
    dist.barrier()
    data_bytes = input.element_size() * input.numel()  # How much data in the input
    sent_bytes = data_bytes * (world_size - 1) / world_size  # How much needs to be sent (no 2x here)
    bandwidth = sent_bytes / duration
    print(f"[reduce_scatter] Rank {rank}: reduce_scatter measured bandwidth = {round(bandwidth / 1024**3, 2)} GB/s", flush=True)
    
    cleanup()
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1762747532924-ee3712cf-fccf-4bc9-8a51-7653aed16e52.png)

<details class="lake-collapse"><summary id="u32bba9e3"><span class="ne-text">tested bandwidth</span></summary><p id="u5836e326" class="ne-p"><span class="ne-text">(base) hurenjun@inspur:~/pymisc$ CUDA_VISIBLE_DEVICES=0,1 python collective_comm.py </span></p><p id="uc29bb5ac" class="ne-p"><span class="ne-text">Running benchmarking with world_size=2</span></p><p id="u448d68fe" class="ne-p"><span class="ne-text">[all_reduce] Rank 0 </span><span class="ne-text" style="color: #DF2A3F">PID 1007</span><span class="ne-text">: all_reduce(world_size=2, num_elements=104857600) took 43.36ms</span></p><p id="uf2e86ce0" class="ne-p"><span class="ne-text">[all_reduce] Rank 1 </span><span class="ne-text" style="color: #DF2A3F">PID 1009</span><span class="ne-text">: all_reduce(world_size=2, num_elements=104857600) took 43.37ms</span></p><p id="u34ced824" class="ne-p"><span class="ne-text">[all_reduce] Rank 0: all_reduce measured bandwidth = </span><span class="ne-text" style="color: #DF2A3F">9.01 GB/s</span></p><p id="uf7efc517" class="ne-p"><span class="ne-text">[all_reduce] Rank 1: all_reduce measured bandwidth = 9.01 GB/s</span></p><p id="u0883ec89" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 1 PID 1300: reduce_scatter(world_size=2, num_elements=104857600) took 49.10ms</span></p><p id="ub4f52507" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 0 PID 1294: reduce_scatter(world_size=2, num_elements=104857600) took 49.09ms</span></p><p id="u1f6b803b" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 0: reduce_scatter measured bandwidth = </span><span class="ne-text" style="color: #DF2A3F">7.96 GB/s</span></p><p id="u95a13c09" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 1: reduce_scatter measured bandwidth = 7.96 GB/s</span></p><p id="u633169ac" class="ne-p"><span class="ne-text"></span></p><p id="u369c527e" class="ne-p"><span class="ne-text">(base) hurenjun@inspur:~/pymisc$ CUDA_VISIBLE_DEVICES=1,2 python collective_comm.py </span></p><p id="u3fe51285" class="ne-p"><span class="ne-text">Running benchmarking with world_size=2</span></p><p id="u305277cd" class="ne-p"><span class="ne-text">[all_reduce] Rank 1 PID 1674: all_reduce(world_size=2, num_elements=104857600) took 8.39ms</span></p><p id="uc14a6da8" class="ne-p"><span class="ne-text">[all_reduce] Rank 0 PID 1673: all_reduce(world_size=2, num_elements=104857600) took 6.96ms</span></p><p id="u1b3330fe" class="ne-p"><span class="ne-text">[all_reduce] Rank 1: all_reduce measured bandwidth = </span><span class="ne-text" style="color: #DF2A3F">46.56 GB/s</span></p><p id="uaa4d6af7" class="ne-p"><span class="ne-text">[all_reduce] Rank 0: all_reduce measured bandwidth = 56.11 GB/s</span></p><p id="u6b793fd5" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 1 PID 1965: reduce_scatter(world_size=2, num_elements=104857600) took 7.92ms</span></p><p id="u3049d319" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 0 PID 1964: reduce_scatter(world_size=2, num_elements=104857600) took 7.91ms</span></p><p id="ube82d526" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 0: reduce_scatter measured bandwidth = </span><span class="ne-text" style="color: #DF2A3F">49.39 GB/s</span></p><p id="u056f295d" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 1: reduce_scatter measured bandwidth = 49.32 GB/s</span></p><p id="ueb578588" class="ne-p"><span class="ne-text"></span></p><p id="u4fdb2fba" class="ne-p"><span class="ne-text">(base) hurenjun@inspur:~/pymisc$ CUDA_VISIBLE_DEVICES=0,1,2 python collective_comm.py </span></p><p id="u60862d9f" class="ne-p"><span class="ne-text">Running benchmarking with world_size=3</span></p><p id="u09797d0c" class="ne-p"><span class="ne-text">[all_reduce] Rank 2 PID 2276: all_reduce(world_size=3, num_elements=104857600) took 67.74ms</span></p><p id="u213dcf11" class="ne-p"><span class="ne-text">[all_reduce] Rank 0 PID 2274: all_reduce(world_size=3, num_elements=104857600) took 67.64ms</span></p><p id="uc34b233a" class="ne-p"><span class="ne-text">[all_reduce] Rank 1 PID 2275: all_reduce(world_size=3, num_elements=104857600) took 67.71ms</span></p><p id="ued5b45b8" class="ne-p"><span class="ne-text">[all_reduce] Rank 0: all_reduce measured bandwidth = </span><span class="ne-text" style="color: #DF2A3F">7.7 GB/s</span></p><p id="uaafef870" class="ne-p"><span class="ne-text">[all_reduce] Rank 2: all_reduce measured bandwidth = 7.69 GB/s</span></p><p id="ue7c77f20" class="ne-p"><span class="ne-text">[all_reduce] Rank 1: all_reduce measured bandwidth = 7.69 GB/s</span></p><p id="u5a7d5aaa" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 0 PID 2547: reduce_scatter(world_size=3, num_elements=104857600) took 100.58ms</span></p><p id="u57c71ee7" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 1 PID 2548: reduce_scatter(world_size=3, num_elements=104857600) took 100.66ms</span></p><p id="u2c0f90ed" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 2 PID 2550: reduce_scatter(world_size=3, num_elements=104857600) took 100.66ms</span></p><p id="u61db80c6" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 2: reduce_scatter measured bandwidth = </span><span class="ne-text" style="color: #DF2A3F">7.76 GB/s</span></p><p id="ude360c24" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 1: reduce_scatter measured bandwidth = 7.76 GB/s</span></p><p id="u871f88f6" class="ne-p"><span class="ne-text">[reduce_scatter] Rank 0: reduce_scatter measured bandwidth = 7.77 GB/s</span></p></details>
### Distributed training
**<u>Util functions</u>**

```python
def generate_sample_data():
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data

def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))

def summarize_tensor(tensor: torch.Tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"
```



**<u>Data parallelism:  Cut up along the batch dimension</u>**

```python
def data_parallelism():
    # Sharding strategy: each rank gets a slice of the data
    world_size = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=world_size, data=data, num_layers=4, num_steps=1)


def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)

    # Get the slice of data for this rank (in practice, each rank should load only its own data)
    batch_size = data.size(0)  
    num_dim = data.size(1) 
    local_batch_size = int_divide(batch_size, world_size)  
    start_index = rank * local_batch_size  
    end_index = start_index + local_batch_size  
    data = data[start_index:end_index].to(get_device(rank))

    # Create MLP parameters params[0], ..., params[num_layers - 1] (each rank has all parameters)
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  

    for step in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude

        # Backward pass
        loss.backward()

        # Sync gradients across workers (only difference between standard training and DDP)
        for layer, param in enumerate(params):
            print(f"[data_parallelism] Rank {rank}: step = {step}, layer = {layer}, grad_before_AR = {summarize_tensor(param.grad*1e4)}", flush=True)
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
            print(f"[data_parallelism] Rank {rank}: step = {step}, layer = {layer}, grad_after_AR = {summarize_tensor(param.grad*1e4)}", flush=True)

        # Update parameters
        optimizer.step()

        print(f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}", flush=True)

    cleanup()
```

```python
[data_parallelism] Rank 1: step = 0, layer = 0, grad_before_AR = 1024x1024[-0.0003...]
[data_parallelism] Rank 0: step = 0, layer = 0, grad_before_AR = 1024x1024[-0.0254...]
[data_parallelism] Rank 1: step = 0, layer = 0, grad_after_AR = 1024x1024[-0.0129...]
[data_parallelism] Rank 0: step = 0, layer = 0, grad_after_AR = 1024x1024[-0.0129...]
[data_parallelism] Rank 1: step = 0, layer = 1, grad_before_AR = 1024x1024[0.1584...]
[data_parallelism] Rank 0: step = 0, layer = 1, grad_before_AR = 1024x1024[0.0442...]
[data_parallelism] Rank 0: step = 0, layer = 1, grad_after_AR = 1024x1024[0.1013...]
[data_parallelism] Rank 1: step = 0, layer = 1, grad_after_AR = 1024x1024[0.1013...]
[data_parallelism] Rank 1: step = 0, layer = 2, grad_before_AR = 1024x1024[0.2189...]
[data_parallelism] Rank 0: step = 0, layer = 2, grad_before_AR = 1024x1024[0.1607...]
[data_parallelism] Rank 1: step = 0, layer = 2, grad_after_AR = 1024x1024[0.1898...]
[data_parallelism] Rank 0: step = 0, layer = 2, grad_after_AR = 1024x1024[0.1898...]
[data_parallelism] Rank 1: step = 0, layer = 3, grad_before_AR = 1024x1024[-0.0111...]
[data_parallelism] Rank 0: step = 0, layer = 3, grad_before_AR = 1024x1024[-0.0127...]
[data_parallelism] Rank 1: step = 0, layer = 3, grad_after_AR = 1024x1024[-0.0119...]
[data_parallelism] Rank 0: step = 0, layer = 3, grad_after_AR = 1024x1024[-0.0119...]
[data_parallelism] Rank 1: step = 0, loss = 0.012629822827875614, params = ['1024x1024[-0.0279...]', '1024x1024[-0.0299...]', '1024x1024[-0.0299...]', '1024x1024[-0.0279...]']
[data_parallelism] Rank 0: step = 0, loss = 0.012605814263224602, params = ['1024x1024[-0.0279...]', '1024x1024[-0.0299...]', '1024x1024[-0.0299...]', '1024x1024[-0.0279...]']
```



**<u>Tensor parallelism:  Cut up along the width dimension</u>**

```python
def tensor_parallelism():
    # Sharding strategy: each rank gets part of each layer, transfer all data/activations
    world_size = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=world_size, data=data, num_layers=world_size*2)


def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)

    data = data.to(get_device(rank))
    batch_size = data.size(0)  
    num_dim = data.size(1)  
    local_num_dim = int_divide(num_dim, world_size)  # Shard `num_dim`  

    # Create model (each rank gets 1/world_size of the parameters)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    # Forward pass
    x = data
    for i in range(num_layers):
        # Compute activations (batch_size x local_num_dim)
        x = x @ params[i]  # Note: this is only on a slice of the parameters
        x = F.gelu(x)
        if i == num_layers - 1:
             print(f"[tensor_parallelism] Rank {rank}: forward pass produced partial activations {summarize_tensor(x)}", flush=True)

        # Allocate memory for activations (world_size x batch_size x local_num_dim)
        activations = [torch.empty(batch_size, local_num_dim, device=get_device(rank)) for _ in range(world_size)]

        # Send activations via all gather
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        # Concatenate them to get batch_size x num_dim
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}", flush=True)

    # Backward pass: homework exercise

    cleanup()
```

<details class="lake-collapse"><summary id="u0187afa1"><span class="ne-text">tensor_parallel result</span></summary><p id="ud3610d23" class="ne-p"><span class="ne-text">[tensor_parallelism] Rank 0: forward pass produced partial activations 128x512[1.0044...]</span></p><p id="uef4d70f3" class="ne-p"><span class="ne-text">[tensor_parallelism] Rank 1: forward pass produced partial activations 128x512[1.0044...]</span></p><p id="u48f5ac9e" class="ne-p"><span class="ne-text">[tensor_parallelism] Rank 0: forward pass produced activations 128x1024[1.0044...]</span></p><p id="u71d34261" class="ne-p"><span class="ne-text">[tensor_parallelism] Rank 1: forward pass produced activations 128x1024[1.0044...]</span></p></details>


**<u>Pipeline parallelism:  Cut up along the depth dimension</u>**

```python
def pipeline_parallelism():
    # Sharding strategy: each rank gets subset of layers, transfer all data/activations
    world_size = len(os.environ.get("CUDA_VISIBLE_DEVICES").split(',')) if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    data = generate_sample_data()
    spawn(pipeline_parallelism_main, world_size=2, data=data, num_layers=4, num_micro_batches=4)


def pipeline_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_micro_batches: int):
    setup(rank, world_size)

    # Use all the data
    data = data.to(get_device(rank))
    batch_size = data.size(0)  
    num_dim = data.size(1)  

    # Split up layers
    local_num_layers = int_divide(num_layers, world_size)  

    # Each rank gets a subset of layers
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # Forward pass

    # Break up into micro batches to minimize the bubble
    micro_batch_size = int_divide(batch_size, num_micro_batches) 
    if rank == 0:
        # The data
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        # Allocate memory for activations
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank)) for _ in range(num_micro_batches)]
    # 多进程分布式训练中，额外注意分支逻辑中不要有同步点，否则会导致进程hang住

    for x in micro_batches:
        # Get activations from previous rank
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # Compute layers assigned to this rank
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # Send to the next rank
        if rank + 1 < world_size:
            print(f"[pipeline_parallelism] Rank {rank}: sending {summarize_tensor(x)} to rank {rank + 1}", flush=True)
            dist.send(tensor=x, dst=rank + 1)

    # Not handled: overlapping communication/computation to eliminate pipeline bubbles

    # Backward pass: homework exercise

    cleanup()
```

<details class="lake-collapse"><summary id="u5ab4d0b6"><span class="ne-text">pipeline_parallel results</span></summary><p id="u81b54b81" class="ne-p"><span class="ne-text">[pipeline_parallelism] Rank 0: sending 32x1024[0.3366...] to rank 1</span></p><p id="uc3a2712d" class="ne-p"><span class="ne-text">[pipeline_parallelism] Rank 0: sending 32x1024[-0.0328...] to rank 1</span></p><p id="u34b0a076" class="ne-p"><span class="ne-text">[pipeline_parallelism] Rank 0: sending 32x1024[-0.0395...] to rank 1</span></p><p id="u0a60f092" class="ne-p"><span class="ne-text">[pipeline_parallelism] Rank 0: sending 32x1024[1.0856...] to rank 1</span></p></details>
---

[collective_comm.py](https://dasellmg.yuque.com/attachments/yuque/0/2025/py/52437949/1762753923623-7b8b0577-dee4-4641-b141-ca8b94ffe6f9.py)

Runnable with base venv in /opt/miniconda3, i.e., /opt/miniconda3/bin/python

