**<u>本节内容：</u>**

+ Benchmarking and profiling
+ kernels
    - CUDA vs. Trition in GeLU
    - Pytorch compilation
    - Triton softmax

# Benchmarking and Profiling
理解（GPU）操作运行效率：

+ Benchmarking：端到端运行时长
    - 对比同一操作不同实现的效率
    - 理解操作性能的 scalability，例如随 dim 的变化
+ Profiling：深入理解一个操作/模型背后涉及的流程（what is being called）， 整体耗时在各个阶段的分布 (where time is being spent)
    - 准备阶段，如模型初始化、kernel launch
    - 计算 vs 存储访问
    - 不同 operator 之间的分布

## Benchmarking
### 基础工具实现
```python
def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] 
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000) 

    mean_time = mean(times) 
    print(f"Benchmarking {description}: {mean_time:.3f} ms (#trail={num_trials})\n")
    return mean_time
```

**<u>GPU 操作 benchmarking 的关键在同步：</u>**

+ 大多数 GPU 操作都是异步执行
    - 当调用一个 GPU 操作（比如 `torch.matmul`），Python 程序（run on CPU）会立即返回
    - 实际的计算会被放入一个 CUDA stream 任务队列，GPU 在后台执行这些任务
+ torch.cuda.synchronize() 的作用是阻塞当前 CPU 线程，直到 GPU 上所有排队的任务全部执行完毕，从而得到准确的时间

### sleep function
```python
benchmark("sleep", lambda : time.sleep(50 / 1000))

# Benchmarking sleep: 50.441 ms (#trail=3)
```

### matmul
```python
def run_operation2(dim: int, operation: Callable, dtype=torch.float16) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, dtype=dtype, device=get_device()) 
    y = torch.randn(dim, dim, dtype=dtype, device=get_device()) 
    # Return a function to perform the operation
    return lambda : operation(x, y)


dims = (1024, 2048, 4096, 8192, 16384)

for dim in dims:
    benchmark(f"matmul(dim={dim})", 
              run_operation2(dim=dim, operation=lambda a, b: a @ b))

# Benchmarking matmul(dim=1024): 0.067 ms (#trail=3)
# Benchmarking matmul(dim=2048): 0.107 ms (#trail=3)
# Benchmarking matmul(dim=4096): 0.573 ms (#trail=3)
# Benchmarking matmul(dim=8192): 4.056 ms (#trail=3)
# Benchmarking matmul(dim=16384): 38.019 ms (#trail=3)
```

**课堂讨论：基于上述结果，float16 matmul 达到 peak FLOP/s 的 dim 在什么范围？**

<details class="lake-collapse"><summary id="u207bbb0d"><span class="ne-text">float16 matmul FLOP/s</span></summary><p id="u118963ce" class="ne-p"><span class="ne-text">Dim=1024, FLOPS = 31.8276 TFLOP/s</span></p><p id="u0dc6d667" class="ne-p"><span class="ne-text">Dim=2048, FLOPS = 160.009 TFLOP/s</span></p><p id="u15f3e19c" class="ne-p"><span class="ne-text">Dim=4096, FLOPS = 239.925 TFLOP/s</span></p><p id="u45fb6868" class="ne-p"><span class="ne-text">Dim=8192, FLOPS = 271.058 TFLOP/s</span></p><p id="u866b901a" class="ne-p"><span class="ne-text">Dim=16384, FLOPS = 231.361 TFLOP/s</span></p><p id="ub231377a" class="ne-p"><span class="ne-text">From A100 spec: FP16 Tensor Core: 312 TFLOPS</span></p><pre data-language="python" id="kbiv7" class="ne-codeblock language-python"><code>for dim, time_ms in matmul_results:
    flops = 2 * dim ** 3 / (time_ms * 0.001) / 1e12
    print(f&quot;Dim={dim}, FLOPS = {flops:.6} TFLOP/s&quot;)</code></pre><hr id="rL5yY" class="ne-hr"><p id="uda3cc687" class="ne-p"><strong><span class="ne-text">If using torch.float32</span></strong></p><p id="ua3b41a8f" class="ne-p"><span class="ne-text">Benchmarking matmul(dim=1024): 0.181 ms (#trail=3)</span></p><p id="ub1796814" class="ne-p"><span class="ne-text">Benchmarking matmul(dim=2048): 1.000 ms (#trail=3)</span></p><p id="u175733dc" class="ne-p"><span class="ne-text">Benchmarking matmul(dim=4096): 7.261 ms (#trail=3)</span></p><p id="ua629e941" class="ne-p"><span class="ne-text">Benchmarking matmul(dim=8192): 57.956 ms (#trail=3)</span></p><p id="u3432ebda" class="ne-p"><span class="ne-text">Benchmarking matmul(dim=16384): 482.757 ms (#trail=3)</span></p><p id="ue72c979c" class="ne-p"><span class="ne-text">Dim=1024, FLOPS = 11.8464 TFLOP/s</span></p><p id="ufbdcbac0" class="ne-p"><span class="ne-text">Dim=2048, FLOPS = 17.1797 TFLOP/s</span></p><p id="u969a2a5b" class="ne-p"><span class="ne-text">Dim=4096, FLOPS = 18.9293 TFLOP/s</span></p><p id="u9e07ac7a" class="ne-p"><span class="ne-text">Dim=8192, FLOPS = 18.9714 TFLOP/s</span></p><p id="u23acb3a0" class="ne-p"><span class="ne-text">Dim=16384, FLOPS = 18.2205 TFLOP/s</span></p><p id="u6c623339" class="ne-p"><span class="ne-text">From A100 spec: TF32 Tensor Core: 156 TFLOPS</span></p></details>
### MLP
```python
class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dim, dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to(get_device())

    # Define an input (random)
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()

            # Backward
            y.backward()

    return run

dim = 256  
num_layers = 4  
batch_size = 256  
num_steps = 2  

mlp_base = benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps)) 

# Benchmarking run_mlp: 2.431 ms (#trail=10)
```

```python
# Scale the number of steps.
for scale in (2, 3, 4, 5):
    result = benchmark(f"run_mlp({scale}x num_steps)", 
                       run_mlp(dim=dim, num_layers=num_layers, 
                               batch_size=batch_size, num_steps=scale * num_steps)) 

# Scale the number of layers.
for scale in (2, 3, 4, 5):
    result = benchmark(f"run_mlp({scale}x num_layers)", 
                       run_mlp(dim=dim, num_layers=scale * num_layers, 
                               batch_size=batch_size, num_steps=num_steps)) 

# Scale the batch size.
for scale in (2, 3, 4, 5):
    result = benchmark(f"run_mlp({scale}x batch_size)", 
                       run_mlp(dim=dim, num_layers=num_layers, 
                               batch_size=scale * batch_size, num_steps=num_steps)) 

# Scale the dimension.
for scale in (2, 3, 4, 5):
    result = benchmark(f"run_mlp({scale}x dim)", 
                       run_mlp(dim=scale * dim, num_layers=num_layers, 
                               batch_size=batch_size, num_steps=num_steps)) 
```

**课堂讨论：上述超参数对 mlp end-to-end 运行时间增长的顺序从高到低如何排序？**

<details class="lake-collapse"><summary id="u30f294e9"><span class="ne-text">Benchmarking results for mlps </span></summary><p id="u641e39e1" class="ne-p"><span class="ne-text">Benchmarking run_mlp: 2.431 ms (#trail=10)</span></p><p id="uecead2e2" class="ne-p"><span class="ne-text"></span></p><p id="u3d6651a0" class="ne-p"><span class="ne-text">Benchmarking run_mlp(2x num_steps): 3.256 ms (#trail=10)</span></p><p id="ue297780d" class="ne-p"><span class="ne-text">Benchmarking run_mlp(3x num_steps): 4.171 ms (#trail=10)</span></p><p id="ua46d6ace" class="ne-p"><span class="ne-text">Benchmarking run_mlp(4x num_steps): 5.548 ms (#trail=10)</span></p><p id="ubc74fe0e" class="ne-p"><span class="ne-text">Benchmarking run_mlp(5x num_steps): 7.120 ms (#trail=10)</span></p><p id="u0b98e968" class="ne-p"><span class="ne-text"></span></p><p id="u278d21e1" class="ne-p"><span class="ne-text">Benchmarking run_mlp(2x num_layers): 2.922 ms (#trail=10)</span></p><p id="uce32a8c0" class="ne-p"><span class="ne-text">Benchmarking run_mlp(3x num_layers): 5.138 ms (#trail=10)</span></p><p id="u05044496" class="ne-p"><span class="ne-text">Benchmarking run_mlp(4x num_layers): 6.195 ms (#trail=10)</span></p><p id="uce698cb4" class="ne-p"><span class="ne-text">Benchmarking run_mlp(5x num_layers): 7.113 ms (#trail=10)</span></p><p id="ucd23d450" class="ne-p"><span class="ne-text"></span></p><p id="ud32d98ec" class="ne-p"><span class="ne-text">Benchmarking run_mlp(2x batch_size): 1.447 ms (#trail=10)</span></p><p id="ue6b58391" class="ne-p"><span class="ne-text">Benchmarking run_mlp(3x batch_size): 1.415 ms (#trail=10)</span></p><p id="u49f40084" class="ne-p"><span class="ne-text">Benchmarking run_mlp(4x batch_size): 2.070 ms (#trail=10)</span></p><p id="ue8852d19" class="ne-p"><span class="ne-text">Benchmarking run_mlp(5x batch_size): 1.417 ms (#trail=10)</span></p><p id="u75153dc3" class="ne-p"><span class="ne-text"></span></p><p id="ue0dde2fb" class="ne-p"><span class="ne-text">Benchmarking run_mlp(2x dim): 1.523 ms (#trail=10)</span></p><p id="u5625b696" class="ne-p"><span class="ne-text">Benchmarking run_mlp(3x dim): 1.843 ms (#trail=10)</span></p><p id="u15c91dc6" class="ne-p"><span class="ne-text">Benchmarking run_mlp(4x dim): 1.960 ms (#trail=10)</span></p><p id="u7994a6dd" class="ne-p"><span class="ne-text">Benchmarking run_mlp(5x dim): 2.599 ms (#trail=10)</span></p></details>


**<u>Further Reading: </u>**

+ torch.utils.benchmark: [https://pytorch.org/tutorials/recipes/recipes/benchmark.html](https://pytorch.org/tutorials/recipes/recipes/benchmark.html)

## Profiling
**<u>Further Reading: </u>**

+ PyTorch profile: [https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### 基础工具实现
```python
def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # Output stack trace for visualization
            with_stack=with_stack,
            # Needed to export stack trace for visualization
            experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
         # Write stack trace visualization
    
    
    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    print(f"## {description}")
    print(table)

    if with_stack:
        text_path = f"./profiling/stacks_{description}.txt"
        svg_path = f"./profiling/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")
    
    return table
```

### sleep function
```python
sleep_function = lambda : time.sleep(50 / 1000)
sleep_profile = profile("sleep", sleep_function, with_stack=True) 
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761481641540-1374df8e-5a55-4f90-b1f2-9704583833cf.png)

### add
```python
add_function = lambda a, b: a + b
add_profile = profile("add", run_operation2(dim=2048, operation=add_function))
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761481671580-b9860fdd-ea06-4374-a7bb-38e3534cf3e2.png)

+ aten::add - PyTorch 算子的一次调度
    - Self CPU 时间反映调度、参数检查、选择后端实现以及可能的 shape 计算等主机端开销
    - CPU total 进一步包含了 kernel launch 时间
+ at::native::xxx - GPU 上真正执行的部分
    - Self CUDA 时间是 kernel 真实执行的时间，该时间也被上述 aten:add 统计

### matmul
```python
matmul_function = lambda a, b: a @ b
matmul_profile = profile("matmul", run_operation2(dim=2048, operation=matmul_function))

matmul_function_128 = lambda a, b: a @ b
matmul_profile_128 = profile("matmul(dim=128)", run_operation2(dim=128, operation=matmul_function_128))
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761481854750-fbe72460-c35e-4b66-aa6f-b8a4d1a8318b.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761481947788-f135dd29-6369-45aa-bc9c-ece4a3715843.png)

+ aten::matmul - 通用矩阵乘法接口
    - 仅分派（dot/mm/bmm_or_broadcast）
+ aten::mm - 针对二维矩阵的底层实现
    - 负责选择对应的 backend kernel 以及执行必要转换， CPU 开销大
    - Tensor dim 不同，mm 调用了不同的 kernel
+ sm80_xxx - 实际负责计算的 GPU kernel
+ cudaFuncSetAttribute - 设置 kernel 执行配置，在 kernel launch 前执行
    - 在 mm 中，通常用于设置动态共享内存大小

### composite operations
```python
cdist_function = lambda a, b: torch.cdist(a, b)
cdist_profile = profile("cdist", run_operation2(dim=2048, operation=cdist_function))

gelu_function = lambda a, b: torch.nn.functional.gelu(a + b)
gelu_profile = profile("gelu", run_operation2(dim=2048, operation=gelu_function))

softmax_function = lambda a, b: torch.nn.functional.softmax(a + b, dim=-1)
softmax_profile = profile("softmax", run_operation2(dim=2048, operation=softmax_function))
```

> torch.cdist(x1, x2, p=2.0) computes batched the p-norm distance between each pair of the two collections of row vectors.
>
> $ \| x_1 -x_2 \| = \sqrt{ \|x_1\|^2 + \|x_2\|^2 - 2 x_1^ \top x_2} $
>

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761482384702-0b59fc88-b44d-4b4c-8f6e-7ccd8b2bf00c.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761482410628-824e9743-997e-40b8-bf47-3563efed8fc9.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761482445153-e38d1e48-ad0a-40bb-a33c-576d92552a03.png)

+ cdist --> euclidean --> matmul (mm) + cat + pow + sum
+ gelu/softmax 已经融合成高效的单算子实现

### mlp
```python
if torch.cuda.is_available():
    mlp_profile = profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2), with_stack=True)
else:
    mlp_profile = profile("mlp", run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=2), with_stack=True)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761482545818-c103a057-e01c-4a07-baf7-98ec78f98fdf.png)

+ 包含多种类型的运算：operators / kernels / backward / autograd ...
+ 每种运算展示是几百次执行的聚合

### Using nvtx & NVIDIA Nsight
```python
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor):
        # Mark the entire forward pass
        for i, layer in enumerate(self.layers):
            # Mark each layer's computation separately
            with nvtx.range(f"layer_{i}"):
                x = layer(x)
                x = torch.nn.functional.gelu(x)
        
        return x

def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int, use_optimizer: bool = False):
    """Run forward and backward passes through an MLP.
    
    Args:
        dim: Dimension of each layer
        num_layers: Number of linear+GeLU layers
        batch_size: Number of samples to process at once
        num_steps: Number of forward/backward iterations
        use_optimizer: Whether to use Adam optimizer for weight updates
    """
    # Define a model (with random weights)
    with nvtx.range("define_model"):
        model = MLP(dim, num_layers).to(get_device())
    
    # Initialize optimizer if requested
    optimizer = torch.optim.Adam(model.parameters()) if use_optimizer else None

    # Define an input (random)
    with nvtx.range("define_input"):
        x = torch.randn(batch_size, dim, device=get_device())

    # Run the model `num_steps` times
    for step in range(num_steps):
        if step > 10:
            # start profiling after 10 warmup iterations
            torch.cuda.cudart().cudaProfilerStart()

        nvtx.range_push(f"step_{step}")
        
        # Zero gradients
        if use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad(set_to_none=True)

        # Forward
        with nvtx.range("forward"):
            y = model(x).mean()

        # Backward
        with nvtx.range("backward"):
            y.backward()

        # Optimizer step if enabled
        if use_optimizer:
            with nvtx.range("optimizer_step"):
                print(f"Step {step}, loss: {y.item():.6f}")
                optimizer.step()
        
        nvtx.range_pop()

def main():
    # Run a larger model if GPU is available
    if torch.cuda.is_available():
        print("Running on GPU")
        run_mlp(dim=4096, num_layers=64, batch_size=1024, num_steps=15, use_optimizer=True)
    else:
        print("Running on CPU")
        run_mlp(dim=128, num_layers=16, batch_size=128, num_steps=15, use_optimizer=True)

if __name__ == "__main__":
    main()
```

<!-- 这是一张图片，ocr 内容为： -->
![nvtx 结合 NVIDIA Nsight Systems可以给出按时间顺序的各操作耗时明细](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761493191920-2e5737dc-8a6e-4797-9b72-68f46a067f60.png)

+ CPU/GPU 独立异步执行
    - CPU 复杂发送任务，GPU 负责具体计算
    - CPU 通常先与 GPU，这种机制可以将 CPU 调度、预处理等耗时隐藏在 GPU 计算中
+ 避免不必要的 CPU-GPU 同步

# Kernels
## Kernel Fusion
Kernel fusion 通过减少 data movement 提升 GPU 计算效率，实际提升效果如何？



```python
def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")

def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
# using tf.float32
# Benchmarking manual_gelu: 14.006 ms (#trail=10)
# Benchmarking pytorch_gelu: 1.285 ms (#trail=10)

manual_gelu_profile = profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
pytorch_gelu_profile = profile("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761528996368-be279ad8-7a4a-4164-bffd-3a0349036bae.png)

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761529032431-69a774ac-8f24-4f5f-8639-1eaa58a521d0.png)

+ Mannul gelu calls 9 kernels (6 x mul + 2 x add + 1 x tanh)
+ Pytorch gelu only calls 1 kernel for all computations, 11.6x faster 

## CUDA Kernels
**<u>CUDA is an extension of C/C++ with APIs for managing GPUs.</u>**

**<u>Kernel is a "small program" that runs on the GPU</u>**

+ Write f(i), CUDA kernels computes f(i) for all i in parallel on thousands of threads
+ 一个函数体，成千上万个线程同时运行它

### GELU CUDA Kernel Implementation
```cpp
#include <math.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

__global__ void gelu_kernel(float* in, float* out, int num_elements) {
    // Get the index into the tensor
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_elements) {  // To handle the case when n < numBlocks * blockDim
        // Do the actual computation
        out[i] = 0.5 * in[i] * (1.0 + tanh(0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])));
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    // Compute ceil(a / b)
    return (a + b - 1) / b;
}

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda());
    TORCH_CHECK(x.is_contiguous());

    // Allocate empty tensor
    torch::Tensor y = torch::empty_like(x);

    // Determine grid (elements divided into blocks)
    int num_elements = x.numel();
    int block_size = 1024;  // Number of threads
    int num_blocks = cdiv(num_elements, block_size);

    // Launch the kernel
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), y.data_ptr<float>(), num_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();  // Catch errors immediately

    return y;
}
```

+ <font style="color:#AE146E;">torch::Tensor gelu(torch::Tensor x)</font> 
    - 该函数在 CPU 上执行
    -  CPU 通过 <font style="color:#AE146E;">gelu_kernel<<<num_blocks, block_size>>>() </font>通知 GPU：请在 GPU 上启动一个 kernel，创建 `num_blocks * block_size` 个线程，每个线程执行一次 `gelu_kernel()`
+ <font style="color:#AE146E;">__global__ void gelu_kernel()</font> 
    - 带 `__global__` 修饰的函数会被编译成 GPU 可执行代码（device code），而且只能由 CPU 通过 `<<< >>>` 启动语法调用
    - blockIdx/blockDim/threadIdx 是 CUDA 的内建变量（类型为 dim3），每个线程启动时都能访问到这些变量，用于识别自己是谁



上述 gelu 实现采用了 1 维的 block/thread 坐标。二维坐标的blockIdx/blockDim/threadIdx 举例：

<!-- 这是一张图片，ocr 内容为： -->
![https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761542651716-76afe11d-a4f9-4684-8b68-e4e32f748658.png)

+ Grid: collection of thread blocks: numBlocks = (2, 4)
+ Thread block / CTA: collection of threads: blockIdx = (0, 1), blockDim = (1, 8)
+ Thread: single unit of operation: threadIdx = (0, 3)

### CUDA GELU Performance
```python
def create_cuda_gelu():
    # Set CUDA_LAUNCH_BLOCKING so that if there are errors, 
    # CUDA will tell you what went wrong.
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # CUDA code: has the full logic
    cuda_gelu_src = open("gelu.cu").read()

    # C++ code: defines the gelu function
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    # Compile the CUDA code and bind it to a Python module.
    if not os.path.exists("var/cuda_gelu"):
        os.mkdir("var/cuda_gelu")
    if not torch.cuda.is_available():
        return None
    
    # "The `load_inline` function makes it convenient to write CUDA code 
    # and bind it to a Python module for immediate use.
    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        verbose=True,
        name="inline_gelu",
        build_directory="var/cuda_gelu",
    )

    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu

cuda_gelu = create_cuda_gelu()
x = manual_gelu 

# Benchmark our CUDA version.
pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) 
manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) 
if cuda_gelu is not None:
    cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu)) 
    cuda_gelu_profile = profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))

# Benchmarking pytorch_gelu: 1.529 ms (#trail=10)
# Benchmarking manual_gelu: 14.116 ms (#trail=10)
# Benchmarking cuda_gelu: 2.726 ms (#trail=10)
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761543415080-277bd966-eb8b-4abd-b33e-495cc9dcdb62.png)

**<u>讨论：</u>**

+ 上述 GELU 的 CUDA 实现比顺序执行的 manual_gelu 更高效，但相比 PyTorch 的实现还有一定差距
+ 用 CUDA 实现 elementwise 算子相对清晰简单
+ matmul/softmax/RMSNorm 等算子每个 elementwise 结果都依然很多输入，需要额外考虑 shared memory 管理

## Triton Kernels
**<u>Short history of Triton: </u>**

1. <font style="color:rgb(26, 26, 26);">Philippe Tillet </font>started the Triton compiler project in 2018 after being frustrated by the difficulty of writing auto-tuners for matrix multiplications in CUDA.
2. P.T. joined OpenAI full time in 2020 to pursue his work on the Triton compiler
3. OpenAI opensource Triton compiler in 2021
4. FlashAttention adopted Triton to speedup attention in 2022



**<u>Make GPU programming more accessible</u>**

+ Write in Python
+ Think about thread blocks rather than threads (compiler does these work)

![Compiler optimizations in CUDA vs Triton. [https://openai.com/index/triton/]](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761543817481-1d013c51-11ce-448b-bee5-fe4bfa28652e.png)

### GELU Triton Kernel Implementation
```python
# The same as CUDA implementation
def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    num_elements = x.numel()
    block_size = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, block_size)

    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)

    return y

"""
@triton.jit Triton 提供的 Just-In-Time 编译器（JIT）装饰器。
会把 Python 函数（含 Triton 张量操作）编译成 GPU 上执行的低级 kernel，
然后在运行时（runtime）自动缓存、调度和执行。
"""
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements

    # 取得当前 program 在第 0 维（1D 网格）上的 program id
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Indices where this thread block should operate
    # Triton 核心是向量化语义, 后续操作对这批 offset 批量执行（SIMD）
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Handle boundary
    # 对最后一个"block"超出位置做保护: 不会访问无效地址，计算会被屏蔽为 no-op
    mask = offsets < num_elements

    # Read: 典型是连续内存，因而自然合并访问
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(0.79788456 * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)
```

+ Triton 的并行模型
    - 启动一个 1D/2D/3D 的 program 网格（grid）
    - 每个 program 类似 CUDA 的一个 block，会（向量化）并行处理若干元素
    - `tl.program_id(axis=0)` 给出当前 program 的“块号”
+ `BLOCK_SIZE: tl.constexpr`：编译期常量，编译器会据此做循环展开、寄存器分配等优化
    - 编译时须确定，也就是针对不同的 BLOCK_SIZE 会编译不同的 kernels
+ **<font style="color:#DF2A3F;">关键：每个 program 一次性处理一个 向量（offsets），并屏蔽了 program 内部的细节</font>**

### Trition GELU Performance
```python
manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) 
pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)) 
cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=create_cuda_gelu())) 
triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) 
# Benchmarking manual_gelu: 14.000 ms (#trail=10)
# Benchmarking pytorch_gelu: 1.280 ms (#trail=10)
# Benchmarking cuda_gelu: 2.698 ms (#trail=10)
# Benchmarking triton_gelu: 1.322 ms (#trail=10)    

triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761546224626-e8087f43-db9b-426e-9637-a823be1e7f7a.png)

**<u>Trition 的 CUDA 实现：</u>**

+ 比 CUDA 实现更加高效，接近 PyTorch 实现
+ Trition 在 Blocks 上执行，这给 Triton compiler 提供更多优化空间 (e.g., thread coarsening)



**<u>PTX (Parallel Thread Execution)</u>**： NVIDIA GPU 的一种中间汇编语言，CUDA/Triton kernels 都会被编译成 PTX，再由驱动编译成 GPU 能直接执行的二进制

```python
def print_ptx(name: str, kernel):
    if os.environ.get("TRITON_INTERPRET") == "1":
        print("PTX is not generated when in interpret mode.")
        return

    """Print out the PTX code generated by Triton for the given `kernel`."""
    ptx_path = f"var/{name}-ptx.txt"

    with open(ptx_path, "w") as f:
        # return list(kernel.cache[0].values())[0].asm["ptx"]
        f.write(list(kernel.cache[0].values())[0].asm["ptx"])

# Look at the generated instructions behind triton_gelu_kernel
print_ptx("triton_gelu", triton_gelu_kernel)
```

<details class="lake-collapse"><summary id="ua7a251d2"><span class="ne-text">triton_gelu_kernel.ptx</span></summary><p id="u36d95e8f" class="ne-p"><span class="ne-text">//</span></p><p id="u4d04152d" class="ne-p"><span class="ne-text">// Generated by LLVM NVPTX Back-End</span></p><p id="ua5b04949" class="ne-p"><span class="ne-text">//</span></p><p id="u5e3a46db" class="ne-p"><span class="ne-text"></span></p><p id="uc7499ffb" class="ne-p"><span class="ne-text">.version 8.1</span></p><p id="u94745da9" class="ne-p"><span class="ne-text">.target sm_80</span></p><p id="u4e03f68c" class="ne-p"><span class="ne-text">.address_size 64</span></p><p id="u253d4acb" class="ne-p"><span class="ne-text"></span></p><p id="uc22f92a8" class="ne-p"><span class="ne-text">	// .globl	triton_gelu_kernel_0d1d2d</span></p><p id="udcedae6f" class="ne-p"><span class="ne-text"></span></p><p id="udc861c76" class="ne-p"><span class="ne-text">.visible .entry triton_gelu_kernel_0d1d2d(</span></p><p id="ue2439afd" class="ne-p"><span class="ne-text">	.param .u64 triton_gelu_kernel_0d1d2d_param_0,</span></p><p id="u42ab23df" class="ne-p"><span class="ne-text">	.param .u64 triton_gelu_kernel_0d1d2d_param_1,</span></p><p id="ud30d12c7" class="ne-p"><span class="ne-text">	.param .u32 triton_gelu_kernel_0d1d2d_param_2</span></p><p id="u4c5f5be6" class="ne-p"><span class="ne-text">)</span></p><p id="ubc168768" class="ne-p"><span class="ne-text">.maxntid 128, 1, 1</span></p><p id="ubfe9553a" class="ne-p"><span class="ne-text">{</span></p><p id="uf0e9969b" class="ne-p"><span class="ne-text">	.reg .pred 	%p&lt;5&gt;;</span></p><p id="u9be2f7eb" class="ne-p"><span class="ne-text">	.reg .b32 	%r&lt;49&gt;;</span></p><p id="u69c84f1d" class="ne-p"><span class="ne-text">	.reg .f32 	%f&lt;113&gt;;</span></p><p id="u448407d0" class="ne-p"><span class="ne-text">	.reg .b64 	%rd&lt;8&gt;;</span></p><p id="u61ed67c5" class="ne-p"><span class="ne-text">	.loc	1 356 0</span></p><p id="ud132783d" class="ne-p"><span class="ne-text">$L__func_begin0:</span></p><p id="u23a561db" class="ne-p"><span class="ne-text">	.loc	1 356 0</span></p><p id="ub91fd9fd" class="ne-p"><span class="ne-text"></span></p><p id="u1b9cea35" class="ne-p"><span class="ne-text">	ld.param.u64 	%rd5, [triton_gelu_kernel_0d1d2d_param_0];</span></p><p id="u04da6e92" class="ne-p"><span class="ne-text">	ld.param.u64 	%rd6, [triton_gelu_kernel_0d1d2d_param_1];</span></p><p id="u21caabeb" class="ne-p"><span class="ne-text">$L__tmp0:</span></p><p id="udb1da289" class="ne-p"><span class="ne-text">	.loc	1 365 41</span></p><p id="ue190c026" class="ne-p"><span class="ne-text">	mov.u32 	%r41, %tid.x;</span></p><p id="u35b4cf8f" class="ne-p"><span class="ne-text">	shl.b32 	%r42, %r41, 2;</span></p><p id="u83daf911" class="ne-p"><span class="ne-text">	ld.param.u32 	%r43, [triton_gelu_kernel_0d1d2d_param_2];</span></p><p id="ucbf091d7" class="ne-p"><span class="ne-text">	and.b32  	%r44, %r42, 508;</span></p><p id="ua9c71024" class="ne-p"><span class="ne-text">	.loc	1 361 24</span></p><p id="u812b88a8" class="ne-p"><span class="ne-text">	mov.u32 	%r45, %ctaid.x;</span></p><p id="u98a869ff" class="ne-p"><span class="ne-text">	.loc	1 362 24</span></p><p id="u1be309a3" class="ne-p"><span class="ne-text">	shl.b32 	%r46, %r45, 10;</span></p><p id="u6825e70c" class="ne-p"><span class="ne-text">	.loc	1 365 28</span></p><p id="u937393d8" class="ne-p"><span class="ne-text">	or.b32  	%r47, %r44, %r46;</span></p><p id="ucf592f41" class="ne-p"><span class="ne-text">	or.b32  	%r48, %r47, 512;</span></p><p id="u78205457" class="ne-p"><span class="ne-text">	.loc	1 368 21</span></p><p id="ueb15235c" class="ne-p"><span class="ne-text">	setp.lt.s32 	%p1, %r47, %r43;</span></p><p id="ufb170d2b" class="ne-p"><span class="ne-text">	setp.lt.s32 	%p2, %r48, %r43;</span></p><p id="u3e80aed5" class="ne-p"><span class="ne-text">	.loc	1 371 24</span></p><p id="u25800cfe" class="ne-p"><span class="ne-text">	mul.wide.s32 	%rd7, %r47, 4;</span></p><p id="u16946b1b" class="ne-p"><span class="ne-text">	add.s64 	%rd1, %rd5, %rd7;</span></p><p id="ue2cc0a70" class="ne-p"><span class="ne-text">	add.s64 	%rd2, %rd1, 2048;</span></p><p id="ubd14cfbd" class="ne-p"><span class="ne-text">	.loc	1 371 16</span></p><p id="u9c927aff" class="ne-p"><span class="ne-text">	mov.u32 %r1, 0x0;</span></p><p id="u330c58c9" class="ne-p"><span class="ne-text">	mov.u32 %r2, 0x0;</span></p><p id="u5be2fc1b" class="ne-p"><span class="ne-text">	mov.u32 %r3, 0x0;</span></p><p id="u365f21a9" class="ne-p"><span class="ne-text">	mov.u32 %r4, 0x0;</span></p><p id="u1fd9b23a" class="ne-p"><span class="ne-text">	@%p1 ld.global.v4.b32 { %r1, %r2, %r3, %r4 }, [ %rd1 + 0 ];</span></p><p id="u6f0942ff" class="ne-p"><span class="ne-text">	mov.b32 	%f17, %r1;</span></p><p id="uaa9beb7d" class="ne-p"><span class="ne-text">	mov.b32 	%f18, %r2;</span></p><p id="ua30b60c1" class="ne-p"><span class="ne-text">	mov.b32 	%f19, %r3;</span></p><p id="ub3b77dfa" class="ne-p"><span class="ne-text">	mov.b32 	%f20, %r4;</span></p><p id="uc255c5dd" class="ne-p"><span class="ne-text">	mov.u32 %r5, 0x0;</span></p><p id="uc9cb4f37" class="ne-p"><span class="ne-text">	mov.u32 %r6, 0x0;</span></p><p id="u08d428a1" class="ne-p"><span class="ne-text">	mov.u32 %r7, 0x0;</span></p><p id="udcc20b81" class="ne-p"><span class="ne-text">	mov.u32 %r8, 0x0;</span></p><p id="u4433d67a" class="ne-p"><span class="ne-text">	@%p2 ld.global.v4.b32 { %r5, %r6, %r7, %r8 }, [ %rd2 + 0 ];</span></p><p id="ueba2aad9" class="ne-p"><span class="ne-text">	mov.b32 	%f21, %r5;</span></p><p id="u24e684d2" class="ne-p"><span class="ne-text">	mov.b32 	%f22, %r6;</span></p><p id="udfc28651" class="ne-p"><span class="ne-text">	mov.b32 	%f23, %r7;</span></p><p id="u4a204af1" class="ne-p"><span class="ne-text">	mov.b32 	%f24, %r8;</span></p><p id="u6b1665b1" class="ne-p"><span class="ne-text">	.loc	1 375 37</span></p><p id="ua3162c59" class="ne-p"><span class="ne-text">	mul.f32 	%f25, %f17, 0f3D372713;</span></p><p id="u9c29b52e" class="ne-p"><span class="ne-text">	mul.f32 	%f26, %f18, 0f3D372713;</span></p><p id="ued8e1dff" class="ne-p"><span class="ne-text">	mul.f32 	%f27, %f19, 0f3D372713;</span></p><p id="u6a8116be" class="ne-p"><span class="ne-text">	mul.f32 	%f28, %f20, 0f3D372713;</span></p><p id="u059522c5" class="ne-p"><span class="ne-text">	mul.f32 	%f29, %f21, 0f3D372713;</span></p><p id="u50934ff9" class="ne-p"><span class="ne-text">	mul.f32 	%f30, %f22, 0f3D372713;</span></p><p id="u4ce578eb" class="ne-p"><span class="ne-text">	mul.f32 	%f31, %f23, 0f3D372713;</span></p><p id="ub63355b1" class="ne-p"><span class="ne-text">	mul.f32 	%f32, %f24, 0f3D372713;</span></p><p id="ub3b2697b" class="ne-p"><span class="ne-text">	.loc	1 375 41</span></p><p id="u547ee3c1" class="ne-p"><span class="ne-text">	mul.f32 	%f33, %f25, %f17;</span></p><p id="u7bf2cd87" class="ne-p"><span class="ne-text">	mul.f32 	%f34, %f26, %f18;</span></p><p id="u31511a92" class="ne-p"><span class="ne-text">	mul.f32 	%f35, %f27, %f19;</span></p><p id="u0428a33f" class="ne-p"><span class="ne-text">	mul.f32 	%f36, %f28, %f20;</span></p><p id="ua4edc2ef" class="ne-p"><span class="ne-text">	mul.f32 	%f37, %f29, %f21;</span></p><p id="u50b1cdf8" class="ne-p"><span class="ne-text">	mul.f32 	%f38, %f30, %f22;</span></p><p id="uc274f4c7" class="ne-p"><span class="ne-text">	mul.f32 	%f39, %f31, %f23;</span></p><p id="ue0507a59" class="ne-p"><span class="ne-text">	mul.f32 	%f40, %f32, %f24;</span></p><p id="u4a414a90" class="ne-p"><span class="ne-text">	.loc	1 375 26</span></p><p id="u8ccc5cff" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f41, %f33, %f17, %f17;</span></p><p id="u9a89ee72" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f42, %f34, %f18, %f18;</span></p><p id="u7eed90a8" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f43, %f35, %f19, %f19;</span></p><p id="u7a9aaea2" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f44, %f36, %f20, %f20;</span></p><p id="u02864946" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f45, %f37, %f21, %f21;</span></p><p id="u9701f802" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f46, %f38, %f22, %f22;</span></p><p id="uc171d398" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f47, %f39, %f23, %f23;</span></p><p id="u87dab0ec" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f48, %f40, %f24, %f24;</span></p><p id="u61bcd55e" class="ne-p"><span class="ne-text">	.loc	1 375 22</span></p><p id="uafa95cfb" class="ne-p"><span class="ne-text">	mul.f32 	%f49, %f41, 0f3F4C422A;</span></p><p id="u00a44cba" class="ne-p"><span class="ne-text">	mul.f32 	%f50, %f42, 0f3F4C422A;</span></p><p id="ua3999304" class="ne-p"><span class="ne-text">	mul.f32 	%f51, %f43, 0f3F4C422A;</span></p><p id="u1f32cd11" class="ne-p"><span class="ne-text">	mul.f32 	%f52, %f44, 0f3F4C422A;</span></p><p id="u4be671c9" class="ne-p"><span class="ne-text">	mul.f32 	%f53, %f45, 0f3F4C422A;</span></p><p id="ude6d1428" class="ne-p"><span class="ne-text">	mul.f32 	%f54, %f46, 0f3F4C422A;</span></p><p id="u8846d052" class="ne-p"><span class="ne-text">	mul.f32 	%f55, %f47, 0f3F4C422A;</span></p><p id="uaae4567e" class="ne-p"><span class="ne-text">	mul.f32 	%f56, %f48, 0f3F4C422A;</span></p><p id="ub15e1e9b" class="ne-p"><span class="ne-text">	.loc	1 376 21</span></p><p id="u96db534e" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f57, %f41, 0f3F4C422A, %f49;</span></p><p id="u1732f5c6" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f58, %f42, 0f3F4C422A, %f50;</span></p><p id="ue78efecb" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f59, %f43, 0f3F4C422A, %f51;</span></p><p id="ud5f941d1" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f60, %f44, 0f3F4C422A, %f52;</span></p><p id="u4d4f4e04" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f61, %f45, 0f3F4C422A, %f53;</span></p><p id="ubf6b974d" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f62, %f46, 0f3F4C422A, %f54;</span></p><p id="u48952b73" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f63, %f47, 0f3F4C422A, %f55;</span></p><p id="uf41b39b7" class="ne-p"><span class="ne-text">	fma.rn.f32 	%f64, %f48, 0f3F4C422A, %f56;</span></p><p id="udf4d52f8" class="ne-p"><span class="ne-text">	.loc	1 376 17</span></p><p id="u2b56b01b" class="ne-p"><span class="ne-text">	mul.f32 	%f2, %f57, 0f3FB8AA3B;</span></p><p id="u8d4129b3" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f1, %f2;</span></p><p id="u61bcc6b5" class="ne-p"><span class="ne-text">	mul.f32 	%f4, %f58, 0f3FB8AA3B;</span></p><p id="u1173f2ac" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f3, %f4;</span></p><p id="u787e33de" class="ne-p"><span class="ne-text">	mul.f32 	%f6, %f59, 0f3FB8AA3B;</span></p><p id="uf9e82bba" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f5, %f6;</span></p><p id="u896258c3" class="ne-p"><span class="ne-text">	mul.f32 	%f8, %f60, 0f3FB8AA3B;</span></p><p id="ufeb9addd" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f7, %f8;</span></p><p id="uf0804fe2" class="ne-p"><span class="ne-text">	mul.f32 	%f10, %f61, 0f3FB8AA3B;</span></p><p id="u9692c2f8" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f9, %f10;</span></p><p id="u20f7dda7" class="ne-p"><span class="ne-text">	mul.f32 	%f12, %f62, 0f3FB8AA3B;</span></p><p id="ue62faa90" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f11, %f12;</span></p><p id="u482808bc" class="ne-p"><span class="ne-text">	mul.f32 	%f14, %f63, 0f3FB8AA3B;</span></p><p id="u0d2b4e39" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f13, %f14;</span></p><p id="ua9d7629f" class="ne-p"><span class="ne-text">	mul.f32 	%f16, %f64, 0f3FB8AA3B;</span></p><p id="ue5773a5b" class="ne-p"><span class="ne-text">	ex2.approx.f32 %f15, %f16;</span></p><p id="u733dd9f0" class="ne-p"><span class="ne-text">	.loc	1 377 18</span></p><p id="u84601899" class="ne-p"><span class="ne-text">	add.f32 	%f65, %f1, 0fBF800000;</span></p><p id="ud1bd6142" class="ne-p"><span class="ne-text">	add.f32 	%f66, %f3, 0fBF800000;</span></p><p id="u40431340" class="ne-p"><span class="ne-text">	add.f32 	%f67, %f5, 0fBF800000;</span></p><p id="u6cf07d7e" class="ne-p"><span class="ne-text">	add.f32 	%f68, %f7, 0fBF800000;</span></p><p id="u1d4d7a29" class="ne-p"><span class="ne-text">	add.f32 	%f69, %f9, 0fBF800000;</span></p><p id="ub0fb59c4" class="ne-p"><span class="ne-text">	add.f32 	%f70, %f11, 0fBF800000;</span></p><p id="u92696c7b" class="ne-p"><span class="ne-text">	add.f32 	%f71, %f13, 0fBF800000;</span></p><p id="ue48fddb5" class="ne-p"><span class="ne-text">	add.f32 	%f72, %f15, 0fBF800000;</span></p><p id="uf19d3092" class="ne-p"><span class="ne-text">	.loc	1 377 30</span></p><p id="ucbace100" class="ne-p"><span class="ne-text">	add.f32 	%f73, %f1, 0f3F800000;</span></p><p id="ua42635fe" class="ne-p"><span class="ne-text">	add.f32 	%f74, %f3, 0f3F800000;</span></p><p id="ufc31376a" class="ne-p"><span class="ne-text">	add.f32 	%f75, %f5, 0f3F800000;</span></p><p id="u28c63bc3" class="ne-p"><span class="ne-text">	add.f32 	%f76, %f7, 0f3F800000;</span></p><p id="u9e12b895" class="ne-p"><span class="ne-text">	add.f32 	%f77, %f9, 0f3F800000;</span></p><p id="ucdc8f166" class="ne-p"><span class="ne-text">	add.f32 	%f78, %f11, 0f3F800000;</span></p><p id="uc3ed5045" class="ne-p"><span class="ne-text">	add.f32 	%f79, %f13, 0f3F800000;</span></p><p id="u399aaf30" class="ne-p"><span class="ne-text">	add.f32 	%f80, %f15, 0f3F800000;</span></p><p id="u37a18b33" class="ne-p"><span class="ne-text">	.loc	1 377 24</span></p><p id="ua391750e" class="ne-p"><span class="ne-text">	mov.b32 	%r10, %f65;</span></p><p id="u8d96a3d0" class="ne-p"><span class="ne-text">	mov.b32 	%r11, %f73;</span></p><p id="u26e1975b" class="ne-p"><span class="ne-text">	div.full.f32 %r9, %r10, %r11;</span></p><p id="ue8d87ba9" class="ne-p"><span class="ne-text">	mov.b32 	%f81, %r9;</span></p><p id="u7f3519e3" class="ne-p"><span class="ne-text">	mov.b32 	%r13, %f66;</span></p><p id="u7a1adbe4" class="ne-p"><span class="ne-text">	mov.b32 	%r14, %f74;</span></p><p id="ufb83e8ee" class="ne-p"><span class="ne-text">	div.full.f32 %r12, %r13, %r14;</span></p><p id="u0cef6482" class="ne-p"><span class="ne-text">	mov.b32 	%f82, %r12;</span></p><p id="u201d696d" class="ne-p"><span class="ne-text">	mov.b32 	%r16, %f67;</span></p><p id="u594a0902" class="ne-p"><span class="ne-text">	mov.b32 	%r17, %f75;</span></p><p id="ubc91372c" class="ne-p"><span class="ne-text">	div.full.f32 %r15, %r16, %r17;</span></p><p id="u647d30ef" class="ne-p"><span class="ne-text">	mov.b32 	%f83, %r15;</span></p><p id="u68983305" class="ne-p"><span class="ne-text">	mov.b32 	%r19, %f68;</span></p><p id="u9cd237c3" class="ne-p"><span class="ne-text">	mov.b32 	%r20, %f76;</span></p><p id="ud276284a" class="ne-p"><span class="ne-text">	div.full.f32 %r18, %r19, %r20;</span></p><p id="ubc5bf1d5" class="ne-p"><span class="ne-text">	mov.b32 	%f84, %r18;</span></p><p id="u17b686e8" class="ne-p"><span class="ne-text">	mov.b32 	%r22, %f69;</span></p><p id="u8282c7bc" class="ne-p"><span class="ne-text">	mov.b32 	%r23, %f77;</span></p><p id="ua5e88910" class="ne-p"><span class="ne-text">	div.full.f32 %r21, %r22, %r23;</span></p><p id="ue04a8db0" class="ne-p"><span class="ne-text">	mov.b32 	%f85, %r21;</span></p><p id="u559e93a3" class="ne-p"><span class="ne-text">	mov.b32 	%r25, %f70;</span></p><p id="uc1e14ec5" class="ne-p"><span class="ne-text">	mov.b32 	%r26, %f78;</span></p><p id="udbc736a8" class="ne-p"><span class="ne-text">	div.full.f32 %r24, %r25, %r26;</span></p><p id="u450b7c9c" class="ne-p"><span class="ne-text">	mov.b32 	%f86, %r24;</span></p><p id="u51b523f5" class="ne-p"><span class="ne-text">	mov.b32 	%r28, %f71;</span></p><p id="u2ce1be06" class="ne-p"><span class="ne-text">	mov.b32 	%r29, %f79;</span></p><p id="u4fe5494b" class="ne-p"><span class="ne-text">	div.full.f32 %r27, %r28, %r29;</span></p><p id="ue5d4dc56" class="ne-p"><span class="ne-text">	mov.b32 	%f87, %r27;</span></p><p id="u5b1016cf" class="ne-p"><span class="ne-text">	mov.b32 	%r31, %f72;</span></p><p id="u5ebd663f" class="ne-p"><span class="ne-text">	mov.b32 	%r32, %f80;</span></p><p id="u2732ac55" class="ne-p"><span class="ne-text">	div.full.f32 %r30, %r31, %r32;</span></p><p id="uc8d1e9c6" class="ne-p"><span class="ne-text">	mov.b32 	%f88, %r30;</span></p><p id="ub57745ce" class="ne-p"><span class="ne-text">	.loc	1 378 14</span></p><p id="u58ca0a5b" class="ne-p"><span class="ne-text">	mul.f32 	%f89, %f17, 0f3F000000;</span></p><p id="u28a5f3a8" class="ne-p"><span class="ne-text">	mul.f32 	%f90, %f18, 0f3F000000;</span></p><p id="u8ea92408" class="ne-p"><span class="ne-text">	mul.f32 	%f91, %f19, 0f3F000000;</span></p><p id="u37d00213" class="ne-p"><span class="ne-text">	mul.f32 	%f92, %f20, 0f3F000000;</span></p><p id="u7cd3b7db" class="ne-p"><span class="ne-text">	mul.f32 	%f93, %f21, 0f3F000000;</span></p><p id="uc264007a" class="ne-p"><span class="ne-text">	mul.f32 	%f94, %f22, 0f3F000000;</span></p><p id="u8c604f03" class="ne-p"><span class="ne-text">	mul.f32 	%f95, %f23, 0f3F000000;</span></p><p id="u78768aeb" class="ne-p"><span class="ne-text">	mul.f32 	%f96, %f24, 0f3F000000;</span></p><p id="ubaab37f7" class="ne-p"><span class="ne-text">	.loc	1 378 23</span></p><p id="uc1a4d879" class="ne-p"><span class="ne-text">	add.f32 	%f97, %f81, 0f3F800000;</span></p><p id="u34999f51" class="ne-p"><span class="ne-text">	add.f32 	%f98, %f82, 0f3F800000;</span></p><p id="u55056f34" class="ne-p"><span class="ne-text">	add.f32 	%f99, %f83, 0f3F800000;</span></p><p id="u399bf215" class="ne-p"><span class="ne-text">	add.f32 	%f100, %f84, 0f3F800000;</span></p><p id="ube39b981" class="ne-p"><span class="ne-text">	add.f32 	%f101, %f85, 0f3F800000;</span></p><p id="u08341ad4" class="ne-p"><span class="ne-text">	add.f32 	%f102, %f86, 0f3F800000;</span></p><p id="ueeeae9c5" class="ne-p"><span class="ne-text">	add.f32 	%f103, %f87, 0f3F800000;</span></p><p id="ua1abf27c" class="ne-p"><span class="ne-text">	add.f32 	%f104, %f88, 0f3F800000;</span></p><p id="ub9fa94ca" class="ne-p"><span class="ne-text">	.loc	1 378 19</span></p><p id="ud3bf58bb" class="ne-p"><span class="ne-text">	mul.f32 	%f105, %f89, %f97;</span></p><p id="u28c6b69b" class="ne-p"><span class="ne-text">	mul.f32 	%f106, %f90, %f98;</span></p><p id="u87192e76" class="ne-p"><span class="ne-text">	mul.f32 	%f107, %f91, %f99;</span></p><p id="u5913cc4a" class="ne-p"><span class="ne-text">	mul.f32 	%f108, %f92, %f100;</span></p><p id="u9dcf3fcb" class="ne-p"><span class="ne-text">	mul.f32 	%f109, %f93, %f101;</span></p><p id="u1ea1f2a9" class="ne-p"><span class="ne-text">	mul.f32 	%f110, %f94, %f102;</span></p><p id="u776400b3" class="ne-p"><span class="ne-text">	mul.f32 	%f111, %f95, %f103;</span></p><p id="uff1d87ef" class="ne-p"><span class="ne-text">	mul.f32 	%f112, %f96, %f104;</span></p><p id="u53869c5d" class="ne-p"><span class="ne-text">	.loc	1 381 21</span></p><p id="udc7a22c9" class="ne-p"><span class="ne-text">	add.s64 	%rd3, %rd6, %rd7;</span></p><p id="u58e9c394" class="ne-p"><span class="ne-text">	add.s64 	%rd4, %rd3, 2048;</span></p><p id="u64e259f4" class="ne-p"><span class="ne-text">	.loc	1 381 30</span></p><p id="u2c1c0389" class="ne-p"><span class="ne-text">	mov.b32 	%r33, %f105;</span></p><p id="u25314044" class="ne-p"><span class="ne-text">	mov.b32 	%r34, %f106;</span></p><p id="u2db5e36f" class="ne-p"><span class="ne-text">	mov.b32 	%r35, %f107;</span></p><p id="ub8c245ff" class="ne-p"><span class="ne-text">	mov.b32 	%r36, %f108;</span></p><p id="ubfe6115f" class="ne-p"><span class="ne-text">	@%p1 st.global.v4.b32 [ %rd3 + 0 ], { %r33, %r34, %r35, %r36 };</span></p><p id="u130b5d8c" class="ne-p"><span class="ne-text">	mov.b32 	%r37, %f109;</span></p><p id="u2c1c59f0" class="ne-p"><span class="ne-text">	mov.b32 	%r38, %f110;</span></p><p id="u25961080" class="ne-p"><span class="ne-text">	mov.b32 	%r39, %f111;</span></p><p id="ue9472592" class="ne-p"><span class="ne-text">	mov.b32 	%r40, %f112;</span></p><p id="ud7ff6de1" class="ne-p"><span class="ne-text">	@%p2 st.global.v4.b32 [ %rd4 + 0 ], { %r37, %r38, %r39, %r40 };</span></p><p id="ue43d0ac5" class="ne-p"><span class="ne-text">	.loc	1 381 4</span></p><p id="ua90223c8" class="ne-p"><span class="ne-text">	ret;</span></p><p id="u39736806" class="ne-p"><span class="ne-text">$L__tmp1:</span></p><p id="u6310b908" class="ne-p"><span class="ne-text">$L__func_end0:</span></p><p id="uce3800ac" class="ne-p"><span class="ne-text"></span></p><p id="u063f347f" class="ne-p"><span class="ne-text">}</span></p><p id="u73e58dbb" class="ne-p"><span class="ne-text">	.file	1 &quot;/ai/data/hurenjun/lmfs-2025/kernels.py&quot;</span></p><p id="u638cca69" class="ne-p"><span class="ne-text">	.section	.debug_abbrev</span></p><p id="u6f40ecbf" class="ne-p"><span class="ne-text">	{</span></p><p id="u413cbe91" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u98f59bcd" class="ne-p"><span class="ne-text">.b8 17</span></p><p id="u2e7e800d" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="ub5da25d5" class="ne-p"><span class="ne-text">.b8 37</span></p><p id="uec4939cc" class="ne-p"><span class="ne-text">.b8 8</span></p><p id="u2d72b958" class="ne-p"><span class="ne-text">.b8 19</span></p><p id="ub2bc2923" class="ne-p"><span class="ne-text">.b8 5</span></p><p id="ua3e98d31" class="ne-p"><span class="ne-text">.b8 3</span></p><p id="uea55eeee" class="ne-p"><span class="ne-text">.b8 8</span></p><p id="u151fd095" class="ne-p"><span class="ne-text">.b8 16</span></p><p id="ub6a4c0c4" class="ne-p"><span class="ne-text">.b8 6</span></p><p id="u068f7932" class="ne-p"><span class="ne-text">.b8 27</span></p><p id="uf9e01769" class="ne-p"><span class="ne-text">.b8 8</span></p><p id="u3fac1120" class="ne-p"><span class="ne-text">.b8 180</span></p><p id="u3e090c9f" class="ne-p"><span class="ne-text">.b8 66</span></p><p id="u5f7d76e3" class="ne-p"><span class="ne-text">.b8 12</span></p><p id="u22fbd3f7" class="ne-p"><span class="ne-text">.b8 17</span></p><p id="ub7eeb171" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u471412d5" class="ne-p"><span class="ne-text">.b8 18</span></p><p id="u0def178d" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u1f610fe2" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u4d7cd5bf" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="uab57587f" class="ne-p"><span class="ne-text">.b8 2</span></p><p id="ubc998345" class="ne-p"><span class="ne-text">.b8 46</span></p><p id="u5e36e6ec" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u9762349a" class="ne-p"><span class="ne-text">.b8 17</span></p><p id="ub0fdee45" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="ue73c5bca" class="ne-p"><span class="ne-text">.b8 18</span></p><p id="uc0a2a4ed" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u06b7d276" class="ne-p"><span class="ne-text">.b8 64</span></p><p id="u4bc51d39" class="ne-p"><span class="ne-text">.b8 10</span></p><p id="u5be14bdc" class="ne-p"><span class="ne-text">.b8 135</span></p><p id="u676d20e7" class="ne-p"><span class="ne-text">.b8 64</span></p><p id="uda354988" class="ne-p"><span class="ne-text">.b8 8</span></p><p id="u24265a7a" class="ne-p"><span class="ne-text">.b8 3</span></p><p id="u4bf50983" class="ne-p"><span class="ne-text">.b8 8</span></p><p id="ucbbb3f1f" class="ne-p"><span class="ne-text">.b8 58</span></p><p id="u1a68d46a" class="ne-p"><span class="ne-text">.b8 11</span></p><p id="u84e7435b" class="ne-p"><span class="ne-text">.b8 59</span></p><p id="u4ad54c83" class="ne-p"><span class="ne-text">.b8 5</span></p><p id="u4e55d2c1" class="ne-p"><span class="ne-text">.b8 63</span></p><p id="udafc2fc0" class="ne-p"><span class="ne-text">.b8 12</span></p><p id="u415fd5d0" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u9c68caa8" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u9dee01f8" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="uebe0e190" class="ne-p"><span class="ne-text">	}</span></p><p id="ue4b3d54f" class="ne-p"><span class="ne-text">	.section	.debug_info</span></p><p id="ube5736cf" class="ne-p"><span class="ne-text">	{</span></p><p id="uff73bf7e" class="ne-p"><span class="ne-text">.b32 153</span></p><p id="ud4b0d8c6" class="ne-p"><span class="ne-text">.b8 2</span></p><p id="u258678dd" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="ufe61f658" class="ne-p"><span class="ne-text">.b32 .debug_abbrev</span></p><p id="ued390f7b" class="ne-p"><span class="ne-text">.b8 8</span></p><p id="u7c288067" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="uafbc30f1" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="u50a53918" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="u8d99d035" class="ne-p"><span class="ne-text">.b8 105</span></p><p id="u023d1904" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="u5edbddb3" class="ne-p"><span class="ne-text">.b8 111</span></p><p id="u711cee05" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="u0c2c2d1a" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u92127c61" class="ne-p"><span class="ne-text">.b8 2</span></p><p id="u5ba223ae" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u5231d4a9" class="ne-p"><span class="ne-text">.b8 107</span></p><p id="ud47cbb8c" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="ucba78113" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="u84d1eba2" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="ud73ec48f" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="u181b4f98" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="u99e68f57" class="ne-p"><span class="ne-text">.b8 115</span></p><p id="u1d306342" class="ne-p"><span class="ne-text">.b8 46</span></p><p id="ub06694a7" class="ne-p"><span class="ne-text">.b8 112</span></p><p id="uf26c23e3" class="ne-p"><span class="ne-text">.b8 121</span></p><p id="ud8a206fc" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="udfa65e6b" class="ne-p"><span class="ne-text">.b32 .debug_line</span></p><p id="udd98b6b8" class="ne-p"><span class="ne-text">.b8 47</span></p><p id="uef634a47" class="ne-p"><span class="ne-text">.b8 97</span></p><p id="u75086afb" class="ne-p"><span class="ne-text">.b8 105</span></p><p id="u68fb5d6d" class="ne-p"><span class="ne-text">.b8 47</span></p><p id="u40511d63" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="uebbf511c" class="ne-p"><span class="ne-text">.b8 97</span></p><p id="u0487fe70" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="u28994b76" class="ne-p"><span class="ne-text">.b8 97</span></p><p id="uc36373d2" class="ne-p"><span class="ne-text">.b8 47</span></p><p id="ud2211e4e" class="ne-p"><span class="ne-text">.b8 104</span></p><p id="u89c0a2fa" class="ne-p"><span class="ne-text">.b8 117</span></p><p id="uaf1cc74c" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="uab69166e" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="u0e969cca" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="u6036b2d0" class="ne-p"><span class="ne-text">.b8 106</span></p><p id="u92bb9f57" class="ne-p"><span class="ne-text">.b8 117</span></p><p id="u473eed70" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="u742d788e" class="ne-p"><span class="ne-text">.b8 47</span></p><p id="ude52cf6a" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="ub95a5767" class="ne-p"><span class="ne-text">.b8 109</span></p><p id="uc9ae1464" class="ne-p"><span class="ne-text">.b8 102</span></p><p id="u76b1cdc2" class="ne-p"><span class="ne-text">.b8 115</span></p><p id="u5f779e37" class="ne-p"><span class="ne-text">.b8 45</span></p><p id="ud4a8c9f9" class="ne-p"><span class="ne-text">.b8 50</span></p><p id="u2fdfddd8" class="ne-p"><span class="ne-text">.b8 48</span></p><p id="u1004758f" class="ne-p"><span class="ne-text">.b8 50</span></p><p id="uba11f973" class="ne-p"><span class="ne-text">.b8 53</span></p><p id="u679275f5" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="uc992bbda" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u687dd646" class="ne-p"><span class="ne-text">.b64 $L__func_begin0</span></p><p id="u4b3e79fa" class="ne-p"><span class="ne-text">.b64 $L__func_end0</span></p><p id="ube1a27d2" class="ne-p"><span class="ne-text">.b8 2</span></p><p id="u6c35cf8e" class="ne-p"><span class="ne-text">.b64 $L__func_begin0</span></p><p id="u53f461ea" class="ne-p"><span class="ne-text">.b64 $L__func_end0</span></p><p id="uce5693ae" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u26235b6c" class="ne-p"><span class="ne-text">.b8 156</span></p><p id="u4c57d761" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="u1837ac36" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="u96bc9c47" class="ne-p"><span class="ne-text">.b8 105</span></p><p id="u8cd4d8f3" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="ud0441eef" class="ne-p"><span class="ne-text">.b8 111</span></p><p id="uc804731f" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="uc52f7b2d" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="ud8bcfc80" class="ne-p"><span class="ne-text">.b8 103</span></p><p id="u4a599e7d" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="uc08380b8" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="ud193e7b8" class="ne-p"><span class="ne-text">.b8 117</span></p><p id="u5749134a" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="u6c22f579" class="ne-p"><span class="ne-text">.b8 107</span></p><p id="ueef59ead" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="u04ca234c" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="ubc42b9f6" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="u7489a506" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="u6848915e" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="u6664cb23" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="u909b8a95" class="ne-p"><span class="ne-text">.b8 48</span></p><p id="ub06c1bcf" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="ua24e40c8" class="ne-p"><span class="ne-text">.b8 49</span></p><p id="u968e1fc9" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="ue2593639" class="ne-p"><span class="ne-text">.b8 50</span></p><p id="u26c21db5" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="ud6e32b8c" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="ua72f9e51" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="ua3b266e0" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="uedf1fac8" class="ne-p"><span class="ne-text">.b8 105</span></p><p id="u6ab3ac51" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="u16eaa6f4" class="ne-p"><span class="ne-text">.b8 111</span></p><p id="uc8931e4a" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="u087ed70f" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="ud9a225b1" class="ne-p"><span class="ne-text">.b8 103</span></p><p id="u96830d7c" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="ubb848d4c" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="u4233c7c1" class="ne-p"><span class="ne-text">.b8 117</span></p><p id="u32380086" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="ua4af964f" class="ne-p"><span class="ne-text">.b8 107</span></p><p id="u81d02976" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="ubfa7f45f" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="u7441e9cd" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="u04837af1" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="ub716699a" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="uc3012ca2" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="ua1e1223b" class="ne-p"><span class="ne-text">.b8 48</span></p><p id="u3e337f44" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="u88ffc7a7" class="ne-p"><span class="ne-text">.b8 49</span></p><p id="u190c2776" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="u9fd1dde3" class="ne-p"><span class="ne-text">.b8 50</span></p><p id="u604ccef5" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="u765475cc" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u4ba64300" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u500a6dc4" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="ude2ee4db" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="ubebdda5c" class="ne-p"><span class="ne-text">.b8 1</span></p><p id="u11abff28" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u20bcbbaa" class="ne-p"><span class="ne-text">	}</span></p><p id="u66d159ad" class="ne-p"><span class="ne-text">	.section	.debug_pubnames</span></p><p id="ua04d4fa0" class="ne-p"><span class="ne-text">	{</span></p><p id="u860615ca" class="ne-p"><span class="ne-text">.b32 $L__pubNames_end0-$L__pubNames_start0</span></p><p id="u42949012" class="ne-p"><span class="ne-text">$L__pubNames_start0:</span></p><p id="u0b8ceb75" class="ne-p"><span class="ne-text">.b8 2</span></p><p id="ub8cda234" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u9ba28b7a" class="ne-p"><span class="ne-text">.b32 .debug_info</span></p><p id="ud59f20e2" class="ne-p"><span class="ne-text">.b32 157</span></p><p id="ud6f530b2" class="ne-p"><span class="ne-text">.b32 81</span></p><p id="u76131a84" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="u67cf4af8" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="ue468eb78" class="ne-p"><span class="ne-text">.b8 105</span></p><p id="ucf9c0ec6" class="ne-p"><span class="ne-text">.b8 116</span></p><p id="u0970cda5" class="ne-p"><span class="ne-text">.b8 111</span></p><p id="u3822b7ca" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="uffd8e436" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="u18a118fb" class="ne-p"><span class="ne-text">.b8 103</span></p><p id="u69e58af1" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="u539cff2e" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="ufdf65e2d" class="ne-p"><span class="ne-text">.b8 117</span></p><p id="uf7a1f250" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="uf1adbf4e" class="ne-p"><span class="ne-text">.b8 107</span></p><p id="u1d1205fa" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="u6fce28ec" class="ne-p"><span class="ne-text">.b8 114</span></p><p id="u74c47b5b" class="ne-p"><span class="ne-text">.b8 110</span></p><p id="uec3368b7" class="ne-p"><span class="ne-text">.b8 101</span></p><p id="u16600af3" class="ne-p"><span class="ne-text">.b8 108</span></p><p id="uc96e4ee3" class="ne-p"><span class="ne-text">.b8 95</span></p><p id="ue85be169" class="ne-p"><span class="ne-text">.b8 48</span></p><p id="u6da4cdad" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="u7d253c50" class="ne-p"><span class="ne-text">.b8 49</span></p><p id="u0096f161" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="ua0106fa7" class="ne-p"><span class="ne-text">.b8 50</span></p><p id="ue406ca06" class="ne-p"><span class="ne-text">.b8 100</span></p><p id="ufab215fa" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="uad2b2167" class="ne-p"><span class="ne-text">.b32 0</span></p><p id="ub54b272f" class="ne-p"><span class="ne-text">$L__pubNames_end0:</span></p><p id="u99d646d3" class="ne-p"><span class="ne-text">	}</span></p><p id="u72eb5c7d" class="ne-p"><span class="ne-text">	.section	.debug_pubtypes</span></p><p id="u698346dd" class="ne-p"><span class="ne-text">	{</span></p><p id="u69073ad6" class="ne-p"><span class="ne-text">.b32 $L__pubTypes_end0-$L__pubTypes_start0</span></p><p id="u545fdd03" class="ne-p"><span class="ne-text">$L__pubTypes_start0:</span></p><p id="u70b34167" class="ne-p"><span class="ne-text">.b8 2</span></p><p id="udd6aa15e" class="ne-p"><span class="ne-text">.b8 0</span></p><p id="u57d1baf2" class="ne-p"><span class="ne-text">.b32 .debug_info</span></p><p id="u955dde34" class="ne-p"><span class="ne-text">.b32 157</span></p><p id="u55886498" class="ne-p"><span class="ne-text">.b32 0</span></p><p id="u3c67230f" class="ne-p"><span class="ne-text">$L__pubTypes_end0:</span></p><p id="u885709f8" class="ne-p"><span class="ne-text">	}</span></p><p id="ue438cc12" class="ne-p"><span class="ne-text">	.section	.debug_loc	{	}</span></p><p id="ud102f07c" class="ne-p"><span class="ne-text"></span></p></details>
+ 从 PTX 看，triton_gelu_kernel 中一个线程同时负责 8 个元素的计算 (thread coarsening)

## Pytorch Compilation
Manual pytorch implementation 执行低效 ==> 在 PyTorch 层面提供统一优化的 compiler 进行算子优化

+ torch.compile() 
+ Triton 是 `torch.compile` 的 GPU 后端引擎之一
+ 整个 `torch.compile` 管线包含多阶段优化，如Python 图层（TorchDynamo）、中间优化层（Inductor）、GPU 编译层（Triton）、驱动层（CUDA PTX → SASS）等

```python
compiled_gelu = torch.compile(manual_gelu)

if not torch.cuda.is_available():
    return

manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu)) 
triton_time = benchmark("triton_gelu", run_operation1(dim=16384, operation=triton_gelu)) 
compiled_time = benchmark("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu)) 
# Benchmarking manual_gelu: 14.000 ms (#trail=10)
# Benchmarking triton_gelu: 1.328 ms (#trail=10)
# Benchmarking compiled_gelu: 1.336 ms (#trail=10)

# Let's look under the hood
compiled_gelu_profile = profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
```

<!-- 这是一张图片，ocr 内容为： -->
![](https://cdn.nlark.com/yuque/0/2025/png/52437949/1761564259125-daee450e-c805-4639-b144-865008434844.png)

## Triton Softmax
> [https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
>
> And many operations implemented in Triton.
>

最后分析一个非 elementwise 的算子：Softmax

+ 考虑矩阵上的 rowwise softmax：每一行进行正则

### Softmax 实现：naive & triton
```python
def manual_softmax(x: torch.Tensor):
    # M: number of rows, N: number of columns
    M, N = x.shape

    # Compute the max of each row (MN reads, M writes)
    x_max = x.max(dim=1)[0]

    # Subtract off the max (MN + M reads, MN writes)
    x = x - x_max[:, None]

    # Exponentiate (MN reads, MN writes)
    numerator = torch.exp(x)

    # Compute normalization constant (MN reads, M writes)
    denominator = numerator.sum(dim=1)

    # Normalize (MN + M reads, MN writes)
    y = numerator / denominator[:, None]

    # Total: 5MN + 2M reads, 3MN + 2M writes
    # In principle, should have MN reads, MN writes (speedup of 4x!)
    return y
```



思考：如果你来实现 triton softmax，会怎么分 block？

```python
def triton_softmax(x: torch.Tensor):
    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid
    M, N = x.shape                          # Number of rows x number of columns
    block_size = triton.next_power_of_2(N)  # Each block contains all the columns
    num_blocks = M                          # Each block is a row

    # Launch kernel
    triton_softmax_kernel[(num_blocks,)](
        x_ptr=x, y_ptr=y,
        x_row_stride=x.stride(0), y_row_stride=y.stride(0),
        num_cols=N, BLOCK_SIZE=block_size
    )

    return y


@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)
```

### Softmax 性能对比
```python
compiled_softmax = torch.compile(manual_softmax)

benchmark("manual_softmax", run_operation1(dim=16384, operation=manual_softmax))
benchmark("compiled_softmax", run_operation1(dim=16384, operation=compiled_softmax))
benchmark("pytorch_softmax", run_operation1(dim=16384, operation=pytorch_softmax))
benchmark("triton_softmax", run_operation1(dim=16384, operation=triton_softmax))

# Benchmarking manual_softmax: 5.453 ms (#trail=10)
# Benchmarking compiled_softmax: 2.047 ms (#trail=10)
# Benchmarking pytorch_softmax: 2.527 ms (#trail=10)
# Benchmarking triton_softmax: 1.384 ms (#trail=10)
```

# Summary
**<u>Understanding the performance of your operators/models</u>**

+ Benchmarking: end-to-end wall clock time，理解 scaling
+ Profiling: diving deep into what is being call behind and how time is spent，理解 internals
    - CPU 和 GPU 的异步执行与分工：将 GPU 计算的额外开销隐藏在 GPU 计算中

**<u>Kernels: a "small program" that runs on the GPU</u>**

+ Kernel fusion: 10x speed for GeLU
+ CUDA kernels: 可以操作最底层线程（Grid/Block/Thread），复杂算子需要考虑 memory management
+ Triton Kernels：只在 Block 层编程，memory management 以及线程并行等由 triton compiler 负责优化
+ torch.compiler：底层依赖 triton，针对 operator fusion 和 matmul 已经足够

**<u>Take-away principles</u>**

+ Organize computation to minimize reads/writes
+ Kernel fusion + tiling, and recomputation memory-bound immediate results
+ Using automatic compilers



---

[kernels.py](https://dasellmg.yuque.com/attachments/yuque/0/2025/py/52437949/1761816145953-ace10b7f-63ec-4d2f-b21a-b44ad0e27d01.py)

[nvtx_profile_mlp.py](https://dasellmg.yuque.com/attachments/yuque/0/2025/py/52437949/1761816145779-ef74d706-0e9d-42c4-99e0-f5a8ff6db2c3.py)

[torch_util.py](https://dasellmg.yuque.com/attachments/yuque/0/2025/py/52437949/1761816145769-44b302a1-6eed-4760-8a92-d065a44981f6.py)

[gelu.cpp](https://dasellmg.yuque.com/attachments/yuque/0/2025/cpp/52437949/1761816179756-701692c0-7738-4c6f-aff8-c96d0514a1c4.cpp)

