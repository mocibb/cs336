# 作业二参考资料（整理中）

## GPU基础

**GPU基础知识**

了解GPU的体系架构。
- 什么是SM（streaming multiprocessor），一个SM中有多少CUDA核心。
- kernel中grid和block的概念。了解block内部线程和不同block间线程的区别。‌同一Block内线程‌在同一个SM上调度，有相同的共享内存，可以通过__syncthreads进行同步。
- block是通过warp在sm中调度的，一个warp由32个线程组成，每个warp有自己的程序计数器和寄存器。了解什么是warp divergence。
- GPU的内存包括，共享内存和HBM，共享内存较少速度很快，HBM速度相对慢。相同block的线程可以访问相同的共享内存。tiling技术通过共享内存用来减少对HBM访问。

**GPU优化总结**

GPU优化主要还是通过减少对HBM访问，来提升效率。

<img src="https://github.com/user-attachments/assets/5ed11434-f122-4dc5-b56c-a27b07bea1ea" alt="matmuls" width="500"/>

Triton是**block-level**的DSL语言，编译后是PTX。Triton提供了很多高级特性，比如显存合并访问，共享内存管理‌，Sm内部调度管理。

Triton通过MLIR编译成IR表示，然后再从IR编译成PTX，所以速度很快，值得学习。

可以通过print_ptx打印Triton生成的PTX代码。

```python

import torch
import os
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def print_ptx(name: str, kernel):
    if os.environ.get("TRITON_INTERPRET") == "1":
        print("PTX is not generated when in interpret mode.")
        return
    return list(kernel.cache[0].values())[0].asm["ptx"]

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=32)
    return output

x = torch.randn(128, device='cuda')
y = torch.randn(128, device='cuda')
print(torch.nn.functional.mse_loss(add(x, y), x+y) < 1e-10)
ptx = print_ptx("add_kernel", add_kernel)
print(ptx)

```

<img src="https://github.com/user-attachments/assets/b7c80a1b-14c4-480c-bcfa-a65452c90434" alt="matmuls" width="500"/>


Torch.compile优化效果很好，对于一般应用可能跟手写cuda性能相当。对想深入了解Torch.compile的同学可以参考[这里](https://www.youtube.com/watch?v=mG8TRTWs9Aw) 和 [这里](https://github.com/pytorch/workshops/tree/master/ASPLOS_2024)

这次作业，只要把作业中weighted_sum看懂就可以上手了。如果对Triton本身比较感兴趣，可以参考这里的[triton-resources](https://github.com/rkinas/triton-resources)

**了解显卡的性能参数**

获取GPU的计算峰值

1.矩阵乘法是计算密集型操作，我们通过torch.mulmat来测试。

```python
# 预热代码....

start_time = time.time()
N = 8192 * 2
a = torch.randn(N, N, device=device, dtype=dtype)
b = torch.randn(N, N, device=device, dtype=dtype)

for i in range(iterations):
    c = torch.matmul(a, b)
torch.cuda.synchronize() # 等待所有迭代的GPU操作完成
end_time = time.time()

total_time = end_time - start_time
avg_time_per_iteration = total_time / iterations

# 矩阵乘法 (C = A * B) 的计算量是 2 * N^3
flops_per_iteration = 2 * (N ** 3)
# TFLOPS = 10^12 FLOPS
tflops = flops_per_iteration / avg_time_per_iteration / 1e12

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds")
print(f"Achieved Performance: {tflops:.2f} TFLOPS ({dtype})")
```

RTX 4060Ti的测试结果，float32为15.19TFLOPS，float16为45.70TFLOPS。


2.获取GPU的带宽

通过英伟达最新的工具，测试带宽  https://github.com/nvidia/nvbandwidth

RTX 4060Ti的测试结果，CPU-GPU拷贝约为10GB/s，而GPU-GPU拷贝约为120GB/s。

GPU优化需要的基础，这部分拷贝了lecture6中slide中的推荐资料。

- [CUDA MODE Lecture 1: how to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM)
- [CUDA MODE Lecture 2: Chapters 1-3 of PPMP book](https://www.youtube.com/watch?v=NQ-0D5Ti2dc)
- [CUDA MODE Lecture 3: Getting started with CUDA for Python Programmers](https://www.youtube.com/watch?v=4sgKnKbR-WE)
- [CUDA MODE Lecture 4: Compute and memory basics](https://www.youtube.com/watch?v=lTmYrKwjSOU)
- [CUDA MODE Lecture 8: CUDA performance checklist](https://www.youtube.com/watch?v=SGhfUhlowB4)
- [HetSys Course: Lecture 1: Programming heterogenous computing systems with GPUs](https://www.youtube.com/watch?v=8JGo2zylE80)
- [HetSys Course: Lecture 2: SIMD processing and GPUs](https://www.youtube.com/watch?v=x1MA4MtO4Tc)
- [HetSys Course: Lecture 3: GPU Software Hierarchy](https://www.youtube.com/watch?v=KGZ00J5MJz0)
- [HetSys Course: Lecture 4: GPU Memory Hierarchy](https://www.youtube.com/watch?v=ZQKMZIP3Fzg)
- [HetSys Course: Lecture 5: GPU performance considerations](https://www.youtube.com/watch?v=ODeprwr3Jho)
- [A100 GPU with NVIDIA Ampere Architecture](https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
- [GPU Puzzles](https://github.com/srush/gpu-puzzles)
- [Triton Paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)
- [PyTorch 2.0 Acceleration](https://medium.com/data-science/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)
- [QuACK](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)

## benchmarking & profiling

### 参考资料

pytorch的[benchmark](https://pytorch.org/tutorials/recipes/recipes/benchmark.html) 和 [profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

### benchmarking
small尺寸下的时间(batch: 4, context_length: 256)

cuda情况 forward时间: 0.034s(0.00036s), backward时间: 0.11s(0.00029s)

cpu情况 forward时间: 0.79s, backward时间: 2.43s

关于warmup的次数，做不做看起来有很大区别，只要做了warmup次数区别并不大。

### nsys profile
1. 安装 [文档](https://developer.nvidia.com/nsight-systems)
2. 命令行 [参考文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
3. 加入nvtx的annotation可以帮助调试，作业指导上给了一个例子。
<img src="https://github.com/user-attachments/assets/ceecfd1c-ef78-43b1-871d-6931c78c1afa" alt="matmuls" width="400"/>

4. 运行 nsys profile
``` sh
uv run nsys profile  -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=none -python-backtrace=cuda --cudabacktrace=true -x true -o result python benchmark.py
```
这里选择trace cuda, nvtx和osrt(OS runtime library)
--cudabacktrace=true需要有管理员权限。

查看kernel调用Summary

<img src="https://github.com/user-attachments/assets/f244eedc-4429-498d-92c9-0309e4736d1c" alt="matmuls" width="800"/>

查看kernel调用参数

<img src="https://github.com/user-attachments/assets/bf10951d-0e7b-4537-9be4-84a2874715aa" alt="matmuls" width="800"/>


### 混合精度

### 内存优化


## Attention性能分析



## Triton


## FlashAttention-2

参考：
- 《FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning》
- 《Self-attention Does Not Need $O(n^2)$ Memory》

<img src="https://github.com/user-attachments/assets/ddcd3763-69f3-47f0-8582-3ce08ed7b9c4" alt="matmuls" width="200"/>

FlashAttention通过减少HBM和SRAM间内存搬运次数提高Attention的执行效率。


### Forward pass

<img src="https://github.com/user-attachments/assets/038096fd-59cd-4dff-ab61-34677643f596" alt="matmuls" width="600"/>

要点：

1）原始的Attention中的softmax是按照行计算的，FlashAttention中也是先行后列。关键点在于如何解决softmax的计算问题。

2）对于第i行，从左到右，

  2.1) $m_i^{(j)}$ 是到第j块为止的最大值， $l_i^{(j)}$ 是到第j块为止的sumexp(已经减掉最大值)
  
  2.2) $\tilde{O}_i^{(j)} |_j$ 表示j块为止，只减掉 $m_i^{(j)}$ 计算后的结果， $\tilde{O}_i^{(T_k)}$ 与 $\tilde{O}_i$ 只差一个 $l_i^{(T_k)}$ 组成的对角矩阵

<img src="https://github.com/user-attachments/assets/b226fd7f-44d0-4a97-9cb0-a624a9b77e15" alt="forward" width="400"/>


### Backward pass
实在不习惯论文中 $dV$ 这些写法，loss对V的导数用 $\nabla_V l$ 来表示。

#### 梯度的推导

关于V的推导

<img src="https://github.com/user-attachments/assets/289abdd4-d65b-43c6-9bef-044de6e7bc5a" alt="grad1" width="400"/>

关于S的推导

<img src="https://github.com/user-attachments/assets/9fa5286e-82fd-440a-af95-e884680549d2" alt="grad2" width="480"/>


算法

<img src="https://github.com/user-attachments/assets/f3bbf7eb-fc16-4cab-88ae-ccf1d124b723" alt="backward" width="600"/>

### 快速FA的Trick整理

- 在causal模式下考虑kernel的负载平衡，一个kernel计算Q的两行，第i行和第N-i-1行。
- 使用两趟方案计算backward梯度，避免atomic操作的开销
- 在casual掩码情况，按行循环时考虑提前中止提升性能
- 区分non-masked分块和对角分块，non-masked不使用casual掩码。


