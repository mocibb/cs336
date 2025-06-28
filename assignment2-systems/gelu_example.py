


from typing import Callable
import torch
from torch.profiler import ProfilerActivity
import torch.cuda.nvtx as nvtx
import triton
import triton.language as tl

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
    # Print out table
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    #text(f"## {description}")
    #text(table, verbatim=True)
    # Write stack trace visualization
    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")
    return table

def run_direct(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    
    # Run the code 
    with nvtx.range(description):
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)
    
    

def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device="cuda")
    # Return a function to perform the operation
    return lambda : operation(x)

def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

def pytorch_gelu(x: torch.Tensor):
    # Use the tanh approximation to match our implementation
    return torch.nn.functional.gelu(x, approximate="tanh")


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


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input is at `x_ptr` and output is at `y_ptr`
    #     |        Block 0            |          Block 1          |      ...      |
    #                            BLOCK_SIZE                                 num_elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # Indices where this thread block should operate
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Handle boundary
    mask = offsets < num_elements
    # Read
    x = tl.load(x_ptr + offsets, mask=mask)
    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    # Store
    tl.store(y_ptr + offsets, y, mask=mask)

compiled_gelu = torch.compile(manual_gelu)
# manual_gelu_profile = profile("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))
pytorch_gelu_profile = profile("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
compiled_gelu_profile = profile("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
triton_gelu_profile = profile("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))
# print(manual_gelu_profile)
print(pytorch_gelu_profile)
# print(compiled_gelu_profile)
print(triton_gelu_profile)

# run_direct("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))
# run_direct("compiled_gelu", run_operation1(dim=16384, operation=compiled_gelu))
# run_direct("triton_gelu", run_operation1(dim=16384, operation=triton_gelu))

