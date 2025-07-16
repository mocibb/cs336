import cs336_basics
import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.model import scaled_dot_product_attention
import torch
import timeit   
import logging
import pandas as pd
import datetime
from contextlib import nullcontext
import torch
import os
import torch.nn as nn
import numpy as np
from cs336_systems.flash_attention import FlashAttentionTriton
import torch.cuda.nvtx as nvtx
from einops import einsum

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def compare_attention_methods(seq_len, d_model, triton, warmup_steps):
    batch_size = 1
    is_causal = True
    device = torch.device('cuda')
    
    rand_Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16, requires_grad=True)
    rand_K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16, requires_grad=True)
    rand_V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16, requires_grad=True)
    
    forward_time = []
    backward_time = []

    forward_memory = []
    backward_memory = []

    if triton:
        attention_module = lambda Q, K, V, is_causal: FlashAttentionTriton.apply(Q, K, V, is_causal)
    else:
        attention_module = Attention()
        attention_module = torch.compile(attention_module)

    name = "triton"
    if not triton:
        name = "basics"

    with nvtx.range(name):
        for i in range(warmup_steps + 100):
            torch.cuda.synchronize()
            start_time = timeit.default_timer()

            out = attention_module(rand_Q, rand_K, rand_V, is_causal)

            torch.cuda.synchronize()
            mid_time = timeit.default_timer()
            mid_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # torch.sum(out, [0,1,2]).backward()

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            end_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            if i > warmup_steps:
                backward_time.append(end_time - mid_time)
                forward_time.append(mid_time - start_time)
                forward_memory.append(mid_memory)
                backward_memory.append(end_memory)

            torch.cuda.reset_peak_memory_stats(device)

    return np.mean(forward_time) * 1000, np.mean(backward_time) * 1000, np.mean(forward_memory), np.mean(backward_memory)
    
def run_triton_benchmark():
    seq_lens = [128, 256, 1024, 4096, 8192, 16384]
    d_models = [16, 32, 64, 128]
    warmup_steps = 2

    rows = []

    for triton in [True, False]:
        for seq_len in seq_lens:
            for d_model in d_models:
                print(f"Running benchmark: seq len {seq_len} d model {d_model}  triton {triton}")
                try:
                    forward_time, backward_time, forward_memory, backward_memory = compare_attention_methods(seq_len, d_model, triton, warmup_steps)
                    rows.append({
                        'seq_len': seq_len,
                        'd_model': d_model,
                        'triton': triton,
                        'forward_time': forward_time,
                        'backward_time': backward_time,
                        'forward_memory': forward_memory,
                        'backward_memory': backward_memory
                    })
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM for seq_len={seq_len}, d_model={d_model}, triton={triton}")
                        torch.cuda.empty_cache()
                        rows.append({
                            'seq_len': seq_len,
                            'd_model': d_model,
                            'triton': triton,
                            'forward_time': None,
                            'backward_time': None,
                            'forward_memory': None,
                            'backward_memory': None
                        })
                    else:
                        raise
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False, floatfmt=".2f"))

class Attention(nn.Module):
    def forward(self, Q, K, V, is_causal):
        _, Nq, d = Q.shape
        scale = 1 / (d ** 0.5)
        P = einsum(Q, K, "... q d, ... k d -> ... q k")
        kv =  einsum(K, V, "... k d1, ... k d2 -> ... d1 d2")
        return einsum(Q, kv, "... q d, ... d1 d2 -> ... q d2")
    
if __name__ == "__main__":
    run_triton_benchmark()





