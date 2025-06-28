import argparse
import timeit
import torch
import numpy as np
import einops
import math
from cs336_basics import model 
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy, softmax
from cs336_basics.data import get_batch
import torch.cuda.nvtx as nvtx


def get_data(batch_size: int, context_length: int, device: str):
    data = torch.randint(10000, (batch_size, context_length+1), dtype=torch.int64, device=device)
    return data[:, :-1], data[:, 1:]

def benchmark(args):
    # Set device
    device = torch.device(args.device)
    
    # Initialize model
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000,
    ).to(device)

    # Loss and optimizer    
    # Generate random data
    inputs, targets = get_data(4, args.context_length, args.device)

    # Warm-up phase
    for _ in range(args.warmup_steps):
        logits = model(inputs)
        
        loss = cross_entropy(
            einops.rearrange(logits, 'b c v -> (b c) v'),
            einops.rearrange(targets, 'b c -> (b c)')
        )
        loss.backward()
        torch.cuda.synchronize()

    # Benchmark phase
    timings = []
    if args.mode == "forward":
        with nvtx.range("forward"):
            for i in range(args.num_steps):
                with nvtx.range(f"it{i}"):
                    start_time = timeit.default_timer()
                    logits = model(inputs)
                    end_time = timeit.default_timer()
                    timings.append(end_time - start_time)
    else:
        with nvtx.range("backward"):
            for i in range(args.num_steps):
                with nvtx.range(f"it{i}"):
                    start_time = timeit.default_timer()
                    logits = model(inputs)
                    loss = cross_entropy(
                        einops.rearrange(logits, 'b c v -> (b c) v'),
                        einops.rearrange(targets, 'b c -> (b c)')
                    )

                    loss.backward()
                    torch.cuda.synchronize()
                    end_time = timeit.default_timer()
                    timings.append(end_time - start_time)
    
    # Report results
    print(f"\nBenchmark results ({args.mode}):")
    print(f"step time: {np.mean(timings):.6f}, std: {np.std(timings):.6f}")
    print(f"Total time for {args.num_steps} steps: {sum(timings):.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Benchmark")
    parser.add_argument("--dataset", type=str, default="./data/TinyStoriesV2-GPT4-valid.npy", help="Dataset path")
    parser.add_argument("--d_model", type=int, default=768, help="Size of model")
    parser.add_argument("--d_ff", type=int, default=3072, help="Size of feed-forward layer")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warm-up iterations")
    parser.add_argument("--num_steps", type=int, default=10, help="Benchmark iterations")

    parser.add_argument("--mode", choices=["forward", "backward"], default="forward",
                        help="Benchmark mode")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()
    benchmark(args)
