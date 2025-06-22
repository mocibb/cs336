import argparse
import timeit
import torch
import numpy as np
import einops
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.data import get_batch


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
    dataset = np.memmap(args.dataset, dtype=np.uint16, mode='r')
    inputs, targets = get_batch(dataset, 4, args.context_length, args.device)

    # Warm-up phase
    for _ in range(args.warmup_steps):
        logits = model(inputs)
        
        loss = cross_entropy(
            einops.rearrange(logits, 'b c v -> (b c) v'),
            einops.rearrange(targets, 'b c -> (b c)')
        )

        if args.mode == "backward":
            loss.backward()
        torch.cuda.synchronize()

    # Benchmark phase
    timings = []
    for _ in range(args.num_steps):
        start_time = timeit.default_timer()
        logits = model(inputs)
        loss = cross_entropy(
            einops.rearrange(logits, 'b c v -> (b c) v'),
            einops.rearrange(targets, 'b c -> (b c)')
        )

        if args.mode == "backward":
            loss.backward()
        
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)
    
    # Report results
    avg_time = sum(timings) / len(timings)
    print(f"\nBenchmark results ({args.mode}):")
    print(f"Average step time: {avg_time:.6f} seconds")
    print(f"Total time for {args.num_steps} steps: {sum(timings):.4f} seconds")
    print(f"Timings: {[f'{t:.6f}' for t in timings]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Model Benchmark")
    parser.add_argument("--dataset", type=str, default="./data/TinyStoriesV2-GPT4-valid.npy", help="Dataset path")
    parser.add_argument("--d_model", type=int, default=1024, help="Size of model")
    parser.add_argument("--d_ff", type=int, default=4096, help="Size of feed-forward layer")
    parser.add_argument("--context_length", type=int, default=128, help="Context length")
    parser.add_argument("--num_layers", type=int, default=24, help="Number of hidden layers")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--warmup_steps", type=int, default=5, help="Warm-up iterations")
    parser.add_argument("--num_steps", type=int, default=10, help="Benchmark iterations")

    parser.add_argument("--mode", choices=["forward", "backward"], default="backward",
                        help="Benchmark mode")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")

    args = parser.parse_args()
    benchmark(args)