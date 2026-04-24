from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from p2.env.rules import compare_7_single_batch
from p2.env.rules_triton import (
    compare_7_cards_single_batch_triton,
    compare_7_single_batch_triton,
    triton_is_available,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_random_batch(
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = torch.zeros((batch_size, 2, 4, 13), dtype=torch.bool, device=device)
    cards_batch = torch.empty((batch_size, 2, 7), dtype=torch.int16, device=device)
    for row in range(batch_size):
        for player in range(2):
            cards = torch.randperm(52, generator=generator, device=device)[:7]
            cards_batch[row, player] = cards.to(torch.int16)
            batch[row, player, cards // 13, cards % 13] = True
    return batch, cards_batch


@torch.no_grad()
def verify_correctness(ab_batch: torch.Tensor, cards_batch: torch.Tensor) -> None:
    baseline = compare_7_single_batch(ab_batch)
    triton_out = compare_7_single_batch_triton(ab_batch)
    if not torch.equal(baseline.to(torch.int32), triton_out):
        mismatch = (baseline.to(torch.int32) != triton_out).nonzero(as_tuple=False)
        first = int(mismatch[0].item())
        raise AssertionError(
            f"Mismatch at row {first}: baseline={int(baseline[first].item())}, "
            f"triton={int(triton_out[first].item())}"
        )
    triton_cards_out = compare_7_cards_single_batch_triton(cards_batch)
    if not torch.equal(baseline.to(torch.int32), triton_cards_out):
        mismatch = (baseline.to(torch.int32) != triton_cards_out).nonzero(as_tuple=False)
        first = int(mismatch[0].item())
        raise AssertionError(
            f"Card-index mismatch at row {first}: baseline={int(baseline[first].item())}, "
            f"triton_cards={int(triton_cards_out[first].item())}"
        )


@torch.no_grad()
def benchmark(
    fn,
    ab_batch: torch.Tensor,
    warmup: int,
    repeats: int,
) -> tuple[float, torch.Tensor]:
    out = None
    for _ in range(warmup):
        out = fn(ab_batch)
    synchronize(ab_batch.device)

    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(ab_batch)
    synchronize(ab_batch.device)
    elapsed = time.perf_counter() - start
    assert out is not None
    return elapsed / repeats, out


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs Triton hand evaluator")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1024,8192,65536",
        help="Comma-separated batch sizes",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not triton_is_available():
        raise SystemExit("Triton is not installed in this environment.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment.")

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Warmup: {args.warmup}, repeats: {args.repeats}")

    for batch_size in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        ab_batch, cards_batch = make_random_batch(batch_size, device, args.seed + batch_size)
        verify_n = min(batch_size, 4096)
        verify_correctness(ab_batch[:verify_n], cards_batch[:verify_n])

        torch_time, torch_out = benchmark(
            compare_7_single_batch,
            ab_batch,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        triton_time, triton_out = benchmark(
            compare_7_single_batch_triton,
            ab_batch,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        triton_cards_time, triton_cards_out = benchmark(
            compare_7_cards_single_batch_triton,
            cards_batch,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        if not torch.equal(torch_out.to(torch.int32), triton_out):
            raise AssertionError(f"Final output mismatch for batch_size={batch_size}")
        if not torch.equal(torch_out.to(torch.int32), triton_cards_out):
            raise AssertionError(f"Final card-index output mismatch for batch_size={batch_size}")

        torch_hands = batch_size / torch_time
        triton_hands = batch_size / triton_time
        triton_cards_hands = batch_size / triton_cards_time
        speedup = torch_time / triton_time
        cards_speedup = torch_time / triton_cards_time

        print(
            f"batch={batch_size:>7,d} | "
            f"torch={torch_time * 1e3:>8.3f} ms ({torch_hands:>12,.0f} hands/s) | "
            f"triton={triton_time * 1e3:>8.3f} ms ({triton_hands:>12,.0f} hands/s) | "
            f"triton_cards={triton_cards_time * 1e3:>8.3f} ms "
            f"({triton_cards_hands:>12,.0f} hands/s) | "
            f"speedup={speedup:>6.2f}x | cards_speedup={cards_speedup:>6.2f}x"
        )


if __name__ == "__main__":
    main()
