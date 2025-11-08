import argparse
import time
from typing import Tuple

import torch


def synchronize_device_if_needed(device: torch.device) -> None:
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def prepare_tensors(
    M: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare [M, 2] tensor and [M] indices tensor."""
    # Generate deterministically on CPU to avoid device-specific generator constraints
    generator = torch.Generator(device="cpu").manual_seed(42)

    # Create [M, 2] tensor with random values
    x_cpu = torch.randn(M, 2, dtype=dtype, generator=generator, device="cpu")

    # Create [M] tensor with random {0, 1} indices
    indices_cpu = torch.randint(0, 2, (M,), generator=generator, device="cpu")

    # Move to target device once
    x = x_cpu.to(device)
    indices = indices_cpu.to(device)

    return x, indices


@torch.no_grad()
def bench_advanced_indexing(
    x: torch.Tensor,
    indices: torch.Tensor,
    repeats: int,
) -> float:
    """Benchmark x[arange(M), i] *= 2.0 approach."""
    device = x.device
    M = x.shape[0]

    # fresh copy per benchmark
    x_work = x.clone()

    # Warmup
    for _ in range(5):
        x_work[torch.arange(M, device=device), indices] *= 2.0

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        x_work[torch.arange(M, device=device), indices] *= 2.0
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(x_work.mean().item())

    return end - start


@torch.no_grad()
def bench_gather_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    repeats: int,
) -> float:
    """Benchmark gather-scatter approach."""
    device = x.device
    M = x.shape[0]

    # fresh copy per benchmark
    x_work = x.clone()

    # Warmup
    for _ in range(5):
        # Gather the values we want to modify
        gathered = torch.gather(x_work, 1, indices.unsqueeze(1))
        # Multiply by 2
        gathered *= 2.0
        # Scatter back
        x_work.scatter_(1, indices.unsqueeze(1), gathered)

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        # Gather the values we want to modify
        gathered = torch.gather(x_work, 1, indices.unsqueeze(1))
        # Multiply by 2
        gathered *= 2.0
        # Scatter back
        x_work.scatter_(1, indices.unsqueeze(1), gathered)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(x_work.mean().item())

    return end - start


@torch.no_grad()
def bench_gather_scatter_alternative(
    x: torch.Tensor,
    indices: torch.Tensor,
    repeats: int,
) -> float:
    """Alternative gather-scatter approach using index_select."""
    device = x.device
    M = x.shape[0]

    # fresh copy per benchmark
    x_work = x.clone()

    # Warmup
    for _ in range(5):
        # Create row indices
        row_indices = torch.arange(M, device=device)
        # Gather values
        gathered = x_work[row_indices, indices]
        # Multiply by 2
        gathered *= 2.0
        # Scatter back using advanced indexing
        x_work[row_indices, indices] = gathered

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        # Create row indices
        row_indices = torch.arange(M, device=device)
        # Gather values
        gathered = x_work[row_indices, indices]
        # Multiply by 2
        gathered *= 2.0
        # Scatter back using advanced indexing
        x_work[row_indices, indices] = gathered
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(x_work.mean().item())

    return end - start


def run_for_device(
    device: torch.device,
    M: int,
    dtype: torch.dtype,
    repeats: int,
) -> None:
    print(f"\nDevice: {device}")
    print(f"M: {M:,}, dtype: {str(dtype).replace('torch.', '')}, repeats: {repeats}")

    x, indices = prepare_tensors(M, dtype, device)

    # Run benchmarks
    t_advanced = bench_advanced_indexing(x, indices, repeats)
    t_gather_scatter = bench_gather_scatter(x, indices, repeats)
    t_gather_scatter_alt = bench_gather_scatter_alternative(x, indices, repeats)

    # Correctness check
    x1 = x.clone()
    x2 = x.clone()
    x3 = x.clone()

    M_device = x.shape[0]
    device_x = x.device

    # Advanced indexing
    x1[torch.arange(M_device, device=device_x), indices] *= 2.0

    # Gather-scatter
    gathered = torch.gather(x2, 1, indices.unsqueeze(1))
    gathered *= 2.0
    x2.scatter_(1, indices.unsqueeze(1), gathered)

    # Alternative gather-scatter
    row_indices = torch.arange(M_device, device=device_x)
    gathered_alt = x3[row_indices, indices]
    gathered_alt *= 2.0
    x3[row_indices, indices] = gathered_alt

    max_diff_12 = (x1 - x2).abs().max().item()
    max_diff_13 = (x1 - x3).abs().max().item()

    print(
        f"Advanced indexing:     {t_advanced:.6f} s total, {t_advanced / repeats:.6e} s/iter"
    )
    print(
        f"Gather-scatter:        {t_gather_scatter:.6f} s total, {t_gather_scatter / repeats:.6e} s/iter"
    )
    print(
        f"Gather-scatter alt:    {t_gather_scatter_alt:.6f} s total, {t_gather_scatter_alt / repeats:.6e} s/iter"
    )
    print(f"Speedup (adv vs gs):   {t_gather_scatter / t_advanced:.2f}x")
    print(f"Speedup (adv vs gsa):  {t_gather_scatter_alt / t_advanced:.2f}x")
    print(f"Max |diff| (adv vs gs): {max_diff_12:.3e}")
    print(f"Max |diff| (adv vs gsa): {max_diff_13:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark advanced indexing vs gather-scatter"
    )
    parser.add_argument("--M", type=int, default=1_000_000, help="Number of rows (M)")
    parser.add_argument(
        "--repeats", type=int, default=20, help="Number of timed iterations"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float64"],
        help="Tensor dtype",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]

    # CPU
    run_for_device(torch.device("cpu"), args.M, dtype, args.repeats)

    # MPS if available
    if torch.backends.mps.is_available():
        run_for_device(torch.device("mps"), args.M, dtype, args.repeats)
    else:
        print("\nMPS not available; skipping MPS benchmarks.")

    # CUDA if available
    if torch.cuda.is_available():
        run_for_device(torch.device("cuda"), args.M, dtype, args.repeats)
    else:
        print("\nCUDA not available; skipping CUDA benchmarks.")


if __name__ == "__main__":
    main()
