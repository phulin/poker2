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
    num_elements: int,
    sparsity: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Generate deterministically on CPU to avoid device-specific generator constraints
    generator = torch.Generator(device="cpu").manual_seed(42)

    a_cpu = torch.randn(num_elements, dtype=dtype, generator=generator, device="cpu")
    b_cpu = torch.zeros(num_elements, dtype=dtype, device="cpu")

    num_active = int(num_elements * (1.0 - sparsity))
    # Choose active indices deterministically on CPU
    indices_cpu = torch.randperm(num_elements, generator=generator)[:num_active]

    b_vals_cpu = torch.randn(num_active, dtype=dtype, generator=generator, device="cpu")
    b_cpu[indices_cpu] = b_vals_cpu

    # Move to target device once
    a = a_cpu.to(device)
    b = b_cpu.to(device)
    indices = indices_cpu.to(device)

    return a, b, indices


@torch.no_grad()
def bench_once(
    a: torch.Tensor,
    b: torch.Tensor,
    indices: torch.Tensor,
    repeats: int,
    mode: str,
) -> float:
    assert mode in {"full", "indexed"}
    device = a.device

    # fresh copy per mode to avoid accumulation across modes
    a_work = a.clone()

    # Warmup
    for _ in range(5):
        if mode == "full":
            a_work += b
        else:
            a_work[indices] += b[indices]

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        if mode == "full":
            a_work += b
        else:
            a_work[indices] += b[indices]
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination and ensure identical results between modes when compared externally
    _ = float(a_work.mean().item())

    return end - start


def run_for_device(
    device: torch.device,
    num_elements: int,
    sparsity: float,
    dtype: torch.dtype,
    repeats: int,
) -> None:
    print(f"\nDevice: {device}")
    print(
        f"Elements: {num_elements:,}, sparsity: {sparsity:.2f}, dtype: {str(dtype).replace('torch.', '')}, repeats: {repeats}"
    )

    a, b, indices = prepare_tensors(num_elements, sparsity, dtype, device)

    t_full = bench_once(a, b, indices, repeats=repeats, mode="full")
    t_indexed = bench_once(a, b, indices, repeats=repeats, mode="indexed")

    # Correctness sanity check
    a1 = a.clone()
    a2 = a.clone()
    a1 += b
    a2[indices] += b[indices]
    max_diff = (a1 - a2).abs().max().item()

    print(f"full add:    {t_full:.6f} s total, {t_full / repeats:.6e} s/iter")
    print(f"indexed add: {t_indexed:.6f} s total, {t_indexed / repeats:.6e} s/iter")
    print(f"max |diff| between methods: {max_diff:.3e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full vs indexed add")
    parser.add_argument(
        "--elements", type=int, default=20_000_000, help="Number of elements in arrays"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="Fraction of zeros in B (0.0 = dense, 1.0 = all zeros)",
    )
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
    run_for_device(
        torch.device("cpu"), args.elements, args.sparsity, dtype, args.repeats
    )

    # MPS if available
    if torch.backends.mps.is_available():
        run_for_device(
            torch.device("mps"), args.elements, args.sparsity, dtype, args.repeats
        )
    else:
        print("\nMPS not available; skipping MPS benchmarks.")


if __name__ == "__main__":
    main()
