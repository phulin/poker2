import argparse
import time
from typing import Tuple

import torch

from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_to_onehot_tensor,
    calculate_unblocked_mass,
)


def synchronize_device_if_needed(device: torch.device) -> None:
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.synchronize()


def prepare_tensors(
    B: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Prepare target tensor [B, NUM_HANDS] with random values."""
    # Generate deterministically on CPU
    generator = torch.Generator(device="cpu").manual_seed(42)

    target_cpu = torch.rand(
        B, NUM_HANDS, dtype=dtype, generator=generator, device="cpu"
    )
    # Normalize so each row sums to 1 (like reach weights)
    target_cpu = target_cpu / target_cpu.sum(dim=-1, keepdim=True)

    # Move to target device
    target = target_cpu.to(device)

    return target


@torch.no_grad()
def bench_current_optimized(
    target: torch.Tensor,
    repeats: int,
) -> Tuple[float, torch.Tensor]:
    """Benchmark the current optimized implementation."""
    device = target.device

    # Warmup
    for _ in range(5):
        _ = calculate_unblocked_mass(target)

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        result = calculate_unblocked_mass(target)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(result.mean().item())

    return end - start, result


@torch.no_grad()
def bench_naive_matrix_mult(
    target: torch.Tensor,
    repeats: int,
) -> Tuple[float, torch.Tensor]:
    """Benchmark using full compatibility matrix multiplication (computes matrix each time)."""
    device = target.device
    target_batched = target.view(-1, NUM_HANDS)

    # Warmup
    for _ in range(5):
        combo_onehot = combo_to_onehot_tensor(device=device).float()
        blocking_matrix = combo_onehot @ combo_onehot.T  # [NUM_HANDS, NUM_HANDS]
        eye = torch.eye(NUM_HANDS, device=device, dtype=torch.float32)
        ones = torch.ones(NUM_HANDS, NUM_HANDS, device=device, dtype=torch.float32)
        compatible_matrix = ones - blocking_matrix + eye  # [NUM_HANDS, NUM_HANDS]
        _ = (compatible_matrix @ target_batched.T).T.view_as(target).clamp(min=0.0)

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        combo_onehot = combo_to_onehot_tensor(device=device).float()
        blocking_matrix = combo_onehot @ combo_onehot.T  # [NUM_HANDS, NUM_HANDS]
        eye = torch.eye(NUM_HANDS, device=device, dtype=torch.float32)
        ones = torch.ones(NUM_HANDS, NUM_HANDS, device=device, dtype=torch.float32)
        compatible_matrix = ones - blocking_matrix + eye  # [NUM_HANDS, NUM_HANDS]
        result = (compatible_matrix @ target_batched.T).T.view_as(target).clamp(min=0.0)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(result.mean().item())

    return end - start, result


@torch.no_grad()
def bench_einsum_version(
    target: torch.Tensor,
    compatible_matrix: torch.Tensor,
    repeats: int,
) -> Tuple[float, torch.Tensor]:
    """Benchmark using einsum for matrix multiplication with precomputed matrix."""
    device = target.device
    target_batched = target.view(-1, NUM_HANDS)

    # Warmup
    for _ in range(5):
        _ = (
            torch.einsum("ij,bj->bi", compatible_matrix, target_batched)
            .view_as(target)
            .clamp(min=0.0)
        )

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        result = (
            torch.einsum("ij,bj->bi", compatible_matrix, target_batched)
            .view_as(target)
            .clamp(min=0.0)
        )
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(result.mean().item())

    return end - start, result


@torch.no_grad()
def bench_manual_optimized(
    target: torch.Tensor,
    repeats: int,
) -> Tuple[float, torch.Tensor]:
    """Benchmark manually implementing the optimized version (same as current but inline)."""
    device = target.device
    target_batched = target.view(-1, NUM_HANDS)
    combo_onehot = combo_to_onehot_tensor(device=device).float()

    # Warmup
    for _ in range(5):
        multiply = combo_onehot @ (combo_onehot.T @ target_batched.T)
        result = (
            target.sum(dim=-1, keepdim=True) - multiply.T.view_as(target) + target
        ).clamp(min=0.0)

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        multiply = combo_onehot @ (combo_onehot.T @ target_batched.T)
        result = (
            target.sum(dim=-1, keepdim=True) - multiply.T.view_as(target) + target
        ).clamp(min=0.0)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(result.mean().item())

    return end - start, result


@torch.no_grad()
def bench_precomputed_compatibility_matrix(
    target: torch.Tensor,
    compatible_matrix: torch.Tensor,
    repeats: int,
) -> Tuple[float, torch.Tensor]:
    """Benchmark using precomputed 1326x1326 compatibility matrix."""
    device = target.device
    target_batched = target.view(-1, NUM_HANDS)

    # Warmup
    for _ in range(5):
        _ = (compatible_matrix @ target_batched.T).T.view_as(target).clamp(min=0.0)

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        result = (compatible_matrix @ target_batched.T).T.view_as(target).clamp(min=0.0)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(result.mean().item())

    return end - start, result


def get_compatibility_matrix(device: torch.device) -> torch.Tensor:
    """Get the precomputed 1326x1326 compatibility matrix."""
    combo_onehot = combo_to_onehot_tensor(device=device).float()
    blocking_matrix = combo_onehot @ combo_onehot.T  # [NUM_HANDS, NUM_HANDS]
    eye = torch.eye(NUM_HANDS, device=device, dtype=torch.float32)
    ones = torch.ones(NUM_HANDS, NUM_HANDS, device=device, dtype=torch.float32)
    compatible_matrix = ones - blocking_matrix + eye  # [NUM_HANDS, NUM_HANDS]
    return compatible_matrix


def verify_correctness(target: torch.Tensor) -> None:
    """Verify that all implementations produce the same result."""
    print("Verifying correctness...")

    # Current optimized
    result_opt = calculate_unblocked_mass(target)

    # Naive matrix mult
    target_batched = target.view(-1, NUM_HANDS)
    compatible_matrix = get_compatibility_matrix(target.device)
    result_naive = (
        (compatible_matrix @ target_batched.T).T.view_as(target).clamp(min=0.0)
    )

    # Einsum
    result_einsum = (
        torch.einsum("ij,bj->bi", compatible_matrix, target_batched)
        .view_as(target)
        .clamp(min=0.0)
    )

    # Manual optimized
    combo_onehot = combo_to_onehot_tensor(device=target.device).float()
    multiply = combo_onehot @ (combo_onehot.T @ target_batched.T)
    result_manual = (
        target.sum(dim=-1, keepdim=True) - multiply.T.view_as(target) + target
    ).clamp(min=0.0)

    max_diff_opt_naive = (result_opt - result_naive).abs().max().item()
    max_diff_opt_einsum = (result_opt - result_einsum).abs().max().item()
    max_diff_opt_manual = (result_opt - result_manual).abs().max().item()

    print(f"Max |diff| optimized vs naive:   {max_diff_opt_naive:.3e}")
    print(f"Max |diff| optimized vs einsum:  {max_diff_opt_einsum:.3e}")
    print(f"Max |diff| optimized vs manual:  {max_diff_opt_manual:.3e}")

    if (
        max_diff_opt_naive > 1e-5
        or max_diff_opt_einsum > 1e-5
        or max_diff_opt_manual > 1e-5
    ):
        print("WARNING: Implementations differ by more than 1e-5!")


def run_for_device(
    device: torch.device,
    B: int,
    dtype: torch.dtype,
    repeats: int,
    tf32_enabled: bool | None = None,
) -> None:
    print(f"\nDevice: {device}")
    if device.type == "cuda" and tf32_enabled is not None:
        print(f"TF32: {'enabled' if tf32_enabled else 'disabled'}")
    print(
        f"B: {B:,}, NUM_HANDS: {NUM_HANDS:,}, dtype: {str(dtype).replace('torch.', '')}, repeats: {repeats}"
    )

    # Set TF32 if CUDA and specified
    if device.type == "cuda" and tf32_enabled is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled

    target = prepare_tensors(B, dtype, device)

    # Verify correctness (only once, not per TF32 setting)
    if tf32_enabled is None or tf32_enabled:
        verify_correctness(target)

    # Precompute compatibility matrix for precomputed versions
    compatible_matrix = get_compatibility_matrix(device)

    # Benchmark all approaches
    t_opt, _ = bench_current_optimized(target, repeats)
    t_naive, _ = bench_naive_matrix_mult(target, repeats)
    t_einsum, _ = bench_einsum_version(target, compatible_matrix, repeats)
    t_manual, _ = bench_manual_optimized(target, repeats)
    t_precomputed, _ = bench_precomputed_compatibility_matrix(
        target, compatible_matrix, repeats
    )

    print(
        f"\nCurrent optimized:        {t_opt:.6f} s total, {t_opt / repeats:.6e} s/iter"
    )
    print(
        f"Naive matrix mult:        {t_naive:.6f} s total, {t_naive / repeats:.6e} s/iter"
    )
    print(
        f"Einsum version:           {t_einsum:.6f} s total, {t_einsum / repeats:.6e} s/iter"
    )
    print(
        f"Manual optimized:         {t_manual:.6f} s total, {t_manual / repeats:.6e} s/iter"
    )
    print(
        f"Precomputed compat matrix: {t_precomputed:.6f} s total, {t_precomputed / repeats:.6e} s/iter"
    )

    print(f"\nSpeedup optimized vs naive:        {t_naive / t_opt:.2f}x")
    print(f"Speedup optimized vs einsum:       {t_einsum / t_opt:.2f}x")
    print(f"Speedup optimized vs manual:        {t_manual / t_opt:.2f}x")
    print(f"Speedup optimized vs precomputed:   {t_precomputed / t_opt:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark different implementations of calculate_unblocked_mass"
    )
    parser.add_argument(
        "--B", type=int, default=10000, help="Batch size (number of nodes)"
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="Number of timed iterations"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float64"],
        help="Tensor dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run benchmark on",
    )
    parser.add_argument(
        "--test-tf32",
        action="store_true",
        help="For CUDA, test both TF32 enabled and disabled",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    dtype = dtype_map[args.dtype]

    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available; exiting.")
        return
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return

    device = torch.device(device_str)

    if device_str == "cuda" and args.test_tf32:
        # Run with TF32 enabled
        run_for_device(device, args.B, dtype, args.repeats, tf32_enabled=True)
        # Run with TF32 disabled
        run_for_device(device, args.B, dtype, args.repeats, tf32_enabled=False)
    else:
        # Run with default TF32 setting (or not applicable for non-CUDA)
        run_for_device(device, args.B, dtype, args.repeats, tf32_enabled=None)


if __name__ == "__main__":
    main()
