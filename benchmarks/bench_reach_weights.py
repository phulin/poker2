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
    elif device.type == "cuda":
        torch.cuda.synchronize()


def prepare_tensors(
    B: int,
    H: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare tensors: weights [B, 2, H], policy [B, H], actor [B]"""
    # Generate deterministically on CPU
    generator = torch.Generator(device="cpu").manual_seed(42)

    weights_cpu = torch.randn(B, 2, H, dtype=dtype, generator=generator, device="cpu")
    policy_cpu = torch.rand(B, H, dtype=dtype, generator=generator, device="cpu")
    # Actor values are 0 or 1
    actor_cpu = torch.randint(
        0, 2, (B,), dtype=torch.long, generator=generator, device="cpu"
    )

    # Move to target device
    weights = weights_cpu.to(device)
    policy = policy_cpu.to(device)
    actor = actor_cpu.to(device)

    return weights, policy, actor


@torch.no_grad()
def bench_indexed_approach(
    weights: torch.Tensor,
    policy: torch.Tensor,
    actor: torch.Tensor,
    repeats: int,
) -> float:
    """Benchmark the indexed assignment approach: target_dest[indices, actor] *= policy"""
    device = weights.device
    B = weights.shape[0]

    # Fresh copy for each repeat
    weights_work = weights.clone()
    indices = torch.arange(B, device=device)

    # Warmup
    for _ in range(5):
        weights_work_copy = weights.clone()
        weights_work_copy[indices, actor] *= policy
        del weights_work_copy

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        weights_work = weights.clone()
        weights_work[indices, actor] *= policy
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(weights_work.mean().item())

    return end - start


@torch.no_grad()
def bench_scatter_reduce_approach(
    weights: torch.Tensor,
    policy: torch.Tensor,
    actor: torch.Tensor,
    repeats: int,
) -> float:
    """Benchmark using scatter_reduce_ to multiply policy values at actor positions"""
    device = weights.device
    B, H = weights.shape[0], weights.shape[2]

    # Fresh copy for each repeat
    weights_work = weights.clone()

    # Create index tensor for scatter_reduce_: [B, H] with actor indices repeated
    # We need to scatter policy values into a [B, 2, H] tensor at positions [:, actor, :]
    actor_expanded = actor.unsqueeze(-1).expand(B, H)  # [B, H]
    actor_index = actor_expanded.unsqueeze(1)  # [B, 1, H]
    policy_src = policy.unsqueeze(1)  # [B, 1, H]

    # Warmup
    for _ in range(5):
        weights_work_copy = weights.clone()
        # Scatter directly into weights_work_copy, multiplying existing values with policy
        weights_work_copy.scatter_reduce_(
            dim=1, index=actor_index, src=policy_src, reduce="prod", include_self=True
        )
        del weights_work_copy

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        weights_work = weights.clone()
        # Scatter directly into weights_work_copy, multiplying existing values with policy
        weights_work.scatter_reduce_(
            dim=1, index=actor_index, src=policy_src, reduce="prod", include_self=True
        )
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(weights_work.mean().item())

    return end - start


@torch.no_grad()
def bench_scatter_approach(
    weights: torch.Tensor,
    policy: torch.Tensor,
    actor: torch.Tensor,
    repeats: int,
) -> float:
    """Benchmark using scatter_ (non-reduce) to place policy values at actor positions"""
    device = weights.device
    B, H = weights.shape[0], weights.shape[2]

    # Fresh copy for each repeat
    weights_work = weights.clone()

    # Create index tensor for scatter_: [B, H] with actor indices repeated
    actor_expanded = actor.unsqueeze(-1).expand(B, H)  # [B, H]
    actor_index = actor_expanded.unsqueeze(1)  # [B, 1, H]
    policy_src = policy.unsqueeze(1)  # [B, 1, H]

    # Warmup
    for _ in range(5):
        weights_work_copy = weights.clone()
        # Create a tensor [B, 2, H] filled with 1s, then scatter policy values
        scatter_target = torch.ones(B, 2, H, dtype=weights.dtype, device=device)
        scatter_target.scatter_(dim=1, index=actor_index, src=policy_src)
        weights_work_copy *= scatter_target
        del weights_work_copy

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        weights_work = weights.clone()
        scatter_target = torch.ones(B, 2, H, dtype=weights.dtype, device=device)
        scatter_target.scatter_(dim=1, index=actor_index, src=policy_src)
        weights_work *= scatter_target
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(weights_work.mean().item())

    return end - start


def verify_correctness(
    weights: torch.Tensor, policy: torch.Tensor, actor: torch.Tensor
) -> None:
    """Verify that all three approaches produce the same result"""
    B = weights.shape[0]
    H = weights.shape[2]
    indices = torch.arange(B, device=weights.device)

    # Indexed approach
    result1 = weights.clone()
    result1[indices, actor] *= policy

    # Scatter reduce approach
    actor_expanded = actor.unsqueeze(-1).expand(B, H)
    actor_index = actor_expanded.unsqueeze(1)  # [B, 1, H]
    policy_src = policy.unsqueeze(1)  # [B, 1, H]
    result2 = weights.clone()
    result2.scatter_reduce_(
        dim=1, index=actor_index, src=policy_src, reduce="prod", include_self=True
    )

    # Scatter approach
    scatter_target2 = torch.ones_like(weights)
    scatter_target2.scatter_(dim=1, index=actor_index, src=policy_src)
    result3 = weights.clone()
    result3 *= scatter_target2

    max_diff_12 = (result1 - result2).abs().max().item()
    max_diff_13 = (result1 - result3).abs().max().item()

    print(f"Max |diff| indexed vs scatter_reduce: {max_diff_12:.3e}")
    print(f"Max |diff| indexed vs scatter: {max_diff_13:.3e}")


def run_for_device(
    device: torch.device, B: int, H: int, dtype: torch.dtype, repeats: int
) -> None:
    print(f"\nDevice: {device}")
    print(
        f"B: {B:,}, H: {H:,}, dtype: {str(dtype).replace('torch.', '')}, repeats: {repeats}"
    )

    weights, policy, actor = prepare_tensors(B, H, dtype, device)

    # Verify correctness
    print("Verifying correctness...")
    verify_correctness(weights, policy, actor)

    # Benchmark all approaches
    t_indexed = bench_indexed_approach(weights, policy, actor, repeats)
    t_scatter_reduce = bench_scatter_reduce_approach(weights, policy, actor, repeats)
    t_scatter = bench_scatter_approach(weights, policy, actor, repeats)

    print(
        f"\nIndexed approach:     {t_indexed:.6f} s total, {t_indexed / repeats:.6e} s/iter"
    )
    print(
        f"Scatter_reduce:       {t_scatter_reduce:.6f} s total, {t_scatter_reduce / repeats:.6e} s/iter"
    )
    print(
        f"Scatter (non-reduce):  {t_scatter:.6f} s total, {t_scatter / repeats:.6e} s/iter"
    )
    print(f"\nSpeedup indexed vs scatter_reduce: {t_scatter_reduce / t_indexed:.2f}x")
    print(f"Speedup indexed vs scatter:        {t_scatter / t_indexed:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark indexed vs scatter_reduce_ for reach weights calculation"
    )
    parser.add_argument(
        "--B", type=int, default=10000, help="Batch size (number of nodes)"
    )
    parser.add_argument("--H", type=int, default=169, help="Number of hands")
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

    run_for_device(torch.device(device_str), args.B, args.H, dtype, args.repeats)


if __name__ == "__main__":
    main()
