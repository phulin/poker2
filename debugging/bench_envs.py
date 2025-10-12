from __future__ import annotations

import argparse
import time
from typing import List

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from alphaholdem.encoding.action_mapping import bin_to_action
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv


def pick_bin_tensor(mask: torch.Tensor) -> torch.Tensor:
    """Greedy per-env bin picker for tensor env: all-in > check/call > first legal."""
    # mask: [N, B]
    N, B = mask.shape
    actions = torch.full((N,), 1, dtype=torch.long)  # default check/call
    allin = mask[:, B - 1]
    actions[allin] = B - 1
    # where check/call not legal, pick first True index
    need_first = ~allin & (~mask[:, 1])
    if need_first.any():
        idxs = need_first.nonzero(as_tuple=False).squeeze(1)
        sub = mask[idxs]
        first = sub.float().argmax(dim=1)
        actions[idxs] = first
    return actions


def pick_bin_scalar(env: HUNLEnv, num_bet_bins: int) -> int:
    bins = env.legal_action_bins(num_bet_bins=num_bet_bins)
    if (num_bet_bins - 1) in bins:
        return num_bet_bins - 1
    if 1 in bins:
        return 1
    return bins[0]


def bench_tensor_env(
    N: int, iters: int, device: str, profile_enabled: bool = False
) -> float:
    # Create RNG
    device_tensor = torch.device(device)
    rng = torch.Generator(device=device_tensor)
    rng.manual_seed(123)

    env = HUNLTensorEnv(
        num_envs=N,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device_tensor,
        rng=rng,
    )
    env.reset()
    B = env.num_bet_bins

    if profile_enabled:
        # Determine activities based on device
        activities = [ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        elif device == "mps":
            activities.append(ProfilerActivity.CPU)  # MPS uses CPU profiler

        print(f"\n=== PROFILING TensorEnv with device={device} ===")
        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            start = time.perf_counter()
            for i in range(iters):
                with record_function(f"iteration_{i}"):
                    mask = env.legal_bins_mask()
                    a = pick_bin_tensor(mask)
                    env.step_bins(a)
            elapsed = time.perf_counter() - start

        # Print profiling results
        print("\n--- Profiling Summary Table ---")
        try:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        except Exception as e:
            print(f"Could not generate table: {e}")

        print("\n--- Top operations by CPU time ---")
        cpu_ops = prof.key_averages()
        cpu_ops.sort(key=lambda x: x.cpu_time_total, reverse=True)
        for op in cpu_ops[:15]:
            gpu_time = getattr(op, "cuda_time_total", 0)
            print(f"{op.key}: CPU={op.cpu_time_total:.4f}ms, GPU={gpu_time:.4f}ms")

        print("\n--- Operations with high CPU overhead (potential syncs) ---")
        high_cpu_ops = [op for op in cpu_ops if op.cpu_time_total > 0.001]  # > 1ms
        high_cpu_ops.sort(key=lambda x: x.cpu_time_total, reverse=True)
        for op in high_cpu_ops[:10]:
            gpu_time = getattr(op, "cuda_time_total", 0)
            print(f"{op.key}: CPU={op.cpu_time_total:.4f}ms, GPU={gpu_time:.4f}ms")

        print("\n--- Memory operations ---")
        mem_ops = [
            op
            for op in cpu_ops
            if "memcpy" in op.key.lower() or "memory" in op.key.lower()
        ]
        for op in mem_ops[:5]:
            print(f"{op.key}: CPU={op.cpu_time_total:.4f}ms")

    else:
        start = time.perf_counter()
        for _ in range(iters):
            mask = env.legal_bins_mask()
            a = pick_bin_tensor(mask)
            env.step_bins(a)
        elapsed = time.perf_counter() - start

    env_steps = N * iters
    sps = env_steps / elapsed
    print(
        f"TensorEnv: N={N} iters={iters} -> {env_steps} steps in {elapsed:.3f}s, {sps:.0f} steps/s"
    )
    return sps


def bench_scalar_envs(N: int, iters: int) -> float:
    num_bet_bins = 8
    envs: List[HUNLEnv] = [HUNLEnv(starting_stack=1000, sb=5, bb=10) for _ in range(N)]
    for i, e in enumerate(envs):
        e.reset(seed=123 + i * 9973)
    start = time.perf_counter()
    for _ in range(iters):
        for i, e in enumerate(envs):
            s = e.state
            if s is None or s.terminal:
                e.reset(seed=123 + i * 9973)
                s = e.state
            assert s is not None
            b = pick_bin_scalar(e, num_bet_bins)
            a = bin_to_action(b, s, num_bet_bins)
            e.step(a)
    elapsed = time.perf_counter() - start
    env_steps = N * iters
    sps = env_steps / elapsed
    print(
        f"List[HUNLEnv]: N={N} iters={iters} -> {env_steps} steps in {elapsed:.3f}s, {sps:.0f} steps/s"
    )
    return sps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TensorEnv vs list of HUNLEnv"
    )
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu", help="cpu|mps|cuda")
    parser.add_argument(
        "--profile", action="store_true", help="Enable PyTorch profiler"
    )
    args = parser.parse_args()

    print(
        f"Running benchmarks with N={args.num_envs}, iters={args.iters}, device={args.device}"
    )
    if args.profile:
        print("Profiling enabled - this will show device sync points")

    sps_tensor = bench_tensor_env(args.num_envs, args.iters, args.device, args.profile)
    sps_scalar = bench_scalar_envs(args.num_envs, args.iters)
    if sps_scalar > 0:
        print(f"Speedup (Tensor/List): {sps_tensor / sps_scalar:.2f}x")


if __name__ == "__main__":
    main()
