from __future__ import annotations

import argparse
import time
from typing import List

import torch

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.encoding.action_mapping import bin_to_action


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


def bench_tensor_env(N: int, iters: int, device: str) -> float:
    env = HUNLTensorEnv(
        num_envs=N,
        starting_stack=1000,
        sb=5,
        bb=10,
        bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0],
        device=torch.device(device),
    )
    env.reset(seed=123)
    B = env.num_bet_bins
    start = time.perf_counter()
    for _ in range(iters):
        mask = env.legal_action_bins_mask()
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
    args = parser.parse_args()

    print(
        f"Running benchmarks with N={args.num_envs}, iters={args.iters}, device={args.device}"
    )
    sps_tensor = bench_tensor_env(args.num_envs, args.iters, args.device)
    sps_scalar = bench_scalar_envs(args.num_envs, args.iters)
    if sps_scalar > 0:
        print(f"Speedup (Tensor/List): {sps_tensor / sps_scalar:.2f}x")


if __name__ == "__main__":
    main()
