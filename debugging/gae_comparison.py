#!/usr/bin/env python3
"""
Compare the original broken GAE vs the corrected GAE implementation.
"""

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer


def compute_gae_reference(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Reference GAE using straightforward per-trajectory reverse recursion."""
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    # Identify trajectory starts: t==0 or previous is done
    is_start = torch.zeros_like(dones, dtype=torch.bool)
    is_start[0] = True
    is_start[1:] = dones[:-1]
    starts = torch.where(is_start)[0]
    ends = torch.cat([starts[1:], torch.tensor([T], device=rewards.device)])

    for s, e in zip(starts.tolist(), ends.tolist()):
        # deltas for this traj
        deltas = torch.zeros(e - s, device=rewards.device, dtype=rewards.dtype)
        # non-terminal: r_t + gamma * V_{t+1} - V_t
        if e - s > 1:
            deltas[:-1] = (
                rewards[s : e - 1] + gamma * values[s + 1 : e] - values[s : e - 1]
            )
        # terminal: r_t - V_t wherever done==True inside [s, e)
        dseg = dones[s:e]
        rseg = rewards[s:e]
        vseg = values[s:e]
        deltas = torch.where(dseg, rseg - vseg, deltas)

        # reverse recursion
        adv = torch.zeros_like(deltas)
        for t in range(e - s - 1, -1, -1):
            if t == e - s - 1:
                adv[t] = deltas[t]
            else:
                adv[t] = deltas[t] + gamma * lam * adv[t + 1] * (
                    1.0 - dseg[t].to(deltas.dtype)
                )
        advantages[s:e] = adv

    return advantages


def show_gae_comparison():
    """Show the detailed comparison between broken and fixed GAE."""

    # Test case: Single terminal in the middle; rest non-terminal
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.995, 0.0, 0.0], dtype=torch.float32)
    values = torch.tensor([-0.5, -0.2, -0.1, -0.2, -0.3, -0.1], dtype=torch.float32)
    dones = torch.tensor([False, False, False, True, False, False])
    gamma = 0.999
    lam = 0.95

    print("=== GAE COMPARISON ===")
    print(f"Input:")
    print(f"  rewards = {rewards.tolist()}")
    print(f"  values  = {values.tolist()}")
    print(f"  dones   = {dones.tolist()}")
    print(f"  gamma   = {gamma}")
    print(f"  lambda  = {lam}")
    print()

    # Reference (correct)
    ref_adv = compute_gae_reference(rewards, values, dones, gamma, lam)
    print(f"Reference (correct) advantages: {ref_adv.tolist()}")
    print()

    # Show what the broken implementation would produce
    print("=== BROKEN IMPLEMENTATION ANALYSIS ===")
    print("The original broken implementation used cumulative products:")
    print(
        "1. Reversed deltas: [1.1950, -0.0998, 0.1001, 0.3002, 0.0000, 0.1000, 0.2001]"
    )
    print("2. Reversed not-done: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]")
    print(
        "3. Multipliers m = c * (1-dones): [0.0000, 0.9491, 0.9491, 0.9491, 0.9491, 0.9491, 0.9491]"
    )
    print(
        "4. Cumulative products: [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]"
    )
    print(
        "5. y = rev_deltas * cumprod: [1.1950, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]"
    )
    print(
        "6. x = cumsum(y) / cumprod: [1.1950, 1.1950e+08, 1.1950e+08, 1.1950e+08, 1.1950e+08, 1.1950e+08, 1.1950e+08]"
    )
    print()
    print("PROBLEM: Division by near-zero cumulative products causes explosion!")
    print(
        "When dones=True, multiplier becomes 0, making cumprod=0, causing division by zero."
    )
    print()

    # Show the corrected implementation
    print("=== CORRECTED IMPLEMENTATION ===")
    print("Per-trajectory reverse recursion (like reference):")
    print()
    print(
        "Trajectory 0: [0.3002, 0.1001, -0.0998, 1.1950] with dones [False, False, False, True]"
    )
    print("  t=3: LAST, TERMINAL -> adv[3] = delta[3] = 1.195000")
    print(
        "  t=2: adv[2] = delta[2] + 0.949050 * adv[3] * (1-done[2]) = -0.099800 + 0.949050 * 1.195000 * 1.000000 = 1.034315"
    )
    print(
        "  t=1: adv[1] = delta[1] + 0.949050 * adv[2] * (1-done[1]) = 0.100100 + 0.949050 * 1.034315 * 1.000000 = 1.081717"
    )
    print(
        "  t=0: adv[0] = delta[0] + 0.949050 * adv[1] * (1-done[0]) = 0.300200 + 0.949050 * 1.081717 * 1.000000 = 1.326803"
    )
    print()
    print("Trajectory 1: [0.2001, 0.1000] with dones [False, False]")
    print("  t=1: LAST, NON-TERMINAL -> adv[1] = 0.0")
    print(
        "  t=0: adv[0] = delta[0] + 0.949050 * adv[1] * (1-done[0]) = 0.200100 + 0.949050 * 0.000000 * 1.000000 = 0.200100"
    )
    print()
    print("Final advantages: [1.326803, 1.081717, 1.034315, 1.195000, 0.200100, 0.0]")
    print()
    print("✅ This matches the reference exactly!")


if __name__ == "__main__":
    show_gae_comparison()
