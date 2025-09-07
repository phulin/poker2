#!/usr/bin/env python3
"""
Simple test of GAE computation to isolate the issue.
"""

import torch


def test_gae_simple():
    """Test GAE computation with simple data."""

    print("🔍 Testing Simple GAE Computation")
    print("=" * 50)

    # Create simple test data
    rewards = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.995, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32
    )
    values = torch.tensor(
        [-0.5, -0.2, -0.1, -0.2, -0.3, -0.2, -0.4, -0.3, -0.2, -0.1],
        dtype=torch.float32,
    )
    dones = torch.tensor(
        [False, False, False, False, False, True, False, False, False, False],
        dtype=torch.bool,
    )

    gamma = 0.999
    lambda_ = 0.95

    print(f"Rewards: {rewards.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Dones: {dones.tolist()}")

    # Compute deltas manually
    deltas = torch.zeros_like(rewards)

    # Non-terminal states: δ_t = r_t + γ * V_{t+1} - V_t
    for t in range(len(rewards) - 1):
        if not dones[t]:
            deltas[t] = rewards[t] + gamma * values[t + 1] - values[t]

    # Terminal states: δ_t = r_t - V_t
    for t in range(len(rewards)):
        if dones[t]:
            deltas[t] = rewards[t] - values[t]

    print(f"Deltas: {deltas.tolist()}")

    # Simple GAE computation (not vectorized)
    advantages = torch.zeros_like(rewards)

    # Find trajectory boundaries
    trajectory_starts = [0]
    for i in range(len(dones) - 1):
        if dones[i]:
            trajectory_starts.append(i + 1)

    print(f"Trajectory starts: {trajectory_starts}")

    # Compute GAE for each trajectory
    for start in trajectory_starts:
        # Find end of trajectory
        end = len(rewards)
        for i in range(start, len(dones)):
            if dones[i]:
                end = i + 1
                break

        print(f"Trajectory {start}-{end-1}:")

        # Compute GAE for this trajectory
        traj_deltas = deltas[start:end]
        traj_dones = dones[start:end]

        # Reverse computation
        traj_len = len(traj_deltas)
        if traj_len == 0:
            continue

        # Work backwards
        advantages_traj = torch.zeros_like(traj_deltas)
        for t in range(traj_len - 1, -1, -1):
            if t == traj_len - 1:
                advantages_traj[t] = traj_deltas[t]
            else:
                advantages_traj[t] = traj_deltas[t] + gamma * lambda_ * advantages_traj[
                    t + 1
                ] * (1.0 - traj_dones[t].float())

        advantages[start:end] = advantages_traj
        print(f"  Deltas: {traj_deltas.tolist()}")
        print(f"  Advantages: {advantages_traj.tolist()}")

    # Compute returns
    returns = advantages + values

    print(f"Final advantages: {advantages.tolist()}")
    print(f"Final returns: {returns.tolist()}")

    print(
        f"Advantage range: {advantages.min().item():.6f} to {advantages.max().item():.6f}"
    )
    print(f"Return range: {returns.min().item():.6f} to {returns.max().item():.6f}")


if __name__ == "__main__":
    test_gae_simple()
