#!/usr/bin/env python3
"""
Test GAE with TRUE constraint: all trajectories end with a single done state.
No terminal states anywhere except at the very end of each trajectory.
"""

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer


def compute_gae_non_vectorized(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Non-vectorized GAE using straightforward per-trajectory reverse recursion."""
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


def test_gae_true_constraint():
    """Test GAE with TRUE constraint: all trajectories end with done=True, no terminal states in middle."""

    # Create a test case where EVERY trajectory ends with done=True
    # NO terminal states anywhere except at the very end of each trajectory
    rewards = torch.tensor(
        [
            0.0,
            0.0,
            0.995,  # Trajectory 0: length 3, ends with done=True
            0.0,
            -0.5,  # Trajectory 1: length 2, ends with done=True
            0.0,
            0.0,
            0.0,
            0.8,  # Trajectory 2: length 4, ends with done=True
            0.0,
            0.0,  # Trajectory 3: length 2, ends with done=True
            0.0,
            0.0,
            0.0,
            0.0,
            -0.3,  # Trajectory 4: length 5, ends with done=True
        ],
        dtype=torch.float32,
    )

    values = torch.tensor(
        [
            -0.1,
            -0.2,
            -0.15,  # Trajectory 0
            -0.3,
            -0.1,  # Trajectory 1
            -0.2,
            -0.25,
            -0.3,
            -0.1,  # Trajectory 2
            -0.4,
            -0.2,  # Trajectory 3
            -0.1,
            -0.15,
            -0.2,
            -0.25,
            -0.05,  # Trajectory 4
        ],
        dtype=torch.float32,
    )

    # TRUE CONSTRAINT: All trajectories end with done=True, NO terminal states in middle
    # Only the last timestep of each trajectory has done=True
    dones = torch.tensor(
        [
            False,
            False,
            True,  # Trajectory 0: only last timestep is done
            False,
            True,  # Trajectory 1: only last timestep is done
            False,
            False,
            False,
            True,  # Trajectory 2: only last timestep is done
            False,
            True,  # Trajectory 3: only last timestep is done
            False,
            False,
            False,
            False,
            True,  # Trajectory 4: only last timestep is done
        ]
    )

    gamma = 0.999
    lam = 0.95

    print("=" * 80)
    print("GAE TEST: TRUE CONSTRAINT - ALL TRAJECTORIES END WITH DONE=TRUE")
    print("=" * 80)
    print(f"Total timesteps: {len(rewards)}")
    print(f"Number of trajectories: {dones.sum().item()}")
    print(f"Dones pattern: {dones.tolist()}")

    # Verify the constraint: check that only the last timestep of each trajectory has done=True
    is_start = torch.zeros_like(dones, dtype=torch.bool)
    is_start[0] = True
    is_start[1:] = dones[:-1]
    starts = torch.where(is_start)[0]
    ends = torch.cat([starts[1:], torch.tensor([len(dones)], device=dones.device)])

    print("Trajectory analysis:")
    for i, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        traj_dones = dones[s:e]
        print(f"  Trajectory {i}: indices {s}-{e-1}, dones={traj_dones.tolist()}")
        # Check that only the last timestep has done=True
        expected_dones = torch.zeros_like(traj_dones)
        expected_dones[-1] = True
        if torch.equal(traj_dones, expected_dones):
            print(f"    ✅ Constraint satisfied: only last timestep is done")
        else:
            print(f"    ❌ Constraint violated: terminal states in middle!")

    print("=" * 80)

    # Non-vectorized reference
    ref_adv = compute_gae_non_vectorized(rewards, values, dones, gamma, lam)

    # Vectorized implementation
    buf = VectorizedReplayBuffer(
        capacity=128, observation_dim=1, legal_mask_dim=1, device=rewards.device
    )
    vec_adv = buf._compute_gae_batch_vectorized(rewards, values, dones, gamma, lam)

    # Compare results
    print(f"Reference advantages: {ref_adv.tolist()}")
    print(f"Vectorized advantages: {vec_adv.tolist()}")

    close = torch.isclose(ref_adv, vec_adv, rtol=1e-5, atol=1e-6)
    print(f"All close: {close.all().item()}")

    if not close.all():
        diff = torch.abs(ref_adv - vec_adv)
        max_diff_idx = diff.argmax().item()
        print(f"Max difference: {diff.max().item():.8f} at index {max_diff_idx}")
        print(f"Reference[{max_diff_idx}] = {ref_adv[max_diff_idx].item():.8f}")
        print(f"Vectorized[{max_diff_idx}] = {vec_adv[max_diff_idx].item():.8f}")
    else:
        print("✅ VECTORIZED IMPLEMENTATION MATCHES REFERENCE!")

    print("=" * 80)


if __name__ == "__main__":
    test_gae_true_constraint()
