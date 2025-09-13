#!/usr/bin/env python3
"""Check advantage distribution during training."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaholdem.rl.replay import compute_gae_returns
from alphaholdem.rl.self_play import SelfPlayTrainer


def check_advantages():
    print("=== Checking Advantage Distribution ===\n")

    trainer = SelfPlayTrainer()

    all_advantages = []
    all_returns = []

    # Collect several trajectories and compute advantages
    for i in range(5):
        print(f"Trajectory {i+1}:")
        trajectory = trainer.collect_trajectory()

        rewards = [t.reward for t in trajectory.transitions]
        values = []

        # Compute values for each transition
        for transition in trajectory.transitions:
            obs = transition.observation
            cards = obs[: (6 * 4 * 13)].reshape(1, 6, 4, 13)
            actions_tensor = obs[(6 * 4 * 13) :].reshape(1, 24, 4, trainer.num_bet_bins)

            with torch.no_grad():
                _, value = trainer.model(cards, actions_tensor)
                values.append(value.item())

        values.append(0.0)  # Final value

        advantages, returns = compute_gae_returns(
            rewards, values, gamma=trainer.gamma, lambda_=trainer.gae_lambda
        )

        all_advantages.extend(advantages)
        all_returns.extend(returns)

        print(f"  Length: {len(trajectory.transitions)}")
        print(f"  Total reward: {sum(rewards):.2f}")
        print(f"  Advantages: {min(advantages):.3f} to {max(advantages):.3f}")
        print(f"  Returns: {min(returns):.3f} to {max(returns):.3f}")
        print()

    # Overall statistics
    print("=== Overall Statistics ===")
    print(f"Total transitions: {len(all_advantages)}")
    print(
        f"Advantages - min: {min(all_advantages):.3f}, max: {max(all_advantages):.3f}, mean: {sum(all_advantages)/len(all_advantages):.3f}"
    )
    print(
        f"Returns - min: {min(all_returns):.3f}, max: {max(all_returns):.3f}, mean: {sum(all_returns)/len(all_returns):.3f}"
    )

    # Check for potential issues
    if abs(sum(all_advantages) / len(all_advantages)) > 1.0:
        print("⚠️  WARNING: Advantages have high mean bias")

    if max(all_advantages) - min(all_advantages) < 0.1:
        print("⚠️  WARNING: Advantages have very low variance")

    if any(abs(a) > 1000 for a in all_advantages):
        print("⚠️  WARNING: Some advantages are extremely large")


if __name__ == "__main__":
    check_advantages()
