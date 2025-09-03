#!/usr/bin/env python3
"""Test training with conservative hyperparameters."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaholdem.rl.self_play import SelfPlayTrainer


def test_conservative_training():
    print("=== Testing Conservative Hyperparameters ===\n")

    # Conservative settings
    trainer = SelfPlayTrainer(
        num_bet_bins=9,
        learning_rate=1e-4,  # Lower learning rate
        batch_size=32,  # Larger batch size
        grad_clip=0.5,  # Less aggressive clipping
    )

    print("Hyperparameters:")
    print(f"  Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Gradient clip: {trainer.grad_clip}")
    print()

    # Track loss over several steps
    losses = []
    rewards = []

    for step in range(10):
        print(f"Step {step + 1}:")

        # Collect initial parameter norms
        if step == 0:
            initial_norms = {}
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    initial_norms[name] = torch.norm(param).item()

        # Training step
        stats = trainer.train_step(num_trajectories=4)

        if "avg_loss" in stats:
            losses.append(stats["avg_loss"])
            print(f"  Loss: {stats['avg_loss']:.6f}")

        rewards.append(stats["avg_reward"])
        print(f"  Avg reward: {stats['avg_reward']:.2f}")
        print(f"  Episodes: {stats['episode_count']}")

        # Check parameter changes
        if step == 9:  # After 10 steps
            final_norms = {}
            for name, param in trainer.model.named_parameters():
                if param.requires_grad:
                    final_norms[name] = torch.norm(param).item()

            print(f"\nParameter changes over 10 steps:")
            total_change = 0
            for name in initial_norms:
                change = final_norms[name] - initial_norms[name]
                total_change += abs(change)
                print(f"  {name}: {change:+.6f}")

            print(f"Total absolute change: {total_change:.6f}")

        print()

    # Analyze trends
    if len(losses) > 1:
        loss_trend = (
            "decreasing"
            if losses[-1] < losses[0]
            else "increasing" if losses[-1] > losses[0] else "stable"
        )
        print(f"Loss trend: {loss_trend}")
        print(f"Loss range: {min(losses):.6f} to {max(losses):.6f}")

    reward_trend = (
        "improving"
        if rewards[-1] > rewards[0]
        else "worsening" if rewards[-1] < rewards[0] else "stable"
    )
    print(f"Reward trend: {reward_trend}")
    print(f"Reward range: {min(rewards):.2f} to {max(rewards):.2f}")


if __name__ == "__main__":
    test_conservative_training()
