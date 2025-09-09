#!/usr/bin/env python3
"""Visualize how trajectories_per_step affects training."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaholdem.rl.self_play import SelfPlayTrainer


def visualize_trajectories_per_step():
    print("=== Understanding trajectories_per_step ===\n")

    # Create trainer
    trainer = SelfPlayTrainer()

    print("Training Flow:")
    print("1. Each training step collects N trajectories (games)")
    print("2. Trajectories are stored in replay buffer")
    print("3. Model is updated using batch_size trajectories")
    print("4. Process repeats\n")

    # Test different values
    test_values = [1, 2, 4, 8]

    for num_trajectories in test_values:
        print(f"Testing trajectories_per_step = {num_trajectories}:")

        # Clear replay buffer
        trainer.replay_buffer.clear()
        initial_episodes = trainer.step_trajectories_collected

        # Run one training step
        stats = trainer.train_step(num_trajectories=num_trajectories)

        print(f"  Trajectories collected: {num_trajectories}")
        print(
            f"  Episodes added: {trainer.step_trajectories_collected - initial_episodes}"
        )
        print(f"  Replay buffer size: {len(trainer.replay_buffer.trajectories)}")
        print(f"  Model updated: {'Yes' if 'avg_loss' in stats else 'No'}")
        if "avg_loss" in stats:
            print(f"  Loss: {stats['avg_loss']:.6f}")
        print(f"  Avg reward: {stats['avg_reward']:.2f}")
        print()

    print("Key Insights:")
    print("• Higher trajectories_per_step = more data before each update")
    print("• More data = more stable gradients but slower updates")
    print("• Too few = noisy updates, too many = slow learning")
    print("• Should be >= batch_size for consistent updates")


def show_training_cycle():
    print("\n=== Training Cycle Visualization ===\n")

    trainer = SelfPlayTrainer()

    print("Step-by-step training cycle:")

    for step in range(3):
        print(f"\n--- Training Step {step + 1} ---")

        # Before step
        buffer_before = len(trainer.replay_buffer.trajectories)
        episodes_before = trainer.step_trajectories_collected

        print(
            f"Before: Buffer has {buffer_before} trajectories, {episodes_before} episodes"
        )

        # Training step
        stats = trainer.train_step(num_trajectories=2)

        # After step
        buffer_after = len(trainer.replay_buffer.trajectories)
        episodes_after = trainer.step_trajectories_collected

        print(f"Collected: {episodes_after - episodes_before} new episodes")
        print(f"After: Buffer has {buffer_after} trajectories")
        print(f"Model updated: {'Yes' if 'avg_loss' in stats else 'No'}")
        if "avg_loss" in stats:
            print(f"Loss: {stats['avg_loss']:.6f}")
        print(f"Avg reward: {stats['avg_reward']:.2f}")


if __name__ == "__main__":
    visualize_trajectories_per_step()
    show_training_cycle()
