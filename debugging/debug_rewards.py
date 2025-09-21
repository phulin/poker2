#!/usr/bin/env python3
"""
Debug script to investigate reward issues in the replay buffer and training.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaholdem.core.config_loader import get_config
from alphaholdem.rl.self_play import SelfPlayTrainer


def debug_rewards():
    """Debug reward computation and replay buffer contents."""

    print("🔍 Debugging Reward Computation")
    print("=" * 50)

    # Initialize trainer
    device = torch.device("cpu")
    config = get_config("configs/default.yaml")

    trainer = SelfPlayTrainer(
        device=device,
        config=config,
        use_tensor_env=True,
        num_envs=8,  # Small number for debugging
        k_best_pool_size=3,
        min_elo_diff=50.0,
        learning_rate=1e-3,
        batch_size=32,
        num_epochs=2,
        gamma=0.999,
        gae_lambda=0.95,
        epsilon=0.2,
        delta1=3.0,
        value_coef=0.5,
        entropy_coef=0.01,
        grad_clip=0.5,
    )

    print(f"Initial ELO: {trainer.current_elo}")

    # Add a snapshot for opponent
    trainer.opponent_pool.add_snapshot(trainer, trainer.current_elo)

    print("\n📊 Test 1: Inspect Replay Buffer Before Training")
    print(f"Buffer size: {trainer.replay_buffer.num_steps()}")

    # Run a small training step to collect some data
    print("\n📊 Test 2: Collect Some Training Data")
    print("Collecting tensor trajectories...")

    # Collect some trajectories manually to inspect
    total_reward = trainer.collect_tensor_trajectories(
        min_steps=50,  # Small number
        all_opponent_snapshots=trainer.opponent_pool.snapshots,
    )

    print(f"Total reward collected: {total_reward}")
    print(f"Buffer size after collection: {trainer.replay_buffer.num_steps()}")

    # Inspect replay buffer contents
    print("\n📊 Test 3: Inspect Replay Buffer Contents")
    if trainer.replay_buffer.num_steps() > 0:
        # Get a sample from the buffer
        rng = torch.Generator(device=trainer.device)
        sample = trainer.replay_buffer.sample_batch(
            rng, min(10, trainer.replay_buffer.num_steps())
        )

        print(f"Sample batch size: {sample.embedding_data.token_ids.shape[0]}")
        print(f"Returns in sample:")
        print(f"  Min: {sample.returns.min().item():.6f}")
        print(f"  Max: {sample.returns.max().item():.6f}")
        print(f"  Mean: {sample.returns.mean().item():.6f}")
        print(f"  Std: {sample.returns.std().item():.6f}")

        # Show first few returns
        print(f"First 10 returns: {sample.returns[:10].tolist()}")

        # Check advantages
        print(f"Advantages in sample:")
        print(f"  Min: {sample.advantages.min().item():.6f}")
        print(f"  Max: {sample.advantages.max().item():.6f}")
        print(f"  Mean: {sample.advantages.mean().item():.6f}")

        # Check delta2 and delta3
        print(f"Delta2 in sample:")
        print(f"  Min: {sample.delta2.min().item():.6f}")
        print(f"  Max: {sample.delta2.max().item():.6f}")
        print(f"  Mean: {sample.delta2.mean().item():.6f}")

        print(f"Delta3 in sample:")
        print(f"  Min: {sample.delta3.min().item():.6f}")
        print(f"  Max: {sample.delta3.max().item():.6f}")
        print(f"  Mean: {sample.delta3.mean().item():.6f}")

    # Run a full training step
    print("\n📊 Test 4: Full Training Step")
    stats = trainer.train_step()

    print(f"Training stats:")
    print(f"  Avg reward: {stats.get('avg_reward', 'N/A')}")
    print(f"  Current ELO: {stats.get('current_elo', 'N/A')}")
    print(f"  First ret: {stats.get('first_ret', 'N/A')}")
    print(f"  First ret clipped: {stats.get('first_ret_clipped', 'N/A')}")
    print(f"  First ret out of bounds: {stats.get('first_ret_out_of_bounds', 'N/A')}")

    print("\n✅ Reward Debug Complete!")


if __name__ == "__main__":
    debug_rewards()
