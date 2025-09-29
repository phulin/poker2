#!/usr/bin/env python3
"""
Debug script to investigate GAE computation specifically.
"""


import torch

from alphaholdem.core.config_loader import get_config
from alphaholdem.rl.self_play import SelfPlayTrainer


def debug_gae():
    """Debug GAE computation specifically."""

    print("🔍 Debugging GAE Computation")
    print("=" * 50)

    # Initialize trainer
    device = torch.device("cpu")
    config = get_config("configs/default.yaml")

    trainer = SelfPlayTrainer(
        device=device,
        config=config,
        use_tensor_env=True,
        num_envs=4,  # Very small for debugging
        k_best_pool_size=3,
        min_elo_diff=50.0,
        learning_rate=1e-3,
        batch_size=16,
        episodes_per_step=1,
        gamma=0.999,
        gae_lambda=0.95,
        epsilon=0.2,
        delta1=3.0,
        value_coef=0.5,
        entropy_coef=0.01,
        grad_clip=0.5,
    )

    # Add a snapshot for opponent
    trainer.opponent_pool.add_snapshot(trainer, trainer.current_elo)

    # Collect some data
    print("Collecting tensor trajectories...")
    total_reward = trainer.collect_tensor_trajectories(
        min_steps=20,  # Very small
        all_opponent_snapshots=trainer.opponent_pool.snapshots,
    )

    print(f"Total reward collected: {total_reward}")
    print(f"Buffer size: {trainer.replay_buffer.num_steps()}")

    # Inspect raw data before GAE
    print("\n📊 Before GAE Computation:")
    if trainer.replay_buffer.num_steps() > 0:
        # Get all data from buffer
        size = trainer.replay_buffer.size
        rewards = trainer.replay_buffer.rewards[:size]
        values = trainer.replay_buffer.values[:size]
        dones = trainer.replay_buffer.dones[:size]

        print(f"Raw rewards:")
        print(f"  Min: {rewards.min().item():.6f}")
        print(f"  Max: {rewards.max().item():.6f}")
        print(f"  Mean: {rewards.mean().item():.6f}")
        print(f"  Non-zero: {(rewards != 0).sum().item()}/{len(rewards)}")
        print(f"  First 10: {rewards[:10].tolist()}")

        print(f"Values:")
        print(f"  Min: {values.min().item():.6f}")
        print(f"  Max: {values.max().item():.6f}")
        print(f"  Mean: {values.mean().item():.6f}")
        print(f"  First 10: {values[:10].tolist()}")

        print(f"Dones:")
        print(f"  True count: {dones.sum().item()}/{len(dones)}")
        print(f"  First 10: {dones[:10].tolist()}")

        # Check if rewards are only non-zero when done=True
        non_zero_rewards = rewards != 0
        print(
            f"Non-zero rewards when done=True: {(non_zero_rewards & dones).sum().item()}"
        )
        print(
            f"Non-zero rewards when done=False: {(non_zero_rewards & ~dones).sum().item()}"
        )

    # Run GAE computation
    print("\n📊 Running GAE Computation...")
    trainer.replay_buffer.compute_gae_returns(gamma=0.999, lambda_=0.95)

    # Inspect results after GAE
    print("\n📊 After GAE Computation:")
    if trainer.replay_buffer.num_steps() > 0:
        size = trainer.replay_buffer.size
        returns = trainer.replay_buffer.returns[:size]
        advantages = trainer.replay_buffer.advantages[:size]

        print(f"Returns:")
        print(f"  Min: {returns.min().item():.6f}")
        print(f"  Max: {returns.max().item():.6f}")
        print(f"  Mean: {returns.mean().item():.6f}")
        print(f"  First 10: {returns[:10].tolist()}")

        print(f"Advantages:")
        print(f"  Min: {advantages.min().item():.6f}")
        print(f"  Max: {advantages.max().item():.6f}")
        print(f"  Mean: {advantages.mean().item():.6f}")
        print(f"  First 10: {advantages[:10].tolist()}")

        # Check for extreme values
        extreme_returns = torch.abs(returns) > 1000
        if extreme_returns.any():
            print(f"Extreme returns (>1000): {extreme_returns.sum().item()}")
            extreme_indices = torch.where(extreme_returns)[0]
            print(f"Extreme return indices: {extreme_indices[:10].tolist()}")
            print(f"Extreme return values: {returns[extreme_indices][:10].tolist()}")

    print("\n✅ GAE Debug Complete!")


if __name__ == "__main__":
    debug_gae()
