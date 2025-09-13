#!/usr/bin/env python3
"""Compare Trinal-Clip PPO vs Standard PPO."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaholdem.rl.losses import standard_ppo_loss, trinal_clip_ppo_loss
from alphaholdem.rl.self_play import SelfPlayTrainer


def compare_ppo_losses():
    print("=== Comparing PPO Loss Functions ===\n")

    # Create trainer
    torch.manual_seed(42)  # For reproducible comparison
    trainer = SelfPlayTrainer()

    # Collect trajectories
    trajectories = []
    for _ in range(4):
        trajectory = trainer.collect_trajectory()
        trajectories.append(trajectory)

    # Compute values and advantages
    for trajectory in trajectories:
        rewards = [t.reward for t in trajectory.transitions]
        values = []

        for transition in trajectory.transitions:
            obs = transition.observation
            cards = obs[: (6 * 4 * 13)].reshape(1, 6, 4, 13)
            actions_tensor = obs[(6 * 4 * 13) :].reshape(1, 24, 4, trainer.num_bet_bins)

            with torch.no_grad():
                _, value = trainer.model(cards, actions_tensor)
                values.append(value.item())

        values.append(0.0)

        from alphaholdem.rl.replay import compute_gae_returns

        advantages, returns = compute_gae_returns(
            rewards, values, gamma=trainer.gamma, lambda_=trainer.gae_lambda
        )

        for i, transition in enumerate(trajectory.transitions):
            transition.advantage = advantages[i]
            transition.return_ = returns[i]

    # Prepare batch
    from alphaholdem.rl.replay import prepare_ppo_batch

    batch = prepare_ppo_batch(trajectories)

    # Get model outputs for the batch
    observations = batch["observations"]
    cards = observations[:, : (6 * 4 * 13)].reshape(-1, 6, 4, 13)
    actions_tensor = observations[:, (6 * 4 * 13) :].reshape(
        -1, 24, 4, trainer.num_bet_bins
    )

    with torch.no_grad():
        logits, values = trainer.model(cards, actions_tensor)

    # Test both loss functions
    print("Testing Trinal-Clip PPO:")
    trinal_loss = trinal_clip_ppo_loss(
        logits=logits,
        values=values,
        actions=batch["actions"],
        log_probs_old=batch["log_probs_old"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        legal_masks=batch["legal_masks"],
        epsilon=0.2,
        delta1=3.0,
        delta2=-100.0,
        delta3=100.0,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    print(f"  Total loss: {trinal_loss['total_loss'].item():.6f}")
    print(f"  Policy loss: {trinal_loss['policy_loss'].item():.6f}")
    print(f"  Value loss: {trinal_loss['value_loss'].item():.6f}")
    print(f"  Ratio mean: {trinal_loss['ratio_mean'].item():.3f}")
    print(f"  Ratio std: {trinal_loss['ratio_std'].item():.3f}")

    print("\nTesting Standard PPO:")
    standard_loss = standard_ppo_loss(
        logits=logits,
        values=values,
        actions=batch["actions"],
        log_probs_old=batch["log_probs_old"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        legal_masks=batch["legal_masks"],
        epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
    )

    print(f"  Total loss: {standard_loss['total_loss'].item():.6f}")
    print(f"  Policy loss: {standard_loss['policy_loss'].item():.6f}")
    print(f"  Value loss: {standard_loss['value_loss'].item():.6f}")
    print(f"  Ratio mean: {standard_loss['ratio_mean'].item():.3f}")
    print(f"  Ratio std: {standard_loss['ratio_std'].item():.3f}")

    # Compare
    print(f"\nComparison:")
    print(
        f"  Loss difference: {trinal_loss['total_loss'].item() - standard_loss['total_loss'].item():+.6f}"
    )
    print(
        f"  Ratio mean difference: {trinal_loss['ratio_mean'].item() - standard_loss['ratio_mean'].item():+.3f}"
    )


if __name__ == "__main__":
    compare_ppo_losses()
