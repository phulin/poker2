#!/usr/bin/env python3
"""Detailed training analysis with comprehensive loss function breakdown."""


import torch

from alphaholdem.rl.losses import trinal_clip_ppo_loss
from alphaholdem.rl.replay import compute_gae_returns, prepare_ppo_batch
from alphaholdem.rl.self_play import SelfPlayTrainer


def detailed_training_analysis():
    print("=== Detailed Training Analysis ===\n")

    # Initialize trainer with conservative settings
    trainer = SelfPlayTrainer()

    print("Initial Model State:")
    print(f"  Total parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"  Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
    print(f"  Batch size: {trainer.batch_size}")
    print(f"  Gradient clip: {trainer.grad_clip}")
    print()

    # Step 1: Perform 10 regular training steps
    print("=== Step 1: Performing 10 Regular Training Steps ===")
    regular_stats = []

    for step in range(10):
        stats = trainer.train_step(num_trajectories=4)
        regular_stats.append(stats)

        print(f"Step {step + 1:2d}: ", end="")
        if "avg_loss" in stats:
            print(f"Loss: {stats['avg_loss']:.6f}, ", end="")
        print(
            f"Reward: {stats['avg_reward']:6.2f}, Trajectories: {stats['trajectories_collected']}"
        )

    print()

    # Step 2: Detailed analysis of one training step
    print("=== Step 2: Detailed Training Step Analysis ===\n")

    # Clear buffer and collect fresh trajectories
    trainer.replay_buffer.clear()
    print("Collecting fresh trajectories for detailed analysis...")

    trajectories = []
    for i in range(4):
        trajectory = trainer.collect_trajectory()
        trajectories.append(trajectory)
        print(
            f"  Trajectory {i+1}: {len(trajectory.transitions)} transitions, "
            f"total reward: {sum(t.reward for t in trajectory.transitions):.2f}"
        )

    print(
        f"\nTotal transitions collected: {sum(len(t.transitions) for t in trajectories)}"
    )

    # Detailed GAE computation
    print("\n--- GAE (Generalized Advantage Estimation) Analysis ---")
    all_advantages = []
    all_returns = []

    for i, trajectory in enumerate(trajectories):
        print(f"\nTrajectory {i+1} GAE computation:")

        rewards = [t.reward for t in trajectory.transitions]
        values = []

        print(f"  Rewards: {rewards}")

        # Compute values for each transition
        for j, transition in enumerate(trajectory.transitions):
            obs = transition.observation
            cards = obs[: (6 * 4 * 13)].reshape(1, 6, 4, 13)
            actions_tensor = obs[(6 * 4 * 13) :].reshape(1, 24, 4, trainer.num_bet_bins)

            with torch.no_grad():
                _, value = trainer.model(cards, actions_tensor)
                values.append(value.item())
                print(
                    f"    Step {j+1}: reward={rewards[j]:.2f}, value={value.item():.6f}"
                )

        values.append(0.0)  # Final value
        print(f"    Final value: 0.0")

        # Compute GAE
        advantages, returns = compute_gae_returns(
            rewards, values, gamma=trainer.gamma, lambda_=trainer.gae_lambda
        )

        print(f"  Advantages: {[f'{a:.6f}' for a in advantages]}")
        print(f"  Returns: {[f'{r:.6f}' for r in returns]}")

        # Update trajectory
        for j, transition in enumerate(trajectory.transitions):
            transition.advantage = advantages[j]
            transition.return_ = returns[j]

        all_advantages.extend(advantages)
        all_returns.extend(returns)

    print(f"\nOverall GAE Statistics:")
    print(
        f"  Advantages - min: {min(all_advantages):.6f}, max: {max(all_advantages):.6f}, "
        f"mean: {sum(all_advantages)/len(all_advantages):.6f}"
    )
    print(
        f"  Returns - min: {min(all_returns):.6f}, max: {max(all_returns):.6f}, "
        f"mean: {sum(all_returns)/len(all_returns):.6f}"
    )

    # Prepare batch
    print("\n--- Batch Preparation ---")
    batch = prepare_ppo_batch(trajectories)

    print(f"Batch statistics:")
    print(f"  Observations shape: {batch['observations'].shape}")
    print(f"  Actions shape: {batch['actions'].shape}")
    print(f"  Log probs old shape: {batch['log_probs_old'].shape}")
    print(f"  Advantages shape: {batch['advantages'].shape}")
    print(f"  Returns shape: {batch['returns'].shape}")
    print(f"  Legal masks shape: {batch['legal_masks'].shape}")

    # Model forward pass
    print("\n--- Model Forward Pass ---")
    observations = batch["observations"]
    cards = observations[:, : (6 * 4 * 13)].reshape(-1, 6, 4, 13)
    actions_tensor = observations[:, (6 * 4 * 13) :].reshape(
        -1, 24, 4, trainer.num_bet_bins
    )

    print(f"Input shapes:")
    print(f"  Cards tensor: {cards.shape}")
    print(f"  Actions tensor: {actions_tensor.shape}")

    # Forward pass (without no_grad for gradient computation)
    logits, values = trainer.model(cards, actions_tensor)

    print(f"Output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Values: {values.shape}")
    print(f"  Logits norm: {torch.norm(logits).item():.6f}")
    print(f"  Values norm: {torch.norm(values).item():.6f}")

    # Detailed loss computation
    print("\n--- Detailed Loss Computation ---")

    # Compute delta bounds
    total_chips_placed = 0
    for trajectory in trajectories:
        for transition in trajectory.transitions:
            total_chips_placed += transition.chips_placed

    if total_chips_placed > 0:
        delta2 = -trajectory.our_chips_committed
        delta3 = trajectory.opp_chips_committed
    else:
        delta2 = -1000.0
        delta3 = 1000.0

    print(f"Delta bounds computation:")
    print(f"  Total chips placed: {total_chips_placed}")
    print(f"  Delta2 (negative bound): {delta2}")
    print(f"  Delta3 (positive bound): {delta3}")

    # Compute loss with detailed breakdown
    loss_dict = trinal_clip_ppo_loss(
        logits=logits,
        values=values,
        actions=batch["actions"],
        log_probs_old=batch["log_probs_old"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        legal_masks=batch["legal_masks"],
        epsilon=trainer.epsilon,
        delta1=trainer.delta1,
        delta2=delta2,
        delta3=delta3,
        value_coef=trainer.value_coef,
        entropy_coef=trainer.entropy_coef,
    )

    print(f"\nLoss components:")
    print(f"  Policy loss: {loss_dict['policy_loss'].item():.6f}")
    print(f"  Value loss: {loss_dict['value_loss'].item():.6f}")
    print(f"  Entropy: {loss_dict['entropy'].item():.6f}")
    print(f"  Total loss: {loss_dict['total_loss'].item():.6f}")
    print(f"  Ratio mean: {loss_dict['ratio_mean'].item():.6f}")
    print(f"  Ratio std: {loss_dict['ratio_std'].item():.6f}")
    print(f"  Clipped ratio mean: {loss_dict['clipped_ratio_mean'].item():.6f}")
    print(f"  Clipped ratio std: {loss_dict['clipped_ratio_std'].item():.6f}")

    # Gradient analysis
    print("\n--- Gradient Analysis ---")

    # Zero gradients
    trainer.optimizer.zero_grad()

    # Backward pass
    loss_dict["total_loss"].backward()

    # Analyze gradients before clipping
    total_grad_norm_before = 0
    grad_info = []

    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm_before += grad_norm**2
            grad_info.append((name, grad_norm))

    total_grad_norm_before = total_grad_norm_before**0.5

    print(f"Gradients before clipping:")
    print(f"  Total gradient norm: {total_grad_norm_before:.6f}")
    print(f"  Gradient clip threshold: {trainer.grad_clip}")

    for name, grad_norm in sorted(grad_info, key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {name}: {grad_norm:.6f}")

    # Apply gradient clipping
    original_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), trainer.grad_clip
    )

    # Compute actual clipped norm
    total_grad_norm_after = 0
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm_after += grad_norm**2
    total_grad_norm_after = total_grad_norm_after**0.5

    print(f"\nGradients after clipping:")
    print(f"  Original norm: {original_norm:.6f}")
    print(f"  Clipped norm: {total_grad_norm_after:.6f}")
    print(f"  Clipping ratio: {total_grad_norm_after / total_grad_norm_before:.6f}")
    print(f"  Gradients clipped: {total_grad_norm_after < total_grad_norm_before}")

    # Parameter update analysis
    print("\n--- Parameter Update Analysis ---")

    # Store parameter norms before update
    param_norms_before = {}
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            param_norms_before[name] = torch.norm(param).item()

    # Perform update
    trainer.optimizer.step()

    # Analyze parameter changes
    print(f"Parameter changes:")
    total_change = 0

    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            norm_before = param_norms_before[name]
            norm_after = torch.norm(param).item()
            change = norm_after - norm_before
            total_change += abs(change)

            if abs(change) > 0.001:  # Only show significant changes
                print(f"  {name}: {change:+.6f}")

    print(f"Total absolute parameter change: {total_change:.6f}")

    # Final statistics
    print("\n--- Final Statistics ---")
    print(f"Training step completed successfully!")
    print(f"Model parameters updated: {total_change > 0}")
    print(f"Loss computed: {loss_dict['total_loss'].item():.6f}")
    print(f"Gradients clipped: {total_grad_norm_after < total_grad_norm_before}")


if __name__ == "__main__":
    detailed_training_analysis()
