#!/usr/bin/env python3
"""Test gradient clipping behavior."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.rl.losses import trinal_clip_ppo_loss
from alphaholdem.rl.replay import compute_gae_returns, prepare_ppo_batch


def test_gradient_clipping():
    print("=== Testing Gradient Clipping Behavior ===\n")

    # Initialize trainer
    trainer = SelfPlayTrainer(
        num_bet_bins=9,
        learning_rate=1e-4,
        batch_size=4,
        grad_clip=0.5,
    )

    # Collect a small batch
    trainer.replay_buffer.clear()
    trajectories = []
    for _ in range(2):
        trajectory = trainer.collect_trajectory()
        trajectories.append(trajectory)

    # Compute GAE
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

        advantages, returns = compute_gae_returns(
            rewards, values, gamma=trainer.gamma, lambda_=trainer.gae_lambda
        )

        for i, transition in enumerate(trajectory.transitions):
            transition.advantage = advantages[i]
            transition.return_ = returns[i]

    # Prepare batch
    batch = prepare_ppo_batch(trajectories)

    # Model forward pass
    observations = batch["observations"]
    cards = observations[:, : (6 * 4 * 13)].reshape(-1, 6, 4, 13)
    actions_tensor = observations[:, (6 * 4 * 13) :].reshape(
        -1, 24, 4, trainer.num_bet_bins
    )

    logits, values = trainer.model(cards, actions_tensor)

    # Compute loss
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
        delta2=-100.0,
        delta3=100.0,
        value_coef=trainer.value_coef,
        entropy_coef=trainer.entropy_coef,
    )

    print(f"Loss: {loss_dict['total_loss'].item():.6f}")

    # Test 1: Manual gradient clipping
    print("\n--- Test 1: Manual Gradient Clipping ---")
    trainer.optimizer.zero_grad()
    loss_dict["total_loss"].backward()

    # Compute total gradient norm manually
    total_norm = 0
    for p in trainer.model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)

    print(f"Manual total gradient norm: {total_norm:.6f}")
    print(f"Clip threshold: {trainer.grad_clip}")

    # Apply clipping manually
    clip_coef = min(1.0, trainer.grad_clip / total_norm)
    print(f"Clip coefficient: {clip_coef:.6f}")

    for p in trainer.model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)

    # Check norm after manual clipping
    total_norm_after = 0
    for p in trainer.model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** (1.0 / 2)

    print(f"Manual clipped norm: {total_norm_after:.6f}")

    # Test 2: PyTorch gradient clipping
    print("\n--- Test 2: PyTorch Gradient Clipping ---")
    trainer.optimizer.zero_grad()
    loss_dict["total_loss"].backward()

    # Use PyTorch's clip_grad_norm_
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), trainer.grad_clip
    )

    print(f"PyTorch clipped norm: {clipped_norm:.6f}")
    print(f"PyTorch clipping ratio: {clipped_norm / total_norm:.6f}")

    # Test 3: Check if gradients are actually clipped
    print("\n--- Test 3: Verify Clipping ---")

    # Reset and compute gradients again
    trainer.optimizer.zero_grad()
    loss_dict["total_loss"].backward()

    # Get gradients before clipping
    grads_before = {}
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grads_before[name] = param.grad.clone()

    # Apply clipping
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.grad_clip)

    # Check if gradients changed
    clipping_occurred = False
    for name, param in trainer.model.named_parameters():
        if param.grad is not None and name in grads_before:
            if not torch.allclose(param.grad, grads_before[name]):
                clipping_occurred = True
                print(f"Gradients changed for {name}")
                print(f"  Before: {torch.norm(grads_before[name]).item():.6f}")
                print(f"  After:  {torch.norm(param.grad).item():.6f}")

    if not clipping_occurred:
        print("No gradient clipping occurred!")

    # Test 4: Check total norm calculation
    print("\n--- Test 4: Total Norm Calculation ---")

    # Reset gradients
    trainer.optimizer.zero_grad()
    loss_dict["total_loss"].backward()

    # Method 1: Sum of squared norms
    total_norm_1 = 0
    for p in trainer.model.parameters():
        if p.grad is not None:
            total_norm_1 += p.grad.data.norm(2) ** 2
    total_norm_1 = total_norm_1**0.5

    # Method 2: Flatten and compute norm
    total_norm_2 = 0
    for p in trainer.model.parameters():
        if p.grad is not None:
            total_norm_2 += p.grad.data.flatten().norm(2) ** 2
    total_norm_2 = total_norm_2**0.5

    # Method 3: PyTorch's clip_grad_norm_ return value
    total_norm_3 = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), float("inf")
    )

    print(f"Method 1 (sum of squared norms): {total_norm_1:.6f}")
    print(f"Method 2 (flattened norms): {total_norm_2:.6f}")
    print(f"Method 3 (PyTorch clip_grad_norm_): {total_norm_3:.6f}")

    # Check if they match
    print(f"Methods 1 and 2 match: {abs(total_norm_1 - total_norm_2) < 1e-6}")
    print(f"Methods 1 and 3 match: {abs(total_norm_1 - total_norm_3) < 1e-6}")


if __name__ == "__main__":
    test_gradient_clipping()
