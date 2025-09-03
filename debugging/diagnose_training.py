#!/usr/bin/env python3
"""Diagnostic script to debug training issues."""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaholdem.rl.self_play import SelfPlayTrainer


def main():
    print("=== AlphaHoldem Training Diagnostics ===\n")

    # Initialize trainer
    trainer = SelfPlayTrainer(
        num_bet_bins=9,
        learning_rate=3e-4,
        batch_size=8,
    )

    # 1. Check model health
    print("1. Model Health Check:")
    diagnostics = trainer.diagnose_model_health()
    print(f"   Total parameters: {diagnostics['total_params']:,}")
    print(f"   Trainable parameters: {diagnostics['trainable_params']:,}")
    print(f"   Has NaN: {diagnostics['has_nan']}")
    print(f"   Has Inf: {diagnostics['has_inf']}")
    print(f"   Learning rate: {diagnostics['optimizer_lr']}")
    print(f"   Replay buffer size: {diagnostics['replay_buffer_size']}")

    # 2. Test forward pass
    print(f"\n2. Forward Pass Test:")
    print(f"   Logits shape: {diagnostics['logits_shape']}")
    print(f"   Values shape: {diagnostics['values_shape']}")
    print(f"   Logits norm: {diagnostics['logits_norm']:.3f}")
    print(f"   Values norm: {diagnostics['values_norm']:.3f}")

    # 3. Test trajectory collection
    print(f"\n3. Trajectory Collection Test:")
    trajectory = trainer.collect_trajectory()
    print(f"   Trajectory length: {len(trajectory.transitions)}")
    print(f"   Total reward: {sum(t.reward for t in trajectory.transitions):.2f}")
    print(
        f"   Rewards range: {min(t.reward for t in trajectory.transitions):.2f} to {max(t.reward for t in trajectory.transitions):.2f}"
    )

    # 4. Test training step
    print(f"\n4. Training Step Test:")
    stats = trainer.train_step(num_trajectories=2)
    print(f"   Episodes: {stats['episode_count']}")
    print(f"   Avg reward: {stats['avg_reward']:.2f}")
    if "avg_loss" in stats:
        print(f"   Avg loss: {stats['avg_loss']:.6f}")

    # 5. Check parameter updates
    print(f"\n5. Parameter Update Check:")
    initial_norms = {}
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            initial_norms[name] = torch.norm(param).item()

    # Run a few more training steps
    for step in range(3):
        trainer.train_step(num_trajectories=2)

    final_norms = {}
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            final_norms[name] = torch.norm(param).item()

    print("   Parameter norm changes:")
    for name in initial_norms:
        change = final_norms[name] - initial_norms[name]
        print(f"     {name}: {change:+.6f}")

    print(f"\n=== Diagnostics Complete ===")


if __name__ == "__main__":
    main()
