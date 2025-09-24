#!/usr/bin/env python3
"""Simple gradient clipping test."""

import os
import sys

import torch

from alphaholdem.rl.self_play import SelfPlayTrainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def simple_gradient_clipping_test():
    print("=== Simple Gradient Clipping Test ===\n")

    # Initialize trainer
    trainer = SelfPlayTrainer()

    # Create a simple loss for testing
    dummy_input = torch.randn(1, 6, 4, 13, requires_grad=True)
    dummy_actions = torch.randn(1, 24, 4, 9, requires_grad=True)

    # Forward pass
    logits, values = trainer.model(dummy_input, dummy_actions)

    # Create a simple loss
    loss = (
        torch.sum(logits**2) + torch.sum(values**2) * 1000
    )  # High value loss to simulate our issue

    print(f"Loss: {loss.item():.6f}")

    # Zero gradients
    trainer.optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Compute total gradient norm before clipping
    total_norm_before = 0
    for p in trainer.model.parameters():
        if p.grad is not None:
            total_norm_before += p.grad.data.norm(2) ** 2
    total_norm_before = total_norm_before**0.5

    print(f"Total gradient norm before clipping: {total_norm_before:.6f}")
    print(f"Clip threshold: {trainer.grad_clip}")

    # Apply gradient clipping
    clipped_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), trainer.grad_clip
    )

    print(f"PyTorch clipped norm: {clipped_norm:.6f}")
    print(f"Clipping ratio: {clipped_norm / total_norm_before:.6f}")

    # Verify clipping occurred
    if clipped_norm < total_norm_before:
        print("✅ Gradient clipping worked!")
    else:
        print("❌ Gradient clipping failed!")

    # Check individual parameter gradients
    print("\nIndividual parameter gradients:")
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"  {name}: {grad_norm:.6f}")

    # Test with different clip values
    print("\n--- Testing Different Clip Values ---")

    for clip_val in [0.1, 0.5, 1.0, 5.0]:
        # Reset gradients
        trainer.optimizer.zero_grad()
        loss.backward()

        # Compute norm before
        total_norm = 0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2) ** 2
        total_norm = total_norm**0.5

        # Apply clipping
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), clip_val
        )

        print(
            f"Clip {clip_val}: {total_norm:.6f} -> {clipped_norm:.6f} (ratio: {clipped_norm/total_norm:.6f})"
        )


if __name__ == "__main__":
    simple_gradient_clipping_test()
