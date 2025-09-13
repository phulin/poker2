#!/usr/bin/env python3
"""Verify gradient clipping behavior."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from alphaholdem.rl.self_play import SelfPlayTrainer


def verify_gradient_clipping():
    print("=== Verifying Gradient Clipping Behavior ===\n")

    # Initialize trainer
    trainer = SelfPlayTrainer(
        learning_rate=1e-4,
        batch_size=4,
        grad_clip=0.5,
    )

    # Create a simple loss for testing
    dummy_input = torch.randn(1, 6, 4, 13)
    dummy_actions = torch.randn(1, 24, 4, 9)

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

    # Store gradients before clipping
    grads_before = {}
    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grads_before[name] = param.grad.clone()

    # Apply gradient clipping
    returned_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), trainer.grad_clip
    )

    print(f"Returned norm from clip_grad_norm_: {returned_norm:.6f}")
    print(f"Expected: {total_norm_before:.6f}")
    print(
        f"Returned norm matches expected: {abs(returned_norm - total_norm_before) < 1e-6}"
    )

    # Compute actual clipped norm
    total_norm_after = 0
    for p in trainer.model.parameters():
        if p.grad is not None:
            total_norm_after += p.grad.data.norm(2) ** 2
    total_norm_after = total_norm_after**0.5

    print(f"Actual clipped norm: {total_norm_after:.6f}")
    print(f"Clipping ratio: {total_norm_after / total_norm_before:.6f}")

    # Verify clipping occurred
    if total_norm_after < total_norm_before:
        print("✅ Gradient clipping worked!")
        print(
            f"Gradients reduced by factor: {total_norm_before / total_norm_after:.6f}"
        )
    else:
        print("❌ Gradient clipping failed!")

    # Check if gradients were actually modified
    gradients_changed = False
    for name, param in trainer.model.named_parameters():
        if param.grad is not None and name in grads_before:
            if not torch.allclose(param.grad, grads_before[name]):
                gradients_changed = True
                print(f"Gradients changed for {name}")
                print(f"  Before: {torch.norm(grads_before[name]).item():.6f}")
                print(f"  After:  {torch.norm(param.grad).item():.6f}")

    if not gradients_changed:
        print("❌ No gradients were modified!")
    else:
        print("✅ Gradients were successfully modified!")

    # Test the detailed analysis script's calculation
    print("\n--- Testing Detailed Analysis Script Calculation ---")

    # Reset for fresh test
    trainer.optimizer.zero_grad()
    loss.backward()

    # Method used in detailed analysis script
    total_grad_norm_before = 0
    grad_info = []

    for name, param in trainer.model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm_before += grad_norm**2
            grad_info.append((name, grad_norm))

    total_grad_norm_before = total_grad_norm_before**0.5

    print(f"Detailed analysis method: {total_grad_norm_before:.6f}")
    print(f"Direct calculation: {total_norm_before:.6f}")
    print(f"Methods match: {abs(total_grad_norm_before - total_norm_before) < 1e-6}")

    # Apply clipping and check return value
    clipped_grad_norm = torch.nn.utils.clip_grad_norm_(
        trainer.model.parameters(), trainer.grad_clip
    )

    print(f"Returned value: {clipped_grad_norm:.6f}")
    print(f"Expected (original norm): {total_grad_norm_before:.6f}")
    print(f"Actual clipped norm: {total_norm_after:.6f}")

    # The issue: detailed analysis script compares returned_norm with total_grad_norm_before
    # But returned_norm IS total_grad_norm_before, not the clipped norm!
    print(f"\n🚨 ISSUE FOUND:")
    print(f"The detailed analysis script compares:")
    print(
        f"  returned_norm ({clipped_grad_norm:.6f}) vs total_grad_norm_before ({total_grad_norm_before:.6f})"
    )
    print(f"But returned_norm IS the original norm, not the clipped norm!")
    print(f"The actual clipped norm is: {total_norm_after:.6f}")


if __name__ == "__main__":
    verify_gradient_clipping()
