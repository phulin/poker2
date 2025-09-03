#!/usr/bin/env python3
"""
Estimate maximum batch size for AlphaHoldem model on 16GB RAM with GPU
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, "/Users/phulin/Documents/Projects/poker2")

from alphaholdem.models.siamese_convnet import SiameseConvNetV1
from alphaholdem.encoding.cards_encoder import CardsPlanesV1
from alphaholdem.encoding.actions_encoder import ActionsHUEncoderV1


def estimate_memory_usage(batch_size, model, device):
    """Estimate memory usage for a given batch size"""

    # Get initial memory
    if device.type == "mps":
        torch.mps.empty_cache()
        initial_memory = torch.mps.current_allocated_memory()
    else:
        initial_memory = 0

    # Input tensors
    cards_tensor = torch.randn(batch_size, 6, 4, 13, requires_grad=True, device=device)
    actions_tensor = torch.randn(
        batch_size, 24, 4, 13, requires_grad=True, device=device
    )

    # Forward pass
    logits, value = model(cards_tensor, actions_tensor)

    # Create a dummy loss that requires gradients
    loss = logits.sum() + value.sum()

    # Backward pass
    loss.backward()

    # Get peak memory usage
    if device.type == "mps":
        peak_memory = torch.mps.current_allocated_memory()
        memory_used = peak_memory - initial_memory
    else:
        # For CPU, estimate based on tensor sizes
        memory_used = (
            cards_tensor.numel() * 4 * 2  # cards tensor (float32) + gradients
            + actions_tensor.numel() * 4 * 2  # actions tensor + gradients
            + logits.numel() * 4 * 2  # logits + gradients
            + value.numel() * 4 * 2  # value + gradients
            + sum(p.numel() * 4 for p in model.parameters())
            * 2  # model params + gradients
        )

    # Clean up
    del cards_tensor, actions_tensor, logits, value, loss
    model.zero_grad()

    if device.type == "mps":
        torch.mps.empty_cache()

    return memory_used


def find_max_batch_size(target_memory_gb=14):  # Leave 2GB buffer
    """Find maximum batch size that fits in target memory"""

    print("🔍 Estimating maximum batch size for 16GB RAM with GPU...")
    print(f"Target memory usage: {target_memory_gb}GB (leaving 2GB buffer)")
    print()

    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = SiameseConvNetV1(
        cards_channels=6, actions_channels=24, fusion_hidden=256, num_actions=9
    )
    model.to(device)

    print("✅ Model initialized successfully")

    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    target_memory_bytes = target_memory_gb * 1024 * 1024 * 1024

    print("📊 Memory usage by batch size:")
    print("Batch Size | Memory (MB) | Memory (GB) | Fits in 16GB")
    print("-" * 55)

    max_safe_batch = 32

    for batch_size in batch_sizes:
        try:
            memory_bytes = estimate_memory_usage(batch_size, model, device)
            memory_mb = memory_bytes / (1024 * 1024)
            memory_gb = memory_bytes / (1024 * 1024 * 1024)

            fits = memory_bytes <= target_memory_bytes
            status = "✅" if fits else "❌"

            print(f"{batch_size:9d} | {memory_mb:10.1f} | {memory_gb:10.2f} | {status}")

            if fits:
                max_safe_batch = batch_size
            else:
                break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{batch_size:9d} | {'OOM':>10} | {'OOM':>10} | ❌")
                break
            else:
                print(f"Error with batch size {batch_size}: {e}")
                break
        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            break

    print()
    print(f"🎯 Recommended maximum batch size: {max_safe_batch}")

    # Calculate memory efficiency
    if max_safe_batch > 32:
        memory_bytes = estimate_memory_usage(max_safe_batch, model, device)
        memory_gb = memory_bytes / (1024 * 1024 * 1024)
        efficiency = (memory_gb / target_memory_gb) * 100
        print(f"📈 Memory efficiency: {efficiency:.1f}% of target")

    # Recommendations
    print()
    print("💡 Recommendations:")
    print(f"• Start with batch_size={max_safe_batch}")
    print(f"• For safety, use batch_size={max_safe_batch//2}")
    print("• Monitor memory usage during training")
    print("• Consider mixed precision training for 20-30% memory savings")

    return max_safe_batch


if __name__ == "__main__":
    find_max_batch_size()
