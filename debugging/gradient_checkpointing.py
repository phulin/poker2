#!/usr/bin/env python3
"""
Test Gradient Checkpointing Implementation

This script tests the gradient checkpointing implementation to verify
it's working correctly and measure memory usage.
"""

import torch

from alphaholdem.models.cnn import SiameseConvNetV1


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def test_gradient_checkpointing(batch_size: int = 1000):
    """Test gradient checkpointing implementation."""

    print(f"🧪 Testing Gradient Checkpointing - Batch Size: {batch_size:,}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Test without gradient checkpointing
    print(f"\n📊 Test 1: Without Gradient Checkpointing")
    model_no_checkpoint = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[2048, 2048],
        num_actions=8,
        use_gradient_checkpointing=False,
    ).to(device)

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Create input tensors
    cards_tensor = torch.randn(batch_size, 6, 4, 13, device=device, dtype=torch.float32)
    actions_tensor = torch.randn(
        batch_size, 24, 4, 8, device=device, dtype=torch.float32
    )

    # Forward pass
    logits, values = model_no_checkpoint(cards_tensor, actions_tensor)

    # Backward pass
    loss = logits.sum() + values.sum()
    loss.backward()

    if device.type == "cuda":
        memory_no_checkpoint = torch.cuda.max_memory_allocated()
        print(f"  Peak Memory: {format_bytes(memory_no_checkpoint)}")

    # Test with gradient checkpointing
    print(f"\n📊 Test 2: With Gradient Checkpointing")
    model_with_checkpoint = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[2048, 2048],
        num_actions=8,
        use_gradient_checkpointing=True,
    ).to(device)

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Forward pass
    logits, values = model_with_checkpoint(cards_tensor, actions_tensor)

    # Backward pass
    loss = logits.sum() + values.sum()
    loss.backward()

    if device.type == "cuda":
        memory_with_checkpoint = torch.cuda.max_memory_allocated()
        print(f"  Peak Memory: {format_bytes(memory_with_checkpoint)}")

        # Compare memory usage
        if memory_no_checkpoint > 0 and memory_with_checkpoint > 0:
            memory_savings = memory_no_checkpoint - memory_with_checkpoint
            savings_percent = (memory_savings / memory_no_checkpoint) * 100

            print(f"\n🎯 Memory Comparison:")
            print(f"  Without Checkpointing: {format_bytes(memory_no_checkpoint)}")
            print(f"  With Checkpointing: {format_bytes(memory_with_checkpoint)}")
            print(
                f"  Memory Savings: {format_bytes(memory_savings)} ({savings_percent:.1f}%)"
            )

            if savings_percent > 20:
                print(f"  ✅ Significant memory savings achieved!")
            elif savings_percent > 5:
                print(f"  ✅ Moderate memory savings achieved")
            else:
                print(f"  ⚠️  Minimal memory savings")

    # Test that outputs are the same
    print(f"\n🔍 Testing Output Consistency:")

    # Clear memory and reset models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Test with same inputs
    cards_tensor = torch.randn(batch_size, 6, 4, 13, device=device, dtype=torch.float32)
    actions_tensor = torch.randn(
        batch_size, 24, 4, 8, device=device, dtype=torch.float32
    )

    # Forward pass without checkpointing
    model_no_checkpoint.eval()
    with torch.no_grad():
        logits_no_checkpoint, values_no_checkpoint = model_no_checkpoint(
            cards_tensor, actions_tensor
        )

    # Forward pass with checkpointing
    model_with_checkpoint.eval()
    with torch.no_grad():
        logits_with_checkpoint, values_with_checkpoint = model_with_checkpoint(
            cards_tensor, actions_tensor
        )

    # Check if outputs are close (should be identical)
    logits_diff = torch.abs(logits_no_checkpoint - logits_with_checkpoint).max().item()
    values_diff = torch.abs(values_no_checkpoint - values_with_checkpoint).max().item()

    print(f"  Logits difference: {logits_diff:.2e}")
    print(f"  Values difference: {values_diff:.2e}")

    if logits_diff < 1e-6 and values_diff < 1e-6:
        print(f"  ✅ Outputs are identical - checkpointing working correctly!")
    else:
        print(f"  ⚠️  Outputs differ - checkpointing may have issues")

    print(f"\n✅ Gradient checkpointing test completed!")


def main():
    """Main test function."""

    if not torch.cuda.is_available():
        print(
            "❌ CUDA not available. This test requires CUDA for accurate memory measurement."
        )
        return 1

    print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
    print(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )

    # Test with different batch sizes
    for batch_size in [1000, 2000, 4000]:
        try:
            test_gradient_checkpointing(batch_size)
            print("\n" + "=" * 80 + "\n")
        except Exception as e:
            print(f"❌ Error with batch_size {batch_size}: {e}")
            continue

    return 0


if __name__ == "__main__":
    main()
