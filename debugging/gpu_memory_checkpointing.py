#!/usr/bin/env python3
"""
MPS Memory Comparison with Gradient Checkpointing

This script compares memory usage with and without gradient checkpointing on MPS.
"""

import sys
import time

import torch

from alphaholdem.models.cnn import SiameseConvNetV1


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def get_memory_info(device):
    """Get current memory information."""
    if device.type == "cuda":
        return torch.cuda.memory_allocated()
    elif device.type == "mps":
        return torch.mps.current_allocated_memory()
    else:
        return 0


def test_gpu_memory_checkpointing():
    """Test memory usage with and without gradient checkpointing on GPU."""

    print("🧪 GPU Memory Comparison with Gradient Checkpointing")
    print("=" * 70)

    # Check for available devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS"
    else:
        print("❌ No GPU available on this system")
        return

    print(f"🖥️  Device: {device_name} ({device})")

    # Test with your actual config settings
    if device.type == "cuda":
        batch_size = 2000  # Larger batch for CUDA
    else:
        batch_size = 1000  # Smaller batch for MPS
    cards_hidden = 256
    actions_hidden = 256

    print(f"📊 Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Cards hidden: {cards_hidden}")
    print(f"  Actions hidden: {actions_hidden}")

    # Test 1: Without checkpointing
    print(f"\n🔍 Test 1: Without Gradient Checkpointing")
    print("-" * 50)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=cards_hidden,
        actions_hidden=actions_hidden,
        fusion_hidden=[2048, 2048],
        num_actions=8,
        use_gradient_checkpointing=False,
    ).to(device)

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()
        time.sleep(0.1)  # Give MPS time to clear

    # Create input tensors
    cards_tensor = torch.randn(batch_size, 6, 4, 13, device=device, dtype=torch.float32)
    actions_tensor = torch.randn(
        batch_size, 24, 4, 8, device=device, dtype=torch.float32
    )

    memory_after_inputs = get_memory_info(device)
    print(f"  Memory after input creation: {format_bytes(memory_after_inputs)}")

    # Forward pass
    start_time = time.time()
    logits, values = model(cards_tensor, actions_tensor)
    forward_time_no_checkpoint = time.time() - start_time

    memory_after_forward = get_memory_info(device)
    print(f"  Memory after forward pass: {format_bytes(memory_after_forward)}")
    print(f"  Forward pass time: {forward_time_no_checkpoint:.3f}s")

    # Backward pass
    start_time = time.time()
    loss = logits.sum() + values.sum()
    loss.backward()
    backward_time_no_checkpoint = time.time() - start_time

    memory_after_backward = get_memory_info(device)
    if device.type == "cuda":
        peak_memory_no_checkpoint = torch.cuda.max_memory_allocated()
        print(f"  Memory after backward pass: {format_bytes(memory_after_backward)}")
        print(
            f"  Peak memory (no checkpoint): {format_bytes(peak_memory_no_checkpoint)}"
        )
    else:
        print(f"  Memory after backward pass: {format_bytes(memory_after_backward)}")
    print(f"  Backward pass time: {backward_time_no_checkpoint:.3f}s")

    # Test 2: With checkpointing
    print(f"\n🔍 Test 2: With Gradient Checkpointing")
    print("-" * 50)

    model.use_gradient_checkpointing = True

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()
        time.sleep(0.1)  # Give MPS time to clear

    # Forward pass
    start_time = time.time()
    logits, values = model(cards_tensor, actions_tensor)
    forward_time_with_checkpoint = time.time() - start_time

    memory_after_forward_checkpoint = get_memory_info(device)
    print(
        f"  Memory after forward pass: {format_bytes(memory_after_forward_checkpoint)}"
    )
    print(f"  Forward pass time: {forward_time_with_checkpoint:.3f}s")

    # Backward pass
    start_time = time.time()
    loss = logits.sum() + values.sum()
    loss.backward()
    backward_time_with_checkpoint = time.time() - start_time

    memory_after_backward_checkpoint = get_memory_info(device)
    if device.type == "cuda":
        peak_memory_with_checkpoint = torch.cuda.max_memory_allocated()
        print(
            f"  Memory after backward pass: {format_bytes(memory_after_backward_checkpoint)}"
        )
        print(
            f"  Peak memory (with checkpoint): {format_bytes(peak_memory_with_checkpoint)}"
        )
    else:
        print(
            f"  Memory after backward pass: {format_bytes(memory_after_backward_checkpoint)}"
        )
    print(f"  Backward pass time: {backward_time_with_checkpoint:.3f}s")

    # Compare results
    print(f"\n📊 Comparison Results:")
    print("=" * 50)

    # Memory comparison
    memory_diff_forward = memory_after_forward - memory_after_forward_checkpoint
    memory_diff_backward = memory_after_backward - memory_after_backward_checkpoint

    print(f"Memory Usage:")
    print(f"  Forward pass difference: {format_bytes(memory_diff_forward)}")
    print(f"  Backward pass difference: {format_bytes(memory_diff_backward)}")

    if device.type == "cuda":
        peak_memory_diff = peak_memory_no_checkpoint - peak_memory_with_checkpoint
        print(f"  Peak memory difference: {format_bytes(peak_memory_diff)}")
        if peak_memory_diff > 0:
            peak_savings_percent = (peak_memory_diff / peak_memory_no_checkpoint) * 100
            print(
                f"  ✅ Peak memory savings: {format_bytes(peak_memory_diff)} ({peak_savings_percent:.1f}%)"
            )
        else:
            print(f"  ⚠️  Peak memory increase: {format_bytes(-peak_memory_diff)}")

    if memory_diff_forward > 0:
        print(f"  ✅ Forward pass memory savings: {format_bytes(memory_diff_forward)}")
    else:
        print(
            f"  ⚠️  Forward pass memory increase: {format_bytes(-memory_diff_forward)}"
        )

    if memory_diff_backward > 0:
        print(
            f"  ✅ Backward pass memory savings: {format_bytes(memory_diff_backward)}"
        )
    else:
        print(
            f"  ⚠️  Backward pass memory increase: {format_bytes(-memory_diff_backward)}"
        )

    # Time comparison
    time_diff_forward = forward_time_with_checkpoint - forward_time_no_checkpoint
    time_diff_backward = backward_time_with_checkpoint - backward_time_no_checkpoint

    print(f"\nTime Performance:")
    print(f"  Forward pass time difference: {time_diff_forward:+.3f}s")
    print(f"  Backward pass time difference: {time_diff_backward:+.3f}s")

    if time_diff_forward > 0:
        print(f"  ⚠️  Forward pass slower by {time_diff_forward:.3f}s (expected)")
    else:
        print(f"  ✅ Forward pass faster by {-time_diff_forward:.3f}s")

    if time_diff_backward > 0:
        print(f"  ⚠️  Backward pass slower by {time_diff_backward:.3f}s (expected)")
    else:
        print(f"  ✅ Backward pass faster by {-time_diff_backward:.3f}s")

    # Analysis
    print(f"\n💡 Analysis:")
    if memory_diff_forward > 0 or memory_diff_backward > 0:
        print(f"  ✅ Gradient checkpointing is providing memory savings")
        print(f"  📈 Memory vs Time trade-off: Less memory, more compute time")
    else:
        print(f"  ⚠️  Minimal memory savings detected")
        print(f"  🔍 Possible reasons:")
        print(f"     - Hidden sizes too small (256) for significant savings")
        print(f"     - MPS memory management overhead")
        print(f"     - Batch size too small to see differences")

    if device.type == "mps":
        print(f"\n⚠️  MPS Limitations:")
        print(f"  - MPS doesn't provide peak memory tracking like CUDA")
        print(f"  - Memory measurements are snapshots, not peak usage")
        print(f"  - Gradient checkpointing benefits may be real but not measurable")
    elif device.type == "cuda":
        print(f"\n✅ CUDA Benefits:")
        print(f"  - Peak memory tracking available")
        print(f"  - More accurate memory measurements")
        print(f"  - Better gradient checkpointing support")


def main():
    """Main test function."""
    test_gpu_memory_checkpointing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
