#!/usr/bin/env python3
"""
Forward Pass Memory Analysis

This script provides detailed memory breakdown during forward pass,
showing memory usage for model, gradients, activations, etc.
"""

import gc
import sys

import torch

from alphaholdem.models.cnn import SiameseConvNetV1


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def get_memory_usage():
    """Get current CUDA memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    return 0


def get_peak_memory():
    """Get peak CUDA memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def analyze_forward_pass_memory(batch_size: int, device: torch.device):
    """Analyze memory usage during forward pass step by step."""

    print(f"🔍 Forward Pass Memory Analysis - Batch Size: {batch_size:,}")
    print("=" * 80)

    # Create model
    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[2048, 2048],
        num_actions=8,
        use_gradient_checkpointing=True,
    ).to(device)

    # Clear memory and reset stats
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

    print(f"\n📊 Model Information:")
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = sum(p.element_size() * p.numel() for p in model.parameters())
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Parameter Memory: {format_bytes(param_memory)}")

    # Step 1: Model creation
    model_memory = get_memory_usage()
    print(f"\n🏗️  Step 1: Model Creation")
    print(f"  Memory: {format_bytes(model_memory)}")

    # Step 2: Input tensors
    cards_shape = (batch_size, 6, 4, 13)
    actions_shape = (batch_size, 24, 4, 8)

    cards_tensor = torch.randn(cards_shape, device=device, dtype=torch.float32)
    actions_tensor = torch.randn(actions_shape, device=device, dtype=torch.float32)

    input_memory = get_memory_usage()
    input_delta = input_memory - model_memory
    print(f"\n📥 Step 2: Input Tensors")
    print(f"  Cards Shape: {cards_shape}")
    print(f"  Actions Shape: {actions_shape}")
    print(f"  Memory: {format_bytes(input_memory)} (+{format_bytes(input_delta)})")

    # Step 3: Forward pass with detailed tracking
    print(f"\n⚡ Step 3: Forward Pass (Detailed)")

    # Track memory at each layer
    memory_snapshots = []

    def track_memory(name):
        current_memory = get_memory_usage()
        memory_snapshots.append((name, current_memory))
        return current_memory

    # Forward pass with memory tracking
    # Cards trunk
    track_memory("Before Cards Trunk")
    x_cards = model.cards_trunk(cards_tensor)
    track_memory("After Cards Trunk")

    # Actions trunk
    track_memory("Before Actions Trunk")
    x_actions = model.actions_trunk(actions_tensor)
    track_memory("After Actions Trunk")

    # Concatenation
    track_memory("Before Concatenation")
    x = torch.cat([x_cards, x_actions], dim=1)
    track_memory("After Concatenation")

    # Fusion layers
    track_memory("Before Fusion")
    h = model.fusion(x)
    track_memory("After Fusion")

    # Policy head
    track_memory("Before Policy Head")
    logits = model.policy_head(h)
    track_memory("After Policy Head")

    # Value head
    track_memory("Before Value Head")
    value = model.value_head(h).squeeze(-1)
    track_memory("After Value Head")

    # Print memory progression
    print(f"  Memory Progression:")
    for i, (name, memory) in enumerate(memory_snapshots):
        if i > 0:
            prev_memory = memory_snapshots[i - 1][1]
            delta = memory - prev_memory
            print(f"    {name}: {format_bytes(memory)} (+{format_bytes(delta)})")
        else:
            print(f"    {name}: {format_bytes(memory)}")

    # Step 4: Forward pass with gradients (autograd)
    print(f"\n🔄 Step 4: Forward Pass with Gradients")

    # Clear memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Forward pass with gradients
    logits, values = model(cards_tensor, actions_tensor)

    forward_memory = get_memory_usage()
    forward_delta = forward_memory - input_memory
    peak_memory = get_peak_memory()

    print(f"  Memory: {format_bytes(forward_memory)} (+{format_bytes(forward_delta)})")
    print(f"  Peak Memory: {format_bytes(peak_memory)}")

    # Step 5: Backward pass
    print(f"\n⬅️  Step 5: Backward Pass")

    loss = logits.sum() + values.sum()
    loss.backward()

    backward_memory = get_memory_usage()
    backward_delta = backward_memory - forward_memory
    peak_memory_after = get_peak_memory()

    print(
        f"  Memory: {format_bytes(backward_memory)} (+{format_bytes(backward_delta)})"
    )
    print(f"  Peak Memory: {format_bytes(peak_memory_after)}")

    # Step 6: Optimizer
    print(f"\n🔧 Step 6: Optimizer Creation")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()

    optimizer_memory = get_memory_usage()
    optimizer_delta = optimizer_memory - backward_memory
    final_peak = get_peak_memory()

    print(
        f"  Memory: {format_bytes(optimizer_memory)} (+{format_bytes(optimizer_delta)})"
    )
    print(f"  Peak Memory: {format_bytes(final_peak)}")

    # Summary
    print(f"\n📋 Memory Summary:")
    print(f"  Model Parameters: {format_bytes(param_memory)}")
    print(f"  Input Tensors: {format_bytes(input_delta)}")
    print(f"  Forward Pass: {format_bytes(forward_delta)}")
    print(f"  Backward Pass: {format_bytes(backward_delta)}")
    print(f"  Optimizer: {format_bytes(optimizer_delta)}")
    print(f"  Total Allocated: {format_bytes(optimizer_memory)}")
    print(f"  Peak Memory: {format_bytes(final_peak)}")

    # Memory efficiency
    theoretical_memory = (
        param_memory + input_delta + forward_delta + backward_delta + optimizer_delta
    )
    efficiency = (theoretical_memory / final_peak) * 100 if final_peak > 0 else 0

    print(f"\n🎯 Memory Efficiency:")
    print(f"  Theoretical: {format_bytes(theoretical_memory)}")
    print(f"  Actual Peak: {format_bytes(final_peak)}")
    print(f"  Efficiency: {efficiency:.1f}%")

    if efficiency < 50:
        print(f"  ⚠️  Low memory efficiency - significant overhead detected")
    elif efficiency < 80:
        print(f"  ⚠️  Moderate memory efficiency - some overhead")
    else:
        print(f"  ✅ Good memory efficiency")


def main():
    """Main analysis function."""

    # Test on CUDA if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🖥️  Testing on device: {device}")
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

        # Test with different batch sizes
        for batch_size in [1000, 2000, 4000]:
            try:
                analyze_forward_pass_memory(batch_size, device)
                print("\n" + "=" * 80 + "\n")
            except Exception as e:
                print(f"❌ Error with batch_size {batch_size}: {e}")
                continue
    else:
        print(
            "❌ CUDA not available. This script requires CUDA for accurate memory analysis."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
