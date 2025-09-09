#!/usr/bin/env python3
"""
MPS Memory Monitor - Track actual memory usage on Apple Silicon GPUs
"""

import torch
import psutil
import time
from typing import Dict, Any


def get_mps_memory_info() -> Dict[str, Any]:
    """Get MPS memory information."""
    if not torch.backends.mps.is_available():
        return {"error": "MPS not available"}

    # Get MPS memory info
    mps_info = {
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }

    # Try to get memory stats (if available)
    try:
        # This might not be available in all PyTorch versions
        if hasattr(torch.mps, "current_allocated_memory"):
            mps_info["current_allocated"] = torch.mps.current_allocated_memory()
        if hasattr(torch.mps, "driver_allocated_memory"):
            mps_info["driver_allocated"] = torch.mps.driver_allocated_memory()
    except Exception as e:
        mps_info["memory_error"] = str(e)

    return mps_info


def get_system_memory_info() -> Dict[str, Any]:
    """Get system memory information."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent_used": memory.percent,
    }


def get_tensor_memory_usage(tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Calculate memory usage of tensors."""
    total_bytes = 0
    tensor_info = {}

    for name, tensor in tensors.items():
        if tensor.device.type == "mps":
            tensor_bytes = tensor.numel() * tensor.element_size()
            total_bytes += tensor_bytes
            tensor_info[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "bytes": tensor_bytes,
                "mb": tensor_bytes / (1024**2),
            }

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024**2),
        "total_gb": total_bytes / (1024**3),
        "tensors": tensor_info,
    }


def monitor_mps_memory():
    """Monitor MPS memory usage."""
    print("🔍 MPS Memory Monitor")
    print("=" * 50)

    # Check MPS availability
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    print()

    # System memory
    sys_mem = get_system_memory_info()
    print("💻 System Memory:")
    print(f"  Total: {sys_mem['total_gb']:.2f} GB")
    print(f"  Available: {sys_mem['available_gb']:.2f} GB")
    print(f"  Used: {sys_mem['used_gb']:.2f} GB ({sys_mem['percent_used']:.1f}%)")
    print()

    # MPS memory info
    mps_info = get_mps_memory_info()
    print("🎯 MPS Memory Info:")
    for key, value in mps_info.items():
        if key != "error":
            print(f"  {key}: {value}")
    if "error" in mps_info:
        print(f"  Error: {mps_info['error']}")
    print()


def create_test_tensors():
    """Create test tensors to measure memory usage."""
    device = torch.device("mps")

    print("🧪 Creating Test Tensors...")

    # Create various sized tensors
    tensors = {}

    # Small tensor
    tensors["small"] = torch.randn(1000, 1000, device=device)

    # Medium tensor
    tensors["medium"] = torch.randn(5000, 5000, device=device)

    # Large tensor (similar to our replay buffer)
    tensors["large"] = torch.randn(10000, 10000, device=device)

    # Bool tensor (like our optimized replay buffer)
    tensors["bool_tensor"] = torch.randn(10000, 10000, device=device).bool()

    # Float32 tensor (like original replay buffer)
    tensors["float32_tensor"] = torch.randn(
        10000, 10000, device=device, dtype=torch.float32
    )

    return tensors


def measure_tensor_memory():
    """Measure memory usage of different tensor types."""
    print("📊 Tensor Memory Measurement")
    print("=" * 50)

    device = torch.device("mps")

    # Test different tensor sizes and types
    test_cases = [
        ("Small Float32", torch.randn(1000, 1000, device=device, dtype=torch.float32)),
        ("Small Bool", torch.randn(1000, 1000, device=device).bool()),
        ("Medium Float32", torch.randn(5000, 5000, device=device, dtype=torch.float32)),
        ("Medium Bool", torch.randn(5000, 5000, device=device).bool()),
        (
            "Large Float32",
            torch.randn(10000, 10000, device=device, dtype=torch.float32),
        ),
        ("Large Bool", torch.randn(10000, 10000, device=device).bool()),
    ]

    print(
        f"{'Tensor Type':<20} {'Shape':<15} {'Dtype':<8} {'Size (MB)':<12} {'Size (GB)':<10}"
    )
    print("-" * 80)

    total_float32_mb = 0
    total_bool_mb = 0

    for name, tensor in test_cases:
        bytes_size = tensor.numel() * tensor.element_size()
        mb_size = bytes_size / (1024**2)
        gb_size = bytes_size / (1024**3)

        print(
            f"{name:<20} {str(list(tensor.shape)):<15} {str(tensor.dtype):<8} {mb_size:<12.2f} {gb_size:<10.4f}"
        )

        if tensor.dtype == torch.float32:
            total_float32_mb += mb_size
        elif tensor.dtype == torch.bool:
            total_bool_mb += mb_size

    print("-" * 80)
    print(
        f"{'Total Float32':<20} {'':<15} {'':<8} {total_float32_mb:<12.2f} {total_float32_mb/1024:<10.4f}"
    )
    print(
        f"{'Total Bool':<20} {'':<15} {'':<8} {total_bool_mb:<12.2f} {total_bool_mb/1024:<10.4f}"
    )

    savings_mb = total_float32_mb - total_bool_mb
    savings_percent = (
        (savings_mb / total_float32_mb) * 100 if total_float32_mb > 0 else 0
    )

    print(
        f"{'Memory Savings':<20} {'':<15} {'':<8} {savings_mb:<12.2f} {savings_mb/1024:<10.4f}"
    )
    print(f"Savings Percentage: {savings_percent:.1f}%")
    print()


def simulate_replay_buffer_memory():
    """Simulate our actual replay buffer memory usage."""
    print("🎮 Replay Buffer Memory Simulation")
    print("=" * 50)

    device = torch.device("mps")

    # Simulate our actual replay buffer dimensions
    capacity = 90112  # trajectories
    max_trajectory_length = 20  # steps per trajectory
    num_bet_bins = 8  # action bins

    print(f"Buffer Configuration:")
    print(f"  Capacity: {capacity:,} trajectories")
    print(f"  Max trajectory length: {max_trajectory_length} steps")
    print(f"  Number of bet bins: {num_bet_bins}")
    print()

    # Create tensors similar to our replay buffer
    tensors = {}

    # Cards features: (capacity, max_trajectory_length, 6, 4, 13)
    tensors["cards_features"] = torch.zeros(
        capacity, max_trajectory_length, 6, 4, 13, dtype=torch.bool, device=device
    )

    # Actions features: (capacity, max_trajectory_length, 24, 4, num_bet_bins)
    tensors["actions_features"] = torch.zeros(
        capacity,
        max_trajectory_length,
        24,
        4,
        num_bet_bins,
        dtype=torch.bool,
        device=device,
    )

    # Action indices: (capacity, max_trajectory_length)
    tensors["action_indices"] = torch.zeros(
        capacity, max_trajectory_length, dtype=torch.long, device=device
    )

    # Legal masks: (capacity, max_trajectory_length, num_bet_bins)
    tensors["legal_masks"] = torch.zeros(
        capacity, max_trajectory_length, num_bet_bins, dtype=torch.bool, device=device
    )

    # Other tensors (float32)
    other_tensors = [
        "log_probs",
        "rewards",
        "dones",
        "chips_placed",
        "delta2",
        "delta3",
        "values",
        "advantages",
        "returns",
    ]

    for name in other_tensors:
        tensors[name] = torch.zeros(
            capacity, max_trajectory_length, dtype=torch.float32, device=device
        )

    # Calculate memory usage
    memory_info = get_tensor_memory_usage(tensors)

    print("Memory Usage Breakdown:")
    print(
        f"{'Tensor':<20} {'Shape':<25} {'Dtype':<8} {'Size (MB)':<12} {'Size (GB)':<10}"
    )
    print("-" * 85)

    for name, info in memory_info["tensors"].items():
        shape_str = str(info["shape"])
        if len(shape_str) > 24:
            shape_str = shape_str[:21] + "..."

        print(
            f"{name:<20} {shape_str:<25} {info['dtype']:<8} {info['mb']:<12.2f} {info['mb']/1024:<10.4f}"
        )

    print("-" * 85)
    print(
        f"{'TOTAL':<20} {'':<25} {'':<8} {memory_info['total_mb']:<12.2f} {memory_info['total_gb']:<10.4f}"
    )
    print()

    # Compare with original (flattened) approach
    original_size = (
        capacity * max_trajectory_length * (6 * 4 * 13 + 24 * 4 * num_bet_bins) * 4
    )  # float32
    optimized_size = memory_info["total_bytes"]
    savings = original_size - optimized_size
    savings_percent = (savings / original_size) * 100

    print("Memory Comparison:")
    print(f"  Original (flattened float32): {original_size / (1024**3):.2f} GB")
    print(f"  Optimized (separate bool):     {optimized_size / (1024**3):.2f} GB")
    print(
        f"  Savings:                       {savings / (1024**3):.2f} GB ({savings_percent:.1f}%)"
    )
    print()


def main():
    """Main function."""
    print("🚀 MPS Memory Monitor")
    print("=" * 50)
    print()

    # Check if MPS is available
    if not torch.backends.mps.is_available():
        print("❌ MPS is not available on this system")
        return

    # Monitor basic MPS memory
    monitor_mps_memory()

    # Measure tensor memory usage
    measure_tensor_memory()

    # Simulate replay buffer memory
    simulate_replay_buffer_memory()

    print("✅ Memory monitoring complete!")


if __name__ == "__main__":
    main()
