#!/usr/bin/env python3
"""
GPU Memory Analysis Script for AlphaHoldem

This script sets up the model with config_high_perf and inspects GPU memory usage
for each component to help estimate VRAM requirements.
"""

import gc
import traceback
from typing import Any, Dict

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Import modules to register components
from alphaholdem.rl.self_play import SelfPlayTrainer


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def get_tensor_memory(tensor: torch.Tensor) -> int:
    """Get memory usage of a tensor in bytes."""
    if tensor.is_cuda or tensor.device.type == "mps":
        return tensor.element_size() * tensor.nelement()
    return 0


def analyze_model_memory(
    model: torch.nn.Module, device: torch.device
) -> Dict[str, Any]:
    """Analyze memory usage of model components."""
    model_info = {
        "total_params": 0,
        "trainable_params": 0,
        "components": {},
        "total_memory_bytes": 0,
    }

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )

            # Calculate memory for parameters
            param_memory = 0
            for param in module.parameters():
                if param.is_cuda or param.device.type == "mps":
                    param_memory += get_tensor_memory(param)

            if param_count > 0:
                model_info["components"][name] = {
                    "params": param_count,
                    "trainable_params": trainable_count,
                    "memory_bytes": param_memory,
                    "memory_formatted": format_bytes(param_memory),
                }
                model_info["total_params"] += param_count
                model_info["trainable_params"] += trainable_count
                model_info["total_memory_bytes"] += param_memory

    return model_info


def analyze_replay_buffer_memory(buffer) -> Dict[str, Any]:
    """Analyze memory usage of replay buffer tensors."""
    buffer_info = {"tensors": {}, "total_memory_bytes": 0}

    # Get all tensor attributes
    tensor_attrs = [
        "observations",
        "actions",
        "log_probs",
        "rewards",
        "dones",
        "legal_masks",
        "delta2",
        "delta3",
        "values",
        "advantages",
        "returns",
        "trajectory_lengths",
        "valid_trajectories",
        "current_step_positions",
    ]

    for attr_name in tensor_attrs:
        if hasattr(buffer, attr_name):
            tensor = getattr(buffer, attr_name)
            if isinstance(tensor, torch.Tensor):
                memory_bytes = get_tensor_memory(tensor)
                buffer_info["tensors"][attr_name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "memory_bytes": memory_bytes,
                    "memory_formatted": format_bytes(memory_bytes),
                }
                buffer_info["total_memory_bytes"] += memory_bytes

    return buffer_info


def analyze_environment_memory(env) -> Dict[str, Any]:
    """Analyze memory usage of environment tensors."""
    env_info = {"tensors": {}, "total_memory_bytes": 0}

    # Get tensor attributes from environment
    tensor_attrs = [
        "stacks",
        "pot",
        "actions_this_round",
        "street",
        "winner",
        "bet_bins_t",
        "history_actions",
        "history_slots",
    ]

    for attr_name in tensor_attrs:
        if hasattr(env, attr_name):
            tensor = getattr(env, attr_name)
            if isinstance(tensor, torch.Tensor):
                memory_bytes = get_tensor_memory(tensor)
                env_info["tensors"][attr_name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "memory_bytes": memory_bytes,
                    "memory_formatted": format_bytes(memory_bytes),
                }
                env_info["total_memory_bytes"] += memory_bytes

    return env_info


def main():
    """Main analysis function."""
    print("🔍 AlphaHoldem GPU Memory Analysis")
    print("=" * 50)

    # Check for available GPU (CUDA or MPS)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"🎯 Using device: {device}")
        print(f"🎯 GPU: {device_name}")
        print(f"🎯 Total GPU memory: {format_bytes(total_memory)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple Silicon GPU (MPS)"
        total_memory = 16 * 1024**3  # Estimate 16GB for M1/M2
        print(f"🎯 Using device: {device}")
        print(f"🎯 GPU: {device_name}")
        print(f"🎯 Estimated GPU memory: {format_bytes(total_memory)}")
    else:
        print("❌ No GPU available (CUDA or MPS). This script requires a GPU.")
        return

    print()

    # Clear GPU memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # Load high-performance config
    GlobalHydra.instance().clear()
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config_high_perf")

    # Disable wandb for memory analysis
    cfg.use_wandb = False

    print("📋 Configuration:")
    print(f"  batch_size: {cfg.train.batch_size}")
    print(f"  num_envs: {cfg.num_envs}")
    print(f"  max_trajectory_length: {cfg.train.max_trajectory_length}")
    print(f"  replay_buffer_batches: {cfg.train.replay_buffer_batches}")
    print(f"  use_mixed_precision: {cfg.train.use_mixed_precision}")
    print()

    # Calculate buffer capacity
    buffer_capacity = cfg.train.batch_size * max(1, 1 + cfg.train.replay_buffer_batches)
    print(f"📊 Buffer capacity: {buffer_capacity:,} trajectories")
    print(
        f"📊 Total buffer steps: {buffer_capacity * cfg.train.max_trajectory_length:,}"
    )
    print()

    try:
        # Create trainer
        print("🏗️  Creating SelfPlayTrainer...")
        trainer = SelfPlayTrainer(cfg=cfg, device=device)
        print("✅ SelfPlayTrainer created successfully")
        print()

        # Analyze model memory
        print("🧠 Model Memory Analysis:")
        print("-" * 30)
        model_info = analyze_model_memory(trainer.model, device)

        print(f"Total parameters: {model_info['total_params']:,}")
        print(f"Trainable parameters: {model_info['trainable_params']:,}")
        print(f"Total model memory: {format_bytes(model_info['total_memory_bytes'])}")
        print()

        print("Model components:")
        for name, info in model_info["components"].items():
            print(f"  {name}:")
            print(f"    Parameters: {info['params']:,}")
            print(f"    Memory: {info['memory_formatted']}")
        print()

        # Analyze replay buffer memory
        print("💾 Replay Buffer Memory Analysis:")
        print("-" * 35)
        buffer_info = analyze_replay_buffer_memory(trainer.replay_buffer)

        print(f"Total buffer memory: {format_bytes(buffer_info['total_memory_bytes'])}")
        print()

        print("Buffer tensors:")
        for name, info in buffer_info["tensors"].items():
            print(f"  {name}:")
            print(f"    Shape: {info['shape']}")
            print(f"    Dtype: {info['dtype']}")
            print(f"    Memory: {info['memory_formatted']}")
        print()

        # Analyze environment memory
        if hasattr(trainer, "tensor_env"):
            print("🎮 Environment Memory Analysis:")
            print("-" * 32)
            env_info = analyze_environment_memory(trainer.tensor_env)

            print(
                f"Total environment memory: {format_bytes(env_info['total_memory_bytes'])}"
            )
            print()

            print("Environment tensors:")
            for name, info in env_info["tensors"].items():
                print(f"  {name}:")
                print(f"    Shape: {info['shape']}")
                print(f"    Dtype: {info['dtype']}")
                print(f"    Memory: {info['memory_formatted']}")
            print()

        # Calculate total memory usage
        total_allocated_memory = (
            model_info["total_memory_bytes"]
            + buffer_info["total_memory_bytes"]
            + (env_info["total_memory_bytes"] if hasattr(trainer, "tensor_env") else 0)
        )

        print("📊 Summary:")
        print("=" * 20)
        print(f"Model memory: {format_bytes(model_info['total_memory_bytes'])}")
        print(f"Buffer memory: {format_bytes(buffer_info['total_memory_bytes'])}")
        if hasattr(trainer, "tensor_env"):
            print(f"Environment memory: {format_bytes(env_info['total_memory_bytes'])}")
        print(f"Total allocated: {format_bytes(total_allocated_memory)}")

        # Get actual GPU memory usage
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            print(f"GPU allocated: {format_bytes(allocated)}")
            print(f"GPU reserved: {format_bytes(reserved)}")

            # Memory efficiency
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            efficiency = (allocated / gpu_total) * 100
            print(f"Memory efficiency: {efficiency:.1f}%")
        else:
            # For MPS, we can't get exact memory usage, so use our estimates
            print(
                f"Estimated total memory usage: {format_bytes(total_allocated_memory)}"
            )
            efficiency = (total_allocated_memory / total_memory) * 100
            print(f"Memory efficiency: {efficiency:.1f}% (estimated)")

        # Recommendations
        print()
        print("💡 Recommendations:")
        if efficiency > 80:
            print("  ⚠️  High memory usage! Consider:")
            print("     - Reducing batch_size")
            print("     - Reducing num_envs")
            print("     - Reducing max_trajectory_length")
            print("     - Enabling mixed precision")
        elif efficiency > 60:
            print("  ⚡ Good memory usage. You could potentially:")
            print("     - Increase batch_size for better training")
            print("     - Increase num_envs for more parallel games")
        else:
            print("  🚀 Low memory usage! You have room to:")
            print("     - Significantly increase batch_size")
            print("     - Increase num_envs")
            print("     - Increase max_trajectory_length")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        # traceback imported at module top

        traceback.print_exc()

    finally:
        # Cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
