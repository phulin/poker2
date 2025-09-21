#!/usr/bin/env python3
"""
SelfPlayTrainer Memory Test - Monitor MPS memory usage during training
"""

import time

import psutil
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from alphaholdem.rl.self_play import SelfPlayTrainer


def get_mps_memory_info():
    """Get current MPS memory information."""
    if not torch.backends.mps.is_available():
        return {"error": "MPS not available"}

    info = {
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }

    try:
        if hasattr(torch.mps, "current_allocated_memory"):
            info["current_allocated"] = torch.mps.current_allocated_memory()
        if hasattr(torch.mps, "driver_allocated_memory"):
            info["driver_allocated"] = torch.mps.driver_allocated_memory()
    except Exception as e:
        info["memory_error"] = str(e)

    return info


def get_system_memory():
    """Get system memory usage."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent_used": memory.percent,
    }


def print_memory_status(label):
    """Print current memory status."""
    print(f"\n📊 {label}")
    print("-" * 50)

    # System memory
    sys_mem = get_system_memory()
    print(
        f"System Memory: {sys_mem['used_gb']:.2f} GB used / {sys_mem['total_gb']:.2f} GB total ({sys_mem['percent_used']:.1f}%)"
    )

    # MPS memory
    mps_info = get_mps_memory_info()
    if "current_allocated" in mps_info:
        current_mb = mps_info["current_allocated"] / (1024**2)
        print(f"MPS Memory: {current_mb:.2f} MB allocated")
    if "driver_allocated" in mps_info:
        driver_mb = mps_info["driver_allocated"] / (1024**2)
        print(f"MPS Driver: {driver_mb:.2f} MB")


def analyze_trainer_memory(trainer):
    """Analyze memory usage of trainer components."""
    print(f"\n🔍 Trainer Memory Analysis")
    print("-" * 50)

    # Model parameters
    model_params = sum(p.numel() for p in trainer.model.parameters())
    model_bytes = model_params * 4  # Assuming float32
    print(f"Model Parameters: {model_params:,} ({model_bytes / (1024**2):.2f} MB)")

    # Replay buffer tensors
    buffer_tensors = [
        ("cards_features", trainer.replay_buffer.cards_features),
        ("actions_features", trainer.replay_buffer.actions_features),
        ("action_indices", trainer.replay_buffer.action_indices),
        ("legal_masks", trainer.replay_buffer.legal_masks),
        ("log_probs", trainer.replay_buffer.log_probs),
        ("rewards", trainer.replay_buffer.rewards),
        ("dones", trainer.replay_buffer.dones),
        ("delta2", trainer.replay_buffer.delta2),
        ("delta3", trainer.replay_buffer.delta3),
        ("values", trainer.replay_buffer.values),
        ("advantages", trainer.replay_buffer.advantages),
        ("returns", trainer.replay_buffer.returns),
    ]

    total_buffer_bytes = 0
    print(f"\nReplay Buffer Tensors:")
    print(f"{'Tensor':<20} {'Shape':<25} {'Dtype':<10} {'Size (MB)':<12}")
    print("-" * 75)

    for name, tensor in buffer_tensors:
        if tensor.device.type == "mps":
            bytes_size = tensor.numel() * tensor.element_size()
            total_buffer_bytes += bytes_size
            mb_size = bytes_size / (1024**2)
            shape_str = str(list(tensor.shape))
            if len(shape_str) > 24:
                shape_str = shape_str[:21] + "..."
            print(
                f"{name:<20} {shape_str:<25} {str(tensor.dtype):<10} {mb_size:<12.2f}"
            )

    print("-" * 75)
    print(
        f"{'TOTAL BUFFER':<20} {'':<25} {'':<10} {total_buffer_bytes / (1024**2):<12.2f}"
    )

    # Environment tensors (if using tensor env)
    if hasattr(trainer, "env") and trainer.env is not None:
        env_memory = 0
        print(f"\nEnvironment Tensors:")
        for attr_name in dir(trainer.env):
            attr = getattr(trainer.env, attr_name)
            if isinstance(attr, torch.Tensor) and attr.device.type == "mps":
                bytes_size = attr.numel() * attr.element_size()
                env_memory += bytes_size
                print(f"  {attr_name}: {bytes_size / (1024**2):.2f} MB")

        if env_memory > 0:
            print(f"  Total Environment: {env_memory / (1024**2):.2f} MB")


def main():
    """Main function."""
    print("🚀 SelfPlayTrainer Memory Test")
    print("=" * 60)

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("❌ MPS is not available on this system")
        return

    device = torch.device("mps")
    print(f"Using device: {device}")

    # Print initial memory status
    print_memory_status("Initial Memory Status")

    # Load high_perf config
    print(f"\n📋 Loading config_high_perf...")
    GlobalHydra.instance().clear()
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config_high_perf")

    # Override device to MPS and reduce replay buffer batches
    cfg.device = "mps"
    cfg.use_wandb = False  # Disable wandb for cleaner output
    cfg.train.use_mixed_precision = False
    cfg.train.replay_buffer_batches = 2  # Reduce memory usage

    print(f"Config loaded:")
    print(f"  Batch size: {cfg.train.batch_size}")
    print(f"  Replay buffer batches: {cfg.train.replay_buffer_batches}")
    print(f"  Max trajectory length: {cfg.train.max_trajectory_length}")
    print(f"  Number of environments: {cfg.num_envs}")
    print(f"  Device: {cfg.device}")

    # Create trainer
    print(f"\n🏗️ Creating SelfPlayTrainer...")
    trainer = SelfPlayTrainer(cfg=cfg, device=device)

    print_memory_status("After Trainer Creation")
    analyze_trainer_memory(trainer)

    # Run one training step
    print(f"\n🏃 Running one training step...")
    start_time = time.time()

    try:
        # Hook into the update_model method to capture batch size
        original_update_model = trainer.update_model

        def wrapped_update_model():
            # Call original method but capture the batch info
            result = original_update_model()

            # Get batch info from the replay buffer (it should have been sampled)
            if trainer.replay_buffer.num_steps() > 0:
                # Sample a small batch just to get the shapes (we'll discard it)
                sample_batch = trainer.replay_buffer.sample_batch(
                    torch.Generator(device=trainer.device),
                    min(trainer.batch_size, trainer.replay_buffer.num_steps()),
                )
                batch_size = sample_batch.embedding_data.cards.shape[0]
                print(f"\n📦 Sample Batch Size: {batch_size}")
                print(f"   Cards shape: {sample_batch.embedding_data.cards.shape}")
                print(f"   Actions shape: {sample_batch.embedding_data.actions.shape}")
                print(f"   Values shape: {sample_batch.returns.shape}")
                print(f"   Logits shape: {sample_batch.all_log_probs.shape}")

            return result

        trainer.update_model = wrapped_update_model

        stats = trainer.train_step(1)
        step_time = time.time() - start_time

        print(f"✅ Training step completed in {step_time:.2f} seconds")
        print(f"Training stats: {stats}")

    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback

        traceback.print_exc()
        return

    print_memory_status("After Training Step")

    # Final analysis
    print(f"\n📈 Memory Usage Summary")
    print("-" * 50)

    initial_mps = get_mps_memory_info()
    final_mps = get_mps_memory_info()

    if "current_allocated" in initial_mps and "current_allocated" in final_mps:
        initial_mb = initial_mps["current_allocated"] / (1024**2)
        final_mb = final_mps["current_allocated"] / (1024**2)
        print(f"MPS Memory: {initial_mb:.2f} MB → {final_mb:.2f} MB")
        print(f"Memory Change: {final_mb - initial_mb:+.2f} MB")

    print(f"\n✅ Memory test complete!")


if __name__ == "__main__":
    main()
