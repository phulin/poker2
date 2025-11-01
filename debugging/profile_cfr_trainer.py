#!/usr/bin/env python3
"""
Profile the RebelCFRTrainer training loop using PyTorch profiler.

This script focuses on identifying hotspots in set_leaf_values and update_policy
functions, showing detailed operator-level timing information.

Usage:
    python debugging/profile_cfr_trainer.py
    python debugging/profile_cfr_trainer.py num_envs=64 train.batch_size=256 search.iterations=30
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

from alphaholdem.core.structured_config import Config
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer


def profile_training_loop(cfg: Config):
    """Run the training loop with PyTorch profiler."""
    device = torch.device(cfg.device)

    print(f"Using device: {device}")
    print(f"Config: batch_size={cfg.train.batch_size}, num_envs={cfg.num_envs}")
    print(f"Search: depth={cfg.search.depth}, iterations={cfg.search.iterations}")
    print("\nInitializing trainer...")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device)

    print("Trainer initialized. Starting profiling...")
    print("=" * 80)

    # Configure profiler activities
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Use profiler with record_shapes and profile_memory for detailed analysis
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        # Run a few training steps with profiling
        num_profile_steps = cfg.num_steps

        for step in range(num_profile_steps):
            print(f"\nProfiling training step {step + 1}/{num_profile_steps}...")

            with record_function("train_step"):
                metrics = trainer.train_step(step)

            loss_val = metrics.get("loss", None)
            loss_str = f"{loss_val:.4f}" if loss_val is not None else "N/A"
            print(f"  Step {step + 1} completed. Loss: {loss_str}")

    print("\n" + "=" * 80)
    print("Profiling complete. Processing results...")

    # Export profiling results
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Key sorting for profiling table
    key_averages = prof.key_averages(group_by_input_shape=True)

    # Print summary table
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY - Top Operations by Self CPU Time")
    print("=" * 80)
    print(
        key_averages.table(
            sort_by="self_cpu_time_total",
            row_limit=50,
            max_name_column_width=80,
        )
    )

    # Filter and print operations related to set_leaf_values
    print("\n" + "=" * 80)
    print("SET_LEAF_VALUES HOTSPOTS")
    print("=" * 80)
    set_leaf_ops = [
        op
        for op in key_averages
        if "set_leaf_values" in op.key
        or any(
            "set_leaf_values" in str(stack_entry) for stack_entry in (op.stack or [])
        )
    ]

    if set_leaf_ops:
        # Sort by self CPU time
        set_leaf_ops_sorted = sorted(
            set_leaf_ops, key=lambda x: x.self_cpu_time_total, reverse=True
        )
        print(
            f"\nFound {len(set_leaf_ops_sorted)} operations related to set_leaf_values\n"
        )

        # Print top 30
        for i, op in enumerate(set_leaf_ops_sorted[:30], 1):
            print(f"{i:2d}. {op.key}")
            print(f"    Self CPU: {op.self_cpu_time_total / 1e6:.3f} ms")
            if hasattr(op, "self_cuda_time_total") and op.self_cuda_time_total > 0:
                print(f"    Self CUDA: {op.self_cuda_time_total / 1e6:.3f} ms")
            print(f"    Calls: {op.count}")
            if op.input_shapes:
                print(f"    Input shapes: {op.input_shapes[:3]}...")  # Show first 3
            print()
    else:
        print("No operations found related to set_leaf_values")

    # Filter and print operations related to update_policy
    print("\n" + "=" * 80)
    print("UPDATE_POLICY HOTSPOTS")
    print("=" * 80)
    update_policy_ops = [
        op
        for op in key_averages
        if "update_policy" in op.key
        or any("update_policy" in str(stack_entry) for stack_entry in (op.stack or []))
    ]

    if update_policy_ops:
        # Sort by self CPU time
        update_policy_ops_sorted = sorted(
            update_policy_ops, key=lambda x: x.self_cpu_time_total, reverse=True
        )
        print(
            f"\nFound {len(update_policy_ops_sorted)} operations related to update_policy\n"
        )

        # Print top 30
        for i, op in enumerate(update_policy_ops_sorted[:30], 1):
            print(f"{i:2d}. {op.key}")
            print(f"    Self CPU: {op.self_cpu_time_total / 1e6:.3f} ms")
            if hasattr(op, "self_cuda_time_total") and op.self_cuda_time_total > 0:
                print(f"    Self CUDA: {op.self_cuda_time_total / 1e6:.3f} ms")
            print(f"    Calls: {op.count}")
            if op.input_shapes:
                print(f"    Input shapes: {op.input_shapes[:3]}...")  # Show first 3
            print()
    else:
        print("No operations related to update_policy found")

    # Export to Chrome trace format
    trace_file = output_dir / f"profile_cfr_trainer_{timestamp}.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\nChrome trace exported to: {trace_file}")
    print("Open this file in chrome://tracing or use Perfetto to visualize")

    # Export stack traces
    stack_file = output_dir / f"profile_cfr_trainer_stacks_{timestamp}.txt"
    try:
        stack_averages = prof.key_averages(group_by_stack_n=10)
        with open(stack_file, "w") as f:
            f.write(
                stack_averages.table(
                    sort_by="self_cpu_time_total",
                    row_limit=100,
                )
            )
        print(f"Stack traces exported to: {stack_file}")
    except Exception as e:
        print(f"Could not export stack traces: {e}")

    # Export detailed table sorted by different metrics
    for metric_name, sort_key in [
        ("cpu_time", "cpu_time_total"),
        ("cuda_time", "cuda_time_total"),
        ("memory", "cpu_memory_usage"),
    ]:
        table_file = output_dir / f"profile_cfr_trainer_{metric_name}_{timestamp}.txt"
        try:
            table_text = key_averages.table(
                sort_by=sort_key,
                row_limit=100,
                max_name_column_width=100,
            )
            with open(table_file, "w") as f:
                f.write(table_text)
            print(f"Table sorted by {metric_name} exported to: {table_file}")
        except Exception as e:
            print(f"Could not export {metric_name} table: {e}")

    print("\n" + "=" * 80)
    print("Profiling analysis complete!")
    print("=" * 80)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config_rebel_cfr",
)
def main(dict_config: DictConfig) -> None:
    """Main entry point with Hydra config."""
    # Convert DictConfig to Config
    config = Config.from_dict_config(dict_config)

    # Override some defaults for profiling
    config.use_wandb = False  # Disable wandb for profiling

    # Default to fewer steps for profiling if not specified
    # User can override via CLI: num_steps=5
    if config.num_steps > 10:
        print(
            f"Warning: num_steps={config.num_steps} is high for profiling. Consider using num_steps=3-5"
        )

    # User can override these via CLI:
    # train.batch_size=512 search.iterations=50 search.depth=2

    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.train.batch_size}")
    print(f"  Num envs: {config.num_envs}")
    print(f"  Search depth: {config.search.depth}")
    print(f"  Search iterations: {config.search.iterations}")
    print()

    profile_training_loop(config)


if __name__ == "__main__":
    main()
