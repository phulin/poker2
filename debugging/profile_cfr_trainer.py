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

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


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

    # Configure profiler activities (CUDA only)
    if device.type != "cuda":
        print(
            f"Warning: CUDA profiling requires CUDA device, but device is {device.type}"
        )
        print("Falling back to CPU profiling...")
        activities = [ProfilerActivity.CPU]
    else:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    # Use profiler with profile_memory, stack traces, and FLOPS for detailed analysis
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
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

    # First get a quick summary without grouping for faster processing
    print("\nComputing summary statistics (this may take a moment)...")
    print("  Step 1/3: Getting top operations...")
    key_averages_summary = prof.key_averages(group_by_input_shape=False)

    # Print summary table first (fast)
    print("\n" + "=" * 80)
    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    print(
        f"PROFILING SUMMARY - Top Operations by Self {('CUDA' if device.type == 'cuda' else 'CPU')} Time"
    )
    print("=" * 80)
    print(
        key_averages_summary.table(
            sort_by=sort_key,
            row_limit=50,
            max_name_column_width=80,
        )
    )

    # Now filter for specific functions before doing detailed analysis
    print("\n  Step 2/3: Filtering for set_leaf_values and update_policy...")

    # Filter operations more efficiently by checking key first
    print("  Step 3/3: Analyzing hotspots...")

    # Filter and print operations related to set_leaf_values
    print("\n" + "=" * 80)
    print("SET_LEAF_VALUES HOTSPOTS")
    print("=" * 80)

    # Use summary for filtering (faster than grouped version)
    set_leaf_ops = []
    for op in key_averages_summary:
        if "set_leaf_values" in op.key:
            set_leaf_ops.append(op)
        elif op.stack:
            # Only check stack if key doesn't match (less common)
            try:
                stack_str = " ".join(str(s) for s in op.stack[:5])  # Limit stack check
                if "set_leaf_values" in stack_str:
                    set_leaf_ops.append(op)
            except:
                pass

    if set_leaf_ops:
        # Sort by CUDA time (or CPU time if CUDA not available)
        sort_key_attr = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        set_leaf_ops_sorted = sorted(
            set_leaf_ops, key=lambda x: getattr(x, sort_key_attr, 0), reverse=True
        )
        print(
            f"\nFound {len(set_leaf_ops_sorted)} operations related to set_leaf_values\n"
        )

        # Print top 30
        for i, op in enumerate(set_leaf_ops_sorted[:30], 1):
            print(f"{i:2d}. {op.key}")
            if device.type == "cuda" and hasattr(op, "self_cuda_time_total"):
                time_ms = (
                    op.self_cuda_time_total / 1e6 if op.self_cuda_time_total > 0 else 0
                )
                print(f"    Self CUDA: {time_ms:.3f} ms")
            else:
                time_ms = op.self_cpu_time_total / 1e6
                print(f"    Self CPU: {time_ms:.3f} ms")
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

    update_policy_ops = []
    for op in key_averages_summary:
        if "update_policy" in op.key:
            update_policy_ops.append(op)
        elif op.stack:
            # Only check stack if key doesn't match (less common)
            try:
                stack_str = " ".join(str(s) for s in op.stack[:5])  # Limit stack check
                if "update_policy" in stack_str:
                    update_policy_ops.append(op)
            except:
                pass

    if update_policy_ops:
        # Sort by CUDA time (or CPU time if CUDA not available)
        sort_key_attr = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        update_policy_ops_sorted = sorted(
            update_policy_ops, key=lambda x: getattr(x, sort_key_attr, 0), reverse=True
        )
        print(
            f"\nFound {len(update_policy_ops_sorted)} operations related to update_policy\n"
        )

        # Print top 30
        for i, op in enumerate(update_policy_ops_sorted[:30], 1):
            print(f"{i:2d}. {op.key}")
            if device.type == "cuda" and hasattr(op, "self_cuda_time_total"):
                time_ms = (
                    op.self_cuda_time_total / 1e6 if op.self_cuda_time_total > 0 else 0
                )
                print(f"    Self CUDA: {time_ms:.3f} ms")
            else:
                time_ms = op.self_cpu_time_total / 1e6
                print(f"    Self CPU: {time_ms:.3f} ms")
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

    # Export raw stack traces using export_stacks
    stacks_file = output_dir / f"profile_cfr_trainer_stacks_{timestamp}.txt"
    try:
        prof.export_stacks(str(stacks_file))
        print(f"Raw stack traces exported to: {stacks_file}")
    except Exception as e:
        print(f"Could not export raw stack traces: {e}")

    # Export formatted stack trace table
    stack_table_file = output_dir / f"profile_cfr_trainer_stack_table_{timestamp}.txt"
    try:
        stack_averages = prof.key_averages(group_by_stack_n=10)
        stack_sort_key = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        with open(stack_table_file, "w") as f:
            f.write(
                stack_averages.table(
                    sort_by=stack_sort_key,
                    row_limit=100,
                )
            )
        print(f"Stack trace table exported to: {stack_table_file}")
    except Exception as e:
        print(f"Could not export stack trace table: {e}")

    # Export detailed table sorted by different metrics (use summary for speed)
    print("\nExporting detailed tables...")
    metrics = []
    if device.type == "cuda":
        metrics = [
            ("cuda_time", "cuda_time_total"),
            ("cuda_memory", "cuda_memory_usage"),
        ]
    else:
        metrics = [
            ("cpu_time", "cpu_time_total"),
            ("memory", "cpu_memory_usage"),
        ]

    for metric_name, sort_key in metrics:
        table_file = output_dir / f"profile_cfr_trainer_{metric_name}_{timestamp}.txt"
        try:
            table_text = key_averages_summary.table(
                sort_by=sort_key,
                row_limit=100,
                max_name_column_width=100,
            )
            with open(table_file, "w") as f:
                f.write(table_text)
            print(f"  Table sorted by {metric_name} exported to: {table_file}")
        except Exception as e:
            print(f"  Could not export {metric_name} table: {e}")

    # Optionally compute grouped version for more detailed analysis (slow, but saved to file)
    print("\nComputing detailed grouped analysis (may take a while)...")
    try:
        key_averages_grouped = prof.key_averages(group_by_input_shape=True)
        grouped_file = output_dir / f"profile_cfr_trainer_grouped_{timestamp}.txt"
        grouped_sort_key = (
            "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        )
        with open(grouped_file, "w") as f:
            f.write(
                key_averages_grouped.table(
                    sort_by=grouped_sort_key,
                    row_limit=200,
                    max_name_column_width=100,
                )
            )
        print(f"  Grouped analysis exported to: {grouped_file}")
    except Exception as e:
        print(f"  Could not export grouped analysis: {e}")

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
