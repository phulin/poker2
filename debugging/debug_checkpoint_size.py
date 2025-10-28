#!/usr/bin/env python3
"""
Debug script to analyze why ReBeL checkpoints are so large.

This script loads checkpoint files and analyzes their contents to identify
which components are taking up the most space.
"""

import argparse
import os

import torch


def analyze_checkpoint(checkpoint_path: str, detailed: bool = False) -> None:
    """Analyze a checkpoint and print size breakdown."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print(f"\n{'='*80}")
    print(f"Analyzing: {checkpoint_path}")
    print(f"{'='*80}")

    # Get file size
    file_size = os.path.getsize(checkpoint_path)
    print(f"\nFile size: {file_size / (1024**2):.2f} MB")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print(f"\nTop-level keys: {list(checkpoint.keys())}")

    # Analyze each key
    total_pytorch_size = 0
    breakdown = []

    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor):
            tensor_size = value.numel() * value.element_size()
            total_pytorch_size += tensor_size
            breakdown.append(
                (
                    key,
                    f"{tensor_size / (1024**2):.2f} MB",
                    value.shape,
                    value.dtype,
                    value.device,
                )
            )
        elif isinstance(value, dict):
            # Recursively analyze nested dicts
            dict_size = analyze_dict(value, f"{key}")
            total_pytorch_size += dict_size
            breakdown.append(
                (
                    key,
                    f"{dict_size / (1024**2):.2f} MB",
                    f"dict with {len(value)} keys",
                    "nested_dict",
                    "N/A",
                )
            )

            if detailed:
                print(f"\n  Nested keys in '{key}':")
                for subkey in value.keys():
                    print(f"    - {subkey}")
        else:
            # Python objects (int, str, etc.)
            import pickle

            try:
                pickled_size = len(pickle.dumps(value))
                breakdown.append(
                    (
                        key,
                        f"{pickled_size / (1024**2):.2f} MB",
                        type(value).__name__,
                        "python_object",
                        "N/A",
                    )
                )
            except:
                breakdown.append(
                    (key, "< 0.01 MB", type(value).__name__, "python_object", "N/A")
                )

    # Sort by size (descending)
    breakdown.sort(key=lambda x: float(x[1].split()[0]), reverse=True)

    # Print breakdown
    print(
        f"\n{'Component':<30} {'Size':<12} {'Shape/Type':<20} {'Dtype':<15} {'Device':<10}"
    )
    print(f"{'-'*80}")
    for item in breakdown:
        component, size, shape_type, dtype, device = item
        print(
            f"{component:<30} {size:<12} {str(shape_type):<20} {str(dtype):<15} {str(device):<10}"
        )

    print(f"\nTotal PyTorch tensors: {total_pytorch_size / (1024**2):.2f} MB")
    print(f"Overhead: {(file_size - total_pytorch_size) / (1024**2):.2f} MB")

    # Detailed analysis for model state dict if present
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        print(f"\n{'='*80}")
        print("Model State Dict Analysis")
        print(f"{'='*80}")

        model_size = 0
        layer_sizes = []

        for name, param in checkpoint["model"].items():
            param_size = param.numel() * param.element_size()
            model_size += param_size
            layer_sizes.append((name, param_size / (1024**2), param.shape, param.dtype))

        # Sort by size
        layer_sizes.sort(key=lambda x: x[1], reverse=True)

        print(
            f"Total model parameters: {sum(p.numel() for p in checkpoint['model'].values()):,}"
        )
        print(f"Total model size: {model_size / (1024**2):.2f} MB")

        if detailed:
            print(f"\nTop 20 largest layers:")
            for i, (name, size, shape, dtype) in enumerate(layer_sizes[:20]):
                print(f"{i+1:2d}. {name:<40} {size:>8.2f} MB  {str(shape):<25} {dtype}")

    # Detailed analysis for optimizer state if present
    if "optimizer" in checkpoint and isinstance(checkpoint["optimizer"], dict):
        print(f"\n{'='*80}")
        print("Optimizer State Analysis")
        print(f"{'='*80}")

        optimizer_size = 0
        optimizer_components = []

        for name, value in checkpoint["optimizer"].items():
            if isinstance(value, dict):
                comp_size = 0
                for param_name, param_data in value.items():
                    if isinstance(param_data, torch.Tensor):
                        comp_size += param_data.numel() * param_data.element_size()
                optimizer_size += comp_size
                optimizer_components.append((name, comp_size / (1024**2), len(value)))
            else:
                import pickle

                try:
                    comp_size = len(pickle.dumps(value))
                    optimizer_size += comp_size
                    optimizer_components.append(
                        (name, comp_size / (1024**2), "various")
                    )
                except:
                    pass

        # Sort by size
        optimizer_components.sort(key=lambda x: x[1], reverse=True)

        print(f"Total optimizer size: {optimizer_size / (1024**2):.2f} MB")

        if detailed and optimizer_components:
            print(f"\nOptimizer components:")
            for name, size, num_items in optimizer_components[:10]:
                print(f"  {name:<40} {size:>8.2f} MB  ({num_items} items)")

        # Analyze optimizer state dict in detail
        if detailed and "state" in checkpoint["optimizer"]:
            print(f"\nAnalyzing optimizer state dict in detail...")
            state_dict = checkpoint["optimizer"]["state"]

            # Map parameter indices to layer names
            if "model" in checkpoint:
                model_params = list(checkpoint["model"].items())
                param_to_name = {i: name for i, (name, _) in enumerate(model_params)}
            else:
                param_to_name = {}

            # Find largest parameter states
            param_sizes = []
            for param_idx, state in state_dict.items():
                param_size = 0
                state_keys_found = []
                for state_key, state_value in state.items():
                    if isinstance(state_value, torch.Tensor):
                        param_size += state_value.numel() * state_value.element_size()
                        state_keys_found.append(state_key)

                # Get parameter name if available
                param_name = param_to_name.get(int(param_idx), f"param_{param_idx}")
                if state_keys_found:
                    state_key_str = ", ".join(state_keys_found)
                    param_name += f" [{state_key_str}]"

                param_sizes.append((param_name, param_size / (1024**2)))

            param_sizes.sort(key=lambda x: x[1], reverse=True)

            print(f"\nTop parameter states in optimizer:")
            total_state_size = sum(size for _, size in param_sizes)
            print(f"Total state size: {total_state_size:.2f} MB")
            for i, (name, size) in enumerate(param_sizes[:15]):
                print(f"  {i+1:2d}. {name:<60} {size:>8.2f} MB")


def analyze_dict(d: dict, prefix: str = "") -> int:
    """Recursively analyze a dict and return total size in bytes."""
    total_size = 0

    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            total_size += value.numel() * value.element_size()
        elif isinstance(value, dict):
            total_size += analyze_dict(value, f"{prefix}.{key}")
        else:
            import pickle

            try:
                total_size += len(pickle.dumps(value))
            except:
                pass

    return total_size


def main():
    parser = argparse.ArgumentParser(description="Analyze ReBeL checkpoint sizes")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed breakdown of model layers and optimizer components",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyze all checkpoints in checkpoints-rebel directory",
    )

    args = parser.parse_args()

    if args.all:
        checkpoint_dir = "checkpoints-rebel"
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, f)
                for f in os.listdir(checkpoint_dir)
                if f.endswith(".pt")
            ]
            checkpoints.sort()

            for ckpt in checkpoints:
                analyze_checkpoint(ckpt, args.detailed)
        else:
            print(f"Error: Directory not found: {checkpoint_dir}")
    else:
        analyze_checkpoint(args.checkpoint, args.detailed)


if __name__ == "__main__":
    main()
