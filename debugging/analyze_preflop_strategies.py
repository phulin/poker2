#!/usr/bin/env python3
"""
Script to analyze preflop strategies from a checkpoint.

This script loads a trained model checkpoint and prints:
1. SB preflop action probabilities and value estimates
2. BB response probabilities when facing SB all-in
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.utils.training_utils import (
    print_preflop_range_grid,
)


def load_checkpoint_and_trainer(
    checkpoint_path: str, device: str = "mps"
) -> SelfPlayTrainer:
    """Load a checkpoint and create a trainer instance."""

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint to inspect its contents
    device_obj = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, weights_only=False, map_location=device_obj
    )

    config = checkpoint["full_config"]
    config.use_wandb = False

    # Create trainer
    trainer = SelfPlayTrainer(config, device_obj)

    # Load the checkpoint
    step, wandb_run_id = trainer.load_checkpoint(checkpoint_path)

    print(f"✅ Checkpoint loaded successfully")
    print(f"   Step: {step}")
    print(f"   ELO: {trainer.opponent_pool.current_elo:.1f}")

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Analyze preflop strategies from a checkpoint"
    )
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument(
        "--device", default="cpu", help="Device to use (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Step number for display (default: from checkpoint)",
    )
    parser.add_argument(
        "--analyze-169", action="store_true", help="Analyze all 169 hands"
    )
    parser.add_argument(
        "--sb-action",
        default="allin",
        choices=["allin", "call", "fold", "bet"],
        help="SB action to simulate (only used with --analyze-169)",
    )

    args = parser.parse_args()

    # Load checkpoint and trainer
    trainer = load_checkpoint_and_trainer(args.checkpoint, args.device)

    # Get step number
    step = args.step
    if step is None:
        # Get step from the loaded checkpoint
        checkpoint = torch.load(
            args.checkpoint, weights_only=False, map_location=args.device
        )
        step = checkpoint.get("step", 0)

        print_preflop_range_grid(trainer, step)


if __name__ == "__main__":
    main()
