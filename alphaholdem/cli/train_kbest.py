#!/usr/bin/env python3
"""
Training script for AlphaHoldem with K-Best self-play.

This script demonstrates the K-Best self-play mechanism as described in the AlphaHoldem paper,
where the agent maintains a pool of K best historical versions and trains against them.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from ..rl.self_play import SelfPlayTrainer
from ..utils.training_utils import (
    print_preflop_range_grid,
    print_training_stats,
    print_evaluation_results,
    print_checkpoint_info,
)
from ..core.config_loader import get_config


def train_kbest(
    num_steps: int,
    k_best_pool_size: int,
    min_elo_diff: float,
    checkpoint_interval: int,
    eval_interval: int,
    checkpoint_dir: str,
    resume_from: str | None = None,
    device: torch.device = None,
    config: str | None = None,
    use_tensor_env: bool = False,
    num_envs: int = 256,
):
    """
    Train AlphaHoldem agent using K-Best self-play.

    Args:
        num_steps: Number of training steps
        k_best_pool_size: Size of K-Best opponent pool
        min_elo_diff: Minimum ELO difference for pool updates
        checkpoint_interval: How often to save checkpoints
        eval_interval: How often to evaluate against pool
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load optional config (centralized defaults)
    cfg = get_config(config)

    # Initialize trainer with K-Best pool
    trainer = SelfPlayTrainer(
        k_best_pool_size=k_best_pool_size,
        min_elo_diff=min_elo_diff,
        device=device,
        config=config,
        use_tensor_env=use_tensor_env,
        num_envs=num_envs,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        start_step = trainer.load_checkpoint(resume_from)

    # Record training start time
    training_start_time = time.time()

    print(f"Starting K-Best training from step {start_step}")
    print(f"K-Best pool size: {k_best_pool_size}")
    print(f"Min ELO difference: {min_elo_diff}")
    if use_tensor_env:
        print(f"Using tensorized environment with {num_envs} parallel environments")
    else:
        print("Using scalar environment")
    # Collection runs until batch_size steps; no fixed trajectories per step
    print(
        f"Training start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}"
    )
    print()

    # Training loop
    for step in range(start_step, num_steps):
        step_start_time = time.time()

        # Training step: collects until batch_size steps
        stats = trainer.train_step()

        # Calculate times
        step_elapsed_time = time.time() - step_start_time
        total_elapsed_time = time.time() - training_start_time

        # Format times
        step_time_str = f"{step_elapsed_time:.2f}s"
        total_time_str = f"{total_elapsed_time:.1f}s"

        # Logging
        print_training_stats(stats, step, num_steps, step_time_str, total_time_str)
        # Optional clipping debug of first sample in batch
        if "first_ret" in stats:
            print(
                "  clip_debug "
                f"ret0 {stats['first_ret']:.4f} "
                f"d2 {stats['first_d2']:.4f} d3 {stats['first_d3']:.4f} "
                f"min {stats['first_min_b']:.4f} max {stats['first_max_b']:.4f} "
                f"retc {stats['first_ret_clipped']:.4f} "
                f"out_of_bounds {bool(stats['first_ret_out_of_bounds'])}"
            )

        # Evaluation against pool
        if (step + 1) % eval_interval == 0:
            eval_results = trainer.evaluate_against_pool(num_games=20)
            print_evaluation_results(eval_results)

        # Checkpointing
        if (step + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_step_{step + 1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, step + 1)
            checkpoint_path = os.path.join(checkpoint_dir, f"latest_model.pt")
            trainer.save_checkpoint(checkpoint_path, step + 1)

            # Also save the best model if it has the highest ELO
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            if (
                not os.path.exists(best_checkpoint_path)
                or stats["current_elo"] > trainer.opponent_pool.get_best_snapshot().elo
            ):
                trainer.save_checkpoint(best_checkpoint_path, step + 1)
                print_checkpoint_info(
                    best_checkpoint_path, stats["current_elo"], is_best=True
                )

            print_preflop_range_grid(trainer, step + 1, seat=0)

    # Final evaluation
    final_total_time = time.time() - training_start_time
    print(f"\nFinal evaluation against opponent pool...")
    final_eval = trainer.evaluate_against_pool(num_games=100)
    print_evaluation_results(final_eval)
    print(
        f"Total training time: {final_total_time:.1f}s ({final_total_time/3600:.2f} hours)"
    )

    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    trainer.save_checkpoint(final_checkpoint_path, num_steps)

    # Print preflop range grid
    print_preflop_range_grid(
        trainer, num_steps, seat=0, title="Final Preflop Range Grid"
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train AlphaHoldem with K-Best self-play"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--k-best-pool-size", type=int, default=10, help="Size of K-Best opponent pool"
    )
    parser.add_argument(
        "--min-elo-diff",
        type=float,
        default=50.0,
        help="Minimum ELO difference for pool updates",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=5, help="Checkpoint save interval"
    )
    parser.add_argument(
        "--eval-interval", type=int, default=50, help="Evaluation interval"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume-from", type=str, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--config", type=str, help="Path to YAML config for components/hparams"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--use-tensor-env",
        action="store_true",
        help="Use tensorized environment for faster training",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Number of parallel environments for tensorized training",
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # Set up device (GPU if available)
    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    if device.type == "mps":
        print("✅ Using Apple M3 GPU (MPS)")
    elif args.device == "cpu":
        print("✅ Using CPU (selected)")
    else:
        print("⚠️ Using CPU (MPS not available)")

    # Train the agent
    trainer = train_kbest(
        num_steps=args.steps,
        # trajectories_per_step removed; collection loops until batch_size steps
        k_best_pool_size=args.k_best_pool_size,
        min_elo_diff=args.min_elo_diff,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=device,
        config=args.config,
        use_tensor_env=args.use_tensor_env,
        num_envs=args.num_envs,
    )

    print("Training completed!")


if __name__ == "__main__":
    main()
