#!/usr/bin/env python3
"""
Utility functions for AlphaHoldem training and analysis.
"""

from typing import Optional


def print_preflop_range_grid(
    trainer, step: int, seat: int = 0, title: Optional[str] = None
):
    """
    Print a preflop range grid for the given trainer and step.

    Args:
        trainer: SelfPlayTrainer instance
        step: Current training step
        seat: Seat position (0 for button, 1 for other)
        title: Optional custom title for the grid
    """
    if title is None:
        title = f"Preflop Range Grid (Step {step})"

    print(f"\n--- {title} ---")
    print("Button play - probability of all-in (%)")
    print("Higher numbers = more likely to go all-in")
    print()
    print(trainer.get_preflop_range_grid(seat=seat))  # Button seat
    print()


def print_training_stats(
    stats: dict, step: int, total_steps: int, step_time: str, total_time: str
):
    """
    Print training statistics in a consistent format.

    Args:
        stats: Dictionary containing training statistics
        step: Current step number
        total_steps: Total number of steps
        step_time: Time for current step
        total_time: Total time elapsed
    """
    # Get loss if available, default to 0
    loss = stats.get("avg_loss", 0)

    print(
        f"Step {step + 1}/{total_steps} | "
        f"Avg Reward: {stats['avg_reward']:.2f} | "
        f"ELO: {stats['current_elo']:.1f} | "
        f"Loss: {loss:.4f} | "
        f"Step Time: {step_time} | "
        f"Total Time: {total_time}"
    )


def print_evaluation_results(eval_results: dict):
    """
    Print evaluation results against opponent pool.

    Args:
        eval_results: Dictionary containing evaluation results
    """
    print("Evaluating against opponent pool...")
    print(f"Overall win rate: {eval_results['overall_win_rate']:.3f}")

    # Print individual opponent results
    for opponent_key, result in eval_results["opponent_results"].items():
        print(
            f"  {opponent_key}: {result['win_rate']:.3f} "
            f"(ELO: {result['opponent_elo']:.1f})"
        )


def print_checkpoint_info(checkpoint_path: str, step: int, is_best: bool = False):
    """
    Print checkpoint information.

    Args:
        checkpoint_path: Path to checkpoint file
        step: Step number
        is_best: Whether this is the best model so far
    """
    if is_best:
        print(f"New best model saved with ELO: {step}")
    else:
        print(f"Checkpoint saved to {checkpoint_path}")
