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
    # Generate both grids
    left = trainer.get_preflop_range_grid(seat=seat, metric="allin").splitlines()
    right = trainer.get_preflop_range_grid(seat=seat, metric="fold").splitlines()

    # Align lengths
    max_lines = max(len(left), len(right))
    while len(left) < max_lines:
        left.append("")
    while len(right) < max_lines:
        right.append("")

    # Compute left column width for padding
    left_width = max((len(line) for line in left), default=0)

    # Headers
    left_hdr = "Small blind (first) - all-in (%)"
    right_hdr = "Small blind (first) - fold (%)"
    print(left_hdr.ljust(left_width) + "   |   " + right_hdr)

    # Separator under headers
    print(("-" * len(left_hdr)).ljust(left_width) + "   |   " + ("-" * len(right_hdr)))

    # Rows side-by-side
    for i in range(max_lines):
        print(left[i].ljust(left_width) + "   |   " + right[i])
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

    # Detailed PPO metrics (if present)
    pol = stats.get("policy_loss")
    val = stats.get("value_loss")
    ent = stats.get("entropy")
    kl = stats.get("approx_kl")
    clip = stats.get("clipfrac")
    ev = stats.get("explained_var")
    d2 = stats.get("delta2_mean")
    d3 = stats.get("delta3_mean")
    parts = []
    if pol is not None:
        parts.append(f"policy {pol:.4f}")
    if val is not None:
        parts.append(f"value {val:.4f}")
    if ent is not None:
        parts.append(f"entropy {ent:.4f}")
    if kl is not None:
        parts.append(f"kl {kl:.4f}")
    if clip is not None:
        parts.append(f"clip {clip:.3f}")
    if ev is not None:
        parts.append(f"ev {ev:.3f}")
    if d2 is not None and d3 is not None:
        parts.append(f"d2 {d2:.1f} d3 {d3:.1f}")
    if parts:
        print("  " + " | ".join(parts))


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
