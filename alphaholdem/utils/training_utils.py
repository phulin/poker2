#!/usr/bin/env python3
"""
Utility functions for AlphaHoldem training and analysis.
"""

from typing import List, Optional, Tuple

from ..env.analyze_tensor_env import (
    get_preflop_betting_grid,
    get_preflop_range_grid,
    get_preflop_range_grid_bb_response,
    get_preflop_value_grid,
    get_preflop_value_grid_bb_response,
)


def print_combined_tables(
    tables: List[Tuple[List[str], str]], title: Optional[str] = None
) -> None:
    """
    Print multiple tables combined horizontally.

    Args:
        tables: List of tuples containing (grid_lines, header) for each table
        title: Optional title for the combined display
    """
    if not tables:
        return

    if title:
        print(f"--- {title} ---")

    # Align all tables to the same length
    max_lines = max(len(table_lines) for table_lines, _ in tables)
    aligned_tables = []

    for table_lines, header in tables:
        # Pad table to max_lines
        padded_lines = table_lines[:]
        while len(padded_lines) < max_lines:
            padded_lines.append("")
        aligned_tables.append((padded_lines, header))

    # Calculate column widths for proper alignment
    widths = []
    for table_lines, header in aligned_tables:
        width = max((len(line) for line in table_lines), default=len(header))
        widths.append(width)

    # Print headers
    header_parts = []
    for i, (_, header) in enumerate(aligned_tables):
        header_parts.append(header.ljust(widths[i]))
    print("   |   ".join(header_parts))

    # Print separator line
    separator_parts = []
    for i, (_, header) in enumerate(aligned_tables):
        separator_parts.append(("-" * len(header)).ljust(widths[i]))
    print("   |   ".join(separator_parts))

    # Print table rows
    for line_idx in range(max_lines):
        row_parts = []
        for i, (table_lines, _) in enumerate(aligned_tables):
            row_parts.append(table_lines[line_idx].ljust(widths[i]))
        print("   |   ".join(row_parts))

    print()


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

    # Generate all grids using the efficient 169-environment approach
    fold_grid = get_preflop_range_grid(
        trainer.model,
        trainer.state_encoder,
        0,
        trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).splitlines()

    call_grid = get_preflop_range_grid(
        trainer.model,
        trainer.state_encoder,
        1,
        trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).splitlines()

    allin_grid = get_preflop_range_grid(
        trainer.model,
        trainer.state_encoder,
        trainer.num_bet_bins - 1,
        trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).splitlines()

    betting_grid = get_preflop_betting_grid(
        trainer.model,
        trainer.state_encoder,
        trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).splitlines()

    # First row: Fold | Call
    print_combined_tables(
        [
            (fold_grid, "Small blind (first) - fold (%)"),
            (call_grid, "Small blind (first) - call (%)"),
        ],
        "First Row: Fold | Call",
    )

    # Second row: Betting | All-in
    print_combined_tables(
        [
            (betting_grid, "Small blind (first) - betting (%)"),
            (allin_grid, "Small blind (first) - all-in (%)"),
        ],
        "Second Row: Betting | All-in",
    )

    # Print value estimates grid
    print("--- Preflop Value Estimates (Step {}) ---".format(step))
    print("Small blind (first) - value estimates (×1000)")

    value_grid = get_preflop_value_grid(
        trainer.model,
        trainer.state_encoder,
        trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    )
    print(value_grid)
    print()

    # Also print BB response (facing SB all-in), matching debug_tensor_env
    print("--- BB Response vs SB All-in (Step {}) ---".format(step))

    bb_fold_grid = get_preflop_range_grid_bb_response(
        trainer.model,
        trainer.state_encoder,
        0,  # fold bin
        device=trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).splitlines()

    bb_call_grid = get_preflop_range_grid_bb_response(
        trainer.model,
        trainer.state_encoder,
        1,  # call bin
        device=trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).splitlines()

    print_combined_tables(
        [
            (bb_fold_grid, "Big blind (facing all-in) - fold (%)"),
            (bb_call_grid, "Big blind (facing all-in) - call (%)"),
        ],
        "BB Response: Fold | Call",
    )

    print("BB value estimates when facing SB all-in (×1000)")
    bb_value_grid = get_preflop_value_grid_bb_response(
        trainer.model,
        trainer.state_encoder,
        device=trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    )
    print(bb_value_grid)
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
        f"Avg Reward: {stats['avg_reward']:5.2f} | "
        f"ELO: {stats['current_elo']:.0f} | "
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
    epsilon = stats.get("epsilon")
    parts = []
    if pol is not None:
        parts.append(f"policy {pol:7.4f}")
    if val is not None:
        parts.append(f"value {val:.4f}")
    if ent is not None:
        parts.append(f"entropy {ent:.2f}")
    if kl is not None:
        parts.append(f"kl {kl:6.3f}")
    if clip is not None:
        parts.append(f"clip {clip:.3f}")
    if ev is not None:
        parts.append(f"ev {ev:6.3f}")
    if d2 is not None and d3 is not None:
        parts.append(f"d2 {d2:.1f} d3 {d3:.1f}")
    if epsilon is not None:
        parts.append(f"eps {epsilon:.3f}")
    if parts:
        print("  " + " | ".join(parts))

    # Minibatch verification metrics (if present)
    mb_ir = stats.get("mb_improve_rate")
    mb_lb = stats.get("mb_loss_before")
    mb_la = stats.get("mb_loss_after")
    if mb_ir is not None or mb_lb is not None or mb_la is not None:
        segs = []
        if mb_ir is not None:
            segs.append(f"verify {mb_ir:.2f}")
        if mb_lb is not None:
            segs.append(f"mb_before {mb_lb:.4f}")
        if mb_la is not None:
            segs.append(f"mb_after {mb_la:.4f}")
        if segs:
            print("  " + " | ".join(segs))

    # Print pool stats
    pool_stats = stats.get("pool_stats")
    if pool_stats:
        # Build pool stats line
        pool_parts = []
        pool_parts.append(f"Pool Size: {pool_stats.get('pool_size', 0)}")
        pool_parts.append(f"Avg ELO: {pool_stats.get('avg_elo', 0):.0f}")
        pool_parts.append(
            f"ELO Range: {pool_stats.get('min_elo', 0):.0f}-{pool_stats.get('max_elo', 0):.0f}"
        )

        # DREDPool specific stats
        if "avg_age" in pool_stats:
            pool_parts.append(f"Avg Age: {pool_stats.get('avg_age', 0):.0f}")
        if "avg_difficulty" in pool_stats:
            pool_parts.append(
                f"Avg Difficulty: {pool_stats.get('avg_difficulty', 0):.3f}"
            )

        # KBestPool specific stats
        if "best_snapshot_step" in pool_stats:
            pool_parts.append(
                f"Best Snapshot: Step {pool_stats.get('best_snapshot_step', 0)}, ELO {pool_stats.get('best_snapshot_elo', 0):.1f}"
            )

        print(f"  {' | '.join(pool_parts)}")


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
