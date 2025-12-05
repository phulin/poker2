#!/usr/bin/env python3
"""
Utility functions for AlphaHoldem training and analysis.
"""

from typing import List, Optional, Tuple

from alphaholdem.env.analyze_tensor_env import (
    PreflopAnalyzer,
    RebelPreflopAnalyzer,
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


def _resolve_search_iterations(
    trainer, iterations_override: Optional[int], step: int
) -> int:
    """Resolve which CFR iteration count to use for printing."""
    if iterations_override is not None:
        return int(iterations_override)
    cfg = getattr(trainer, "cfg", None)
    # Prefer the same interpolation schedule used in training when iterations_final
    # is configured.
    if (
        cfg is not None
        and getattr(cfg.search, "iterations_final", None) is not None
        and hasattr(trainer, "initial_iterations")
    ):
        total_steps = max(1, int(getattr(cfg, "num_steps", 1)))
        t = min(1.0, max(0.0, step / float(total_steps)))
        iterations_start = int(trainer.initial_iterations)
        iterations_final = int(cfg.search.iterations_final)
        iterations_now = int(
            round(iterations_start + (iterations_final - iterations_start) * t)
        )
        warm = getattr(cfg.search, "warm_start_iterations", 0)
        iterations_now = max(warm + 1, iterations_now)
        return iterations_now
    cfr_iters = getattr(getattr(trainer, "cfr_evaluator", None), "cfr_iterations", None)
    if cfr_iters is not None:
        return int(cfr_iters)
    return int(trainer.cfg.search.iterations)


def print_preflop_range_grid(
    trainer,
    step: int,
    title: Optional[str] = None,
    rebel: bool = False,
    iterations: Optional[int] = None,
):
    """
    Print preflop range grids for the given trainer and step.

    Args:
        trainer: SelfPlayTrainer instance
        step: Current training step (0-indexed)
        seat: Seat position (0 for button, 1 for other)
        title: Optional custom title for the grid
    """
    if title is None:
        title = f"Preflop Range Grid (Step {step + 1})"

    if rebel:
        # Use model_avg if available, matching trainer logic
        eval_model = getattr(trainer, "model_avg", None)
        if eval_model is None:
            eval_model = trainer.model

        analyzer = RebelPreflopAnalyzer(
            eval_model,
            trainer.cfg,
            button=0,
            device=trainer.device,
            rng=trainer.rng,
            popart_normalizer=getattr(trainer, "popart_normalizer", None),
        )
        # Keep analyzer iterations aligned with the trainer's current schedule.
        resolved_iters = _resolve_search_iterations(trainer, iterations, step)
        if getattr(analyzer, "cfr_evaluator", None) is not None:
            analyzer.cfr_evaluator.cfr_iterations = max(
                analyzer.cfr_evaluator.warm_start_iterations + 1, resolved_iters
            )
    else:
        analyzer = PreflopAnalyzer(
            trainer.model,
            button=0,
            starting_stack=trainer.cfg.env.stack,
            sb=trainer.cfg.env.sb,
            bb=trainer.cfg.env.bb,
            bet_bins=trainer.cfg.env.bet_bins,
            device=trainer.device,
            rng=trainer.rng,
            flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
            popart_normalizer=getattr(trainer, "popart_normalizer", None),
        )

    print(f"\n--- {title} ---")

    # Generate all grids using the efficient 169-environment approach
    grids = analyzer.get_preflop_grids()
    fold_grid = grids["ranges"][0].splitlines()
    call_grid = grids["ranges"][1].splitlines()
    allin_grid = grids["ranges"][trainer.num_bet_bins - 1].splitlines()
    betting_grid = grids["betting"].splitlines()
    value_grid = grids["value"]
    suited_vs_offsuit = grids["suited_vs_offsuit"]

    # First row: Fold | Call
    print_combined_tables(
        [
            (fold_grid, "Small blind (first) - fold (%)"),
            (call_grid, "Small blind (first) - call (%)"),
        ],
    )

    # Second row: Betting | All-in
    print_combined_tables(
        [
            (betting_grid, "Small blind (first) - betting (%)"),
            (allin_grid, "Small blind (first) - all-in (%)"),
        ],
    )

    # Print value estimates grid
    print("--- Preflop Value Estimates (Step {}) ---".format(step + 1))
    print("Small blind (first) - value estimates (×1000)")

    print(value_grid)
    print()

    print("--- Preflop Suited vs Offsuit (Step {}) ---".format(step + 1))
    print(
        f"Fold:   Suited {100 * suited_vs_offsuit[0][0]:3.0f}%, Offsuit {100 * suited_vs_offsuit[0][1]:3.0f}%"
    )
    print(
        f"Call:   Suited {100 * suited_vs_offsuit[1][0]:3.0f}%, Offsuit {100 * suited_vs_offsuit[1][1]:3.0f}%"
    )
    betting_suited_vs_offsuit = suited_vs_offsuit[2:-1].sum(dim=0)
    print(
        f"Bet:    Suited {100 * betting_suited_vs_offsuit[0]:3.0f}%, Offsuit {100 * betting_suited_vs_offsuit[1]:3.0f}%"
    )

    print(
        f"All-in: Suited {100 * suited_vs_offsuit[-1][0]:3.0f}%, Offsuit {100 * suited_vs_offsuit[-1][1]:3.0f}%"
    )
    print()

    # Also print BB response (facing SB all-in), matching debug_tensor_env
    print("--- BB Response vs SB All-in (Step {}) ---".format(step + 1))

    grids = analyzer.get_preflop_grids_allin_response()
    bb_fold_grid = grids["ranges"][0].splitlines()
    bb_call_grid = grids["ranges"][1].splitlines()
    bb_value_grid = grids["value"]

    print_combined_tables(
        [
            (bb_fold_grid, "Big blind (facing all-in) - fold (%)"),
            (bb_call_grid, "Big blind (facing all-in) - call (%)"),
        ],
    )

    print("BB value estimates when facing SB all-in (×1000)")
    print(bb_value_grid)
    print()


def print_training_stats(
    stats: dict,
    step: int,
    total_steps: int,
    episodes: int,
    step_time: str,
    total_time: str,
):
    """
    Print training statistics in a consistent format.

    Args:
        stats: Dictionary containing training statistics
        step: Current step number (0-indexed)
        total_steps: Total number of steps
        step_time: Time for current step
        total_time: Total time elapsed
    """
    # Get loss if available, default to 0
    loss = stats.get("avg_loss", 0)

    print(
        f"Step {step + 1}/{total_steps} | "
        f"Episodes {episodes} | "
        f"Avg Reward {stats['avg_reward']:5.2f} | "
        f"Elo {stats['current_elo']:.0f} | "
        f"Loss {loss:.4f} | "
        f"Step Time {step_time} | "
        f"Total Time {total_time}"
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
    lr = stats.get("learning_rate")
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
    # if epsilon is not None:
    #     parts.append(f"eps {epsilon:.3f}")
    if lr is not None:
        parts.append(f"lr {lr:.5f}")
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
        step: Step number (0-indexed)
        is_best: Whether this is the best model so far
    """
    if is_best:
        print(f"New best model saved with ELO: {step + 1}")
    else:
        print(f"Checkpoint saved to {checkpoint_path}")
