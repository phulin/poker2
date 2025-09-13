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
import torch
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alphaholdem.core.structured_config import Config
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.utils.training_utils import (
    print_preflop_range_grid,
    print_combined_tables,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.analyze_tensor_env import (
    create_169_hand_analysis_setup,
    create_169_hand_combinations,
    step_sb_action,
    get_preflop_range_grid_bb_response,
    get_preflop_value_grid_bb_response,
    get_probabilities,
    create_state_encoder_for_model,
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

    # Create a basic config with reasonable defaults
    from alphaholdem.core.structured_config import (
        TrainingConfig,
        ModelConfig,
        EnvConfig,
        StateEncoderConfig,
    )

    # Detect model type from checkpoint
    model_keys = list(checkpoint["model_state_dict"].keys())
    if any("transformer" in key for key in model_keys):
        model_name = "poker_transformer_v1"
        state_encoder_name = "transformer"
        model_kwargs = {
            "d_model": 256,
            "n_layers": 2,
            "n_heads": 2,
            "vocab_size": 80,
            "num_actions": 8,
            "dropout": 0.1,
            "use_auxiliary_loss": False,
        }
    else:
        # Default to CNN model
        model_name = "siamese_convnet_v1"
        state_encoder_name = "cnn"
        model_kwargs = {
            "cards_channels": 6,
            "actions_channels": 24,
            "cards_hidden": 192,
            "actions_hidden": 192,
            "fusion_hidden": [2048, 2048],
            "num_actions": 8,
        }

    print(f"Detected model type: {model_name}")
    print(f"Using state encoder: {state_encoder_name}")

    cfg = Config(
        train=TrainingConfig(
            batch_size=32,
            learning_rate=3e-4,
            num_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            ppo_eps=0.2,
            replay_buffer_batches=4,
            max_trajectory_length=1000,
            ppo_delta1=0.01,
            value_coef=0.5,
            entropy_coef=0.01,
            grad_clip=0.5,
            use_mixed_precision=True,
            loss_scale=1.0,
        ),
        model=ModelConfig(
            name=model_name,
            kwargs=model_kwargs,
            use_gradient_checkpointing=False,
        ),
        state_encoder=StateEncoderConfig(name=state_encoder_name),
        env=EnvConfig(
            stack=1000,
            sb=5,
            bb=10,
            bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0],
            flop_showdown=False,
        ),
        use_tensor_env=True,
        num_envs=16,
        k_best_pool_size=10,
        min_elo_diff=50.0,
        min_step_diff=1,
        k_factor=32.0,
        use_wandb=False,
        device=device,
    )

    # Create trainer
    trainer = SelfPlayTrainer(cfg, device_obj)

    # Load the checkpoint
    step, wandb_run_id = trainer.load_checkpoint(checkpoint_path)

    print(f"✅ Checkpoint loaded successfully")
    print(f"   Step: {step}")
    print(f"   ELO: {trainer.opponent_pool.current_elo:.1f}")

    return trainer


def print_sb_preflop_tables(trainer: SelfPlayTrainer, step: int):
    """Print SB preflop action probabilities and value estimates."""
    print("\n" + "=" * 80)
    print("SMALL BLIND PREFLOP STRATEGY")
    print("=" * 80)

    # Print the standard preflop range grid (includes value estimates)
    print_preflop_range_grid(trainer, step, seat=0, title="SB Preflop Strategy")


def print_bb_response_tables(trainer: SelfPlayTrainer, step: int):
    """Print BB response probabilities when facing SB all-in."""
    print("\n" + "=" * 80)
    print("BIG BLIND RESPONSE TO SB ALL-IN")
    print("=" * 80)

    # Get all the grids for BB response analysis
    # Note: The grid functions now create their own 169-hand environments internally
    fold_grid_lines = get_preflop_range_grid_bb_response(
        trainer.model,
        seat=1,
        metric="fold",
        device=trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).split("\n")

    call_grid_lines = get_preflop_range_grid_bb_response(
        trainer.model,
        seat=1,
        metric="call",
        device=trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    ).split("\n")

    # Display BB response strategy in the same format as SB strategy
    print("\n--- BB Response Strategy vs SB All-in ---")

    # First row: Fold | Call
    print_combined_tables(
        [
            (fold_grid_lines, "Big blind (facing all-in) - fold (%)"),
            (call_grid_lines, "Big blind (facing all-in) - call (%)"),
        ]
    )

    # Value estimates for BB when facing all-in
    print("\n--- BB Value Estimates vs SB All-in ---")
    print("BB value estimates when facing SB all-in (×100)")
    value_grid = get_preflop_value_grid_bb_response(
        trainer.model,
        seat=1,
        device=trainer.device,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    )
    print(value_grid)


def analyze_169_hands_vectorized(trainer: SelfPlayTrainer, sb_action: str = "allin"):
    """Analyze all 169 possible preflop hands using debug utilities."""
    print(
        f"\n=== ANALYZING 169 HANDS WITH SB ACTION: {sb_action.upper()} (VECTORIZED) ==="
    )

    # Create 169-hand environment using debug utilities
    temp_env, _ = create_169_hand_analysis_setup(
        trainer.model,
        starting_stack=trainer.cfg.env.stack,
        sb=trainer.cfg.env.sb,
        bb=trainer.cfg.env.bb,
        bet_bins=trainer.cfg.env.bet_bins,
        device=trainer.device,
        rng=trainer.rng,
        flop_showdown=getattr(trainer.cfg.env, "flop_showdown", False),
    )

    print(f"Total hands to analyze: 169")

    # Simulate SB action using debug utilities
    step_sb_action(temp_env, sb_action, trainer.device)

    # Analyze BB responses using debug utilities
    analysis_results = analyze_bb_responses(
        temp_env,
        trainer.model,
        trainer.device,
    )

    results = analysis_results["results"]

    # Print summary statistics
    print(f"\n=== SUMMARY STATISTICS FOR SB {sb_action.upper()} ===")

    # Count hands by type
    pocket_pairs = [r for r in results if r["cards"][0][:-1] == r["cards"][1][:-1]]
    suited_hands = [
        r
        for r in results
        if r["cards"][0][-1] == r["cards"][1][-1]
        and r["cards"][0][:-1] != r["cards"][1][:-1]
    ]
    offsuit_hands = [r for r in results if r["cards"][0][-1] != r["cards"][1][-1]]

    print(f"Pocket pairs: {len(pocket_pairs)}")
    print(f"Suited hands: {len(suited_hands)}")
    print(f"Off-suit hands: {len(offsuit_hands)}")

    # Analyze call rates
    def get_call_rate(hand_list):
        call_counts = {
            "0": 0,
            "1-10": 0,
            "11-25": 0,
            "26-50": 0,
            "51-75": 0,
            "76-99": 0,
            "100": 0,
        }
        for hand in hand_list:
            call_prob = hand["call_prob"]
            if call_prob == " 0":
                call_counts["0"] += 1
            elif call_prob == "██":
                call_counts["100"] += 1
            else:
                try:
                    prob_val = int(call_prob)
                    if 1 <= prob_val <= 10:
                        call_counts["1-10"] += 1
                    elif 11 <= prob_val <= 25:
                        call_counts["11-25"] += 1
                    elif 26 <= prob_val <= 50:
                        call_counts["26-50"] += 1
                    elif 51 <= prob_val <= 75:
                        call_counts["51-75"] += 1
                    elif 76 <= prob_val <= 99:
                        call_counts["76-99"] += 1
                except ValueError:
                    pass
        return call_counts

    print(f"\nCall rate distribution:")
    print(f"Pocket pairs: {get_call_rate(pocket_pairs)}")
    print(f"Suited hands: {get_call_rate(suited_hands)}")
    print(f"Off-suit hands: {get_call_rate(offsuit_hands)}")

    # Show some example hands
    print(f"\n=== EXAMPLE HANDS ===")
    example_hands = [
        ("As", "Kh"),  # AK suited
        ("Ah", "Ad"),  # AA
        ("2s", "7d"),  # 72 off-suit
        ("Ks", "Qs"),  # KQ suited
        ("9h", "9d"),  # 99
    ]

    for card1, card2 in example_hands:
        result = next((r for r in results if r["cards"] == (card1, card2)), None)
        if result:
            print(
                f"{card1} {card2}: Call={result['call_prob']}, Fold={result['fold_prob']}, Value={result['value']:.4f}"
            )

    # Print range table for SB all-in
    if sb_action == "allin":
        print_bb_call_range_table(results, sb_action)

    return results


def print_bb_call_range_table(results: List[Dict[str, Any]], sb_action: str = "allin"):
    """Print a range table showing BB call probabilities when facing SB action."""
    print(f"\n--- BB Call Range Table (SB {sb_action.upper()}) ---")

    # Create a 13x13 grid representing all possible hole card combinations
    # Rows/cols: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

    # Initialize grid
    grid = []
    header = "    " + " ".join(f"{rank:>2}" for rank in ranks)
    grid.append(header)
    grid.append("   " + "-" * 39)  # Separator line

    # Create a lookup dictionary for quick access
    results_dict = {
        (result["cards"][0], result["cards"][1]): result for result in results
    }

    for i, rank1 in enumerate(ranks):
        row = [f"{rank1:>2} |"]
        for j, rank2 in enumerate(ranks):
            if i == j:
                # Same rank (pairs) - always suited
                card1, card2 = f"{rank1}s", f"{rank1}h"  # Suited pair
            elif i < j:
                # Top-right triangle: suited hands (e.g., AKs, AQs)
                card1, card2 = f"{rank1}s", f"{rank2}s"  # Suited
            else:
                # Bottom-left triangle: off-suit hands (e.g., KAs, QAs)
                card1, card2 = f"{rank2}s", f"{rank1}h"  # Off-suit

            # Get call probability for this hand
            if (card1, card2) in results_dict:
                call_percentage = results_dict[(card1, card2)]["call_percentage"]
                prob_str = f"{call_percentage:>2}"
            else:
                prob_str = " 0"  # Default if not found

            row.append(prob_str)

        grid.append(" ".join(row))

    # Print the grid
    for line in grid:
        print(line)

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze preflop strategies from a checkpoint"
    )
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument(
        "--device", default="mps", help="Device to use (mps, cuda, cpu)"
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

    try:
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

        if args.analyze_169:
            # Analyze all 169 hands using vectorized approach
            print(f"\nAnalyzing all 169 hands at step {step}")
            results = analyze_169_hands_vectorized(trainer, args.sb_action)
            print(f"\n=== ANALYSIS COMPLETE ===")
            print(f"Analyzed {len(results)} hands with SB action: {args.sb_action}")
        else:
            # Print SB preflop tables
            print(f"\nAnalyzing preflop strategies at step {step}")
            print_sb_preflop_tables(trainer, step)

            # Print BB response tables
            print_bb_response_tables(trainer, step)

            print("\n" + "=" * 80)
            print("ANALYSIS COMPLETE")
            print("=" * 80)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def analyze_bb_responses(
    env: HUNLTensorEnv,
    model,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Analyze BB responses to SB actions across all 169 hands.

    Args:
        env: HUNLTensorEnv with 169 environments
        model: Trained model for prediction
        device: Device for tensor operations

    Returns:
        Dictionary containing results for all 169 hands
    """
    if device is None:
        device = env.device

    hands = create_169_hand_combinations()

    # Create appropriate state encoder for the model
    state_encoder = create_state_encoder_for_model(model, env, device)

    # Get probabilities and values using the centralized function
    probs, values, legal_masks = get_probabilities(
        model, state_encoder, env, seat=1, device=device
    )

    # Get call and fold probabilities
    call_probs = probs[:, 1]  # Second bin is call
    fold_probs = probs[:, 0]  # First bin is fold

    # Convert to percentage and format
    call_percentages = torch.round(call_probs * 100).int()
    fold_percentages = torch.round(fold_probs * 100).int()

    # Create results
    results = []
    for i, (card1, card2) in enumerate(hands):
        call_percentage = call_percentages[i].item()
        fold_percentage = fold_percentages[i].item()

        if call_percentage >= 100:
            call_prob_str = "██"
        else:
            call_prob_str = f"{call_percentage:2d}"

        if fold_percentage >= 100:
            fold_prob_str = "██"
        else:
            fold_prob_str = f"{fold_percentage:2d}"

        results.append(
            {
                "cards": (card1, card2),
                "call_prob": call_prob_str,
                "fold_prob": fold_prob_str,
                "call_percentage": call_percentage,
                "fold_percentage": fold_percentage,
                "value": values[i].item(),
            }
        )

    return {
        "results": results,
        "hands": hands,
        "values": values,
        "legal_masks": legal_masks,
        "probs": probs,
    }


if __name__ == "__main__":
    main()
