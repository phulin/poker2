#!/usr/bin/env python3
"""
Demonstration script for K-Best self-play in AlphaHoldem.

This script shows how to use the K-Best self-play mechanism as described in the AlphaHoldem paper.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphaholdem.rl.self_play import SelfPlayTrainer


def demonstrate_kbest():
    """Demonstrate K-Best self-play functionality."""
    print("=== AlphaHoldem K-Best Self-Play Demonstration ===\n")

    # Initialize trainer with K-Best pool
    trainer = SelfPlayTrainer()

    print("Initial setup:")
    print(f"  - K-Best pool size: {trainer.opponent_pool.k}")
    print(f"  - Min ELO difference: {trainer.opponent_pool.min_elo_diff}")
    print(f"  - Starting ELO: {trainer.current_elo}")
    print(f"  - Pool size: {trainer.opponent_pool.get_pool_stats()['pool_size']}")
    print()

    # Run training for a few steps
    print("Running training steps...")
    for step in range(10):
        stats = trainer.train_step(num_trajectories=3)

        print(f"Step {step + 1}:")
        print(f"  - Episodes: {stats['episode_count']}")
        print(f"  - Avg Reward: {stats['avg_reward']:.2f}")
        print(f"  - Current ELO: {stats['current_elo']:.1f}")
        print(f"  - Pool Size: {stats['pool_stats']['pool_size']}")

        # Show pool details every few steps
        if step % 3 == 0:
            pool_stats = stats["pool_stats"]
            if pool_stats["pool_size"] > 0:
                print(f"  - Best opponent ELO: {pool_stats['best_snapshot_elo']:.1f}")
                print(
                    f"  - Pool ELO range: {pool_stats['min_elo']:.1f} - {pool_stats['max_elo']:.1f}"
                )
        print()

    # Evaluate against the pool
    print("Evaluating against opponent pool...")
    eval_results = trainer.evaluate_against_pool(num_games=20)

    print(f"Overall win rate: {eval_results['overall_win_rate']:.3f}")
    print("Individual opponent results:")
    for opponent_key, result in eval_results["opponent_results"].items():
        print(
            f"  {opponent_key}: {result['win_rate']:.3f} (ELO: {result['opponent_elo']:.1f})"
        )
    print()

    # Show preflop range
    print("Preflop range grid (button play):")
    print(trainer.get_preflop_range_grid(seat=0))
    print()

    # Demonstrate checkpointing
    print("Saving checkpoint...")
    trainer.save_checkpoint("demo_checkpoint.pt", 10)

    print("Loading checkpoint...")
    trainer.load_checkpoint("demo_checkpoint.pt")

    # Cleanup
    if os.path.exists("demo_checkpoint.pt"):
        os.remove("demo_checkpoint.pt")
    if os.path.exists("demo_checkpoint_pool.pt"):
        os.remove("demo_checkpoint_pool.pt")

    print("=== Demonstration completed! ===")


def explain_kbest_concept():
    """Explain the K-Best self-play concept."""
    print("=== K-Best Self-Play Concept ===\n")

    print("K-Best self-play is a training mechanism that addresses the problem")
    print("of agents getting trapped in local minima during self-play training.\n")

    print("Key components:")
    print("1. Opponent Pool: Maintains K best historical versions of the agent")
    print("2. ELO Rating: Tracks the strength of each version")
    print(
        "3. Diverse Sampling: Trains against different opponents to avoid overfitting"
    )
    print(
        "4. Dynamic Updates: Adds new snapshots when significant improvement occurs\n"
    )

    print("Benefits:")
    print("- Prevents strategy cycling and local minima")
    print("- Ensures diverse training data")
    print("- Maintains strong opponents for challenging training")
    print("- Enables better exploration of the strategy space\n")

    print("As described in the AlphaHoldem paper, this approach:")
    print("- Trains only one main agent (efficient)")
    print("- Maintains a pool of competing agents (diverse)")
    print("- Samples opponents by ELO rating (challenging)")
    print("- Updates pool based on performance (adaptive)\n")


def main():
    """Run the demonstration."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Explain the concept
    explain_kbest_concept()

    # Run demonstration
    demonstrate_kbest()


if __name__ == "__main__":
    main()
