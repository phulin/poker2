#!/usr/bin/env python3
"""
Debug script to investigate ELO rating system issues.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.core.config import RootConfig


def debug_elo_system():
    """Debug the ELO rating system to find why it's not updating."""

    print("🔍 Debugging ELO Rating System")
    print("=" * 50)

    # Initialize trainer
    device = torch.device("cpu")  # Use CPU for debugging
    from alphaholdem.core.config_loader import get_config

    config = get_config("configs/default.yaml")

    trainer = SelfPlayTrainer(
        device=device,
        config=config,
        use_tensor_env=True,  # Test with tensorized environment
        num_envs=4,  # Small number for debugging
        k_best_pool_size=3,
        min_elo_diff=50.0,
        learning_rate=1e-3,
        batch_size=64,
        num_epochs=4,
        gamma=0.999,
        gae_lambda=0.95,
        epsilon=0.2,
        delta1=3.0,
        value_coef=0.5,
        entropy_coef=0.01,
        grad_clip=0.5,
    )

    print(f"Initial ELO: {trainer.current_elo}")
    print(f"Pool size: {len(trainer.opponent_pool.snapshots)}")
    print(f"Pool stats: {trainer.opponent_pool.get_pool_stats()}")

    # Test 1: Check if opponent pool is empty initially
    print("\n📊 Test 1: Initial Pool State")
    print(f"Snapshots: {len(trainer.opponent_pool.snapshots)}")
    print(f"Current ELO: {trainer.current_elo}")

    # Test 2: Add a snapshot manually
    print("\n📊 Test 2: Adding Initial Snapshot")
    trainer.opponent_pool.add_snapshot(trainer, trainer.current_elo)
    print(f"Snapshots after add: {len(trainer.opponent_pool.snapshots)}")
    print(f"Pool stats: {trainer.opponent_pool.get_pool_stats()}")

    # Test 3: Run a training step
    print("\n📊 Test 3: Running Training Step")
    print(f"ELO before training step: {trainer.current_elo}")

    stats = trainer.train_step()
    print(f"ELO after training step: {trainer.current_elo}")
    print(f"Training stats: {stats}")

    # Test 4: Check if ELO was updated during training
    print("\n📊 Test 4: ELO Update Check")
    print(f"Current ELO: {trainer.current_elo}")
    print(f"Pool current ELO: {trainer.opponent_pool.current_elo}")

    # Test 5: Manual ELO update test
    print("\n📊 Test 5: Manual ELO Update Test")
    if trainer.opponent_pool.snapshots:
        opponent = trainer.opponent_pool.snapshots[0]
        print(
            f"Before update - Current ELO: {trainer.current_elo}, Opponent ELO: {opponent.elo}"
        )

        # Simulate a win
        trainer.opponent_pool.update_elo_after_game(opponent, "win")
        print(
            f"After win - Current ELO: {trainer.current_elo}, Opponent ELO: {opponent.elo}"
        )

        # Simulate a loss
        trainer.opponent_pool.update_elo_after_game(opponent, "loss")
        print(
            f"After loss - Current ELO: {trainer.current_elo}, Opponent ELO: {opponent.elo}"
        )

    # Test 6: Check evaluation logic
    print("\n📊 Test 6: Evaluation Logic Check")
    eval_results = trainer.evaluate_against_pool(num_games=5)
    print(f"Evaluation results: {eval_results}")

    # Test 7: Check if evaluation updates ELO
    print(f"ELO after evaluation: {trainer.current_elo}")

    print("\n✅ ELO Debugging Complete!")


if __name__ == "__main__":
    debug_elo_system()
