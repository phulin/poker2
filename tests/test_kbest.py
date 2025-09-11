#!/usr/bin/env python3
"""
Test script for K-Best self-play functionality.

This script tests the basic functionality of the K-Best opponent pool and self-play training.
"""

import os
import sys
import torch
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphaholdem.rl.k_best_pool import KBestOpponentPool, AgentSnapshot
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.models.cnn import SiameseConvNetV1
from alphaholdem.core.structured_config import (
    Config,
    TrainingConfig,
    ModelConfig,
    EnvConfig,
)


def test_kbest_pool():
    """Test basic K-Best pool functionality."""
    print("Testing K-Best opponent pool...")

    # Create a pool
    pool = KBestOpponentPool(k=3, min_elo_diff=50.0)

    # Create some dummy models
    model1 = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=128,
        actions_hidden=128,
        fusion_hidden=[256, 256],
        num_actions=8,
    )
    model2 = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=128,
        actions_hidden=128,
        fusion_hidden=[256, 256],
        num_actions=8,
    )
    model3 = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=128,
        actions_hidden=128,
        fusion_hidden=[256, 256],
        num_actions=8,
    )

    # Create snapshots
    snapshot1 = AgentSnapshot(model1, step=100, elo=1200.0)
    snapshot2 = AgentSnapshot(model2, step=200, elo=1250.0)
    snapshot3 = AgentSnapshot(model3, step=300, elo=1300.0)

    # Add snapshots
    pool.add_snapshot(model1, 100, rating=1200.0)
    pool.add_snapshot(model2, 200, rating=1250.0)
    pool.add_snapshot(model3, 300, rating=1300.0)

    # Test sampling
    opponents = pool.sample(k=2)
    print(f"Sampled {len(opponents)} opponents")

    # Test pool stats
    stats = pool.get_pool_stats()
    print(f"Pool stats: {stats}")

    # Test ELO updates
    if opponents:
        pool.update_elo_after_game(opponents[0], "win")
        print(f"After win, current ELO: {pool.current_elo}")

    print("K-Best pool test completed!\n")


def test_selfplay_with_kbest():
    """Test self-play training with K-Best opponents."""
    print("Testing self-play with K-Best opponents...")

    # Create a Hydra config with small parameters for fast testing
    cfg = Config(
        train=TrainingConfig(
            learning_rate=1e-3,
            batch_size=4,  # Small batch for testing
            num_epochs=1,  # Only 1 epoch instead of 4
        ),
        model=ModelConfig(),
        env=EnvConfig(),
        k_best_pool_size=2,  # Smaller pool
        min_elo_diff=20.0,  # Lower threshold
        use_tensor_env=True,  # Use faster tensor environment
        num_envs=2,  # Much smaller than default 256
        device="cpu",  # Set device to cpu for testing
    )

    # Set device for testing
    device = torch.device("cpu")

    trainer = SelfPlayTrainer(
        cfg=cfg,
        device=device,
    )

    print(f"Initial ELO: {trainer.opponent_pool.current_elo}")
    print(f"Initial pool size: {trainer.opponent_pool.get_pool_stats()['pool_size']}")

    # Run fewer training steps for faster testing
    for step in range(2):  # Reduced from 5 to 2
        stats = trainer.train_step(step + 1)
        print(
            f"Step {step + 1}: ELO={stats['current_elo']:.1f}, "
            f"Pool size={stats['pool_stats']['pool_size']}, "
            f"Avg reward={stats['avg_reward']:.2f}"
        )

    # Test evaluation with fewer games
    print("Testing evaluation against pool...")
    eval_results = trainer.evaluate_against_pool(min_games=3)  # Reduced from 10 to 3
    print(f"Evaluation results: {eval_results}")

    # Test checkpointing
    print("Testing checkpointing...")
    trainer.save_checkpoint("test_checkpoint.pt", 2)
    trainer.load_checkpoint("test_checkpoint.pt")

    # Cleanup
    if os.path.exists("test_checkpoint.pt"):
        os.remove("test_checkpoint.pt")
    if os.path.exists("test_checkpoint_pool.pt"):
        os.remove("test_checkpoint_pool.pt")

    print("Self-play with K-Best test completed!\n")


def test_opponent_sampling():
    """Test that opponent sampling works correctly."""
    print("Testing opponent sampling...")

    # Create a Hydra config with small parameters for testing
    cfg = Config(
        train=TrainingConfig(batch_size=4),  # Smaller batch
        model=ModelConfig(),
        env=EnvConfig(),
        k_best_pool_size=3,  # Smaller pool
        min_elo_diff=15.0,  # Lower threshold
        use_tensor_env=True,  # Use faster tensor environment
        num_envs=2,  # Very small for testing
        device="cpu",  # Set device to cpu for testing
    )

    # Set device for testing
    device = torch.device("cpu")

    trainer = SelfPlayTrainer(
        cfg=cfg,
        device=device,
    )

    # Add some snapshots with different ELOs
    for i in range(5):
        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )

        trainer.opponent_pool.add_snapshot(model, (i + 1) * 100, rating=1200.0 + i * 50)

    print(f"Pool size: {trainer.opponent_pool.get_pool_stats()['pool_size']}")

    # Test sampling
    opponents = trainer.opponent_pool.sample(k=3)
    print(f"Sampled {len(opponents)} opponents")

    # Check that sampling returns different opponents
    opponent_steps = [opp.step for opp in opponents]
    print(f"Opponent steps: {opponent_steps}")

    print("Opponent sampling test completed!\n")


def main():
    """Run all tests."""
    print("Running K-Best self-play tests...\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)

    try:
        test_kbest_pool()
        test_selfplay_with_kbest()
        test_opponent_sampling()

        print("All tests passed! ✅")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
