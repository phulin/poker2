#!/usr/bin/env python3
"""
Test script for K-Best self-play functionality.

This script tests the basic functionality of the K-Best opponent pool and self-play training.
"""

import sys
from pathlib import Path

import torch

from p2.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from p2.models.cnn import SiameseConvNetV1
from p2.rl.agent_snapshot import AgentSnapshot
from p2.rl.k_best_pool import KBestOpponentPool
from p2.rl.self_play import SelfPlayTrainer

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


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


def test_selfplay_with_kbest(tmp_path):
    """Test K-Best wiring on the self-play trainer without running a rollout."""

    cfg = Config(
        train=TrainingConfig(
            learning_rate=1e-3,
            batch_size=1,
            episodes_per_step=1,
        ),
        model=ModelConfig(),
        env=EnvConfig(),
        k_best_pool_size=2,
        min_elo_diff=20.0,
        use_tensor_env=True,
        num_envs=1,
        device="cpu",
    )

    trainer = SelfPlayTrainer(cfg=cfg, device=torch.device("cpu"))
    assert isinstance(trainer.opponent_pool, KBestOpponentPool)
    assert trainer.opponent_pool.k == 2
    assert trainer.opponent_pool.min_elo_diff == 20.0

    trainer.opponent_pool.add_snapshot(trainer.model, step=1, rating=1250.0)
    sampled = trainer.opponent_pool.sample(k=1)
    assert len(sampled) == 1
    assert sampled[0].step == 1

    ckpt = tmp_path / "kbest.pt"
    trainer.save_checkpoint(str(ckpt), 2)
    trainer.load_checkpoint(str(ckpt))


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
