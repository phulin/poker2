#!/usr/bin/env python3
"""
Debug script to test ELO update logic specifically.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from alphaholdem.rl.k_best_pool import KBestOpponentPool, AgentSnapshot
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.core.config_loader import get_config


def debug_elo_update():
    """Debug the ELO update logic specifically."""

    print("🔍 Debugging ELO Update Logic")
    print("=" * 50)

    # Test 1: Direct ELO update test
    print("\n📊 Test 1: Direct ELO Update")
    pool = KBestOpponentPool(k=3, min_elo_diff=50.0)

    # Create a dummy opponent
    class DummyModel:
        def __init__(self):
            pass

    opponent = AgentSnapshot(DummyModel(), step=0, elo=1200.0)
    pool.snapshots.append(opponent)

    print(
        f"Before update - Current ELO: {pool.current_elo}, Opponent ELO: {opponent.elo}"
    )

    # Test single game update
    pool.update_elo_after_game(opponent, "win")
    print(
        f"After single win - Current ELO: {pool.current_elo}, Opponent ELO: {opponent.elo}"
    )

    # Test vectorized update
    print(f"\n📊 Test 2: Vectorized ELO Update")
    print(
        f"Before vectorized - Current ELO: {pool.current_elo}, Opponent ELO: {opponent.elo}"
    )

    # Create rewards tensor for vectorized update
    rewards = torch.tensor(
        [1.0, -1.0, 0.0], device=torch.device("cpu")
    )  # win, loss, draw
    opponents = [opponent, opponent, opponent]

    pool.update_elo_batch_vectorized(opponents, rewards)
    print(
        f"After vectorized - Current ELO: {pool.current_elo}, Opponent ELO: {opponent.elo}"
    )

    # Test 3: Check if the issue is in the training loop
    print(f"\n📊 Test 3: Training Loop ELO Update")
    device = torch.device("cpu")
    config = get_config("configs/default.yaml")

    trainer = SelfPlayTrainer(
        device=device,
        config=config,
        use_tensor_env=True,
        num_envs=4,
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
    print(f"Pool current ELO: {trainer.opponent_pool.current_elo}")

    # Add a snapshot
    trainer.opponent_pool.add_snapshot(trainer, trainer.current_elo)
    print(f"After adding snapshot - Current ELO: {trainer.current_elo}")
    print(f"Pool current ELO: {trainer.opponent_pool.current_elo}")

    # Run a training step
    stats = trainer.train_step()
    print(f"After training step - Current ELO: {trainer.current_elo}")
    print(f"Pool current ELO: {trainer.opponent_pool.current_elo}")
    print(f"Training stats current_elo: {stats.get('current_elo', 'NOT_FOUND')}")

    print("\n✅ ELO Update Debug Complete!")


if __name__ == "__main__":
    debug_elo_update()
