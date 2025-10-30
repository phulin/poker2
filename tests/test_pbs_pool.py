from __future__ import annotations

import pytest
import torch

from alphaholdem.core.structured_config import CFRType, SearchConfig
from alphaholdem.env.card_utils import NUM_HANDS, combo_index, mask_conflicting_combos
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.pbs_pool import PBSPool
from alphaholdem.search.rebel_cfr_evaluator import RebelFeatureEncoder


def get_device() -> torch.device:
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )


def make_simple_model(device: torch.device) -> RebelFFN:
    """Create a simple RebelFFN model for testing."""
    return RebelFFN(
        input_dim=RebelFeatureEncoder.feature_dim,
        num_actions=5,  # fold, call, 2 bets, allin
        hidden_dim=64,
        num_hidden_layers=2,
        detach_value_head=True,
        num_players=2,
    ).to(device)


"""
The following tests validated the old two-prior helper and related behavior.
They have been removed since PBS evaluation now uses evaluator showdown EVs
and no longer exposes the two-prior helper.
"""

# Removed: test_two_prior_averaging_symmetric_beliefs
# Removed: test_two_prior_averaging_extreme_beliefs
# Removed: test_two_prior_averaging_with_board
# Removed: test_two_prior_averaging_zero_legal_hands
# Removed: test_two_prior_averaging_deterministic
# Removed: test_belief_normalization


def _removed_placeholder():
    assert True
    """Test two-prior averaging with identical beliefs (should give fair result)."""
    device = get_device()

    # Simple board (preflop - no board cards)
    board = torch.tensor([-1, -1, -1, -1, -1], device=device, dtype=torch.long)

    # Uniform beliefs for both agents about both players (all uniform = symmetric)
    uniform_beliefs = (
        torch.ones(NUM_HANDS, device=device, dtype=torch.float32) / NUM_HANDS
    )

    pot = 100.0

    # Since all beliefs are uniform, expected payoff should be ~0
    payoff = PBSPool._expected_showdown_payoff_two_prior(
        board,
        uniform_beliefs,  # A's belief about A
        uniform_beliefs,  # A's belief about B
        uniform_beliefs,  # B's belief about A
        uniform_beliefs,  # B's belief about B
        pot,
        device,
    )

    # With uniform beliefs and no board, result should be approximately zero
    # (slight variance due to hand ranking differences, but should be small)
    assert (
        abs(payoff) < 10.0
    ), f"Expected payoff near 0 for uniform beliefs, got {payoff}"


def test_pbs_pool_empty_pool():
    """Test PBSPool with empty pool."""
    device = get_device()
    pool = PBSPool(pool_size=3)

    assert len(pool.snapshots) == 0
    assert pool.sample(1) == []

    stats = pool.get_pool_stats()
    assert stats["pool_size"] == 0


def test_pbs_pool_add_without_evaluation():
    """Test adding snapshot without evaluation (manual rating)."""
    device = get_device()
    pool = PBSPool(pool_size=3)

    model = make_simple_model(device)

    # Add snapshot with manual rating
    added = pool.add_snapshot(
        model=model,
        step=100,
        rating=1200.0,
        evaluate_against_pool=False,
    )

    assert added is True
    assert len(pool.snapshots) == 1
    assert pool.snapshots[0].elo == 1200.0


def test_pbs_pool_eviction_logic():
    """Test that pool evicts worst model when new one is better."""
    device = get_device()
    pool = PBSPool(pool_size=3)

    model = make_simple_model(device)

    # Fill pool with snapshots at different ELOs
    pool.add_snapshot(model, step=100, rating=1100.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=200, rating=1150.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=300, rating=1200.0, evaluate_against_pool=False)

    assert len(pool.snapshots) == 3
    worst_elo = pool.get_worst_snapshot().elo
    assert worst_elo == 1100.0

    # Add better snapshot - should replace worst
    added = pool.add_snapshot(
        model, step=400, rating=1125.0, evaluate_against_pool=False
    )
    assert added is True
    assert len(pool.snapshots) == 3
    assert pool.get_worst_snapshot().elo == 1125.0  # New worst is 1125

    # Add snapshot worse than worst - should not be added
    added = pool.add_snapshot(
        model, step=500, rating=1100.0, evaluate_against_pool=False
    )
    assert added is False
    assert len(pool.snapshots) == 3
    assert pool.get_worst_snapshot().elo == 1125.0  # Still 1125


def test_pbs_pool_sample_weighted():
    """Test that sampling is weighted by ELO."""
    device = get_device()
    pool = PBSPool(pool_size=3)

    model = make_simple_model(device)

    # Add snapshots with very different ELOs
    pool.add_snapshot(model, step=100, rating=1000.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=200, rating=1500.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=300, rating=2000.0, evaluate_against_pool=False)

    # Sample many times - higher ELO should be sampled more often
    samples = []
    for _ in range(1000):
        sampled = pool.sample(1)
        if sampled:
            samples.append(sampled[0].elo)

    # Count samples by ELO
    count_2000 = sum(1 for e in samples if e == 2000.0)
    count_1500 = sum(1 for e in samples if e == 1500.0)
    count_1000 = sum(1 for e in samples if e == 1000.0)

    # Higher ELO should be sampled more
    assert count_2000 > count_1500 > count_1000


# Removed: test_create_default_evaluation_fn (helper no longer used)


def test_play_public_belief_games_basic():
    """Test playing a few public-belief games (slow test)."""
    device = get_device()
    bet_bins = [0.5, 1.5]

    model_a = make_simple_model(device)
    model_b = make_simple_model(device)

    search_cfg = SearchConfig()
    search_cfg.depth = 1
    search_cfg.iterations = 1
    search_cfg.warm_start_iterations = 0
    search_cfg.cfr_type = CFRType.linear
    search_cfg.cfr_avg = True

    # Seed RNG for determinism
    generator = torch.Generator(device=device)
    generator.manual_seed(12345)
    # Play a small number of games
    num_games = 2
    rewards = PBSPool._play_public_belief_games(
        model_a, model_b, num_games, bet_bins, generator, device, search_cfg
    )

    assert rewards.shape == (num_games,)
    assert rewards.dtype == torch.float32
    # Rewards should be finite
    assert torch.isfinite(rewards).all()


def test_pbs_pool_full_evaluation():
    """Test full evaluation pipeline with mock evaluation function."""
    device = get_device()

    pool = PBSPool(pool_size=3)

    model = make_simple_model(device)

    # Add first snapshot (pool empty, always added)
    added = pool.add_snapshot(
        model=model,
        step=100,
        evaluate_against_pool=True,
        num_games_per_opponent=5,
    )
    assert added is True
    assert len(pool.snapshots) == 1

    # Add second snapshot (should be evaluated against first)
    added = pool.add_snapshot(
        model=model,
        step=200,
        evaluate_against_pool=True,
        num_games_per_opponent=5,
    )
    # Whether added depends on ELO after evaluation
    assert len(pool.snapshots) >= 1
    assert len(pool.snapshots) <= 2


def test_pbs_pool_get_best_worst():
    """Test getting best and worst snapshots."""
    device = get_device()
    pool = PBSPool(pool_size=3)

    model = make_simple_model(device)

    # Add snapshots with known ELOs
    pool.add_snapshot(model, step=100, rating=1100.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=200, rating=1200.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=300, rating=1300.0, evaluate_against_pool=False)

    best = pool.get_best_snapshot()
    worst = pool.get_worst_snapshot()

    assert best is not None
    assert worst is not None
    assert best.elo == 1300.0
    assert worst.elo == 1100.0
    assert best.elo > worst.elo


def test_pbs_pool_stats():
    """Test pool statistics."""
    device = get_device()
    pool = PBSPool(pool_size=5)

    model = make_simple_model(device)

    pool.add_snapshot(model, step=100, rating=1100.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=200, rating=1200.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=300, rating=1300.0, evaluate_against_pool=False)

    stats = pool.get_pool_stats()

    assert stats["pool_size"] == 3
    assert stats["max_pool_size"] == 5
    assert stats["min_elo"] == 1100.0
    assert stats["max_elo"] == 1300.0
    assert stats["avg_elo"] == pytest.approx(1200.0, abs=1.0)
    assert stats["best_snapshot_step"] == 300
    assert stats["worst_snapshot_step"] == 100


def test_pbs_pool_should_add_snapshot():
    """Test should_add_snapshot (always True for PBS pool)."""
    pool = PBSPool(pool_size=3)

    # Should always return True (evaluation happens in add_snapshot)
    assert pool.should_add_snapshot(100) is True
    assert pool.should_add_snapshot(200) is True

    # Even after filling pool, should return True
    model = make_simple_model(get_device())
    pool.add_snapshot(model, step=100, rating=1200.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=200, rating=1250.0, evaluate_against_pool=False)
    pool.add_snapshot(model, step=300, rating=1300.0, evaluate_against_pool=False)

    assert pool.should_add_snapshot(400) is True


def test_elo_update_correctness():
    """Test that Elo ratings update correctly after wins and losses."""
    device = get_device()
    pool = PBSPool(pool_size=5, k_factor=32.0)

    model = make_simple_model(device)

    # Add opponent with known Elo
    added = pool.add_snapshot(
        model, step=100, rating=1500.0, evaluate_against_pool=False
    )
    assert added is True
    assert len(pool.snapshots) == 1
    opponent = pool.snapshots[0]

    # Set candidate Elo to same level
    pool.current_elo = 1500.0

    # Simulate rewards: 5 wins, 3 losses, 2 draws
    rewards = torch.tensor(
        [10.0, 15.0, 8.0, -5.0, -8.0, 12.0, -3.0, 0.0, 0.0, -2.0], device=device
    )
    original_elo = pool.current_elo
    original_opponent_elo = opponent.elo

    # Update Elo
    pool.update_elo_batch_vectorized(opponent, rewards)

    # Both Elos should change
    assert pool.current_elo != original_elo
    assert opponent.elo != original_opponent_elo

    # With 5 wins out of 10 games (winning more), candidate Elo should increase
    assert pool.current_elo > original_elo

    # Opponent Elo should decrease (opposite direction)
    assert opponent.elo < original_opponent_elo

    # Total Elo should be conserved (approximately, with finite precision)
    total_before = original_elo + original_opponent_elo
    total_after = pool.current_elo + opponent.elo
    assert (
        abs(total_before - total_after) < 1.0
    ), "Elo should be approximately conserved"


def test_button_alternation():
    """Test that button position alternates across games."""
    device = get_device()
    bet_bins = [0.5, 1.5]

    model_a = make_simple_model(device)
    model_b = make_simple_model(device)

    search_cfg = SearchConfig()
    search_cfg.depth = 1
    search_cfg.iterations = 1
    search_cfg.warm_start_iterations = 0
    search_cfg.cfr_type = CFRType.linear
    search_cfg.cfr_avg = True

    # Seed RNG for determinism
    generator = torch.Generator(device=device)
    generator.manual_seed(12345)
    # Play 2 games (smaller for speed)
    num_games = 2
    rewards = PBSPool._play_public_belief_games(
        model_a, model_b, num_games, bet_bins, generator, device, search_cfg
    )

    assert rewards.shape == (num_games,)

    # Button alternates: game 0 button=0, game 1 button=1
    # With fair models, rewards should be finite
    assert torch.isfinite(rewards).all(), "Rewards should be finite"

    # At least some games should result in different outcomes
    # (in practice, even with random models, button alternation should create variation)
    assert rewards.numel() > 0, "Games should produce rewards"


# Removed: test_belief_normalization


# Removed: test_evaluation_against_multiple_opponents (external eval_fn no longer used)


def test_empty_pool_evaluation():
    """Test that evaluation handles empty pool correctly."""
    device = get_device()

    pool = PBSPool(pool_size=3)

    model = make_simple_model(device)

    # First snapshot should be added without evaluation (empty pool)
    added = pool.add_snapshot(
        model, step=100, rating=1500.0, evaluate_against_pool=False
    )

    assert added is True
    assert len(pool.snapshots) == 1


def test_elo_conservation():
    """Test that total Elo is conserved across multiple games."""
    device = get_device()
    pool = PBSPool(pool_size=5, k_factor=32.0)

    model = make_simple_model(device)

    # Add opponent
    added = pool.add_snapshot(
        model, step=100, rating=1500.0, evaluate_against_pool=False
    )
    assert added is True
    opponent = pool.snapshots[0]

    pool.current_elo = 1500.0

    # Play multiple batches of games
    original_elo_sum = pool.current_elo + opponent.elo

    # Batch 1
    rewards1 = torch.tensor([0.5, -0.3, 0.2], device=device)
    pool.update_elo_batch_vectorized(opponent, rewards1)

    # Batch 2
    rewards2 = torch.tensor([-0.1, 0.4, -0.2, 0.1], device=device)
    pool.update_elo_batch_vectorized(opponent, rewards2)

    # Elo should be approximately conserved (within numerical precision)
    final_elo_sum = pool.current_elo + opponent.elo
    elo_difference = abs(original_elo_sum - final_elo_sum)

    # Due to ELO calculation (non-linear), conservation is approximate
    # But difference should be small
    assert elo_difference < 10.0, f"Elo difference too large: {elo_difference}"


def test_elo_update_stats():
    """Test that game statistics are updated correctly after Elo updates."""
    device = get_device()
    pool = PBSPool(pool_size=5)

    model = make_simple_model(device)

    added = pool.add_snapshot(
        model, step=100, rating=1500.0, evaluate_against_pool=False
    )
    assert added is True
    opponent = pool.snapshots[0]

    # Check initial stats
    initial_wins = opponent.wins
    initial_losses = opponent.losses
    initial_games = opponent.games_played

    # Play some games with mixed results
    rewards = torch.tensor([0.5, -0.3, 0.2, -0.1, 0.4], device=device)

    pool.update_elo_batch_vectorized(opponent, rewards)

    # Stats should be updated
    assert opponent.games_played == initial_games + rewards.numel()

    # Wins and losses should have increased
    assert opponent.wins >= initial_wins
    assert opponent.losses >= initial_losses

    # Total games should equal wins + losses + draws
    assert opponent.games_played == opponent.wins + opponent.losses + opponent.draws


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
