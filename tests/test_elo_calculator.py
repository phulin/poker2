#!/usr/bin/env python3
"""
Tests for ELO calculator functionality.
"""

import pytest
import torch

from alphaholdem.models.cnn import SiameseConvNetV1
from alphaholdem.rl.agent_snapshot import AgentSnapshot
from alphaholdem.rl.elo_calculator import ELOCalculator


class TestELOCalculator:
    """Test cases for ELO calculator."""

    def test_elo_calculator_initialization(self):
        """Test ELO calculator initialization."""
        calc = ELOCalculator(k_factor=32.0)
        assert calc.k_factor == 32.0

        # Test default initialization
        calc_default = ELOCalculator()
        assert calc_default.k_factor == 32.0

    def test_elo_update_win(self):
        """Test ELO update after a win."""
        calc = ELOCalculator(k_factor=32.0)

        # Create opponent with higher ELO
        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1300.0)

        # Current player has lower ELO, wins against higher ELO opponent
        current_elo = 1200.0
        new_elo = calc.update_elo_after_game(current_elo, opponent, "win")

        # Should gain ELO points (upset win)
        assert new_elo > current_elo
        # Expected score = 1/(1 + 10^((1300-1200)/400)) = 1/(1 + 10^0.25) ≈ 0.36
        # ELO change = 32 * (1.0 - 0.36) = 32 * 0.64 ≈ 20.5
        assert new_elo == pytest.approx(1220.5, abs=1.0)

    def test_elo_update_loss(self):
        """Test ELO update after a loss."""
        calc = ELOCalculator(k_factor=32.0)

        # Create opponent with lower ELO
        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1100.0)

        # Current player has higher ELO, loses to lower ELO opponent
        current_elo = 1200.0
        new_elo = calc.update_elo_after_game(current_elo, opponent, "loss")

        # Should lose ELO points (upset loss)
        assert new_elo < current_elo
        # Expected score = 1/(1 + 10^((1100-1200)/400)) = 1/(1 + 10^-0.25) ≈ 0.64
        # ELO change = 32 * (0.0 - 0.64) = 32 * -0.64 ≈ -20.5
        assert new_elo == pytest.approx(1179.5, abs=1.0)

    def test_elo_update_draw(self):
        """Test ELO update after a draw."""
        calc = ELOCalculator(k_factor=32.0)

        # Create opponent with same ELO
        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1200.0)

        # Current player has same ELO, draws
        current_elo = 1200.0
        new_elo = calc.update_elo_after_game(current_elo, opponent, "draw")

        # Should have minimal change (expected score = 0.5, actual score = 0.5)
        assert new_elo == pytest.approx(current_elo, abs=0.1)

    def test_elo_update_custom_k_factor(self):
        """Test ELO update with custom K-factor."""
        calc = ELOCalculator(k_factor=16.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1300.0)

        current_elo = 1200.0
        new_elo = calc.update_elo_after_game(
            current_elo, opponent, "win", k_factor=16.0
        )

        # Should gain fewer points with smaller K-factor
        assert new_elo > current_elo
        # Expected score = 1/(1 + 10^((1300-1200)/400)) = 1/(1 + 10^0.25) ≈ 0.36
        # ELO change = 16 * (1.0 - 0.36) = 16 * 0.64 ≈ 10.2
        assert new_elo == pytest.approx(1210.2, abs=1.0)

    def test_elo_update_expected_score_calculation(self):
        """Test that expected score calculation is correct."""
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )

        # Test with equal ELO (expected score should be 0.5)
        opponent_equal = AgentSnapshot(model, step=100, elo=1200.0)
        current_elo = 1200.0

        # Win against equal opponent
        new_elo_win = calc.update_elo_after_game(current_elo, opponent_equal, "win")
        assert new_elo_win == pytest.approx(1216.0, abs=1.0)  # +16 points

        # Loss against equal opponent
        new_elo_loss = calc.update_elo_after_game(current_elo, opponent_equal, "loss")
        assert new_elo_loss == pytest.approx(1184.0, abs=1.0)  # -16 points

    def test_vectorized_elo_update(self):
        """Test vectorized ELO update for multiple games."""
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1200.0)

        # Test with mixed results against a stronger opponent
        opponent = AgentSnapshot(model, step=100, elo=1300.0)  # Stronger opponent
        rewards = torch.tensor(
            [1.0, -1.0, 0.0, 0.5, -0.5]
        )  # win, loss, draw, partial win, partial loss
        current_elo = 1200.0

        new_elo = calc.update_elo_batch_vectorized(current_elo, opponent, rewards)

        # Should have some change based on the mixed results
        assert new_elo != current_elo

        # Note: Opponent stats are no longer updated in ELOCalculator
        # They are updated in the pool implementations instead
        assert opponent.games_played == 0  # Stats not updated in calculator
        assert opponent.wins == 0
        assert opponent.losses == 0
        assert opponent.draws == 0

    def test_vectorized_elo_update_empty_rewards(self):
        """Test vectorized ELO update with empty rewards."""
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1200.0)

        rewards = torch.tensor([])
        current_elo = 1200.0

        new_elo = calc.update_elo_batch_vectorized(current_elo, opponent, rewards)

        # Should return unchanged ELO
        assert new_elo == current_elo
        assert opponent.games_played == 0

    def test_vectorized_elo_update_none_opponent(self):
        """Test vectorized ELO update with None opponent."""
        calc = ELOCalculator(k_factor=32.0)

        rewards = torch.tensor([1.0, -1.0])
        current_elo = 1200.0

        new_elo = calc.update_elo_batch_vectorized(current_elo, None, rewards)

        # Should return unchanged ELO
        assert new_elo == current_elo

    def test_vectorized_elo_update_extreme_rewards(self):
        """Test vectorized ELO update with rewards outside [-1, 1] range."""
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1200.0)

        # Test with extreme rewards (should trigger warning)
        rewards = torch.tensor([2.0, -2.0, 1.0])  # Extreme values
        current_elo = 1200.0

        # This should not crash and should handle extreme values gracefully
        new_elo = calc.update_elo_batch_vectorized(current_elo, opponent, rewards)

        # Should still update ELO
        assert new_elo != current_elo
        # Note: Opponent stats are no longer updated in ELOCalculator
        assert opponent.games_played == 0

    def test_elo_convergence(self):
        """Test that ELO ratings converge over time."""
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1200.0)

        # Simulate many games with equal skill
        current_elo = 1200.0

        # Use fixed seed for deterministic results
        import random

        random.seed(42)

        for _ in range(100):
            # Randomly win/lose/draw
            result = random.choice(["win", "loss", "draw"])
            current_elo = calc.update_elo_after_game(current_elo, opponent, result)

        # ELO should stay close to starting value (1200) with equal skill
        # Increased tolerance to account for randomness in equal skill scenarios
        assert abs(current_elo - 1200.0) < 100.0  # Should be within 100 points

    def test_elo_magnitude_scoring(self):
        """Test that magnitude-based scoring works correctly."""
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )
        opponent = AgentSnapshot(model, step=100, elo=1200.0)

        # Test different reward magnitudes against stronger opponent
        opponent = AgentSnapshot(model, step=100, elo=1300.0)  # Stronger opponent

        # Test with all positive rewards (should gain more ELO)
        rewards_positive = torch.tensor(
            [0.5, 0.3, 0.1]
        )  # All wins, different magnitudes
        # Test with all negative rewards (should lose more ELO)
        rewards_negative = torch.tensor(
            [-0.5, -0.3, -0.1]
        )  # All losses, different magnitudes

        current_elo = 1200.0

        elo_positive = calc.update_elo_batch_vectorized(
            current_elo, opponent, rewards_positive
        )
        elo_negative = calc.update_elo_batch_vectorized(
            current_elo, opponent, rewards_negative
        )

        # Positive rewards should increase ELO, negative should decrease ELO
        assert elo_positive > current_elo
        assert elo_negative < current_elo

        # The magnitude should matter - larger rewards should have larger impact
        change_positive = elo_positive - current_elo
        change_negative = current_elo - elo_negative

        # Both should be positive changes (in opposite directions)
        assert change_positive > 0
        assert change_negative > 0

    def test_elo_conservation(self):
        """Test ELO conservation behavior.

        NOTE: The current ELO calculator does NOT conserve ELO because it only updates
        the current player's rating, not the opponent's. This is intentional for the
        poker training system where opponent snapshots are static historical versions.
        This test verifies this behavior and documents the design choice.
        """
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )

        # Test with different scenarios
        scenarios = [
            # (current_elo, opponent_elo, result)
            (1200.0, 1300.0, "win"),  # Upset win
            (1200.0, 1100.0, "loss"),  # Upset loss
            (1200.0, 1200.0, "draw"),  # Equal ELO draw
            (1500.0, 1000.0, "win"),  # Expected win
            (1000.0, 1500.0, "loss"),  # Expected loss
        ]

        for current_elo, opponent_elo, result in scenarios:
            opponent = AgentSnapshot(model, step=100, elo=opponent_elo)

            # Calculate initial total ELO
            initial_total = current_elo + opponent_elo

            # Update ELO after game
            new_elo = calc.update_elo_after_game(current_elo, opponent, result)

            # Calculate final total ELO
            final_total = new_elo + opponent_elo

            # ELO is NOT conserved in this system (opponent ELO doesn't change)
            # This is intentional - opponent snapshots are static historical versions
            # Exception: ELO is conserved when both players have equal ELO and result is draw
            if current_elo == opponent_elo and result == "draw":
                assert (
                    abs(final_total - initial_total) < 1e-10
                ), f"ELO should be conserved for equal ELO draw: initial={initial_total}, final={final_total}"
            else:
                assert final_total != initial_total, (
                    f"ELO unexpectedly conserved: initial={initial_total}, final={final_total}, "
                    f"current_elo={current_elo}->{new_elo}, opponent_elo={opponent_elo}, result={result}"
                )

            # Verify that only the current player's ELO changed
            assert opponent.elo == opponent_elo, "Opponent ELO should not change"

            # Verify that the change is in the expected direction
            if result == "win":
                assert new_elo > current_elo, "Should gain ELO on win"
            elif result == "loss":
                assert new_elo < current_elo, "Should lose ELO on loss"
            else:  # draw
                # For draws, change depends on expected score
                expected_score = 1.0 / (
                    1.0 + 10 ** ((opponent_elo - current_elo) / 400.0)
                )
                if expected_score < 0.5:  # Expected to lose
                    assert new_elo > current_elo, "Should gain ELO on unexpected draw"
                elif expected_score > 0.5:  # Expected to win
                    assert new_elo < current_elo, "Should lose ELO on unexpected draw"
                else:  # Equal expected
                    assert (
                        abs(new_elo - current_elo) < 1e-10
                    ), "Should have minimal change on equal draw"

    def test_elo_conservation_vectorized(self):
        """Test ELO conservation behavior with vectorized updates.

        NOTE: Like the single-game version, this does NOT conserve ELO because
        opponent snapshots are static historical versions.
        """
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )

        # Test with mixed rewards against different opponents
        test_cases = [
            (1200.0, 1300.0, torch.tensor([1.0, -1.0, 0.5, -0.5])),  # Mixed results
            (1500.0, 1000.0, torch.tensor([1.0, 1.0, 0.8])),  # Mostly wins
            (1000.0, 1500.0, torch.tensor([-1.0, -1.0, -0.8])),  # Mostly losses
        ]

        for current_elo, opponent_elo, rewards in test_cases:
            opponent = AgentSnapshot(model, step=100, elo=opponent_elo)

            # Calculate initial total ELO
            initial_total = current_elo + opponent_elo

            # Update ELO with vectorized method
            new_elo = calc.update_elo_batch_vectorized(current_elo, opponent, rewards)

            # Calculate final total ELO
            final_total = new_elo + opponent_elo

            # ELO is NOT conserved in this system (opponent ELO doesn't change)
            assert final_total != initial_total, (
                f"ELO unexpectedly conserved in vectorized update: initial={initial_total}, final={final_total}, "
                f"current_elo={current_elo}->{new_elo}, opponent_elo={opponent_elo}, rewards={rewards.tolist()}"
            )

            # Verify that only the current player's ELO changed
            assert opponent.elo == opponent_elo, "Opponent ELO should not change"

    def test_elo_conservation_multiple_games(self):
        """Test ELO conservation behavior over multiple games.

        NOTE: This does NOT conserve ELO because opponent snapshots are static.
        """
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )

        # Start with two players
        player1_elo = 1200.0
        player2_elo = 1300.0
        initial_total = player1_elo + player2_elo

        # Play multiple games
        results = ["win", "loss", "draw", "win", "loss", "draw", "win", "loss"]

        for result in results:
            opponent = AgentSnapshot(model, step=100, elo=player2_elo)
            player1_elo = calc.update_elo_after_game(player1_elo, opponent, result)

        # After all games, total ELO should NOT be conserved (opponent ELO doesn't change)
        final_total = player1_elo + player2_elo
        assert (
            abs(final_total - initial_total) > 1e-10
        ), f"ELO unexpectedly conserved over multiple games: initial={initial_total}, final={final_total}"

        # Verify that player2's ELO never changed
        assert player2_elo == 1300.0, "Player2 ELO should remain unchanged"

    def test_elo_conservation_realistic_training(self):
        """Test ELO conservation under realistic training conditions.

        This test mirrors the actual usage pattern in KBestOpponentPool and DREDPool
        where both players' ELOs are updated symmetrically to maintain conservation.
        """
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )

        # Simulate realistic training scenario
        current_player_elo = 1200.0
        opponent_elo = 1300.0
        initial_total = current_player_elo + opponent_elo

        # Create opponent snapshot
        opponent = AgentSnapshot(model, step=100, elo=opponent_elo)

        # Simulate multiple games with realistic rewards
        # These are the types of rewards that come from poker games
        rewards = torch.tensor(
            [
                0.8,  # Big win
                -0.3,  # Small loss
                0.1,  # Small win
                -0.9,  # Big loss
                0.0,  # Draw
                0.5,  # Medium win
                -0.2,  # Small loss
                0.7,  # Big win
            ]
        )

        # Update current player's ELO (as done in training)
        new_current_elo = calc.update_elo_batch_vectorized(
            current_player_elo, opponent, rewards
        )

        # Update opponent's ELO with opposite rewards (as done in training)
        # Use ORIGINAL current_elo for opponent calculation (not updated one)
        temp_snapshot = AgentSnapshot(model, step=200, elo=current_player_elo)
        new_opponent_elo = calc.update_elo_batch_vectorized(
            opponent_elo, temp_snapshot, -rewards
        )

        # Calculate final total ELO
        final_total = new_current_elo + new_opponent_elo

        # ELO should be conserved in realistic training conditions
        # Use a more realistic tolerance for floating-point arithmetic
        assert abs(final_total - initial_total) < 1e-5, (
            f"ELO not conserved in realistic training: initial={initial_total}, final={final_total}, "
            f"current_player: {current_player_elo}->{new_current_elo}, "
            f"opponent: {opponent_elo}->{new_opponent_elo}"
        )

        # Verify that both players' ELOs changed
        assert (
            abs(new_current_elo - current_player_elo) > 1e-10
        ), "Current player ELO should change"
        assert (
            abs(new_opponent_elo - opponent_elo) > 1e-10
        ), "Opponent ELO should change"

        # Verify that changes are opposite (one gains, other loses)
        current_change = new_current_elo - current_player_elo
        opponent_change = new_opponent_elo - opponent_elo
        assert (
            abs(current_change + opponent_change) < 1e-5
        ), f"ELO changes should be opposite: current_change={current_change}, opponent_change={opponent_change}"

    def test_elo_conservation_multiple_opponents(self):
        """Test ELO conservation when playing against multiple opponents."""
        calc = ELOCalculator(k_factor=32.0)

        model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=128,
            actions_hidden=128,
            fusion_hidden=[256, 256],
            num_actions=8,
        )

        # Start with current player and multiple opponents
        current_player_elo = 1200.0
        opponents = [
            AgentSnapshot(model, step=100, elo=1100.0),  # Weaker opponent
            AgentSnapshot(model, step=200, elo=1300.0),  # Stronger opponent
            AgentSnapshot(model, step=300, elo=1250.0),  # Similar opponent
        ]

        initial_total = current_player_elo + sum(opp.elo for opp in opponents)

        # Play games against each opponent
        opponent_rewards = [
            torch.tensor([0.5, 0.3, -0.2]),  # Mostly wins against weaker opponent
            torch.tensor([-0.8, -0.4, 0.1]),  # Mostly losses against stronger opponent
            torch.tensor([0.2, -0.1, 0.0]),  # Mixed results against similar opponent
        ]

        new_current_elo = current_player_elo

        for opponent, rewards in zip(opponents, opponent_rewards):
            # Store the current ELO before updating (needed for ELO conservation)
            current_elo_before_update = new_current_elo

            # Update current player's ELO
            new_current_elo = calc.update_elo_batch_vectorized(
                new_current_elo, opponent, rewards
            )

            # Update opponent's ELO with opposite rewards
            # Use the ELO from BEFORE the current player update to maintain conservation
            temp_snapshot = AgentSnapshot(
                model, step=400, elo=current_elo_before_update
            )
            opponent.elo = calc.update_elo_batch_vectorized(
                opponent.elo, temp_snapshot, -rewards
            )

        # Calculate final total ELO
        final_total = new_current_elo + sum(opp.elo for opp in opponents)

        # ELO should be conserved across all opponents
        # Use the same tolerance as single opponent test
        assert (
            abs(final_total - initial_total) < 1e-5
        ), f"ELO not conserved with multiple opponents: initial={initial_total}, final={final_total}"

        # Verify that all ELOs changed
        assert (
            abs(new_current_elo - current_player_elo) > 1e-10
        ), "Current player ELO should change"
        for i, opponent in enumerate(opponents):
            assert (
                abs(opponent.elo - [1100.0, 1300.0, 1250.0][i]) > 1e-10
            ), f"Opponent {i} ELO should change"


if __name__ == "__main__":
    # Run tests if executed directly
    import sys

    pytest.main([__file__] + sys.argv[1:])
