#!/usr/bin/env python3
"""
Unit tests for legal action generation in HUNLEnv.

Tests various poker situations to ensure legal actions are correctly generated.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.types import Action


class TestLegalActions:
    """Test legal action generation for different poker situations."""

    def test_preflop_small_blind_actions(self):
        """Test preflop actions when small blind acts first."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Small blind acts first preflop
        assert state.to_act == 0  # Small blind
        assert state.players[0].committed == 10  # Small blind posted
        assert state.players[1].committed == 20  # Big blind posted
        assert state.pot == 30

        legal_actions = env.legal_actions()
        action_kinds = [action.kind for action in legal_actions]
        action_amounts = [action.amount for action in legal_actions]

        # Should be able to fold, call, raise, or all-in
        assert "fold" in action_kinds
        assert "call" in action_kinds
        assert "raise" in action_kinds
        assert "allin" in action_kinds

        # Call amount should be 10 (to match big blind)
        call_action = next(a for a in legal_actions if a.kind == "call")
        assert call_action.amount == 10

        # Raise amounts should be call + pot fractions
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        assert len(raise_actions) >= 3  # Should have multiple raise sizes

        # All-in should be remaining stack
        allin_action = next(a for a in legal_actions if a.kind == "allin")
        assert allin_action.amount == 990  # 1000 - 10 (already committed)

    def test_preflop_big_blind_actions(self):
        """Test preflop actions when big blind acts."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Make small blind call
        call_action = Action("call", amount=10)
        state, _, _, _ = env.step(call_action)

        # Now big blind acts
        assert state.to_act == 1  # Big blind
        assert state.players[0].committed == 20  # Both committed 20
        assert state.players[1].committed == 20
        assert state.pot == 40

        legal_actions = env.legal_actions()
        action_kinds = [action.kind for action in legal_actions]

        # Should be able to check or bet (no one has bet yet)
        assert "check" in action_kinds
        assert "bet" in action_kinds
        assert "allin" in action_kinds
        assert "fold" not in action_kinds  # Can't fold when no bet to call

        # Should have multiple bet sizes
        bet_actions = [a for a in legal_actions if a.kind == "bet"]
        assert len(bet_actions) >= 3  # 1/2, 3/4, pot, 1.5x, 2x pot

    def test_postflop_betting_actions(self):
        """Test postflop actions when no one has bet."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))  # Small blind calls
        env.step(Action("check", amount=0))  # Big blind checks

        # Now on flop
        state = env.state
        assert state.street == "flop"
        assert state.pot == 40
        assert state.players[0].committed == 0
        assert state.players[1].committed == 0

        legal_actions = env.legal_actions()
        action_kinds = [action.kind for action in legal_actions]

        # Should be able to check or bet
        assert "check" in action_kinds
        assert "bet" in action_kinds
        assert "allin" in action_kinds
        assert "fold" not in action_kinds

        # Should have multiple bet sizes based on pot
        bet_actions = [a for a in legal_actions if a.kind == "bet"]
        expected_sizes = [20, 30, 40, 60, 80]  # 1/2, 3/4, pot, 1.5x, 2x pot
        actual_sizes = [a.amount for a in bet_actions]

        for expected in expected_sizes:
            assert (
                expected in actual_sizes
            ), f"Expected bet size {expected} not found in {actual_sizes}"

    def test_facing_bet_actions(self):
        """Test actions when facing a bet."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))  # Small blind calls
        env.step(Action("check", amount=0))  # Big blind checks

        # Small blind bets on flop
        env.step(Action("bet", amount=40))  # Pot-sized bet

        # Now big blind faces a bet
        state = env.state
        assert (
            state.players[1].committed == 40
        )  # Player 1 (big blind) committed the bet
        assert (
            state.players[0].committed == 0
        )  # Player 0 (small blind) hasn't acted yet
        assert state.pot == 80

        legal_actions = env.legal_actions()
        action_kinds = [action.kind for action in legal_actions]

        # Should be able to fold, call, or raise
        assert "fold" in action_kinds
        assert "call" in action_kinds
        assert "raise" in action_kinds
        assert "allin" in action_kinds
        assert "check" not in action_kinds  # Can't check when facing bet

        # Call amount should be 40
        call_action = next(a for a in legal_actions if a.kind == "call")
        assert call_action.amount == 40

        # Raise amounts should be call + pot fractions
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        assert len(raise_actions) >= 3

    def test_all_in_situation(self):
        """Test actions when player is all-in."""
        env = HUNLEnv(starting_stack=50, sb=10, bb=20, seed=123)
        state = env.reset()

        # Small blind has only 50 chips, already committed 10
        assert state.players[0].stack == 40  # 50 - 10
        assert state.players[1].stack == 30  # 50 - 20

        legal_actions = env.legal_actions()
        action_kinds = [action.kind for action in legal_actions]

        # Should be able to fold, call, raise, or all-in
        assert "fold" in action_kinds
        assert "call" in action_kinds
        assert "raise" in action_kinds
        assert "allin" in action_kinds

        # All-in should be remaining stack
        allin_action = next(a for a in legal_actions if a.kind == "allin")
        assert allin_action.amount == 40

    def test_zero_pot_betting(self):
        """Test betting when pot is zero (edge case)."""
        env = HUNLEnv(starting_stack=1000, sb=0, bb=0, seed=123)
        state = env.reset()

        # No blinds posted
        assert state.pot == 0
        assert state.players[0].committed == 0
        assert state.players[1].committed == 0

        legal_actions = env.legal_actions()
        action_kinds = [action.kind for action in legal_actions]

        # Should be able to check or all-in (no pot to bet)
        assert "check" in action_kinds
        assert "allin" in action_kinds
        assert "bet" not in action_kinds  # No pot to bet against
        assert "fold" not in action_kinds  # No bet to fold to

    def test_minimum_bet_sizes(self):
        """Test that bet sizes respect minimum bet rules."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))
        env.step(Action("check", amount=0))

        # On flop with pot = 40
        state = env.state
        assert state.pot == 40

        legal_actions = env.legal_actions()
        bet_actions = [a for a in legal_actions if a.kind == "bet"]

        # All bet sizes should be positive and within stack
        for action in bet_actions:
            assert action.amount > 0, f"Bet amount {action.amount} should be positive"
            assert action.amount <= 1000, f"Bet amount {action.amount} exceeds stack"

    def test_raise_sizes_after_bet(self):
        """Test that raise sizes are correct after a bet."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))
        env.step(Action("check", amount=0))

        # Small blind bets 40 (pot-sized)
        env.step(Action("bet", amount=40))

        # Big blind should be able to raise by pot fractions
        state = env.state
        legal_actions = env.legal_actions()
        raise_actions = [a for a in legal_actions if a.kind == "raise"]

        # Should have multiple raise sizes
        assert len(raise_actions) >= 3

        # All raise amounts should be > call amount (40)
        for action in raise_actions:
            assert (
                action.amount > 40
            ), f"Raise amount {action.amount} should be > call amount 40"


class TestQuantizedRaiseSizes:
    """Test quantized legal raise sizes across different scenarios."""

    def test_preflop_raise_sizes(self):
        """Test preflop raise sizes with standard blinds."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Preflop: pot=30, SB committed=10, BB committed=20
        # Total committed = 30 + 10 + 20 = 60
        # Call amount = 10
        # Expected raise sizes: 10 + [30, 45, 60, 90, 120] = [40, 55, 70, 100, 130]

        legal_actions = env.legal_actions()
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        raise_amounts = [a.amount for a in raise_actions]

        expected_raises = [40, 55, 70, 100, 130]
        assert (
            raise_amounts == expected_raises
        ), f"Expected {expected_raises}, got {raise_amounts}"

    def test_postflop_bet_sizes(self):
        """Test postflop bet sizes with no previous action."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))  # SB calls
        env.step(Action("check", amount=0))  # BB checks

        # Postflop: pot=40, both committed=0
        # Total committed = 40 + 0 + 0 = 40
        # Expected bet sizes: [20, 30, 40, 60, 80]

        legal_actions = env.legal_actions()
        bet_actions = [a for a in legal_actions if a.kind == "bet"]
        bet_amounts = [a.amount for a in bet_actions]

        expected_bets = [20, 30, 40, 60, 80]
        assert (
            bet_amounts == expected_bets
        ), f"Expected {expected_bets}, got {bet_amounts}"

    def test_facing_pot_sized_bet(self):
        """Test raise sizes when facing a pot-sized bet."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))
        env.step(Action("check", amount=0))

        # SB bets 40 (pot-sized)
        env.step(Action("bet", amount=40))

        # BB facing bet: pot=80, SB committed=0, BB committed=40
        # Total committed = 80 + 0 + 40 = 120
        # Call amount = 40
        # Expected raise sizes: 40 + [60, 90, 120, 180, 240] = [100, 130, 160, 220, 280]

        legal_actions = env.legal_actions()
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        raise_amounts = [a.amount for a in raise_actions]

        expected_raises = [100, 130, 160, 220, 280]
        assert (
            raise_amounts == expected_raises
        ), f"Expected {expected_raises}, got {raise_amounts}"

    def test_facing_half_pot_bet(self):
        """Test raise sizes when facing a half-pot bet."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))
        env.step(Action("check", amount=0))

        # SB bets 20 (half pot)
        env.step(Action("bet", amount=20))

        # BB facing bet: pot=60, SB committed=0, BB committed=20
        # Total committed = 60 + 0 + 20 = 80
        # Call amount = 20
        # Expected raise sizes: 20 + [40, 60, 80, 120, 160] = [60, 80, 100, 140, 180]

        legal_actions = env.legal_actions()
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        raise_amounts = [a.amount for a in raise_actions]

        expected_raises = [60, 80, 100, 140, 180]
        assert (
            raise_amounts == expected_raises
        ), f"Expected {expected_raises}, got {raise_amounts}"

    def test_facing_overbet(self):
        """Test raise sizes when facing an overbet (1.5x pot)."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))
        env.step(Action("check", amount=0))

        # SB bets 60 (1.5x pot)
        env.step(Action("bet", amount=60))

        # BB facing bet: pot=100, SB committed=0, BB committed=60
        # Total committed = 100 + 0 + 60 = 160
        # Call amount = 60
        # Expected raise sizes: 60 + [80, 120, 160, 240, 320] = [140, 180, 220, 300, 380]

        legal_actions = env.legal_actions()
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        raise_amounts = [a.amount for a in raise_actions]

        expected_raises = [140, 180, 220, 300, 380]
        assert (
            raise_amounts == expected_raises
        ), f"Expected {expected_raises}, got {raise_amounts}"

    def test_multiple_rounds_of_betting(self):
        """Test raise sizes after multiple rounds of betting."""
        env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
        state = env.reset()

        # Complete preflop
        env.step(Action("call", amount=10))
        env.step(Action("check", amount=0))

        # SB bets 40 on flop
        env.step(Action("bet", amount=40))

        # BB calls
        env.step(Action("call", amount=40))

        # SB bets 120 on turn
        env.step(Action("bet", amount=120))

        # BB facing bet: pot=240, SB committed=0, BB committed=120
        # Total committed = 240 + 0 + 120 = 360
        # Call amount = 120
        # Expected raise sizes: 120 + [180, 270, 360, 540, 720] = [300, 390, 480, 660, 840]

        legal_actions = env.legal_actions()
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        raise_amounts = [a.amount for a in raise_actions]

        expected_raises = [300, 390, 480, 660, 840]
        assert (
            raise_amounts == expected_raises
        ), f"Expected {expected_raises}, got {raise_amounts}"

    def test_small_stack_scenario(self):
        """Test raise sizes with small stacks."""
        env = HUNLEnv(starting_stack=100, sb=10, bb=20, seed=123)
        state = env.reset()

        # Small stacks: SB has 90 remaining, BB has 80 remaining
        # Preflop: pot=30, SB committed=10, BB committed=20
        # Total committed = 30 + 10 + 20 = 60
        # Call amount = 10
        # Expected raise sizes: 10 + [30, 45, 60, 90, 120] = [40, 55, 70, 100, 130]
        # But some may be limited by stack size

        legal_actions = env.legal_actions()
        raise_actions = [a for a in legal_actions if a.kind == "raise"]
        raise_amounts = [a.amount for a in raise_actions]

        # All raise amounts should be <= 90 (SB's remaining stack)
        for amount in raise_amounts:
            assert amount <= 90, f"Raise amount {amount} exceeds stack size 90"

        # Should still have multiple raise options
        assert len(raise_amounts) >= 3

    def test_zero_blinds_scenario(self):
        """Test bet sizes with zero blinds."""
        env = HUNLEnv(starting_stack=1000, sb=0, bb=0, seed=123)
        state = env.reset()

        # Zero blinds: pot=0, both committed=0
        # Total committed = 0 + 0 + 0 = 0
        # Should not offer any bet sizes when total_committed is 0

        legal_actions = env.legal_actions()
        bet_actions = [a for a in legal_actions if a.kind == "bet"]

        assert (
            len(bet_actions) == 0
        ), f"Should not offer bet actions when total_committed is 0, got {bet_actions}"

    def test_raise_sizes_consistency(self):
        """Test that raise sizes are consistent across different scenarios."""
        scenarios = [
            # (sb, bb, expected_total_committed, expected_call_amount, expected_raises)
            (10, 20, 60, 10, [40, 55, 70, 100, 130]),
            (5, 10, 30, 5, [20, 27, 35, 50, 65]),
            (25, 50, 150, 25, [100, 137, 175, 250, 325]),
        ]

        for sb, bb, expected_total, expected_call, expected_raises in scenarios:
            env = HUNLEnv(starting_stack=1000, sb=sb, bb=bb, seed=123)
            state = env.reset()

            # Verify total committed calculation
            total_committed = (
                state.pot + state.players[0].committed + state.players[1].committed
            )
            assert (
                total_committed == expected_total
            ), f"Expected total_committed {expected_total}, got {total_committed}"

            # Verify call amount
            to_call = state.players[1].committed - state.players[0].committed
            assert (
                to_call == expected_call
            ), f"Expected call amount {expected_call}, got {to_call}"

            # Verify raise sizes
            legal_actions = env.legal_actions()
            raise_actions = [a for a in legal_actions if a.kind == "raise"]
            raise_amounts = [a.amount for a in raise_actions]

            assert (
                raise_amounts == expected_raises
            ), f"Expected raises {expected_raises}, got {raise_amounts}"


if __name__ == "__main__":
    # Run tests
    test_suite = TestLegalActions()

    print("Running legal action tests...")

    # Run each test
    test_methods = [method for method in dir(test_suite) if method.startswith("test_")]

    for method_name in test_methods:
        print(f"\n--- Running {method_name} ---")
        try:
            method = getattr(test_suite, method_name)
            method()
            print(f"✅ {method_name} PASSED")
        except Exception as e:
            print(f"❌ {method_name} FAILED: {e}")

    print("\n=== Test Summary ===")
    print("All legal action tests completed!")
