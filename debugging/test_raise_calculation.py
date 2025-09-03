#!/usr/bin/env python3
"""
Test Raise Calculation Script

Tests that raises are calculated correctly based on pot size.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaholdem.env.hunl_env import HUNLEnv


def test_raise_calculation():
    """Test that raises are calculated correctly."""
    print("=== Testing Raise Calculation ===")

    # Create environment
    env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
    state = env.reset()

    print(f"Initial state:")
    print(f"  Button: {state.button}")
    print(f"  To act: {state.to_act}")
    print(f"  Pot: {state.pot}")
    print(f"  Player 0 committed: {state.players[0].committed}")
    print(f"  Player 1 committed: {state.players[1].committed}")
    print(f"  Last aggressive amount: {state.last_aggressive_amount}")
    print()

    # Step 1: Player 0 raises
    print("--- Step 1: Player 0 raises ---")
    legal_actions = env.legal_actions()
    print(f"Legal actions: {legal_actions}")

    # Find raise action
    raise_action = None
    for action in legal_actions:
        if action.kind == "raise":
            raise_action = action
            break

    if raise_action:
        print(f"Taking raise action: {raise_action}")
        state, reward, done, _ = env.step(raise_action)
        print(f"After raise:")
        print(f"  Pot: {state.pot}")
        print(f"  Player 0 committed: {state.players[0].committed}")
        print(f"  Player 1 committed: {state.players[1].committed}")
        print(f"  Last aggressive amount: {state.last_aggressive_amount}")
        print()

        # Step 2: Player 1 raises
        print("--- Step 2: Player 1 raises ---")
        legal_actions = env.legal_actions()
        print(f"Legal actions: {legal_actions}")

        # Find raise action
        raise_action = None
        for action in legal_actions:
            if action.kind == "raise":
                raise_action = action
                break

        if raise_action:
            print(f"Taking raise action: {raise_action}")
            state, reward, done, _ = env.step(raise_action)
            print(f"After second raise:")
            print(f"  Pot: {state.pot}")
            print(f"  Player 0 committed: {state.players[0].committed}")
            print(f"  Player 1 committed: {state.players[1].committed}")
            print(f"  Last aggressive amount: {state.last_aggressive_amount}")

    print("=== Raise Calculation Test Complete ===")


if __name__ == "__main__":
    test_raise_calculation()
