#!/usr/bin/env python3
"""
Test Turn Order Script

Tests that players alternate turns correctly in the poker environment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphaholdem.env.hunl_env import HUNLEnv


def test_turn_order():
    """Test that players alternate turns correctly."""
    print("=== Testing Turn Order ===")
    
    # Create environment
    env = HUNLEnv(starting_stack=400, sb=1, bb=2, seed=456)  # Different seed
    state = env.reset()
    
    print(f"Initial state:")
    print(f"  Button: {state.button}")
    print(f"  To act: {state.to_act}")
    print(f"  Street: {state.street}")
    print(f"  Player 0 hole cards: {state.players[0].hole_cards}")
    print(f"  Player 1 hole cards: {state.players[1].hole_cards}")
    print()
    
    step_count = 0
    max_steps = 20
    
    while not state.terminal and step_count < max_steps:
        step_count += 1
        
        print(f"--- Step {step_count} ---")
        print(f"  Street: {state.street}")
        print(f"  To act: {state.to_act}")
        print(f"  Pot: {state.pot}")
        print(f"  Player 0 committed: {state.players[0].committed}")
        print(f"  Player 1 committed: {state.players[1].committed}")
        print(f"  Actions this round: {env.actions_this_round}")
        
        if state.board:
            print(f"  Board: {state.board}")
        
        # Get legal actions
        legal_actions = env.legal_actions()
        print(f"  Legal actions: {legal_actions}")
        
        # Take first legal action (usually call/check)
        action = legal_actions[0]
        print(f"  Taking action: {action}")
        
        # Prefer call/check over fold
        for action in legal_actions:
            if action.kind in ['call', 'check']:
                break
        print(f"  Taking action: {action}")
        
        # Step
        state, reward, done, _ = env.step(action)
        
        if done:
            print(f"  Hand finished! Winner: {state.winner}, Reward: {reward}")
            break
        
        print(f"  Next to act: {state.to_act}")
        print()
    
    print("=== Turn Order Test Complete ===")


if __name__ == "__main__":
    test_turn_order()
