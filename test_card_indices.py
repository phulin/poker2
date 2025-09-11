#!/usr/bin/env python3
"""Test script to verify card index tracking in HUNLTensorEnv."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv


def test_card_indices():
    """Test that card indices are properly tracked."""
    print("Testing card index tracking in HUNLTensorEnv...")

    # Create environment
    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=2,
        starting_stack=1000,
        sb=5,
        bb=10,
        bet_bins=[0.5, 1.0, 2.0],
        device=device,
    )

    # Reset environments
    env.reset()

    print(f"Environment 0:")
    print(f"  Hole indices: {env.get_hole_card_indices(0)}")
    print(f"  Board indices: {env.get_board_card_indices(0)}")
    print(f"  Player 0 visible: {env.get_visible_card_indices(0, 0)}")
    print(f"  Player 1 visible: {env.get_visible_card_indices(0, 1)}")
    print(f"  All cards: {env.get_all_card_indices(0)}")

    print(f"\nEnvironment 1:")
    print(f"  Hole indices: {env.get_hole_card_indices(1)}")
    print(f"  Board indices: {env.get_board_card_indices(1)}")
    print(f"  Player 0 visible: {env.get_visible_card_indices(1, 0)}")
    print(f"  Player 1 visible: {env.get_visible_card_indices(1, 1)}")
    print(f"  All cards: {env.get_all_card_indices(1)}")

    # Test that indices match onehot representation
    print(f"\nVerifying consistency:")
    env_idx = 0

    # Check hole cards
    p0_cards, p1_cards = env.get_hole_card_indices(env_idx)
    print(f"  Player 0 hole cards: {p0_cards}")
    print(f"  Player 1 hole cards: {p1_cards}")

    # Check that onehot matches indices
    for player in [0, 1]:
        for card_idx in [0, 1]:
            card_index = env.hole_indices[env_idx, player, card_idx]
            if card_index >= 0:
                onehot_card = env.hole_onehot[env_idx, player, card_idx]
                expected_onehot = env.card_onehot_cache[card_index]
                assert torch.equal(
                    onehot_card, expected_onehot
                ), f"Mismatch for player {player}, card {card_idx}"

    print("✅ Card index tracking test passed!")


if __name__ == "__main__":
    test_card_indices()
