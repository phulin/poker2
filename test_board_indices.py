#!/usr/bin/env python3
"""Test script to verify board card index tracking in HUNLTensorEnv."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv


def test_board_card_indices():
    """Test that board card indices are properly tracked when dealt."""
    print("Testing board card index tracking...")

    # Create environment
    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        bet_bins=[0.5, 1.0, 2.0],
        device=device,
    )

    # Reset environment
    env.reset()

    print(f"Initial state:")
    print(f"  Board indices: {env.get_board_card_indices(0)}")
    print(f"  Street: {env.street[0].item()}")

    # Force some actions to trigger board dealing
    # First, make both players call to get to flop
    env.step_bins(torch.tensor([1], device=device))  # Player 0 calls
    env.step_bins(torch.tensor([1], device=device))  # Player 1 calls

    print(f"\nAfter preflop actions:")
    print(f"  Board indices: {env.get_board_card_indices(0)}")
    print(f"  Street: {env.street[0].item()}")

    # Continue to get to turn
    env.step_bins(torch.tensor([1], device=device))  # Player 0 checks
    env.step_bins(torch.tensor([1], device=device))  # Player 1 checks

    print(f"\nAfter flop actions:")
    print(f"  Board indices: {env.get_board_card_indices(0)}")
    print(f"  Street: {env.street[0].item()}")

    # Continue to get to river
    env.step_bins(torch.tensor([1], device=device))  # Player 0 checks
    env.step_bins(torch.tensor([1], device=device))  # Player 1 checks

    print(f"\nAfter turn actions:")
    print(f"  Board indices: {env.get_board_card_indices(0)}")
    print(f"  Street: {env.street[0].item()}")

    # Verify that board indices are properly set
    board_indices = env.get_board_card_indices(0)
    print(f"\nFinal board indices: {board_indices}")

    # Check that indices match onehot representation
    for i, card_index in enumerate(board_indices):
        if card_index >= 0:
            onehot_card = env.board_onehot[0, i]
            expected_onehot = env.card_onehot_cache[card_index]
            assert torch.equal(
                onehot_card, expected_onehot
            ), f"Mismatch for board card {i}"
            print(f"  Board card {i}: index {card_index.item()} ✅")

    print("✅ Board card index tracking test passed!")


if __name__ == "__main__":
    test_board_card_indices()
