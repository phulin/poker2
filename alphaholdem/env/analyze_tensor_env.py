#!/usr/bin/env python3
"""
Debug utilities for HUNLTensorEnv.
Contains functions for creating test environments and analyzing specific scenarios.
"""

import torch
from typing import List, Tuple, Dict, Any
from .hunl_tensor_env import HUNLTensorEnv


def create_169_hand_combinations() -> List[Tuple[str, str]]:
    """Create all 169 possible preflop hand combinations.

    Returns:
        List of tuples containing (card1, card2) strings like ("As", "Kh")
        - 13 pocket pairs: ("As", "Ah"), ("Ks", "Kh"), etc.
        - 78 suited hands: ("As", "Ks"), ("As", "Qs"), etc.
        - 78 off-suit hands: ("Ah", "Kd"), ("Ah", "Qd"), etc.
    """
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    hands = []

    for rank1 in ranks:
        for rank2 in ranks:
            if rank1 < rank2:
                hands.append((f"{rank1}s", f"{rank2}s"))
            elif rank1 >= rank2:
                hands.append((f"{rank1}s", f"{rank2}h"))

    return hands


def create_169_hand_env(
    starting_stack: int = 20000,
    sb: int = 50,
    bb: int = 100,
    bet_bins: List[int] = None,
    device: torch.device = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> HUNLTensorEnv:
    """Create a tensor environment with all 169 preflop hands set up.

    Args:
        starting_stack: Starting stack size
        sb: Small blind amount
        bb: Big blind amount
        bet_bins: List of bet bin values
        device: Device to use
        rng: Random number generator
        flop_showdown: Whether to showdown after flop

    Returns:
        HUNLTensorEnv with 169 environments, each with a different preflop hand
    """
    if bet_bins is None:
        bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]

    if device is None:
        device = torch.device("cpu")

    if rng is None:
        rng = torch.Generator(device=device)

    # Create environment with 169 states
    temp_env = HUNLTensorEnv(
        num_envs=169,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
        flop_showdown=flop_showdown,
    )

    # Reset the environment
    temp_env.reset()

    # Set all environments to BB to act (seat 1)
    temp_env.button[:] = 1  # BB is button
    temp_env.to_act[:] = 1  # BB to act

    # Get all 169 hand combinations
    hands = create_169_hand_combinations()

    # Set hole cards for each environment
    for i, (card1, card2) in enumerate(hands):
        # Convert card strings to integers (assuming standard mapping)
        card1_int = _card_str_to_int(card1)
        card2_int = _card_str_to_int(card2)
        opp_card1_int = temp_env.hole_indices[i, 1, 0]
        opp_card2_int = temp_env.hole_indices[i, 1, 1]

        # Convert cards to one-hot representation
        suit1, rank1 = card1_int // 13, card1_int % 13
        suit2, rank2 = card2_int // 13, card2_int % 13

        # Set hole cards in the tensor environment
        temp_env.hole_onehot[i, 0, :, :, :] = False
        temp_env.hole_onehot[i, 0, 0, suit1, rank1] = True
        temp_env.hole_onehot[i, 0, 1, suit2, rank2] = True

        # Set hole card indices
        temp_env.hole_indices[i, 0, 0] = card1_int
        temp_env.hole_indices[i, 0, 1] = card2_int

        # Reshuffle deck without these cards
        # Remove card1_int and card2_int from the deck and reshuffle
        # Get the full deck (0..51)
        full_deck = list(range(52))
        # Remove the two hole cards
        deck_minus_hole = [
            c
            for c in full_deck
            if c not in (card1_int, card2_int, opp_card1_int, opp_card2_int)
        ]
        # Shuffle the deck
        # Place the shuffled deck into temp_env.deck for this env
        temp_env.deck[i, 0] = card1_int
        temp_env.deck[i, 1] = card2_int
        temp_env.deck[i, 2] = opp_card1_int
        temp_env.deck[i, 3] = opp_card2_int
        temp_env.deck[i, 4:9] = torch.tensor(deck_minus_hole, device=temp_env.device)[
            torch.randperm(48)[:5]
        ]

        # Set deck_pos to 4 (after hole cards)
        temp_env.deck_pos[i] = 4

    return temp_env


def _card_str_to_int(card_str: str) -> int:
    """Convert card string like 'As' to integer index.

    Args:
        card_str: Card string like 'As', 'Kh', 'Qd', 'Jc'

    Returns:
        Integer index (0-51) representing the card
    """
    rank_map = {
        "A": 12,
        "K": 11,
        "Q": 10,
        "J": 9,
        "T": 8,
        "9": 7,
        "8": 6,
        "7": 5,
        "6": 4,
        "5": 3,
        "4": 2,
        "3": 1,
        "2": 0,
    }
    suit_map = {"s": 0, "h": 1, "d": 2, "c": 3}

    rank = card_str[:-1]
    suit = card_str[-1]

    return rank_map[rank] * 4 + suit_map[suit]


def simulate_sb_action(
    env: HUNLTensorEnv, sb_action: str = "allin", device: torch.device = None
) -> None:
    """Simulate a specific SB action across all environments.

    Args:
        env: HUNLTensorEnv with 169 environments
        sb_action: SB action to simulate ("allin", "call", "fold", "bet")
        device: Device for tensor operations
    """
    if device is None:
        device = env.device

    if sb_action == "allin":
        # Set SB (seat 0) to all-in for all environments
        env.step_bins(torch.full((169,), 7, device=device))
    elif sb_action == "call":
        # SB calls (checks) for all environments
        env.step_bins(torch.full((169,), 1, device=device))
    elif sb_action == "fold":
        # SB folds for all environments
        env.step_bins(torch.full((169,), 0, device=device))
    elif sb_action == "bet":
        # SB bets (half pot) for all environments
        env.step_bins(torch.full((169,), 2, device=device))
