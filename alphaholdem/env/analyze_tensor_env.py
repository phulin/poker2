#!/usr/bin/env python3
"""
Debug utilities for HUNLTensorEnv.
Contains functions for creating test environments and analyzing specific scenarios.
"""

from typing import Any, List, Tuple

import torch

from ..models.state_encoder import CNNStateEncoder
from ..models.transformer.state_encoder import TransformerStateEncoder
from .hunl_tensor_env import HUNLTensorEnv


def create_state_encoder_for_model(model, env: HUNLTensorEnv, device: torch.device):
    """Create the appropriate state encoder based on the model type.

    Args:
        model: The trained model instance
        env: The 169-hand tensor environment
        device: Device for tensor operations

    Returns:
        Appropriate state encoder for the model
    """
    # Detect model type from the model's class name
    model_class_name = model.__class__.__name__.lower()

    if "transformer" in model_class_name:
        # For transformer models, create TransformerStateEncoder
        return TransformerStateEncoder(env, device)
    elif "siamese" in model_class_name or "cnn" in model_class_name:
        # For CNN models, create CNNStateEncoder with tensor_env and device
        return CNNStateEncoder(env, device)
    else:
        raise ValueError(f"Unknown model type: {model_class_name}")


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

    for i, rank1 in enumerate(ranks):
        for j, rank2 in enumerate(ranks):
            if i == j:
                # Same rank (pairs) - always suited
                hands.append((f"{rank1}s", f"{rank1}h"))
            elif i < j:
                # Top-right triangle: suited hands (e.g., AKs, AQs)
                hands.append((f"{rank1}s", f"{rank2}s"))
            else:
                # Bottom-left triangle: off-suit hands (e.g., KAs, QAs)
                hands.append((f"{rank1}s", f"{rank2}h"))

    return hands


def get_probabilities(
    model,
    state_encoder,
    env: HUNLTensorEnv,
    seat: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get model probabilities and values for all 169 preflop hands.

    Args:
        model: The trained model instance
        state_encoder: The state encoder for the model
        env: The 169-hand tensor environment
        seat: Seat position (0 for SB, 1 for BB)
        device: Device for tensor operations

    Returns:
        Tuple of (probabilities [169, num_bet_bins], values [169], legal_masks [169, num_bet_bins])
    """
    # Get model prediction using tensor environment
    with torch.no_grad():
        embedding_data = state_encoder.encode_tensor_states(
            seat, torch.arange(169, device=device)  # All environments
        )
        outputs = model(embedding_data)

        logits = outputs["policy_logits"]  # [169, num_bet_bins]
        values = outputs["value"]  # [169]

        # Get legal actions from tensor environment
        legal_masks = env.legal_bins_mask()  # [169, num_bet_bins]

        # Apply legal mask
        masked_logits = torch.where(legal_masks == 0, -1e9, logits)

        # Get probabilities
        probs = torch.softmax(masked_logits, dim=-1)  # [169, num_bet_bins]

    return probs, values, legal_masks


def create_169_hand_analysis_setup(
    model,
    button: int,
    starting_stack: int = 20000,
    sb: int = 50,
    bb: int = 100,
    bet_bins: List[int] = None,
    device: torch.device = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> Tuple[HUNLTensorEnv, Any]:
    """Create a tensor environment with all 169 preflop hands set up for player 0 and appropriate state encoder.

    Args:
        model: The trained model instance
        button: Which player is the button
        starting_stack: Starting stack size
        sb: Small blind amount
        bb: Big blind amount
        bet_bins: List of bet bin values
        device: Device to use
        rng: Random number generator
        flop_showdown: Whether to showdown after flop

    Returns:
        Tuple of (HUNLTensorEnv with 169 environments, state encoder for the model)
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
    temp_env.reset(
        force_button=torch.full((169,), button, dtype=torch.long, device=device)
    )

    # Get all 169 hand combinations
    hands = create_169_hand_combinations()

    # Set hole cards for each environment
    for i, (card1, card2) in enumerate(hands):
        # Convert card strings to integers (assuming standard mapping)
        card1_int = _card_str_to_int(card1)
        card2_int = _card_str_to_int(card2)

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
        deck_minus_hole = [c for c in full_deck if c not in (card1_int, card2_int)]
        # Shuffle the deck
        # Place the shuffled deck into temp_env.deck for this env
        temp_env.deck[i, 0] = card1_int
        temp_env.deck[i, 1] = card2_int
        temp_env.deck[i, 2:9] = torch.tensor(deck_minus_hole, device=temp_env.device)[
            torch.randperm(50)[:7]
        ]

        # Deal opponent hole cards
        opp_card1_int = temp_env.deck[i, 2]
        opp_card2_int = temp_env.deck[i, 3]

        opp_suit1, opp_rank1 = opp_card1_int // 13, opp_card1_int % 13
        opp_suit2, opp_rank2 = opp_card2_int // 13, opp_card2_int % 13

        # Set hole cards in the tensor environment
        temp_env.hole_onehot[i, 1, :, :, :] = False
        temp_env.hole_onehot[i, 1, 0, opp_suit1, opp_rank1] = True
        temp_env.hole_onehot[i, 1, 1, opp_suit2, opp_rank2] = True

        # Set hole card indices
        temp_env.hole_indices[i, 1, 0] = opp_card1_int
        temp_env.hole_indices[i, 1, 1] = opp_card2_int

        # Set deck_pos to 4 (after hole cards)
        temp_env.deck_pos[i] = 4

    # Create appropriate state encoder for the model
    state_encoder = create_state_encoder_for_model(model, temp_env, device)

    return temp_env, state_encoder


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

    # Use suit-major indexing consistent with HUNLTensorEnv (card // 13 = suit, card % 13 = rank)
    return suit_map[suit] * 13 + rank_map[rank]


def _create_169_grid(
    values: torch.Tensor, value_type: str = "probability", default_value: str = " 0.0"
) -> str:
    """Create a standardized 13x13 grid for 169 preflop hands.

    Args:
        values: Tensor of 169 values for each hand
        value_type: Type of values - "probability", "value", or "raw"
        default_value: Default value to use if hand not found

    Returns:
        String representation of the 13x13 grid
    """
    # Create a 13x13 grid representing all possible hole card combinations
    # Rows/cols: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

    # Format values based on type
    if value_type == "probability":
        # Convert to percentages and format
        percentages = (values * 100).round().int()
        formatted_values = []
        for pct in percentages:
            if pct >= 100:
                formatted_values.append("99")  # Cap at 99%
            else:
                formatted_values.append(f"{pct.item():2d}")
    elif value_type == "value":
        # Format value estimates (multiply by 1000 for readability, show 3 sig figs, max 4 chars)
        formatted_values = []
        for value in values:
            val_scaled = value.item() * 1000
            formatted_values.append(f"{val_scaled:4.0f}")
    else:  # raw
        # Use values as-is (assumes they're already formatted strings)
        formatted_values = values

    # Initialize grid
    grid = []
    header = "    " + " ".join(f"{rank:>2}" for rank in ranks)
    grid.append(header)
    grid.append("   " + "-" * 39)  # Separator line

    # Create mapping from hand index to grid position
    hand_to_index = {}
    hands = create_169_hand_combinations()
    for i, (card1, card2) in enumerate(hands):
        # Convert to grid coordinates
        rank1 = card1[:-1]
        rank2 = card2[:-1]
        grid_row = ranks.index(rank1)
        grid_col = ranks.index(rank2)
        hand_to_index[(grid_row, grid_col)] = i

    for i, rank1 in enumerate(ranks):
        row = [f"{rank1:>2} |"]
        for j, rank2 in enumerate(ranks):
            if (i, j) in hand_to_index:
                hand_idx = hand_to_index[(i, j)]
                value_str = formatted_values[hand_idx]
            else:
                value_str = default_value
            row.append(value_str)

        grid.append(" ".join(row))

    return "\n".join(grid)


def get_preflop_betting_grid(
    model,
    device: torch.device = None,
    starting_stack: int = 20000,
    sb: int = 50,
    bb: int = 100,
    bet_bins: List[int] = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> str:
    """Get preflop betting probabilities as a grid showing all bet options combined.

    Args:
        model: Trained model for prediction
        seat: Seat position (0 for SB, 1 for BB)
        device: Device for tensor operations
        starting_stack: Starting stack size
        sb: Small blind amount
        bb: Big blind amount
        bet_bins: List of bet bin values
        rng: Random number generator
        flop_showdown: Whether to showdown after flop

    Returns:
        String representation of the preflop betting grid (all bet options combined)
    """
    if device is None:
        device = torch.device("cpu")

    # Create 169-hand environment and state encoder
    env, state_encoder = create_169_hand_analysis_setup(
        model,
        button=0,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
        flop_showdown=flop_showdown,
    )

    # Get probabilities and values
    probs, _, _ = get_probabilities(model, state_encoder, env, 0, device)

    # Sum all betting probabilities (exclude fold=0, call=1, and all-in=num_bet_bins-1)
    # Betting actions are typically indices 2 through num_bet_bins-2 (excluding all-in)
    num_bet_bins = probs.shape[1]
    betting_probs = probs[:, 2 : num_bet_bins - 1].sum(
        dim=1
    )  # Sum bet actions excluding all-in [169]

    return _create_169_grid(betting_probs, "probability", " 0")


def step_sb_action(
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


def get_preflop_range_grid(
    model,
    bin_index: int,
    device: torch.device = None,
    starting_stack: int = 20000,
    sb: int = 50,
    bb: int = 100,
    bet_bins: List[int] = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> str:
    """Get preflop range as a grid showing selected action probabilities.

    Args:
        model: Trained model for prediction
        bin_index: Index of the betting bin to check
        device: Device for tensor operations
        starting_stack: Starting stack size
        sb: Small blind amount
        bb: Big blind amount
        bet_bins: List of bet bin values
        rng: Random number generator
        flop_showdown: Whether to showdown after flop

    Returns:
        String representation of the preflop range grid
    """
    if device is None:
        device = torch.device("cpu")

    # Create 169-hand environment and state encoder
    env, state_encoder = create_169_hand_analysis_setup(
        model,
        button=0,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
        flop_showdown=flop_showdown,
    )

    # Get probabilities and values
    probs, _, _ = get_probabilities(model, state_encoder, env, 0, device)

    # Get probabilities for the selected metric
    selected_probs = probs[:, bin_index]  # [169]

    return _create_169_grid(selected_probs, "probability", " 0")


def get_preflop_range_grid_bb_response(
    model,
    bin_index: int,
    device: torch.device = None,
    starting_stack: int = 20000,
    sb: int = 50,
    bb: int = 100,
    bet_bins: List[int] = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> str:
    """Get preflop range as a grid for BB response after SB all-in.

    Args:
        model: Trained model for prediction
        bin_index: Index of the betting bin to check
        device: Device for tensor operations
        starting_stack: Starting stack size
        sb: Small blind amount
        bb: Big blind amount
        bet_bins: List of bet bin values
        rng: Random number generator
        flop_showdown: Whether to showdown after flop

    Returns:
        String representation of the preflop range grid
    """
    if device is None:
        device = torch.device("cpu")

    # Create 169-hand environment and state encoder
    # Seat 0 (model perspective) is BB, button/SB is p1.
    env, state_encoder = create_169_hand_analysis_setup(
        model,
        button=1,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
        flop_showdown=flop_showdown,
    )

    # Simulate SB all-in action
    step_sb_action(env, "allin", device)

    # Now next-to-act is p0 (BB)

    # Get probabilities and values
    probs, _, _ = get_probabilities(model, state_encoder, env, 0, device)

    # Get probabilities for the selected metric
    selected_probs = probs[:, bin_index]  # [169]

    return _create_169_grid(selected_probs, "probability", " 0")


def get_preflop_value_grid_bb_response(
    model,
    device: torch.device = None,
    starting_stack: int = 20000,
    sb: int = 50,
    bb: int = 100,
    bet_bins: List[int] = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> str:
    """Get preflop value estimates as a grid for BB response after SB all-in.

    Args:
        model: Trained model for prediction
        device: Device for tensor operations
        starting_stack: Starting stack size
        sb: Small blind amount
        bb: Big blind amount
        bet_bins: List of bet bin values
        rng: Random number generator
        flop_showdown: Whether to showdown after flop

    Returns:
        String representation of the preflop value grid
    """
    if device is None:
        device = torch.device("cpu")

    # Create 169-hand environment and state encoder
    env, state_encoder = create_169_hand_analysis_setup(
        model,
        button=1,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
        flop_showdown=flop_showdown,
    )

    # Simulate SB all-in action
    step_sb_action(env, "allin", device)

    # Get probabilities and values
    _, values, _ = get_probabilities(model, state_encoder, env, 0, device)

    return _create_169_grid(values, "value", " 0.0")


def get_preflop_value_grid(
    model,
    device: torch.device = None,
    starting_stack: int = 20000,
    sb: int = 50,
    bb: int = 100,
    bet_bins: List[int] = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> str:
    """Get preflop value estimates as a grid showing value estimates.

    Args:
        model: Trained model for prediction
        seat: Seat position (0 for SB, 1 for BB)
        device: Device for tensor operations
        starting_stack: Starting stack size
        sb: Small blind amount
        bb: Big blind amount
        bet_bins: List of bet bin values
        rng: Random number generator
        flop_showdown: Whether to showdown after flop

    Returns:
        String representation of the preflop value grid
    """
    if device is None:
        device = torch.device("cpu")

    # Create 169-hand environment and state encoder
    env, state_encoder = create_169_hand_analysis_setup(
        model,
        button=0,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
        flop_showdown=flop_showdown,
    )

    # Get probabilities and values
    _, values, _ = get_probabilities(model, state_encoder, env, 0, device)

    return _create_169_grid(values, "value", " 0.0")
