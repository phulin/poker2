#!/usr/bin/env python3
"""
Debug utilities for HUNLTensorEnv.
Contains functions for creating test environments and analyzing specific scenarios.
"""

from typing import Any, List, Tuple, Union

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.cnn.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.state_encoder import CNNStateEncoder
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder


class DummyStateEncoder:
    def encode_tensor_states(self, player: int, idxs: torch.Tensor) -> torch.Tensor:
        return idxs


def create_state_encoder_for_model(model, env: HUNLTensorEnv, device: torch.device):
    """Create the appropriate state encoder based on the model type.

    Args:
        model: The trained model instance
        env: The 169-hand tensor environment
        device: Device for tensor operations

    Returns:
        Appropriate state encoder for the model
    """

    if isinstance(model, PokerTransformerV1):
        # For transformer models, create TokenSequenceBuilder with sane defaults
        return TokenSequenceBuilder(
            tensor_env=env,
            sequence_length=model.max_sequence_length,
            num_bet_bins=env.num_bet_bins,
            device=device,
            float_dtype=torch.float32,
        )
    elif isinstance(model, SiameseConvNetV1):
        # For CNN models, create CNNStateEncoder with tensor_env and device
        return CNNStateEncoder(env, device)
    else:
        # for testing.
        return DummyStateEncoder()


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


def create_1326_hand_combinations() -> List[Tuple[str, str]]:
    """Create all 1326 distinct preflop combinations (ordered hole cards, no overlap).

    Returns pairs like ("As", "Kh"). Offsuit/suited and pairs fully enumerated.
    """
    suits = ["s", "h", "d", "c"]
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    # Build full deck strings
    deck = [r + s for r in ranks for s in suits]
    hands: List[Tuple[str, str]] = []
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            c1, c2 = deck[i], deck[j]
            # Exclude same card obviously, and allow any suit/rank
            hands.append((c1, c2))
    return hands


def _grid_coords_for_hand(card1: str, card2: str) -> Tuple[int, int]:
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    r1 = card1[:-1]
    r2 = card2[:-1]
    i = ranks.index(r1)
    j = ranks.index(r2)
    # Suited if same suit and not pair → top-right triangle; else bottom-left
    if r1 == r2:
        return i, j
    if card1[-1] == card2[-1]:
        # suited → place at (min(i,j), max(i,j)) where higher rank is column
        return (min(i, j), max(i, j))
    else:
        # offsuit → place at (max(i,j), min(i,j))
        return (max(i, j), min(i, j))


def get_probabilities(
    model,
    state_encoder,
    env: HUNLTensorEnv,
    seat: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get model probabilities and values for all environments in `env`.

    Args:
        model: The trained model instance
        state_encoder: The state encoder for the model
        env: The 169-hand tensor environment
        seat: Seat position (0 for SB, 1 for BB)
        device: Device for tensor operations

    Returns:
        Tuple of (probabilities [N, num_bet_bins], values [N], legal_masks [N, num_bet_bins])
    """
    # Get model prediction using tensor environment
    with torch.no_grad():
        embedding_data = state_encoder.encode_tensor_states(
            seat, torch.arange(env.N, device=device)  # All environments
        )
        outputs = model(embedding_data)

        # Get legal actions from tensor environment
        legal_masks = env.legal_bins_mask()  # [N, num_bet_bins]

        # Apply legal mask
        masked_logits = torch.where(legal_masks == 0, -1e9, outputs.policy_logits)

        # Get probabilities
        probs = torch.softmax(masked_logits, dim=-1)  # [N, num_bet_bins]

    return probs, outputs.value, legal_masks


def create_169_hand_analysis_setup(
    model: Union[PokerTransformerV1, SiameseConvNetV1],
    button: int,
    starting_stack: int = 1000,
    sb: int = 5,
    bb: int = 10,
    bet_bins: List[int] = None,
    device: torch.device = None,
    rng: torch.Generator = None,
    flop_showdown: bool = False,
) -> Tuple[HUNLTensorEnv, Any]:
    """Create a tensor environment with all 1326 preflop hands set up for player 0 and appropriate state encoder.

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

    # Create environment with 1326 states
    temp_env = HUNLTensorEnv(
        num_envs=1326,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
        flop_showdown=flop_showdown,
    )

    # Get all 1326 hand combinations
    hands = create_1326_hand_combinations()

    cards = torch.tensor(
        [(_card_str_to_int(hand[0]), _card_str_to_int(hand[1])) for hand in hands],
        dtype=torch.long,
        device=device,
    )

    # Reset the environment
    temp_env.reset(
        force_button=torch.full((temp_env.N,), button, dtype=torch.long, device=device),
        force_deck=cards,
    )

    state_encoder = create_state_encoder_for_model(model, temp_env, device)
    if isinstance(state_encoder, TokenSequenceBuilder):
        state_encoder.add_game(torch.arange(temp_env.N, device=device))
        state_encoder.add_card(torch.arange(temp_env.N, device=device), cards[:, 0])
        state_encoder.add_card(torch.arange(temp_env.N, device=device), cards[:, 1])
        state_encoder.add_context(torch.arange(temp_env.N, device=device))

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
    starting_stack: int = 1000,
    sb: int = 5,
    bb: int = 10,
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

    # Create 1326-hand environment and state encoder
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

    # Get probabilities and values for all 1326 combos
    probs, _, _ = get_probabilities(model, state_encoder, env, 0, device)

    # Sum all betting probabilities (exclude fold=0, call=1, and all-in=num_bet_bins-1)
    # Betting actions are typically indices 2 through num_bet_bins-2 (excluding all-in)
    num_bet_bins = probs.shape[1]
    betting_probs = probs[:, 2 : num_bet_bins - 1].sum(dim=1)  # [1326]

    # Aggregate to 169 grid by averaging combos per cell
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    grid_sums = torch.zeros(13, 13, dtype=probs.dtype, device=device)
    grid_counts = torch.zeros(13, 13, dtype=torch.long, device=device)

    hands = create_1326_hand_combinations()
    for idx, (c1, c2) in enumerate(hands):
        i, j = _grid_coords_for_hand(c1, c2)
        grid_sums[i, j] += betting_probs[idx]
        grid_counts[i, j] += 1

    averaged = torch.where(
        grid_counts > 0,
        grid_sums / grid_counts.clamp(min=1),
        torch.zeros_like(grid_sums),
    )
    values_169 = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            values_169.append(averaged[i, j])
    values_169 = torch.stack(values_169)

    return _create_169_grid(values_169, "probability", " 0")


def step_sb_action(
    env: HUNLTensorEnv,
    state_encoder,
    sb_action: str = "allin",
    device: torch.device = None,
) -> None:
    """Simulate a specific SB action across all environments.

    Args:
        env: HUNLTensorEnv with N environments
        sb_action: SB action to simulate ("allin", "call", "fold", "bet")
        device: Device for tensor operations
    """
    if device is None:
        device = env.device

    N = env.N
    bin = None
    if sb_action == "allin":
        bin = 7
    elif sb_action == "call":
        bin = 1
    elif sb_action == "fold":
        bin = 0
    elif sb_action == "bet":
        bin = 2

    assert bin is not None
    env.step_bins(torch.full((N,), bin, dtype=torch.long, device=device))

    if isinstance(state_encoder, TokenSequenceBuilder):
        state_encoder.add_action(
            torch.arange(N, device=device),
            torch.ones(N, device=device),
            torch.full((N,), bin, device=device),
            torch.full((N, env.num_bet_bins + 3), True, device=device),
            torch.full((N,), 0, device=device),
        )
        state_encoder.add_context(torch.arange(N, device=device))


def get_preflop_range_grid(
    model,
    bin_index: int,
    device: torch.device = None,
    starting_stack: int = 1000,
    sb: int = 5,
    bb: int = 10,
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

    # Aggregate 1326 → 169 by averaging combos that map to same grid cell
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    grid_sums = torch.zeros(13, 13, dtype=probs.dtype, device=device)
    grid_counts = torch.zeros(13, 13, dtype=torch.long, device=device)

    hands = create_1326_hand_combinations()
    selected = probs[:, bin_index]  # [1326]
    for idx, (c1, c2) in enumerate(hands):
        i, j = _grid_coords_for_hand(c1, c2)
        grid_sums[i, j] += selected[idx]
        grid_counts[i, j] += 1

    # Avoid div by zero; cells with zero combos should not happen
    averaged = torch.where(
        grid_counts > 0,
        grid_sums / grid_counts.clamp(min=1),
        torch.zeros_like(grid_sums),
    )
    # Flatten in 169 order matching _create_169_grid traversal
    values_169 = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            values_169.append(averaged[i, j])
    values_169 = torch.stack(values_169)

    return _create_169_grid(values_169, "probability", " 0")


def get_preflop_range_grid_bb_response(
    model: Union[PokerTransformerV1, SiameseConvNetV1],
    bin_index: int,
    device: torch.device = None,
    starting_stack: int = 1000,
    sb: int = 5,
    bb: int = 10,
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
    step_sb_action(env, state_encoder, "allin", device)

    # Now next-to-act is p0 (BB)

    # Get probabilities and values for all 1326 combos
    probs, _, _ = get_probabilities(model, state_encoder, env, 0, device)

    # Aggregate 1326 → 169 by averaging combos that map to same grid cell
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    grid_sums = torch.zeros(13, 13, dtype=probs.dtype, device=device)
    grid_counts = torch.zeros(13, 13, dtype=torch.long, device=device)

    hands = create_1326_hand_combinations()
    selected = probs[:, bin_index]  # [1326]
    for idx, (c1, c2) in enumerate(hands):
        i, j = _grid_coords_for_hand(c1, c2)
        grid_sums[i, j] += selected[idx]
        grid_counts[i, j] += 1

    # Avoid div by zero
    averaged = torch.where(
        grid_counts > 0,
        grid_sums / grid_counts.clamp(min=1),
        torch.zeros_like(grid_sums),
    )
    # Flatten in 169 order matching _create_169_grid traversal
    values_169 = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            values_169.append(averaged[i, j])
    values_169 = torch.stack(values_169)

    return _create_169_grid(values_169, "probability", " 0")


def get_preflop_value_grid_bb_response(
    model: Union[PokerTransformerV1, SiameseConvNetV1],
    device: torch.device = None,
    starting_stack: int = 1000,
    sb: int = 5,
    bb: int = 10,
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
    step_sb_action(env, state_encoder, "allin", device)

    # Get values for all 1326 combos
    _, values, _ = get_probabilities(model, state_encoder, env, 0, device)

    # Aggregate to 169 grid by averaging
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    grid_sums = torch.zeros(13, 13, dtype=values.dtype, device=device)
    grid_counts = torch.zeros(13, 13, dtype=torch.long, device=device)

    hands = create_1326_hand_combinations()
    for idx, (c1, c2) in enumerate(hands):
        i, j = _grid_coords_for_hand(c1, c2)
        grid_sums[i, j] += values[idx]
        grid_counts[i, j] += 1

    averaged = torch.where(
        grid_counts > 0,
        grid_sums / grid_counts.clamp(min=1),
        torch.zeros_like(grid_sums),
    )
    values_169 = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            values_169.append(averaged[i, j])
    values_169 = torch.stack(values_169)

    return _create_169_grid(values_169, "value", " 0.0")


def get_preflop_value_grid(
    model: Union[PokerTransformerV1, SiameseConvNetV1],
    device: torch.device = None,
    starting_stack: int = 1000,
    sb: int = 5,
    bb: int = 10,
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

    # Create 1326-hand environment and state encoder
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

    # Get values for all 1326 combos
    _, values, _ = get_probabilities(model, state_encoder, env, 0, device)

    # Aggregate to 169 grid by averaging
    ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    grid_sums = torch.zeros(13, 13, dtype=values.dtype, device=device)
    grid_counts = torch.zeros(13, 13, dtype=torch.long, device=device)

    hands = create_1326_hand_combinations()
    for idx, (c1, c2) in enumerate(hands):
        i, j = _grid_coords_for_hand(c1, c2)
        grid_sums[i, j] += values[idx]
        grid_counts[i, j] += 1

    averaged = torch.where(
        grid_counts > 0,
        grid_sums / grid_counts.clamp(min=1),
        torch.zeros_like(grid_sums),
    )
    values_169 = []
    for i, r1 in enumerate(ranks):
        for j, r2 in enumerate(ranks):
            values_169.append(averaged[i, j])
    values_169 = torch.stack(values_169)

    return _create_169_grid(values_169, "value", " 0.0")
