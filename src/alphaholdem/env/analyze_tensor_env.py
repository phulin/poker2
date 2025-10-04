#!/usr/bin/env python3
"""
Debug utilities for HUNLTensorEnv.
Contains functions for creating test environments and analyzing specific scenarios.
"""

from typing import Dict, List, Tuple, Union

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.cnn.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.cnn.state_encoder import CNNStateEncoder
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder

RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS = ["s", "h", "d", "c"]


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


def create_1326_hand_combinations() -> List[Tuple[str, str]]:
    """Create all 1326 distinct preflop combinations (ordered hole cards, no overlap).

    Returns pairs like ("As", "Kh"). Offsuit/suited and pairs fully enumerated.
    """
    # Build full deck strings
    deck = [r + s for r in RANKS for s in SUITS]
    hands: List[Tuple[str, str]] = []
    for i in range(len(deck)):
        for j in range(i + 1, len(deck)):
            c1, c2 = deck[i], deck[j]
            # Exclude same card obviously, and allow any suit/rank
            hands.append((c1, c2))
    return hands


def _grid_coords_for_hand(card1: int, card2: int) -> Tuple[int, int]:
    s1, s2 = card1 // 13, card2 // 13

    # Grid is reversed, higher ranks first.
    i, j = 12 - card1 % 13, 12 - card2 % 13

    # Suited if same suit and not pair → top-right triangle; else bottom-left
    if s1 == s2:
        # suited → place at (min(i,j), max(i,j)) where higher rank is column
        return (min(i, j), max(i, j))
    else:
        # offsuit → place at (max(i,j), min(i,j))
        return (max(i, j), min(i, j))


class PreflopAnalyzer:
    """Cached analyzer for preflop hand analysis with 1326-hand environment.

    This class caches the 1326-hand environment and hand combinations to avoid
    recreating them for each analysis. It provides methods to convert 1326-hand
    data to 169-hand-bucket grids.
    """

    def __init__(
        self,
        model: Union[PokerTransformerV1, SiameseConvNetV1],
        button: int = 0,
        starting_stack: int = 1000,
        sb: int = 5,
        bb: int = 10,
        bet_bins: List[int] = None,
        device: torch.device = None,
        rng: torch.Generator = None,
        flop_showdown: bool = False,
        popart_normalizer=None,
    ):
        """Initialize the analyzer with cached environment and hands.

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
            popart_normalizer: Optional PopArt normalizer for denormalizing values
        """
        if bet_bins is None:
            bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]

        if device is None:
            device = torch.device("cpu")

        if rng is None:
            rng = torch.Generator(device=device)

        self.model = model
        self.device = device
        self.popart_normalizer = popart_normalizer

        # Create and cache the environment
        self.env = HUNLTensorEnv(
            num_envs=1326,
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            bet_bins=bet_bins,
            device=device,
            rng=rng,
            flop_showdown=flop_showdown,
        )

        self.state_encoder = create_state_encoder_for_model(model, self.env, device)

        # Cache the 1326 hand combinations
        self.all_hands_str = create_1326_hand_combinations()

        # Set up the hands in the environment
        self.all_hands = torch.tensor(
            [
                (_card_str_to_int(hand[0]), _card_str_to_int(hand[1]))
                for hand in self.all_hands_str
            ],
            dtype=torch.long,
            device=device,
        )

        self.reset(button)

    def reset(self, button: int):
        """Reset the environment and state encoder to initial state."""

        self.env.reset(
            force_button=torch.full(
                (self.env.N,), button, dtype=torch.long, device=self.device
            ),
            force_deck=self.all_hands,
        )

        if isinstance(self.state_encoder, TokenSequenceBuilder):
            self.state_encoder.reset()
            self.state_encoder.add_game(torch.arange(self.env.N, device=self.device))
            self.state_encoder.add_card(
                torch.arange(self.env.N, device=self.device), self.all_hands[:, 0]
            )
            self.state_encoder.add_card(
                torch.arange(self.env.N, device=self.device), self.all_hands[:, 1]
            )
            self.state_encoder.add_context(torch.arange(self.env.N, device=self.device))

    def convert_1326_to_169_tensor(self, values_1326: torch.Tensor) -> torch.Tensor:
        """Convert 1326-hand values to a 169-hand-bucket tensor.

        Args:
            values_1326: Tensor of 1326 values for each hand

        Returns:
            Tensor of shape [169] representing the 169-hand buckets
        """
        # Aggregate 1326 → 169 by averaging combos that map to same grid cell
        grid_sums = torch.zeros(13, 13, dtype=values_1326.dtype, device=self.device)
        grid_counts = torch.zeros(13, 13, dtype=torch.long, device=self.device)

        for idx, (c1, c2) in enumerate(self.all_hands):
            i, j = _grid_coords_for_hand(c1, c2)
            grid_sums[i, j] += values_1326[idx]
            grid_counts[i, j] += 1

        # Avoid div by zero
        averaged = torch.where(
            grid_counts > 0,
            grid_sums / grid_counts.clamp(min=1),
            torch.zeros_like(grid_sums),
        )

        return averaged

    def make_range_grid(
        self,
        values_1326: torch.Tensor,
        value_type: str = "probability",
    ) -> str:
        """Convert 1326-hand values to a 169-hand-bucket grid table.

        Args:
            values_1326: Tensor of 1326 values for each hand
            value_type: Type of values - "probability", "value", or "raw"

        Returns:
            String representation of the 13x13 grid
        """
        values_169 = self.convert_1326_to_169_tensor(values_1326)
        return _create_169_grid(values_169, value_type)

    def get_probabilities(
        self, seat: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get model probabilities and values for all 1326 hands.

        Args:
            seat: Seat position (0 for SB, 1 for BB)

        Returns:
            Tuple of (probabilities [1326, num_bet_bins], values [1326], legal_masks [1326, num_bet_bins])
        """

        # Get model prediction using tensor environment
        with torch.no_grad():
            embedding_data = self.state_encoder.encode_tensor_states(
                seat, torch.arange(self.env.N, device=self.device)  # All environments
            )
            outputs = self.model(embedding_data)

            # Get legal actions from tensor environment
            legal_masks = self.env.legal_bins_mask()  # [N, num_bet_bins]

            # Apply legal mask
            masked_logits = torch.where(legal_masks == 0, -1e9, outputs.policy_logits)

            # Get probabilities
            probs = torch.softmax(masked_logits, dim=-1)  # [N, num_bet_bins]

            # Denormalize values if PopArt normalizer is available
            values = outputs.value
            if self.popart_normalizer is not None:
                values = self.popart_normalizer.denormalize_value(values)

        return probs, values, legal_masks

    def step_sb_action(self, sb_action: str = "allin") -> None:
        """Simulate a specific SB action across all environments.

        Args:
            sb_action: SB action to simulate ("allin", "call", "fold", "bet")
        """
        N = self.env.N
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
        legal_masks = self.env.legal_bins_mask()
        assert legal_masks[:, bin].all()
        self.env.step_bins(torch.full((N,), bin, dtype=torch.long, device=self.device))

        if isinstance(self.state_encoder, TokenSequenceBuilder):
            self.state_encoder.add_action(
                torch.arange(N, device=self.device),
                torch.ones(N, dtype=torch.long, device=self.device),
                torch.full((N,), bin, dtype=torch.long, device=self.device),
                legal_masks,
                torch.zeros(N, dtype=torch.long, device=self.device),
            )
            self.state_encoder.add_context(torch.arange(N, device=self.device))

    def get_preflop_grids(self) -> Dict[str, Union[str, List[str]]]:
        """Get preflop range as a grid showing selected action probabilities.

        Args:
            bin_index: Index of the betting bin to check

        Returns:
            String representation of the preflop range grid
        """
        self.reset(0)

        # Get probabilities and values
        probs, values, _ = self.get_probabilities(0)

        return {
            "ranges": [
                self.make_range_grid(probs[:, bin_index], "probability")
                for bin_index in range(probs.shape[1])
            ],
            "betting": self.make_range_grid(
                probs[:, 2 : probs.shape[1] - 1].sum(dim=1), "probability"
            ),
            "value": self.make_range_grid(values, "value"),
        }

    def get_preflop_range_grid(self, bin_index: int) -> str:
        """Get preflop range as a grid for the given bin index.

        Args:
            bin_index: Index of the betting bin to check

        Returns:
            String representation of the preflop range grid
        """
        return self.get_preflop_grids()["ranges"][bin_index]

    def get_preflop_betting_grid(self) -> str:
        """Get preflop betting range (sum of all betting bins) as a grid.

        Returns:
            String representation of the preflop betting grid
        """
        return self.get_preflop_grids()["betting"]

    def get_preflop_value_grid(self) -> str:
        """Get preflop value as a grid for the given bin index.

        Returns:
            String representation of the preflop value grid
        """
        return self.get_preflop_grids()["value"]

    def get_preflop_grids_allin_response(self) -> Dict[str, Union[str, List[str]]]:
        """Get preflop grids for all-in response.

        Returns:
            Dictionary of preflop grids
        """
        # p1 leads as button/sb
        self.reset(1)
        self.step_sb_action("allin")

        # Get probabilities and values
        probs, values, _ = self.get_probabilities(0)

        return {
            "ranges": [
                self.make_range_grid(probs[:, bin_index], "probability")
                for bin_index in range(probs.shape[1])
            ],
            "betting": self.make_range_grid(
                probs[:, 2 : probs.shape[1] - 1].sum(dim=1), "probability"
            ),
            "value": self.make_range_grid(values, "value"),
        }

    def get_preflop_range_grid_allin_response(self, bin_index: int) -> str:
        """Get preflop range as a grid for BB response after SB all-in.

        Args:
            bin_index: Index of the betting bin to check

        Returns:
            String representation of the preflop range grid
        """
        return self.get_preflop_grids_allin_response()["ranges"][bin_index]

    def get_preflop_value_grid_allin_response(self) -> str:
        """Get preflop value estimates as a grid for BB response after SB all-in.

        Returns:
            String representation of the preflop value grid
        """
        return self.get_preflop_grids_allin_response()["value"]


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


def _create_169_grid(values: torch.Tensor, value_type: str = "probability") -> str:
    """Create a standardized 13x13 grid for 169 preflop hands.

    Args:
        values: Tensor of 13x13 values for each hand
        value_type: Type of values - "probability", "value", or "raw"

    Returns:
        String representation of the 13x13 grid
    """

    # Format values based on type, assuming values is a 13x13 grid
    if value_type == "probability":
        # Convert to percentages and format as 2-digit strings, cap at 99
        formatted_values = []
        for i in range(13):
            row = []
            for j in range(13):
                val = min(0.99, values[i, j].item())
                row.append(f"{val * 100:3.0f}")
            formatted_values.append(row)
    elif value_type == "value":
        # Format value estimates (multiply by 1000 for readability, show 3 sig figs, max 4 chars)
        formatted_values = []
        for i in range(13):
            row = []
            for j in range(13):
                val_scaled = values[i, j].item() * 1000
                row.append(f"{val_scaled:5.0f}")
            formatted_values.append(row)
    else:  # raw
        raise ValueError(f"Invalid value type: {value_type}")

    # Initialize grid
    grid = []
    if value_type == "probability":
        grid.append("    " + "".join(f"{rank:>3}" for rank in RANKS))
        grid.append("   +" + "-" * 39)  # Separator line
    else:
        grid.append("    " + "".join(f"{rank:>5}" for rank in RANKS))
        grid.append("   +" + "-" * 65)  # Separator line

    for i, row in enumerate(formatted_values):
        grid.append(f"{RANKS[i]:>2} |" + "".join(row))

    return "\n".join(grid)
