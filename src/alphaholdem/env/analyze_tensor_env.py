#!/usr/bin/env python3
"""
Debug utilities for HUNLTensorEnv.
Contains functions for creating test environments and analyzing specific scenarios.
"""

from typing import Dict, List, Tuple, Union

import torch

from alphaholdem.env.card_utils import (
    combo_lookup_tensor,
    hand_combos_tensor,
    NUM_HANDS,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.cnn.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.cnn.state_encoder import CNNStateEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.token_sequence_builder import TokenSequenceBuilder
from alphaholdem.search.rebel_cfr_evaluator import (
    T_WARM,
    PublicBeliefState,
    RebelCFREvaluator,
)

GRID_RANKS = "AKQJT98765432"


class DummyStateEncoder:
    def encode_tensor_states(self, player: int, idxs: torch.Tensor) -> torch.Tensor:
        return idxs


class RebelStateEncoderWrapper:
    """Adapter so PreflopAnalyzer can use RebelFeatureEncoder like other encoders."""

    def __init__(self, env: HUNLTensorEnv, device: torch.device):
        self.encoder = RebelFeatureEncoder(env=env, device=device)
        self.env = env
        self.device = device

    def encode_tensor_states(self, player: int, idxs: torch.Tensor) -> torch.Tensor:
        beliefs = torch.full(
            (idxs.numel(), 2, NUM_HANDS),
            fill_value=1.0 / NUM_HANDS,
            device=idxs.device,
            dtype=torch.float32,
        )
        return self.encoder.encode(beliefs)[idxs]


def create_state_encoder_for_model(
    model,
    env: HUNLTensorEnv,
    device: torch.device,
    bet_bins: list[float],
):
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
            bet_bins=bet_bins,
            device=device,
            float_dtype=torch.float32,
        )
    elif isinstance(model, SiameseConvNetV1):
        # For CNN models, create CNNStateEncoder with tensor_env and device
        return CNNStateEncoder(env, device)
    elif isinstance(model, RebelFFN):
        return RebelStateEncoderWrapper(env, device)
    else:
        # for testing.
        return DummyStateEncoder()


class PreflopAnalyzer:
    """Cached analyzer for preflop hand analysis with 1326-hand environment.

    This class caches the 1326-hand environment and hand combinations to avoid
    recreating them for each analysis. It provides methods to convert 1326-hand
    data to 169-hand-bucket grids.
    """

    def __init__(
        self,
        model: Union[PokerTransformerV1, SiameseConvNetV1, RebelFFN],
        button: int = 0,
        starting_stack: int = 1000,
        sb: int = 5,
        bb: int = 10,
        bet_bins: List[int] = None,
        device: torch.device = None,
        rng: torch.Generator = None,
        flop_showdown: bool = False,
        popart_normalizer=None,
        reset=True,
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
        self.combo_lookup = combo_lookup_tensor(device=self.device)

        # Create and cache the environment
        self.env = HUNLTensorEnv(
            num_envs=1326,
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            default_bet_bins=bet_bins,
            device=device,
            rng=rng,
            flop_showdown=flop_showdown,
        )

        self.state_encoder = create_state_encoder_for_model(
            model, self.env, device, bet_bins
        )

        # Set up the hands in the environment
        self.all_hands = hand_combos_tensor(device=device)

        if reset:
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

        suits = self.all_hands // 13
        ranks = self.all_hands % 13  # With new rank mapping: A=0, K=1, ..., 2=12
        ranks_sorted = torch.sort(ranks, dim=1).values
        ranks_flat = torch.where(
            suits[:, 0] == suits[:, 1],
            ranks_sorted[:, 0] * 13 + ranks_sorted[:, 1],
            ranks_sorted[:, 1] * 13 + ranks_sorted[:, 0],
        )
        grid_counts.flatten().scatter_add_(0, ranks_flat, torch.ones_like(ranks_flat))
        grid_sums.flatten().scatter_add_(0, ranks_flat, values_1326)

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
            env_indices = torch.arange(self.env.N, device=self.device)
            embedding_data = self.state_encoder.encode_tensor_states(seat, env_indices)
            outputs = self.model(embedding_data)

            # Get legal actions from tensor environment
            legal_masks = self.env.legal_bins_mask()  # [N, num_bet_bins]

            if isinstance(self.model, RebelFFN):
                hero_cards = self.env.hole_indices[env_indices, seat]
                card_a = hero_cards[:, 0]
                card_b = hero_cards[:, 1]
                combo_idx = self.combo_lookup[card_a, card_b]

                batch_idx = torch.arange(self.env.N, device=self.device)
                logits = outputs.policy_logits[batch_idx, combo_idx]
                if outputs.hand_values is None:
                    raise ValueError(
                        "RebelFFN must provide hand_values for per-combo analysis."
                    )
                values = outputs.hand_values[batch_idx, seat, combo_idx]
            else:
                logits = outputs.policy_logits
                values = outputs.value

            # Apply legal mask with broadcast handling
            masked_logits = torch.where(legal_masks == 0, -1e9, logits)

            # Get probabilities
            probs = torch.softmax(masked_logits, dim=-1)  # [N, num_bet_bins]

            # Denormalize values if PopArt normalizer is available
            if self.popart_normalizer is not None:
                values = self.popart_normalizer.denormalize_value(values)
            if values.dim() > 1:
                values = values.squeeze(-1)

        return probs, values, legal_masks

    def step_sb_action(self, sb_action: str = "allin") -> None:
        """Simulate a specific SB action across all environments.

        Args:
            sb_action: SB action to simulate ("allin", "call", "fold", "bet")
        """
        N = self.env.N
        bin = None
        if sb_action == "allin":
            bin = len(self.env.default_bet_bins) + 3 - 1
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
                idxs=torch.arange(N, device=self.device),
                actors=torch.ones(N, dtype=torch.long, device=self.device),
                action_ids=torch.full((N,), bin, dtype=torch.long, device=self.device),
                legal_masks=legal_masks,
                action_amounts=self.env.pot // 2,
                token_streets=torch.zeros(N, dtype=torch.long, device=self.device),
            )
            self.state_encoder.add_context(torch.arange(N, device=self.device))

    def calculate_suited_vs_offsuit(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate the suited vs offsuit probability for the given action for each hand."""
        values_169 = self.convert_1326_to_169_tensor(probs)
        suited_rows, suited_cols = torch.triu_indices(13, 13, offset=1)
        offsuit_rows, offsuit_cols = torch.tril_indices(13, 13, offset=-1)
        suited_probs = values_169[suited_rows, suited_cols].mean()
        offsuit_probs = values_169[offsuit_rows, offsuit_cols].mean()
        return torch.stack([suited_probs, offsuit_probs], dim=0)

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
            "suited_vs_offsuit": torch.stack(
                [
                    self.calculate_suited_vs_offsuit(probs[:, bin_index])
                    for bin_index in range(probs.shape[1])
                ],
                dim=0,
            ),
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


class RebelPreflopAnalyzer(PreflopAnalyzer):
    """Proper ReBeL preflop analyzer using uniform belief states and CFR search.

    This analyzer correctly implements ReBeL by:
    1. Using uniform belief states for both players (no private information leakage)
    2. Running CFR search to compute proper policies
    3. Generating range grids from the CFR-computed policies
    """

    def __init__(
        self,
        model: RebelFFN,
        button: int = 0,
        starting_stack: int = 1000,
        sb: int = 5,
        bb: int = 10,
        bet_bins: List[int] = None,
        device: torch.device = None,
        rng: torch.Generator = None,
        flop_showdown: bool = False,
        popart_normalizer=None,
        cfr_iterations: int = 100,
        max_depth: int = 2,
    ):
        """Initialize the ReBeL analyzer with CFR search capabilities.

        Args:
            model: The trained RebelFFN model
            button: Which player is the button
            starting_stack: Starting stack size
            sb: Small blind amount
            bb: Big blind amount
            bet_bins: List of bet bin values
            device: Device to use
            rng: Random number generator
            flop_showdown: Whether to showdown after flop
            popart_normalizer: Optional PopArt normalizer for denormalizing values
            cfr_iterations: Number of CFR iterations to run
            max_depth: Maximum search depth
        """
        if bet_bins is None:
            bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]

        if device is None:
            device = torch.device("cpu")

        if rng is None:
            rng = torch.Generator(device=device)

        if cfr_iterations <= T_WARM:
            raise ValueError(
                f"RebelPreflopAnalyzer requires cfr_iterations > T_WARM ({T_WARM}); "
                f"got {cfr_iterations}."
            )

        super().__init__(
            model=model,
            button=button,
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            bet_bins=bet_bins,
            device=device,
            rng=rng,
            flop_showdown=flop_showdown,
            popart_normalizer=popart_normalizer,
            reset=False,
        )

        self.cfr_iterations = cfr_iterations
        self.max_depth = max_depth
        self.combo_lookup = combo_lookup_tensor(device=self.device)

        # Create single environment for CFR search
        self.cfr_env = HUNLTensorEnv(
            num_envs=1,  # Single environment for analysis
            starting_stack=starting_stack,
            sb=sb,
            bb=bb,
            default_bet_bins=bet_bins,
            device=device,
            rng=rng,
            flop_showdown=flop_showdown,
        )

        self.cfr_evaluator = RebelCFREvaluator(
            search_batch_size=1,  # Single environment
            env_proto=self.cfr_env,
            model=self.model,
            bet_bins=bet_bins,
            max_depth=max_depth,
            cfr_iterations=cfr_iterations,
            device=device,
            float_dtype=torch.float32,
            generator=rng,
        )

        # Reinitialize both the base and CFR environments now that CFR state is set up.
        self.reset(button)

    def reset(self, button: int):
        """Reset cached environments for both direct model inference and CFR search."""
        super().reset(button)

        self.cfr_env.reset(
            force_button=torch.tensor([button], dtype=torch.long, device=self.device),
        )

        uniform_beliefs = torch.full(
            (1, 2, NUM_HANDS),
            1.0 / NUM_HANDS,
            device=self.device,
            dtype=torch.float32,
        )

        self.pbs = PublicBeliefState.from_proto(
            env_proto=self.cfr_env,
            beliefs=uniform_beliefs,
        )
        root_idx = torch.tensor([0], dtype=torch.long, device=self.device)
        self.pbs.env.copy_state_from(
            self.cfr_env,
            root_idx,
            root_idx,
            copy_deck=True,
        )

        self.current_index = 1
        self.current_depth = 1

    def get_probabilities_from_cfr(
        self, seat: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get probabilities and values by running CFR search with uniform beliefs.

        Args:
            seat: Seat position (0 for SB, 1 for BB)

        Returns:
            Tuple of (probabilities [1326, num_bet_bins], values [1326], legal_masks [1326, num_bet_bins])
        """
        assert (
            self.cfr_evaluator is not None and self.cfr_env is not None
        ), "RebelPreflopAnalyzer must initialize CFR evaluator."

        # Run CFR search to compute policies
        self.cfr_evaluator.initialize_search(
            self.pbs.env,
            torch.tensor([0], dtype=torch.long, device=self.device),
            self.pbs.beliefs,
        )
        self.cfr_evaluator.self_play_iteration(training_mode=False)

        # Get the root node policy (index 0) [NUM_HANDS, num_actions]
        actions_slice = slice(
            self.current_index,
            self.current_index + self.cfr_evaluator.num_actions,
        )
        root_policy = self.cfr_evaluator.policy_probs_avg[actions_slice]
        root_policy /= root_policy.sum(dim=0, keepdim=True).clamp_min(1e-12)

        # Get legal actions
        legal_masks = self.cfr_evaluator.valid_mask[actions_slice]
        legal_masks = legal_masks[None, :].expand(NUM_HANDS, -1)

        # Get values from CFR
        root_values = (
            self.cfr_evaluator.values[actions_slice, seat] * root_policy
        ).sum(dim=0)

        # Denormalize values if PopArt normalizer is available
        if self.popart_normalizer is not None:
            root_values = self.popart_normalizer.denormalize_value(root_values)

        return root_policy.T, root_values, legal_masks

    def get_probabilities(
        self, seat: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Override to use CFR search when configured, else belief-averaged model outputs."""
        return self.get_probabilities_from_cfr(seat)

    def step_sb_action(self, sb_action: str = "allin") -> None:
        """Override to apply the SB action inside the CFR root environment."""
        super().step_sb_action(sb_action)

        bin_map = {
            "fold": 0,
            "call": 1,
            "bet": 2,
            "allin": self.cfr_evaluator.num_actions - 1,
        }
        action_bin = bin_map[sb_action]

        legal_masks = self.cfr_env.legal_bins_mask()
        bin_tensor = torch.full(
            (self.cfr_env.N,), action_bin, dtype=torch.long, device=self.device
        )
        self.cfr_env.step_bins(bin_tensor, legal_masks=legal_masks)

        root_indices = torch.arange(self.cfr_env.N, device=self.device)
        self.pbs.env.copy_state_from(
            self.cfr_env,
            root_indices,
            root_indices,
            copy_deck=True,
        )


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
        for i in range(12, -1, -1):
            row = []
            for j in range(12, -1, -1):
                val = min(0.99, values[i, j].item())
                row.append(f"{val * 100:3.0f}")
            formatted_values.append(row)
    elif value_type == "value":
        # Format value estimates (multiply by 1000 for readability, show 3 sig figs, max 4 chars)
        formatted_values = []
        for i in range(12, -1, -1):
            row = []
            for j in range(12, -1, -1):
                val_scaled = values[i, j].item() * 1000
                row.append(f"{val_scaled:5.0f}")
            formatted_values.append(row)
    else:  # raw
        raise ValueError(f"Invalid value type: {value_type}")

    # Initialize grid
    grid = []
    if value_type == "probability":
        grid.append("    " + "".join(f"{rank:>3}" for rank in GRID_RANKS))
        grid.append("   +" + "-" * 39)  # Separator line
    else:
        grid.append("    " + "".join(f"{rank:>5}" for rank in GRID_RANKS))
        grid.append("   +" + "-" * 65)  # Separator line

    for i, row in enumerate(formatted_values):
        grid.append(f"{GRID_RANKS[i]:>2} |" + "".join(row))

    return "\n".join(grid)
