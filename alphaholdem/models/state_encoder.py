"""State encoding utilities for poker models."""

from __future__ import annotations

from typing import Dict, Tuple
import torch

from ..env.hunl_tensor_env import HUNLTensorEnv
from ..env.hunl_env import HUNLEnv
from .cnn import CardsPlanesV1, ActionsHUEncoderV1


class StateEncoder:
    """Encodes poker game states into model inputs."""

    def __init__(
        self,
        cards_encoder: CardsPlanesV1,
        actions_encoder: ActionsHUEncoderV1,
        device: torch.device,
    ):
        self.cards_encoder = cards_encoder
        self.actions_encoder = actions_encoder
        self.device = device

    def encode_tensor_states(
        self, tensor_env: HUNLTensorEnv, num_envs: int, num_bet_bins: int
    ) -> Dict[str, torch.Tensor]:
        """
        Encode states for all tensor_environments in the tensorized environment.

        Args:
            tensor_env: The tensorized environment
            num_envs: Number of environments
            num_bet_bins: Number of betting bins

        Returns:
            Dictionary with 'cards' and 'actions' tensors
        """
        batch_size = num_envs

        # Vectorized card encoding - much faster than Python loops
        hole_cards = tensor_env.hole_onehot[:, 0]  # [N, 2, 4, 13]
        board_cards = tensor_env.board_onehot  # [N, 5, 4, 13]

        # Initialize cards tensor as bool
        cards = torch.zeros(batch_size, 6, 4, 13, dtype=torch.bool, device=self.device)

        # Channel 0: hole cards (sum over 2 hole cards)
        cards[:, 0] = hole_cards.any(dim=1)  # [N, 4, 13]

        # Channel 1: flop cards (first 3 board cards)
        cards[:, 1] = board_cards[:, :3].any(dim=1)  # [N, 4, 13]

        # Channel 2: turn card (4th board card)
        cards[:, 2] = board_cards[:, 3]  # [N, 4, 13]

        # Channel 3: river card (5th board card)
        cards[:, 3] = board_cards[:, 4]  # [N, 4, 13]

        # Channel 4: public cards (all board cards)
        cards[:, 4] = board_cards.any(dim=1)  # [N, 4, 13]

        # Channel 5: all cards (hole + board)
        cards[:, 5] = hole_cards.any(dim=1) | board_cards.any(dim=1)  # [N, 4, 13]

        # Get action history directly from tensor environment
        # Shape: [N, 4_streets, 6_slots, 4_players, num_bet_bins]
        action_history = tensor_env.get_action_history()

        # Reshape to match ActionsHUEncoderV1 format: [N, 24_channels, 4_players, num_bet_bins]
        # Flatten streets and slots: [N, 4*6, 4, num_bet_bins] = [N, 24, 4, num_bet_bins]
        actions = action_history.view(batch_size, 24, 4, num_bet_bins)

        return {
            "cards": cards,
            "actions": actions,
        }

    def encode_single_state(
        self, state: HUNLEnv, seat: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single game state for non-tensorized environments.

        Args:
            state: The game state
            seat: Player seat (0 or 1)

        Returns:
            Tuple of (cards_tensor, actions_tensor)
        """
        cards = self.cards_encoder.encode_cards(state, seat=seat)
        actions_tensor = self.actions_encoder.encode_actions(state, seat=seat)
        return cards, actions_tensor
