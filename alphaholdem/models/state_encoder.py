"""State encoding utilities for poker models."""

from __future__ import annotations

from typing import Dict, Tuple
import torch

from ..env.hunl_tensor_env import HUNLTensorEnv
from ..env.hunl_env import HUNLEnv
from .cnn import CardsPlanesV1, ActionsHUEncoderV1
from .cnn_embedding_data import CNNEmbeddingData


class CNNStateEncoder:
    """Encodes poker game states into model inputs."""

    def __init__(
        self,
        tensor_env: HUNLTensorEnv,
        device: torch.device,
    ):
        self.tensor_env = tensor_env
        self.device = device
        self.num_bet_bins = tensor_env.num_bet_bins

    def encode_tensor_states(self, player: int, idxs: torch.Tensor) -> CNNEmbeddingData:
        """
        Encode states for all tensor_environments in the tensorized environment.

        Args:
            player: Player index (0 or 1)
            idxs: Tensor of indices to filter environments

        Returns:
            CNNEmbeddingData containing cards and actions tensors
        """
        M = idxs.numel()
        env_indices = idxs

        hole_cards = self.tensor_env.hole_onehot[env_indices, 0]  # [M, 2, 4, 13]
        board_cards = self.tensor_env.board_onehot[env_indices]  # [M, 5, 4, 13]

        # Initialize cards tensor as bool with correct batch size
        cards = torch.zeros(M, 6, 4, 13, dtype=torch.bool, device=self.device)

        # Channel 0: hole cards (sum over 2 hole cards)
        cards[:, 0] = hole_cards.any(dim=1)  # [M, 4, 13]

        # Channel 1: flop cards (first 3 board cards)
        cards[:, 1] = board_cards[:, :3].any(dim=1)  # [M, 4, 13]

        # Channel 2: turn card (4th board card)
        cards[:, 2] = board_cards[:, 3]  # [M, 4, 13]

        # Channel 3: river card (5th board card)
        cards[:, 3] = board_cards[:, 4]  # [M, 4, 13]

        # Channel 4: public cards (all board cards)
        cards[:, 4] = board_cards.any(dim=1)  # [M, 4, 13]

        # Channel 5: all cards (hole + board)
        cards[:, 5] = hole_cards.any(dim=1) | board_cards.any(dim=1)  # [M, 4, 13]

        # Get action history directly from tensor environment
        # Shape: [M, 4_streets, 6_slots, 4_players, num_bet_bins]
        action_history = self.tensor_env.get_action_history()[env_indices]

        # Reshape to match ActionsHUEncoderV1 format: [M, 24_channels, 4_players, num_bet_bins]
        # Flatten streets and slots: [M, 4*6, 4, num_bet_bins] = [M, 24, 4, num_bet_bins]
        actions = action_history.view(M, 24, 4, self.num_bet_bins)

        return CNNEmbeddingData(
            cards=cards,
            actions=actions,
        )
