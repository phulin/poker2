"""State encoding utilities for poker models."""

from __future__ import annotations

from typing import Tuple

import torch

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData


class CNNStateEncoder:
    """Encodes poker game states into model inputs."""

    def __init__(
        self,
        tensor_env: HUNLTensorEnv,
        device: torch.device,
        num_bet_bins: int,
    ):
        self.tensor_env = tensor_env
        self.device = device
        self.num_bet_bins = num_bet_bins

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

        hole_cards = self.tensor_env.hole_onehot[env_indices, player]  # [M, 2, 4, 13]
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
        # TODO: Implement proper action history tracking in tensor environment
        action_history = torch.zeros(M, 4, 6, 4, self.num_bet_bins, device=self.device)

        # Reshape to match ActionsHUEncoderV1 format: [M, 24_channels, 4_players, num_bet_bins]
        # Flatten streets and slots: [M, 4*6, 4, num_bet_bins] = [M, 24, 4, num_bet_bins]
        actions = action_history.view(M, 24, 4, self.num_bet_bins)
        if player == 1:
            # Always present the state to the model as if the model is player 0.
            actions = actions.clone()
            actions[:, :, [0, 1], :] = actions[:, :, [1, 0], :]

        return CNNEmbeddingData(
            cards=cards,
            actions=actions,
        )

    def encode_single_state(
        self, game_state: HUNLEnv, seat: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a single game state for CNN model.

        Args:
            game_state: Single HUNLEnv game state
            seat: Player index (0 or 1)

        Returns:
            Tuple of (cards_tensor, actions_tensor)
        """
        # Create a temporary tensor environment with this state
        # We'll need to manually construct the tensors from the game state

        # Convert hole cards to one-hot
        hole_cards = torch.zeros(2, 4, 13, dtype=torch.bool, device=self.device)
        for i, card in enumerate(game_state.players[seat].hole_cards):
            if card is not None:
                suit = card // 13
                rank = card % 13
                hole_cards[i, suit, rank] = True

        # Convert board cards to one-hot
        board_cards = torch.zeros(5, 4, 13, dtype=torch.bool, device=self.device)
        for i, card in enumerate(game_state.board):
            if card is not None:
                suit = card // 13
                rank = card % 13
                board_cards[i, suit, rank] = True

        # Create cards tensor [6, 4, 13]
        cards = torch.zeros(6, 4, 13, dtype=torch.bool, device=self.device)

        # Channel 0: hole cards (sum over 2 hole cards)
        cards[0] = hole_cards.any(dim=0)

        # Channel 1: flop cards (first 3 board cards)
        cards[1] = board_cards[:3].any(dim=0)

        # Channel 2: turn card (4th board card)
        cards[2] = board_cards[3]

        # Channel 3: river card (5th board card)
        cards[4] = board_cards[4]

        # Channel 4: public cards (all board cards)
        cards[4] = board_cards.any(dim=0)

        # Channel 5: all cards (hole + board)
        cards[5] = hole_cards.any(dim=0) | board_cards.any(dim=0)

        # Create actions tensor [24, 4, num_bet_bins]
        # For now, create empty actions tensor - this would need to be populated
        # with actual action history from the game state
        actions = torch.zeros(
            24, 4, self.num_bet_bins, dtype=torch.bool, device=self.device
        )

        return cards, actions
