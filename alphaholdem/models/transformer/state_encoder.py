"""State encoder for transformer-based poker models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...env.hunl_env import HUNLEnv
from ...env.hunl_tensor_env import HUNLTensorEnv
from ...utils.profiling import profile
from .embedding_data import StructuredEmbeddingData
from .tokens import Context, Special


class TransformerStateEncoder:
    """State encoder for transformer models using token-based representation.

    This encoder converts poker game states into structured embeddings suitable for
    transformer processing, leveraging the 0-51 card indices from HUNLTensorEnv.
    """

    def __init__(
        self,
        tensor_env: HUNLTensorEnv,
        device: torch.device,
    ):
        self.tensor_env = tensor_env
        self.N = tensor_env.N
        self.device = device
        self.num_bet_bins = tensor_env.num_bet_bins

        self.stages = torch.tensor([0, 0, 1, 1, 1, 2, 3], device=self.device)
        self.arange_context = torch.arange(
            Context.NUM_CONTEXT.value, device=self.device
        )

        # Token ID offsets
        self.special_offset = 0
        self.card_token_offset = self.special_offset + Special.NUM_SPECIAL.value
        self.action_token_offset = self.card_token_offset + 52
        self.context_token_offset = self.action_token_offset + self.num_bet_bins

        # Index offsets
        self.cls_index_offset = 0
        self.card_index_offset = self.cls_index_offset + 1
        self.action_index_offset = self.card_index_offset + 7
        self.context_index_offset = self.action_index_offset + 24
        self.L = self.context_index_offset + Context.NUM_CONTEXT.value

        N, L = self.N, self.L

        # Preallocate tensors
        self.token_ids = torch.full((N, L), -1, dtype=torch.int8, device=device)

        self.card_ranks = torch.full((N, L), -1, dtype=torch.uint8, device=device)
        self.card_suits = torch.full((N, L), -1, dtype=torch.uint8, device=device)
        self.card_streets = torch.full((N, L), -1, dtype=torch.uint8, device=device)

        self.action_actors = torch.zeros(N, L, dtype=torch.uint8, device=device)
        self.action_streets = torch.zeros(N, L, dtype=torch.uint8, device=device)
        self.action_legal_masks = torch.zeros(
            N, L, self.num_bet_bins, dtype=torch.bool, device=device
        )

        # Consolidated context tensor [N, L, 10]
        self.context_features = torch.zeros(N, L, 10, dtype=torch.long, device=device)

    def encode_tensor_states(
        self, player: int, idxs: torch.Tensor
    ) -> StructuredEmbeddingData:
        """Encode states for all environments in the tensorized environment.

        Args:
            player: Player index (0 or 1)

        Returns:
            StructuredEmbeddingData containing all embedding components
        """
        # M < N is the number of environments we're encoding
        M = idxs.numel()

        # Add CLS token for all environments (position 0)
        self.token_ids[:, 0] = 52  # Special CLS token index

        # Process all components
        self._process_cards_vectorized(player, idxs)
        self._process_actions_vectorized(player, idxs)
        self._process_context_vectorized(
            player,
            idxs,
        )

        return StructuredEmbeddingData(
            token_ids=self.token_ids[:M].clone(),
            card_ranks=self.card_ranks[:M].clone(),
            card_suits=self.card_suits[:M].clone(),
            card_streets=self.card_streets[:M].clone(),
            action_actors=self.action_actors[:M].clone(),
            action_streets=self.action_streets[:M].clone(),
            action_legal_masks=self.action_legal_masks[:M].clone(),
            context_features=self.context_features[:M].clone(),
        )

    def _process_cards_vectorized(
        self,
        player: int,
        idxs: torch.Tensor,
    ) -> None:
        """Process cards for all environments in a vectorized manner."""
        M = idxs.numel()
        k = self.card_index_offset

        # Get visible cards for this player across all environments
        player_hole = self.tensor_env.hole_indices[
            idxs, player, :
        ]  # [N, 2] - select player and all their cards
        board = self.tensor_env.board_indices[idxs]  # [N, 5]

        # Combine hole and board cards for all environments
        visible_cards = torch.cat([player_hole, board], dim=1)  # [N, 7]
        self.token_ids[:M, k : k + 7] = visible_cards
        self.card_ranks[:M, k : k + 7] = torch.where(
            visible_cards >= 0, visible_cards % 13, -1
        )
        self.card_suits[:M, k : k + 7] = torch.where(
            visible_cards >= 0, visible_cards // 13, -1
        )
        self.card_streets[:M, k : k + 7] = torch.where(
            visible_cards >= 0, self.stages.view(1, 7), -1
        )

    def _process_actions_vectorized(
        self,
        player: int,
        idxs: torch.Tensor,
    ) -> None:
        """Process actions for all environments in a vectorized manner."""
        N = self.N
        M = idxs.numel()
        k = self.action_index_offset

        # Encode the whole action history.
        action_history = self.tensor_env.get_action_history().view(
            N, 24, 4, self.num_bet_bins
        )[idxs]

        # Determine which player acted in each slot using argmax
        # action_history[:, :, :2, :] gives us [M, 24, 2, num_bet_bins]
        # .any(dim=3) gives us [M, 24, 2] - whether each player acted
        action_count = action_history[:, :, :2, :].any(dim=3).float()  # [M, 24, 2]
        # .argmax(dim=2) gives us [M, 24] - which player acted (0 or 1)
        action_actor = action_count.argmax(dim=2)  # [M, 24]

        if player == 1:
            # Always present the state to the model as if we are player 0.
            action_actor = 1 - action_actor

        action_id = action_history[:, :, 3, :].float().argmax(dim=2)  # [N, 24]

        # Fill all 24 action slots (NO available_slots logic. DO NOT CHANGE THIS.)
        self.token_ids[:M, k : k + 24] = self.action_token_offset + action_id
        self.action_actors[:M, k : k + 24] = action_actor
        self.action_streets[:M, k : k + 24] = torch.arange(24, device=self.device) // 6
        self.action_legal_masks[:M, k : k + 24] = action_history[
            :, :, 3, :
        ]  # [N, 24, 8]

    def _process_context_vectorized(
        self,
        player: int,
        idxs: torch.Tensor,
    ) -> None:
        """Process context for all environments in a vectorized manner."""
        M = idxs.numel()
        k = self.context_index_offset

        self.token_ids[:M, k : k + Context.NUM_CONTEXT.value] = (
            self.context_token_offset + self.arange_context
        )

        # Consolidate all context features into a single tensor [M, 10]
        context_features = torch.stack(
            [
                self.tensor_env.pot[idxs].long(),  # 0: pot size
                self.tensor_env.stacks[idxs, 0].long(),  # 1: our stack
                self.tensor_env.stacks[idxs, 1].long(),  # 2: opponent stack
                self.tensor_env.committed[idxs, 0].long(),  # 3: our committed
                self.tensor_env.committed[idxs, 1].long(),  # 4: opponent committed
                torch.where(
                    self.tensor_env.button[idxs] == player, 0, 1
                ),  # 5: position
                self.tensor_env.street[idxs].long(),  # 6: street
                self.tensor_env.actions_this_round[
                    idxs
                ].long(),  # 7: actions this round
                self.tensor_env.min_raise[idxs].long(),  # 8: min raise
                (
                    self.tensor_env.committed[idxs, 1 - player]
                    - self.tensor_env.committed[idxs, player]
                ).long(),  # 9: bet to call
            ],
            dim=1,
        )  # [M, 10]

        self.context_features[:M, k, :] = context_features

    def encode_single_state(
        self, game_state: HUNLEnv, player: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Not implemented for transformer")

    def get_sequence_length(self) -> int:
        """Get maximum sequence length."""
        return self.L

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.context_token_offset + Context.NUM_CONTEXT.value
