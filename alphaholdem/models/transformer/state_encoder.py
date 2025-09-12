"""State encoder for transformer-based poker models."""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List, Any
from line_profiler import profile
import torch
from enum import Enum

from ...env.hunl_tensor_env import HUNLTensorEnv
from ...env.hunl_env import HUNLEnv
from .embedding_data import StructuredEmbeddingData


# not using most of these for now
class Special(Enum):
    CLS = 0
    SEP = 1
    MASK = 2
    PAD = 3
    NUM_SPECIAL = 4


class Context(Enum):
    POT = 0
    STACK_P0 = 1
    STACK_P1 = 2
    COMMITTED_P0 = 3
    COMMITTED_P1 = 4
    POSITION = 5
    STREET = 6
    ACTIONS_ROUND = 7
    MIN_RAISE = 8
    BET_TO_CALL = 9
    NUM_CONTEXT = 10


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
        self.token_ids = torch.full((N, L), -1, dtype=torch.long, device=device)

        self.card_ranks = torch.full((N, L), -1, dtype=torch.long, device=device)
        self.card_suits = torch.full((N, L), -1, dtype=torch.long, device=device)
        self.card_stages = torch.full((N, L), -1, dtype=torch.long, device=device)

        self.action_actors = torch.zeros(N, L, dtype=torch.long, device=device)
        self.action_streets = torch.zeros(N, L, dtype=torch.long, device=device)
        self.action_legal_masks = torch.zeros(N, L, 8, dtype=torch.bool, device=device)

        self.context_pot_sizes = torch.zeros(N, L, 1, dtype=torch.float, device=device)
        self.context_stack_sizes = torch.zeros(
            N, L, 2, dtype=torch.float, device=device
        )
        self.context_committed_sizes = torch.zeros(
            N, L, 2, dtype=torch.float, device=device
        )
        self.context_positions = torch.zeros(N, L, dtype=torch.long, device=device)
        self.context_street = torch.zeros(N, L, 4, dtype=torch.float, device=device)
        self.context_actions_this_round = torch.zeros(
            N, L, dtype=torch.long, device=device
        )
        self.context_min_raise = torch.zeros(N, L, dtype=torch.float, device=device)
        self.context_bet_to_call = torch.zeros(N, L, dtype=torch.float, device=device)

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
            card_indices=self.token_ids[:M],
            card_stages=self.card_stages[:M],
            action_actors=self.action_actors[:M],
            action_streets=self.action_streets[:M],
            action_legal_masks=self.action_legal_masks[:M],
            context_pot_sizes=self.context_pot_sizes[:M],
            context_stack_sizes=self.context_stack_sizes[:M],
            context_committed_sizes=self.context_committed_sizes[:M],
            context_positions=self.context_positions[:M],
            context_street=self.context_street[:M],
            context_actions_this_round=self.context_actions_this_round[:M],
            context_min_raise=self.context_min_raise[:M],
            context_bet_to_call=self.context_bet_to_call[:M],
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
        self.card_stages[:M, k : k + 7] = torch.where(
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
        action_count = action_history[:, :, :2, :].any(dim=3).float()  # [N, 24, 2]
        action_actor = action_count.argmax(dim=2)  # [N, 24]
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
        self.context_pot_sizes[:M, k, 0] = self.tensor_env.pot[idxs].float()
        self.context_stack_sizes[:M, k, :] = self.tensor_env.stacks[idxs].float()
        self.context_committed_sizes[:M, k, :] = self.tensor_env.committed[idxs].float()
        self.context_positions[:M, k] = torch.where(
            self.tensor_env.button[idxs] == player, 0, 1
        )
        # Create street context tensor with 4 dimensions
        street_context = torch.stack(
            [
                self.tensor_env.street[idxs].float(),
                self.tensor_env.actions_this_round[idxs].float(),
                self.tensor_env.min_raise[idxs].float(),
                (
                    self.tensor_env.committed[idxs, 1 - player]
                    - self.tensor_env.committed[idxs, player]
                ).float(),
            ],
            dim=1,
        )  # [M, 4]
        self.context_street[:M, k, :] = street_context

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
