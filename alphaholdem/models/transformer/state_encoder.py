"""State encoder for transformer-based poker models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from ...env.hunl_env import HUNLEnv
from ...env.hunl_tensor_env import HUNLTensorEnv
from .embedding_data import StructuredEmbeddingData
from .tokens import Context, Special


@dataclass(frozen=True)
class StreetLayout:
    """Helper describing how many cards belong to each street."""

    index: int
    special: Special
    num_cards: int


class TransformerStateEncoder:
    """Convert tensorized poker state into variable-length token sequences."""

    # Order in which streets are emitted. Preflop includes private cards.
    STREETS: Tuple[StreetLayout, ...] = (
        StreetLayout(0, Special.STREET_PREFLOP, 2),
        StreetLayout(1, Special.STREET_FLOP, 3),
        StreetLayout(2, Special.STREET_TURN, 1),
        StreetLayout(3, Special.STREET_RIVER, 1),
    )

    def __init__(
        self,
        tensor_env: HUNLTensorEnv,
        device: torch.device,
    ) -> None:
        self.tensor_env = tensor_env
        self.N = tensor_env.N
        self.device = device
        self.num_bet_bins = tensor_env.num_bet_bins
        self.history_slots = tensor_env.history_slots

        # Derived constants
        self.max_sequence_length = self.get_max_sequence_length(self.history_slots)
        self.special_offset = self.get_special_token_offset(self.num_bet_bins)
        self.card_offset = self.get_card_token_offset(self.num_bet_bins)
        self.action_offset = self.get_action_token_offset(self.num_bet_bins)

        # Pre-allocate reusable buffers. Values are overwritten on each encode call.
        N, L = self.N, self.max_sequence_length
        self.token_ids = torch.full((N, L), -1, dtype=torch.int32, device=device)
        self.card_ranks = torch.full((N, L), -1, dtype=torch.int16, device=device)
        self.card_suits = torch.full((N, L), -1, dtype=torch.int16, device=device)
        self.card_streets = torch.full((N, L), -1, dtype=torch.int16, device=device)
        self.action_actors = torch.full((N, L), -1, dtype=torch.int16, device=device)
        self.action_streets = torch.full((N, L), -1, dtype=torch.int16, device=device)
        self.action_legal_masks = torch.zeros(
            N, L, self.num_bet_bins, dtype=torch.bool, device=device
        )
        self.context_features = torch.zeros(
            N, L, 10, dtype=torch.float32, device=device
        )
        self.length_buffer = torch.zeros(N, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ Helpers
    @classmethod
    def get_special_token_offset(cls, num_bet_bins: int) -> int:
        """Offset where special tokens start."""

        del num_bet_bins  # unused but kept for API compatibility
        return 0

    @classmethod
    def get_card_token_offset(cls, num_bet_bins: int) -> int:
        """Offset where card tokens start."""

        return cls.get_special_token_offset(num_bet_bins) + Special.NUM_SPECIAL.value

    @classmethod
    def get_action_token_offset(cls, num_bet_bins: int) -> int:
        """Offset where action tokens start."""

        return cls.get_card_token_offset(num_bet_bins) + 52

    @classmethod
    def get_max_sequence_length(cls, history_slots: int = 6) -> int:
        """Maximum possible sequence length for a single observation."""

        # CLS + context tokens
        base_tokens = 2
        # Street markers plus cards
        base_tokens += sum(1 + layout.num_cards for layout in cls.STREETS)
        # Maximum number of actions (history slots per street)
        base_tokens += len(cls.STREETS) * history_slots
        return base_tokens

    # ------------------------------------------------------------------ Encoding
    def encode_tensor_states(
        self, player: int, idxs: torch.Tensor
    ) -> StructuredEmbeddingData:
        """Encode a batch of tensorized environments for the requested player."""

        batch = idxs.numel()
        if batch == 0:
            return StructuredEmbeddingData(
                token_ids=self.token_ids[:0].clone(),
                card_ranks=self.card_ranks[:0].clone(),
                card_suits=self.card_suits[:0].clone(),
                card_streets=self.card_streets[:0].clone(),
                action_actors=self.action_actors[:0].clone(),
                action_streets=self.action_streets[:0].clone(),
                action_legal_masks=self.action_legal_masks[:0].clone(),
                context_features=self.context_features[:0].clone(),
                lengths=self.length_buffer[:0].clone(),
            )

        self._reset_buffers(batch)

        env = self.tensor_env
        device = self.device
        idxs = idxs.to(torch.long)

        token_ids = self.token_ids[:batch]
        card_ranks = self.card_ranks[:batch]
        card_suits = self.card_suits[:batch]
        card_streets = self.card_streets[:batch]
        action_actors = self.action_actors[:batch]
        action_streets = self.action_streets[:batch]
        action_legal_masks = self.action_legal_masks[:batch]
        context_features = self.context_features[:batch]
        lengths = self.length_buffer[:batch]

        # CLS token (game invariants)
        token_ids[:, 0] = self.special_offset + Special.CLS.value
        context_features[:, 0, 0].fill_(float(env.sb))
        context_features[:, 0, 1].fill_(float(env.bb))
        button = env.button[idxs]
        hero_on_button = (button == player).to(context_features.dtype)
        context_features[:, 0, 2] = hero_on_button

        # CONTEXT token (dynamic scalars)
        token_ids[:, 1] = self.special_offset + Special.CONTEXT.value
        opp = 1 - player
        context_features[:, 1, Context.POT.value] = env.pot[idxs].to(
            context_features.dtype
        )
        context_features[:, 1, Context.STACK_P0.value] = env.stacks[idxs, player].to(
            context_features.dtype
        )
        context_features[:, 1, Context.STACK_P1.value] = env.stacks[idxs, opp].to(
            context_features.dtype
        )
        context_features[:, 1, Context.COMMITTED_P0.value] = env.committed[
            idxs, player
        ].to(context_features.dtype)
        context_features[:, 1, Context.COMMITTED_P1.value] = env.committed[
            idxs, opp
        ].to(context_features.dtype)
        context_features[:, 1, Context.POSITION.value] = (button != player).to(
            context_features.dtype
        )
        context_features[:, 1, Context.STREET.value] = env.street[idxs].to(
            context_features.dtype
        )
        context_features[:, 1, Context.ACTIONS_ROUND.value] = env.actions_this_round[
            idxs
        ].to(context_features.dtype)
        context_features[:, 1, Context.MIN_RAISE.value] = env.min_raise[idxs].to(
            context_features.dtype
        )
        bet_to_call = env.committed[idxs, opp] - env.committed[idxs, player]
        context_features[:, 1, Context.BET_TO_CALL.value] = bet_to_call.to(
            context_features.dtype
        )

        hole_indices = env.hole_indices[idxs]
        board_indices = env.board_indices[idxs]
        action_history = env.get_action_history()[idxs]

        # Track next write position per environment
        pos = torch.full((batch,), 2, dtype=torch.long, device=device)

        for street_layout in self.STREETS:
            street_idx = street_layout.index

            if street_idx == 0:
                cards = hole_indices[:, player]
            else:
                offset, count = self._board_slice_for_street(street_layout)
                cards = board_indices[:, offset : offset + count]

            cards_valid = cards >= 0
            actions_tensor = action_history[:, street_idx]
            action_taken_mask = actions_tensor[:, :, 2].any(dim=-1)
            has_actions = action_taken_mask.any(dim=1)
            has_cards = cards_valid.any(dim=1)
            has_content = has_cards | has_actions

            active = has_content.nonzero(as_tuple=True)[0]
            if active.numel() == 0:
                continue

            marker_positions = pos[active]
            token_ids[active, marker_positions] = (
                self.special_offset + street_layout.special.value
            )
            card_streets[active, marker_positions] = street_idx
            pos[active] += 1

            card_base = pos.clone()
            if cards_valid.any():
                rows, cols = cards_valid.nonzero(as_tuple=True)
                card_values = cards[rows, cols]
                dest = card_base[rows] + cols
                token_ids[rows, dest] = self.card_offset + card_values.to(
                    token_ids.dtype
                )
                card_ranks[rows, dest] = (card_values % 13).to(card_ranks.dtype)
                card_suits[rows, dest] = (card_values // 13).to(card_suits.dtype)
                card_streets[rows, dest] = street_idx

            num_cards = cards_valid.sum(dim=1)
            pos += num_cards

            action_base = pos.clone()
            if action_taken_mask.any():
                rows, slots = action_taken_mask.nonzero(as_tuple=True)
                action_offsets = action_taken_mask.long().cumsum(dim=1) - 1
                dest = action_base[rows] + action_offsets[rows, slots]

                action_ids = actions_tensor[rows, slots, 2].float().argmax(dim=-1)
                actor_is_p0 = actions_tensor[rows, slots, 0].any(dim=-1)
                actor = (~actor_is_p0).long()
                if player == 0:
                    hero_actor = actor
                else:
                    hero_actor = 1 - actor

                token_ids[rows, dest] = self.action_offset + action_ids.to(
                    token_ids.dtype
                )
                action_actors[rows, dest] = hero_actor.to(action_actors.dtype)
                action_streets[rows, dest] = street_idx
                action_legal_masks[rows, dest] = actions_tensor[rows, slots, 3]

            num_actions = action_taken_mask.sum(dim=1)
            pos += num_actions

        lengths.copy_(pos)

        return StructuredEmbeddingData(
            token_ids=token_ids.clone(),
            card_ranks=card_ranks.clone(),
            card_suits=card_suits.clone(),
            card_streets=card_streets.clone(),
            action_actors=action_actors.clone(),
            action_streets=action_streets.clone(),
            action_legal_masks=action_legal_masks.clone(),
            context_features=context_features.clone(),
            lengths=lengths.clone(),
        )

    # ---------------------------------------------------------------- Private
    def _reset_buffers(self, batch: int) -> None:
        """Reset buffer slices that will be written this call."""

        self.token_ids[:batch].fill_(-1)
        self.card_ranks[:batch].fill_(-1)
        self.card_suits[:batch].fill_(-1)
        self.card_streets[:batch].fill_(-1)
        self.action_actors[:batch].fill_(-1)
        self.action_streets[:batch].fill_(-1)
        self.action_legal_masks[:batch].zero_()
        self.context_features[:batch].zero_()
        self.length_buffer[:batch].zero_()

    def _board_slice_for_street(self, street_layout: StreetLayout) -> Tuple[int, int]:
        """Return slice (offset, length) for board cards of a street."""

        if street_layout.index == 1:
            return 0, 3
        if street_layout.index == 2:
            return 3, 1
        return 4, 1

    # ---------------------------------------------------------------- Interface
    def encode_single_state(
        self, game_state: HUNLEnv, player: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Single-state encoding is not implemented")

    def get_sequence_length(self) -> int:
        """Return the maximum padded sequence length for this encoder."""

        return self.max_sequence_length

    def get_vocab_size(self) -> int:
        """Total number of distinct token ids."""

        return self.action_offset + self.num_bet_bins
