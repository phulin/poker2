"""State encoder for transformer-based poker models."""

from __future__ import annotations

from typing import Tuple

import torch

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.embedding_data import StructuredEmbeddingData
from alphaholdem.models.transformer.tokens import Context, Special


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

        # CLS token
        token_ids[:, 0] = self.special_offset + Special.GAME.value
        context_dtype = context_features.dtype
        context_features[:, 0, 0].fill_(float(env.sb))
        context_features[:, 0, 1].fill_(float(env.bb))
        button = env.button[idxs]
        context_features[:, 0, 2] = (button == player).to(context_dtype)

        # CONTEXT token scalars
        opp = 1 - player
        token_ids[:, 1] = self.special_offset + Special.CONTEXT.value
        context_features[:, 1, Context.POT.value] = env.pot[idxs].to(context_dtype)
        context_features[:, 1, Context.STACK_P0.value] = env.stacks[idxs, player].to(
            context_dtype
        )
        context_features[:, 1, Context.STACK_P1.value] = env.stacks[idxs, opp].to(
            context_dtype
        )
        context_features[:, 1, Context.COMMITTED_P0.value] = env.committed[
            idxs, player
        ].to(context_dtype)
        context_features[:, 1, Context.COMMITTED_P1.value] = env.committed[
            idxs, opp
        ].to(context_dtype)
        context_features[:, 1, Context.POSITION.value] = (button != player).to(
            context_dtype
        )
        context_features[:, 1, Context.STREET.value] = env.street[idxs].to(
            context_dtype
        )
        context_features[:, 1, Context.ACTIONS_ROUND.value] = env.actions_this_round[
            idxs
        ].to(context_dtype)
        context_features[:, 1, Context.MIN_RAISE.value] = env.min_raise[idxs].to(
            context_dtype
        )
        bet_to_call = env.committed[idxs, opp] - env.committed[idxs, player]
        context_features[:, 1, Context.BET_TO_CALL.value] = bet_to_call.to(
            context_dtype
        )

        num_streets = len(self.STREETS)
        street_indices = torch.arange(num_streets, device=device)
        street_special = torch.tensor(
            [layout.special.value for layout in self.STREETS],
            device=device,
            dtype=token_ids.dtype,
        )

        hole_indices = env.hole_indices[idxs]
        board_indices = env.board_indices[idxs]
        action_history = env.get_action_history()[idxs]

        max_cards = max(layout.num_cards for layout in self.STREETS)
        cards_tensor = torch.full(
            (batch, num_streets, max_cards),
            -1,
            dtype=torch.long,
            device=device,
        )
        cards_tensor[:, 0, :2] = hole_indices[:, player]
        cards_tensor[:, 1, :3] = board_indices[:, 0:3]
        cards_tensor[:, 2, 0] = board_indices[:, 3]
        cards_tensor[:, 3, 0] = board_indices[:, 4]

        cards_valid = cards_tensor >= 0
        card_counts = cards_valid.sum(dim=2)

        actions_tensor = action_history
        action_taken_mask = actions_tensor[:, :, :, 2].any(dim=-1)
        action_counts = action_taken_mask.sum(dim=2)

        marker_mask = (card_counts > 0) | (action_counts > 0)
        marker_counts = marker_mask.long()
        card_counts_long = card_counts.to(torch.long)
        action_counts_long = action_counts.to(torch.long)

        street_token_counts = marker_counts + card_counts_long + action_counts_long
        street_start = 2 + torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.long, device=device),
                torch.cumsum(street_token_counts[:, :-1], dim=1),
            ],
            dim=1,
        )

        lengths.copy_((2 + street_token_counts.sum(dim=1)).to(lengths.dtype))

        batch_ids = torch.arange(batch, device=device)

        # Street markers
        if marker_mask.any():
            marker_batch = batch_ids.unsqueeze(1).expand(-1, num_streets)[marker_mask]
            marker_pos = street_start[marker_mask]
            marker_streets = street_indices.unsqueeze(0).expand(batch, -1)[marker_mask]
            token_ids[marker_batch, marker_pos] = (
                self.special_offset + street_special[marker_streets]
            )
            card_streets[marker_batch, marker_pos] = marker_streets.to(
                card_streets.dtype
            )

        # Cards
        if cards_valid.any():
            card_offsets = torch.arange(max_cards, device=device)
            card_positions = (
                street_start.unsqueeze(-1) + marker_counts.unsqueeze(-1) + card_offsets
            )

            rows, streets, card_idx = cards_valid.nonzero(as_tuple=True)
            dest = card_positions[rows, streets, card_idx]
            card_values = cards_tensor[rows, streets, card_idx]
            token_ids[rows, dest] = self.card_offset + card_values.to(token_ids.dtype)
            card_ranks[rows, dest] = (card_values % 13).to(card_ranks.dtype)
            card_suits[rows, dest] = (card_values // 13).to(card_suits.dtype)
            card_streets[rows, dest] = streets.to(card_streets.dtype)

        # Actions
        if action_taken_mask.any():
            action_offsets = action_taken_mask.long().cumsum(dim=2) - 1
            action_positions = (
                street_start.unsqueeze(-1)
                + marker_counts.unsqueeze(-1)
                + card_counts_long.unsqueeze(-1)
                + action_offsets
            )

            action_rows, action_streets_idx, action_slot = action_taken_mask.nonzero(
                as_tuple=True
            )
            dest = action_positions[action_rows, action_streets_idx, action_slot]

            action_ids_all = actions_tensor[:, :, :, 2].float().argmax(dim=-1)
            action_ids = action_ids_all[action_rows, action_streets_idx, action_slot]

            actor_is_p0 = actions_tensor[:, :, :, 0].any(dim=-1)
            actor = torch.where(
                actor_is_p0,
                torch.zeros_like(actor_is_p0, dtype=torch.long),
                torch.ones_like(actor_is_p0, dtype=torch.long),
            )
            hero_actor = actor[action_rows, action_streets_idx, action_slot]
            if player != 0:
                hero_actor = 1 - hero_actor

            token_ids[action_rows, dest] = self.action_offset + action_ids.to(
                token_ids.dtype
            )
            action_actors[action_rows, dest] = hero_actor.to(action_actors.dtype)
            action_streets[action_rows, dest] = action_streets_idx.to(
                action_streets.dtype
            )
            action_legal_masks[action_rows, dest] = actions_tensor[
                action_rows, action_streets_idx, action_slot, 3
            ]

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
