"""State encoder for transformer-based poker models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

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
        base_tokens += len(cls.STREETS) * history_slots  # action tokens
        base_tokens += len(cls.STREETS) * history_slots  # per-action context tokens
        return base_tokens

    # ------------------------------------------------------------------ Encoding
    def encode_tensor_states(
        self, player: int, idxs: torch.Tensor
    ) -> StructuredEmbeddingData:
        """Encode a batch of tensorized environments for the requested player."""

        batch = idxs.numel()
        if batch == 0:
            return StructuredEmbeddingData(
                token_ids=torch.empty(0, 0, device=self.device),
                card_ranks=torch.empty(0, 0, device=self.device),
                card_suits=torch.empty(0, 0, device=self.device),
                card_streets=torch.empty(0, 0, device=self.device),
                action_actors=torch.empty(0, 0, device=self.device),
                action_streets=torch.empty(0, 0, device=self.device),
                action_legal_masks=torch.empty(
                    0, 0, self.num_bet_bins, device=self.device
                ),
                context_features=torch.empty(
                    0, 0, Context.NUM_CONTEXT.value, device=self.device
                ),
                lengths=torch.empty(0, dtype=torch.long, device=self.device),
            )

        self._reset_buffers(batch)

        rows = torch.arange(batch, device=self.device)
        positions = torch.zeros(batch, dtype=torch.long, device=self.device)

        token_ids = self.token_ids[:batch]
        card_ranks = self.card_ranks[:batch]
        card_suits = self.card_suits[:batch]
        card_streets = self.card_streets[:batch]
        action_actors = self.action_actors[:batch]
        action_streets = self.action_streets[:batch]
        action_legal_masks = self.action_legal_masks[:batch]
        context_features = self.context_features[:batch]

        special_offset = self.special_offset
        context_offset = self.special_offset + Special.CONTEXT.value

        base_context_world = self._gather_env_context(idxs)
        base_context_hero = self._to_hero_context(base_context_world, player)

        # CLS token (stores blinds and hero button flag)
        token_ids[rows, positions] = special_offset + Special.CLS.value
        cf_dtype = context_features.dtype
        context_features[rows, positions, 0] = torch.full(
            (batch,), float(self.tensor_env.sb), device=self.device, dtype=cf_dtype
        )
        context_features[rows, positions, 1] = torch.full(
            (batch,), float(self.tensor_env.bb), device=self.device, dtype=cf_dtype
        )
        hero_on_button = (self.tensor_env.button[idxs] == player).to(cf_dtype)
        context_features[rows, positions, 2] = hero_on_button
        positions = positions + 1

        # Initial context token for current state
        token_ids[rows, positions] = context_offset
        context_features[rows, positions] = base_context_hero
        positions = positions + 1

        hole_cards = self.tensor_env.hole_indices[idxs, player]
        board_cards = self.tensor_env.board_indices[idxs]
        action_history = self.tensor_env.get_action_history()[idxs]
        action_context_world = self.tensor_env.action_context[idxs]

        for street_layout in self.STREETS:
            street_idx = street_layout.index

            if street_idx == 0:
                street_cards = hole_cards
            elif street_idx == 1:
                street_cards = board_cards[:, :3]
            elif street_idx == 2:
                street_cards = board_cards[:, 3:4]
            else:
                street_cards = board_cards[:, 4:5]

            has_cards = street_cards >= 0
            card_presence = (
                has_cards.any(dim=1)
                if street_cards.shape[1] > 0
                else torch.zeros(batch, dtype=torch.bool, device=self.device)
            )

            street_actions = action_history[:, street_idx]
            action_taken_mask = street_actions[:, :, 2, :].any(dim=2)
            actions_present = action_taken_mask.any(dim=1)

            street_mask = card_presence | actions_present
            if not street_mask.any():
                continue

            active_rows = torch.where(street_mask)[0]
            active_positions = positions[active_rows]
            token_ids[active_rows, active_positions] = (
                special_offset + street_layout.special.value
            )
            card_streets[active_rows, active_positions] = torch.full(
                active_positions.shape,
                street_idx,
                device=self.device,
                dtype=card_streets.dtype,
            )
            positions[active_rows] = active_positions + 1

            # Street cards
            for card_slot in range(street_cards.shape[1]):
                slot_cards = street_cards[:, card_slot]
                mask = street_mask & (slot_cards >= 0)
                if not mask.any():
                    continue
                idx = torch.where(mask)[0]
                pos = positions[idx]
                vals = slot_cards[idx]
                token_ids[idx, pos] = (self.card_offset + vals).to(token_ids.dtype)
                card_ranks[idx, pos] = (vals % 13).to(card_ranks.dtype)
                card_suits[idx, pos] = (vals // 13).to(card_suits.dtype)
                card_streets[idx, pos] = torch.full(
                    pos.shape,
                    street_idx,
                    device=self.device,
                    dtype=card_streets.dtype,
                )
                positions[idx] = pos + 1

            # Actions with per-step context
            street_context = action_context_world[:, street_idx]
            for slot in range(self.history_slots):
                slot_mask = street_mask & action_taken_mask[:, slot]
                if not slot_mask.any():
                    continue

                idx = torch.where(slot_mask)[0]
                pos = positions[idx]

                ctx_world = street_context[idx, slot]
                ctx_hero = self._to_hero_context(ctx_world, player)
                token_ids[idx, pos] = context_offset
                context_features[idx, pos] = ctx_hero
                positions[idx] = pos + 1

                action_sum = street_actions[:, slot, 2, :]
                action_ids_all = action_sum.float().argmax(dim=1)
                action_ids = action_ids_all[idx]

                actor_flags = street_actions[:, slot, :2, :].any(dim=2).float()
                actors = actor_flags.argmax(dim=1)[idx]
                if player == 1:
                    actors = 1 - actors

                legal_masks = street_actions[:, slot, 3, :][idx]

                pos = positions[idx]
                token_ids[idx, pos] = (self.action_offset + action_ids).to(
                    token_ids.dtype
                )
                action_actors[idx, pos] = actors.to(action_actors.dtype)
                action_streets[idx, pos] = torch.full(
                    pos.shape,
                    street_idx,
                    device=self.device,
                    dtype=action_streets.dtype,
                )
                action_legal_masks[idx, pos] = legal_masks
                positions[idx] = pos + 1

        self.length_buffer[:batch] = positions

        return StructuredEmbeddingData(
            token_ids=token_ids.clone(),
            card_ranks=card_ranks.clone(),
            card_suits=card_suits.clone(),
            card_streets=card_streets.clone(),
            action_actors=action_actors.clone(),
            action_streets=action_streets.clone(),
            action_legal_masks=action_legal_masks.clone(),
            context_features=context_features.clone(),
            lengths=positions.clone(),
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

    def _gather_env_context(self, idxs: torch.Tensor) -> torch.Tensor:
        env = self.tensor_env
        result = torch.zeros(
            idxs.numel(), Context.NUM_CONTEXT.value, device=self.device
        )
        target_dtype = result.dtype
        result[:, Context.POT.value].copy_(env.pot[idxs].to(target_dtype))
        result[:, Context.STACK_P0.value].copy_(env.stacks[idxs, 0].to(target_dtype))
        result[:, Context.STACK_P1.value].copy_(env.stacks[idxs, 1].to(target_dtype))
        result[:, Context.COMMITTED_P0.value].copy_(
            env.committed[idxs, 0].to(target_dtype)
        )
        result[:, Context.COMMITTED_P1.value].copy_(
            env.committed[idxs, 1].to(target_dtype)
        )
        result[:, Context.POSITION.value].copy_(env.button[idxs].to(target_dtype))
        result[:, Context.STREET.value].copy_(env.street[idxs].to(target_dtype))
        result[:, Context.ACTIONS_ROUND.value].copy_(
            env.actions_this_round[idxs].to(target_dtype)
        )
        result[:, Context.MIN_RAISE.value].copy_(env.min_raise[idxs].to(target_dtype))
        diff = env.committed[idxs, 1] - env.committed[idxs, 0]
        result[:, Context.BET_TO_CALL.value].copy_(diff.to(target_dtype))
        return result

    def _to_hero_context(
        self, context_world: torch.Tensor, player: int
    ) -> torch.Tensor:
        result = context_world.clone()
        if player == 1:
            result[:, Context.STACK_P0.value], result[:, Context.STACK_P1.value] = (
                context_world[:, Context.STACK_P1.value],
                context_world[:, Context.STACK_P0.value],
            )
            (
                result[:, Context.COMMITTED_P0.value],
                result[:, Context.COMMITTED_P1.value],
            ) = (
                context_world[:, Context.COMMITTED_P1.value],
                context_world[:, Context.COMMITTED_P0.value],
            )

        button = context_world[:, Context.POSITION.value]
        hero_not_button = (button != player).to(context_world.dtype)
        result[:, Context.POSITION.value] = hero_not_button
        result[:, Context.BET_TO_CALL.value] = (
            result[:, Context.COMMITTED_P1.value]
            - result[:, Context.COMMITTED_P0.value]
        )
        return result

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
