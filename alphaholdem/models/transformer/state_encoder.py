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
        base_tokens += len(cls.STREETS) * history_slots
        return base_tokens

    # ------------------------------------------------------------------ Encoding
    def encode_tensor_states(
        self, player: int, idxs: torch.Tensor
    ) -> StructuredEmbeddingData:
        """Encode a batch of tensorized environments for the requested player."""

        batch = idxs.numel()
        self._reset_buffers(batch)

        action_history = self.tensor_env.get_action_history()[idxs]
        lengths = self.length_buffer[:batch]

        for row, env_index in enumerate(idxs.tolist()):
            lengths[row] = self._encode_single(
                row, env_index, action_history[row], player
            )

        return StructuredEmbeddingData(
            token_ids=self.token_ids[:batch].clone(),
            card_ranks=self.card_ranks[:batch].clone(),
            card_suits=self.card_suits[:batch].clone(),
            card_streets=self.card_streets[:batch].clone(),
            action_actors=self.action_actors[:batch].clone(),
            action_streets=self.action_streets[:batch].clone(),
            action_legal_masks=self.action_legal_masks[:batch].clone(),
            context_features=self.context_features[:batch].clone(),
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

    def _encode_single(
        self,
        row: int,
        env_index: int,
        action_history: torch.Tensor,
        player: int,
    ) -> int:
        """Encode one environment row and return the produced sequence length."""

        pos = 0

        # Game-level invariants (blinds, seating) live on CLS token.
        self.token_ids[row, pos] = self.special_offset + Special.CLS.value
        self._write_game_invariants(row, pos, env_index, player)
        pos += 1

        # Dynamic observation context lives on a dedicated token.
        self.token_ids[row, pos] = self.special_offset + Special.CONTEXT.value
        self._write_dynamic_context(row, pos, env_index, player)
        pos += 1

        # Cards and actions grouped by street.
        for street_layout in self.STREETS:
            pos = self._encode_street(
                row=row,
                env_index=env_index,
                street_layout=street_layout,
                action_history=action_history[street_layout.index],
                pos=pos,
                player=player,
            )

        return pos

    def _write_game_invariants(
        self, row: int, pos: int, env_index: int, player: int
    ) -> None:
        """Populate CLS token features with blinds and hero position."""

        env = self.tensor_env
        features = self.context_features
        features[row, pos, 0] = float(env.sb)
        features[row, pos, 1] = float(env.bb)
        hero_on_button = 1.0 if int(env.button[env_index]) == player else 0.0
        features[row, pos, 2] = hero_on_button

    def _write_dynamic_context(
        self, row: int, pos: int, env_index: int, player: int
    ) -> None:
        """Populate hand-level context token with numeric scalars."""

        env = self.tensor_env
        opp = 1 - player
        features = self.context_features

        features[row, pos, Context.POT.value] = env.pot[env_index].float()
        features[row, pos, Context.STACK_P0.value] = env.stacks[
            env_index, player
        ].float()
        features[row, pos, Context.STACK_P1.value] = env.stacks[env_index, opp].float()
        features[row, pos, Context.COMMITTED_P0.value] = env.committed[
            env_index, player
        ].float()
        features[row, pos, Context.COMMITTED_P1.value] = env.committed[
            env_index, opp
        ].float()
        hero_button_flag = 0.0 if int(env.button[env_index]) == player else 1.0
        features[row, pos, Context.POSITION.value] = hero_button_flag
        features[row, pos, Context.STREET.value] = env.street[env_index].float()
        features[row, pos, Context.ACTIONS_ROUND.value] = env.actions_this_round[
            env_index
        ].float()
        features[row, pos, Context.MIN_RAISE.value] = env.min_raise[env_index].float()
        bet_to_call = env.committed[env_index, opp] - env.committed[env_index, player]
        features[row, pos, Context.BET_TO_CALL.value] = bet_to_call.float()

    def _encode_street(
        self,
        row: int,
        env_index: int,
        street_layout: StreetLayout,
        action_history: torch.Tensor,
        pos: int,
        player: int,
    ) -> int:
        """Emit street marker, visible cards, and historical actions."""

        cards = list(self._iter_street_cards(env_index, street_layout, player))
        actions = list(self._iter_actions(action_history))

        if not cards and not actions:
            return pos

        # Street marker token
        self.token_ids[row, pos] = self.special_offset + street_layout.special.value
        self.card_streets[row, pos] = street_layout.index
        pos += 1

        # Visible cards from hero perspective
        for card in cards:
            self.token_ids[row, pos] = self.card_offset + card
            self.card_ranks[row, pos] = card % 13
            self.card_suits[row, pos] = card // 13
            self.card_streets[row, pos] = street_layout.index
            pos += 1

        # Historical actions in chronological order
        for action_id, actor, legal_mask in actions:
            hero_actor = actor if player == 0 else 1 - actor
            self.token_ids[row, pos] = self.action_offset + action_id
            self.action_actors[row, pos] = hero_actor
            self.action_streets[row, pos] = street_layout.index
            self.action_legal_masks[row, pos] = legal_mask
            pos += 1

        return pos

    def _iter_street_cards(
        self, env_index: int, street_layout: StreetLayout, player: int
    ) -> Iterable[int]:
        """Yield visible card indices for the given street."""

        env = self.tensor_env
        if street_layout.index == 0:
            hole_cards = env.hole_indices[env_index, player]
            for card in hole_cards.tolist():
                if card >= 0:
                    yield int(card)
            return

        # Board cards: indices 0-2 flop, 3 turn, 4 river
        board = env.board_indices[env_index]
        offset, count = self._board_slice_for_street(street_layout)
        for idx in range(offset, offset + count):
            card = int(board[idx])
            if card >= 0:
                yield card

    def _board_slice_for_street(self, street_layout: StreetLayout) -> Tuple[int, int]:
        """Return slice (offset, length) for board cards of a street."""

        if street_layout.index == 1:
            return 0, 3
        if street_layout.index == 2:
            return 3, 1
        return 4, 1

    def _iter_actions(
        self, street_history: torch.Tensor
    ) -> Iterable[Tuple[int, int, torch.Tensor]]:
        """Yield (action_id, actor, legal_mask) tuples for populated slots."""

        for slot in range(self.history_slots):
            slot_view = street_history[slot]
            action_taken = slot_view[2]
            if not action_taken.any():
                break

            # Determine the actor by checking which player's row fired.
            actor = 0 if slot_view[0].any() else 1
            action_id = slot_view[2].float().argmax().item()
            legal_mask = slot_view[3].clone()
            yield int(action_id), actor, legal_mask

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
