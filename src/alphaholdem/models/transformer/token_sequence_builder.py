from typing import Optional, Union

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.tokens import (
    GAME_INDEX,
    HOLE0_INDEX,
    HOLE1_INDEX,
    Context,
    Game,
    Special,
    get_action_token_id_offset,
    get_card_token_id_offset,
)


class TokenSequenceBuilder:
    """Vectorized staging buffer for building transformer token sequences per trajectory.

    Usage:
      - Instantiate once per collection batch with capacity, sequence_length, device, num_bet_bins
      - Call add_context/add_action/add_card with (traj_indices, ...)
      - Call commit(replay_buffer, traj_indices) to append staged tokens into the buffer
    """

    tensor_env: HUNLTensorEnv
    sequence_length: int
    bet_bins: list[float]
    device: torch.device
    float_dtype: torch.dtype

    token_ids: torch.Tensor
    token_streets: torch.Tensor
    card_ranks: torch.Tensor
    card_suits: torch.Tensor
    action_actors: torch.Tensor
    action_legal_masks: torch.Tensor
    action_amounts: torch.Tensor
    context_features: torch.Tensor
    lengths: torch.Tensor

    def __init__(
        self,
        tensor_env: HUNLTensorEnv,
        sequence_length: int,
        bet_bins: list[float],
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> None:
        self.tensor_env = tensor_env
        self.sequence_length = sequence_length
        self.bet_bins = bet_bins
        self.num_bet_bins = len(bet_bins) + 3
        self.device = device
        self.float_dtype = float_dtype

        L = sequence_length
        N = tensor_env.N
        self.token_ids = torch.full((N, L), -1, dtype=torch.long, device=device)
        self.token_streets = torch.zeros(N, L, dtype=torch.long, device=device)
        self.card_ranks = torch.zeros(N, L, dtype=torch.long, device=device)
        self.card_suits = torch.zeros(N, L, dtype=torch.long, device=device)
        self.action_actors = torch.zeros(N, L, dtype=torch.long, device=device)
        self.action_legal_masks = torch.zeros(
            N, L, self.num_bet_bins, dtype=torch.bool, device=device
        )
        self.action_amounts = torch.zeros(N, L, dtype=torch.long, device=device)
        self.context_features = torch.zeros(
            N, L, Context.NUM_RAW_CONTEXT.value, dtype=torch.int16, device=device
        )
        self.lengths = torch.zeros(N, dtype=torch.long, device=device)

        self.add_cls()

    def _reserve(self, idxs: Union[torch.Tensor, slice], k: int) -> torch.Tensor:
        pos = self.lengths[idxs]
        if isinstance(idxs, slice):
            pos = pos.clone()
        if ((pos + k) > self.sequence_length).any():
            raise ValueError("Token sequence exceeds configured sequence length")
        start = pos
        self.lengths[idxs] = pos + k
        return start

    def add_context(self, idxs: torch.Tensor) -> None:
        if idxs.numel() == 0:
            return
        player = 0
        opp = 1 - player
        start = self._reserve(idxs, 1)
        self.token_ids[idxs, start] = Special.CONTEXT.value
        self.token_streets[idxs, start] = self.tensor_env.street[idxs]

        # store current legal mask (shown on context + on action itself later)
        self.action_legal_masks[idxs, start] = self.tensor_env.legal_bins_mask(
            self.bet_bins
        )[idxs]

        # Store raw unscaled values as int16
        self.context_features[idxs, start, Context.POT.value] = self.tensor_env.pot[
            idxs
        ].to(torch.int16)
        self.context_features[idxs, start, Context.STACK_P0.value] = (
            self.tensor_env.stacks[idxs, player].to(torch.int16)
        )
        self.context_features[idxs, start, Context.STACK_P1.value] = (
            self.tensor_env.stacks[idxs, opp].to(torch.int16)
        )
        self.context_features[idxs, start, Context.COMMITTED_P0.value] = (
            self.tensor_env.committed[idxs, player].to(torch.int16)
        )
        self.context_features[idxs, start, Context.COMMITTED_P1.value] = (
            self.tensor_env.committed[idxs, opp].to(torch.int16)
        )
        self.context_features[idxs, start, Context.POSITION.value] = (
            # position 0 is the button, 1 is BB
            (self.tensor_env.button[idxs] != self.tensor_env.to_act[idxs]).to(
                torch.int16
            )
        )
        self.context_features[idxs, start, Context.ACTIONS_ROUND.value] = (
            self.tensor_env.actions_this_round[idxs].to(torch.int16)
        )
        self.context_features[idxs, start, Context.MIN_RAISE.value] = (
            self.tensor_env.min_raise[idxs].to(torch.int16)
        )
        bet_to_call = (
            self.tensor_env.committed[idxs, opp]
            - self.tensor_env.committed[idxs, player]
        )
        self.context_features[idxs, start, Context.BET_TO_CALL.value] = bet_to_call.to(
            torch.int16
        )

    def add_cls(self, idxs: Optional[torch.Tensor] = None) -> None:
        if idxs is not None and idxs.numel() == 0:
            return

        select = idxs if idxs is not None else slice(None)

        start = self._reserve(select, 1)
        self.token_ids[select, start] = Special.CLS.value

    def add_game(self, idxs: torch.Tensor) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        self.token_ids[idxs, start] = Special.GAME.value
        # GAME slot (index 1) - store raw values as int16
        self.context_features[idxs, start, Game.SB.value] = self.tensor_env.sb
        self.context_features[idxs, start, Game.BB.value] = self.tensor_env.bb
        self.context_features[idxs, start, Game.HERO_POSITION.value] = (
            self.tensor_env.button[idxs] == 1
        ).to(torch.int16)

    def add_action(
        self,
        idxs: torch.Tensor,
        actors: torch.Tensor,
        action_ids: torch.Tensor,
        legal_masks: torch.Tensor,
        action_amounts: torch.Tensor,
        token_streets: torch.Tensor,
    ) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        self.token_ids[idxs, start] = Special.NUM_SPECIAL.value + 52 + action_ids
        self.token_streets[idxs, start] = token_streets
        self.action_actors[idxs, start] = actors
        self.action_legal_masks[idxs, start, :] = legal_masks
        self.action_amounts[idxs, start] = action_amounts

    def add_card(
        self,
        idxs: torch.Tensor,
        card_indices: torch.Tensor,
    ) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        token_vals = (Special.NUM_SPECIAL.value + card_indices).to(torch.long)
        self.token_ids[idxs, start] = token_vals
        self.token_streets[idxs, start] = self.tensor_env.street[idxs]
        self.card_ranks[idxs, start] = card_indices % 13
        self.card_suits[idxs, start] = card_indices // 13

    def add_street(self, idxs: torch.Tensor, street_ids: torch.Tensor) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        special_tokens = Special.STREET_PREFLOP.value + street_ids.to(
            torch.int64
        )  # assumes 0=preflop,1=flop,2=turn,3=river
        self.token_ids[idxs, start] = special_tokens.to(torch.long)
        self.token_streets[idxs, start] = street_ids.to(torch.long)

    def encode_tensor_states(
        self,
        player: int,
        idxs: torch.Tensor,
    ) -> StructuredEmbeddingData:
        """Build minimal context features (CLS + CONTEXT) from the live tensor env.

        This mirrors the transformer state encoder's context computation so the
        TSB can act as the state-encoder during rollout.
        """

        result = StructuredEmbeddingData(
            token_ids=self.token_ids[idxs],
            token_streets=self.token_streets[idxs],
            card_ranks=self.card_ranks[idxs],
            card_suits=self.card_suits[idxs],
            action_actors=self.action_actors[idxs],
            action_legal_masks=self.action_legal_masks[idxs],
            action_amounts=self.action_amounts[idxs],
            context_features=self.context_features[idxs],
            lengths=self.lengths[idxs],
        )

        if player == 1:
            # hero_on_button flag in CLS (binary)
            button = self.tensor_env.button[idxs]
            result.context_features[:, GAME_INDEX, Game.HERO_POSITION.value] = (
                button == int(player)
            ).to(self.float_dtype)

            # Set hole cards
            result.token_ids[:, HOLE0_INDEX] = (
                get_card_token_id_offset()
                + self.tensor_env.hole_indices[idxs, player, 0]
            )
            result.card_ranks[:, HOLE0_INDEX] = (
                self.tensor_env.hole_indices[idxs, player, 0] % 13
            )
            result.card_suits[:, HOLE0_INDEX] = (
                self.tensor_env.hole_indices[idxs, player, 0] // 13
            )
            result.token_ids[:, HOLE1_INDEX] = (
                get_card_token_id_offset()
                + self.tensor_env.hole_indices[idxs, player, 1]
            )
            result.card_ranks[:, HOLE1_INDEX] = (
                self.tensor_env.hole_indices[idxs, player, 1] % 13
            )
            result.card_suits[:, HOLE1_INDEX] = (
                self.tensor_env.hole_indices[idxs, player, 1] // 13
            )

            result.context_features[
                :,
                1:,  # skip CLS token
                [
                    Context.STACK_P0.value,
                    Context.STACK_P1.value,
                    Context.COMMITTED_P0.value,
                    Context.COMMITTED_P1.value,
                ],
            ] = result.context_features[
                :,
                1:,  # skip CLS token
                [
                    Context.STACK_P1.value,
                    Context.STACK_P0.value,
                    Context.COMMITTED_P1.value,
                    Context.COMMITTED_P0.value,
                ],
            ]

            # Reverse all CONTEXT bet_to_call values. Will be 0 for non-context tokens, so this is fine.
            result.context_features[
                :, :, Context.BET_TO_CALL.value
            ] = -result.context_features[:, :, Context.BET_TO_CALL.value]

            # Reverse all CONTEXT position values.
            rows, cols = torch.where(result.token_ids == Special.CONTEXT.value)
            result.context_features[rows, cols, Context.POSITION.value] = (
                1 - result.context_features[rows, cols, Context.POSITION.value]
            )

            # Reverse all actor indices, but only where the token is an action.
            action_token_mask = result.token_ids >= get_action_token_id_offset()
            result.action_actors[action_token_mask] = (
                1 - result.action_actors[action_token_mask]
            )

        return result

    def reset(self, idxs: Optional[torch.Tensor] = None) -> None:
        if idxs is not None and idxs.numel() == 0:
            return

        select = idxs if idxs is not None else slice(None)

        self.token_ids[select] = -1
        self.token_streets[select] = 0
        self.card_ranks[select] = 0
        self.card_suits[select] = 0
        self.action_actors[select] = 0
        self.action_legal_masks[select] = False
        self.context_features[select] = 0
        self.lengths[select] = 0

        self.add_cls(idxs)

    def clone_tokens(
        self, dst_children: torch.Tensor, src_parents: torch.Tensor
    ) -> None:
        """Clone token buffers for a set of rows (vectorized)."""
        if dst_children.numel() == 0:
            return
        assert dst_children.shape[0] == src_parents.shape[0]
        self.token_ids[dst_children] = self.token_ids[src_parents]
        self.token_streets[dst_children] = self.token_streets[src_parents]
        self.card_ranks[dst_children] = self.card_ranks[src_parents]
        self.card_suits[dst_children] = self.card_suits[src_parents]
        self.action_actors[dst_children] = self.action_actors[src_parents]
        self.action_legal_masks[dst_children] = self.action_legal_masks[src_parents]
        self.action_amounts[dst_children] = self.action_amounts[src_parents]
        self.context_features[dst_children] = self.context_features[src_parents]
        self.lengths[dst_children] = self.lengths[src_parents]

    def copy_from_structured(
        self, dst_indices: torch.Tensor, data: StructuredEmbeddingData
    ) -> None:
        """Copy token streams from a StructuredEmbeddingData batch into dst rows."""
        if dst_indices.numel() == 0:
            return
        if dst_indices.shape[0] != len(data):
            raise ValueError(
                "dst_indices and StructuredEmbeddingData batch size must match"
            )

        self.token_ids[dst_indices] = data.token_ids.to(self.token_ids.dtype)
        self.token_streets[dst_indices] = data.token_streets.to(
            self.token_streets.dtype
        )
        self.card_ranks[dst_indices] = data.card_ranks.to(self.card_ranks.dtype)
        self.card_suits[dst_indices] = data.card_suits.to(self.card_suits.dtype)
        self.action_actors[dst_indices] = data.action_actors.to(
            self.action_actors.dtype
        )
        self.action_legal_masks[dst_indices] = data.action_legal_masks.to(
            self.action_legal_masks.dtype
        )
        self.action_amounts[dst_indices] = data.action_amounts.to(
            self.action_amounts.dtype
        )
        self.context_features[dst_indices] = data.context_features.to(
            self.context_features.dtype
        )
        self.lengths[dst_indices] = data.lengths.to(self.lengths.dtype)
