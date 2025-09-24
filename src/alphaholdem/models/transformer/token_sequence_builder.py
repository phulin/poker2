from typing import Optional, Union

import torch

from ...env.hunl_tensor_env import HUNLTensorEnv
from .structured_embedding_data import StructuredEmbeddingData
from .tokens import (
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
    num_bet_bins: int
    device: torch.device
    float_dtype: torch.dtype

    token_ids: torch.Tensor
    token_streets: torch.Tensor
    card_ranks: torch.Tensor
    card_suits: torch.Tensor
    action_actors: torch.Tensor
    action_legal_masks: torch.Tensor
    context_features: torch.Tensor
    lengths: torch.Tensor

    def __init__(
        self,
        tensor_env: HUNLTensorEnv,
        sequence_length: int,
        num_bet_bins: int,
        device: torch.device,
        float_dtype: torch.dtype,
    ) -> None:
        self.tensor_env = tensor_env
        self.sequence_length = sequence_length
        self.num_bet_bins = num_bet_bins
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
            N, L, num_bet_bins, dtype=torch.bool, device=device
        )
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
        token_streets: torch.Tensor,
    ) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        self.token_ids[idxs, start] = Special.NUM_SPECIAL.value + 52 + action_ids
        self.token_streets[idxs, start] = token_streets
        self.action_actors[idxs, start] = actors
        self.action_legal_masks[idxs, start, :] = legal_masks

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
            result.context_features[:, :, Context.BET_TO_CALL.value] = (
                -result.context_features[:, :, Context.BET_TO_CALL.value]
            )

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

    def reset_envs(self, idxs: Optional[torch.Tensor] = None) -> None:
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
