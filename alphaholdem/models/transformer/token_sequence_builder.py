import torch

from ...env.hunl_tensor_env import HUNLTensorEnv
from .tokens import (
    Special,
    Context,
    Cls,
    get_card_token_id_offset,
    HOLE0_INDEX,
    HOLE1_INDEX,
)
from .embedding_data import StructuredEmbeddingData


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
    card_ranks: torch.Tensor
    card_suits: torch.Tensor
    card_streets: torch.Tensor
    action_actors: torch.Tensor
    action_streets: torch.Tensor
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
        self.card_ranks = torch.zeros(N, L, dtype=torch.long, device=device)
        self.card_suits = torch.zeros(N, L, dtype=torch.long, device=device)
        self.card_streets = torch.zeros(N, L, dtype=torch.long, device=device)
        self.action_actors = torch.full((N, L), -1, dtype=torch.long, device=device)
        self.action_streets = torch.full((N, L), -1, dtype=torch.long, device=device)
        self.action_legal_masks = torch.zeros(
            N, L, num_bet_bins, dtype=torch.bool, device=device
        )
        self.context_features = torch.zeros(N, L, 10, dtype=float_dtype, device=device)
        self.lengths = torch.zeros(N, dtype=torch.long, device=device)

    def _reserve(self, idxs: torch.Tensor, k: int) -> torch.Tensor:
        pos = self.lengths[idxs]
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
        self.context_features[idxs, start, Context.POT.value] = self.tensor_env.pot[
            idxs
        ].to(self.float_dtype)
        self.context_features[idxs, start, Context.STACK_P0.value] = (
            self.tensor_env.stacks[idxs, player].to(self.float_dtype)
        )
        self.context_features[idxs, start, Context.STACK_P1.value] = (
            self.tensor_env.stacks[idxs, opp].to(self.float_dtype)
        )
        self.context_features[idxs, start, Context.COMMITTED_P0.value] = (
            self.tensor_env.committed[idxs, player].to(self.float_dtype)
        )
        self.context_features[idxs, start, Context.COMMITTED_P1.value] = (
            self.tensor_env.committed[idxs, opp].to(self.float_dtype)
        )
        self.context_features[idxs, start, Context.POSITION.value] = (
            self.tensor_env.button[idxs] != player
        ).to(self.float_dtype)
        self.context_features[idxs, start, Context.STREET.value] = (
            self.tensor_env.street[idxs].to(self.float_dtype)
        )
        self.context_features[idxs, start, Context.ACTIONS_ROUND.value] = (
            self.tensor_env.actions_this_round[idxs].to(self.float_dtype)
        )
        self.context_features[idxs, start, Context.MIN_RAISE.value] = (
            self.tensor_env.min_raise[idxs].to(self.float_dtype)
        )
        bet_to_call = (
            self.tensor_env.committed[idxs, opp]
            - self.tensor_env.committed[idxs, player]
        )
        self.context_features[idxs, start, Context.BET_TO_CALL.value] = bet_to_call.to(
            self.float_dtype
        )

    def add_cls(self, idxs: torch.Tensor) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        self.token_ids[idxs, start] = Special.CLS.value
        # CLS slot (index 0)
        self.context_features[idxs, start, Cls.SB.value] = float(self.tensor_env.sb)
        self.context_features[idxs, start, Cls.BB.value] = float(self.tensor_env.bb)
        self.context_features[idxs, start, Cls.HERO_ON_BUTTON.value] = (
            self.tensor_env.button[idxs] == 0
        ).to(self.float_dtype)

    def add_action(
        self,
        idxs: torch.Tensor,
        actors: torch.Tensor,
        action_ids: torch.Tensor,
        legal_masks: torch.Tensor,
    ) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        self.token_ids[idxs, start] = Special.NUM_SPECIAL.value + 52 + action_ids
        self.action_actors[idxs, start] = actors
        self.action_streets[idxs, start] = self.tensor_env.street[idxs]
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
        self.card_ranks[idxs, start] = card_indices % 13
        self.card_suits[idxs, start] = card_indices // 13
        self.card_streets[idxs, start] = self.tensor_env.street[idxs]

    def add_street(self, idxs: torch.Tensor, street_ids: torch.Tensor) -> None:
        if idxs.numel() == 0:
            return
        start = self._reserve(idxs, 1)
        special_tokens = Special.STREET_PREFLOP.value + street_ids.to(
            torch.int64
        )  # assumes 0=preflop,1=flop,2=turn,3=river
        self.token_ids[idxs, start] = special_tokens.to(torch.long)
        self.card_streets[idxs, start] = street_ids.to(torch.long)

    def encode_tensor_states(
        self,
        player: int,
        idxs: torch.Tensor,
    ) -> StructuredEmbeddingData:
        """Build minimal context features (CLS + CONTEXT) from the live tensor env.

        This mirrors the transformer state encoder's context computation so the
        TSB can act as the state-encoder during rollout.
        """
        M = idxs.numel()

        result = StructuredEmbeddingData(
            token_ids=self.token_ids[:M].clone(),
            card_ranks=self.card_ranks[:M].clone(),
            card_suits=self.card_suits[:M].clone(),
            card_streets=self.card_streets[:M].clone(),
            action_actors=self.action_actors[:M].clone(),
            action_streets=self.action_streets[:M].clone(),
            action_legal_masks=self.action_legal_masks[:M].clone(),
            context_features=self.context_features[:M].clone(),
            lengths=self.lengths[:M].clone(),
        )

        # hero_on_button flag in CLS (binary)
        button = self.tensor_env.button[idxs]
        result.context_features[:, 0, Cls.HERO_ON_BUTTON.value] = (
            button == int(player)
        ).to(self.float_dtype)

        # Set hole cards
        result.token_ids[:, HOLE0_INDEX] = (
            get_card_token_id_offset() + self.tensor_env.hole_indices[idxs, player, 0]
        )
        result.card_ranks[:, HOLE0_INDEX] = (
            self.tensor_env.hole_indices[idxs, player, 0] % 13
        )
        result.card_suits[:, HOLE0_INDEX] = (
            self.tensor_env.hole_indices[idxs, player, 0] // 13
        )
        result.token_ids[:, HOLE1_INDEX] = (
            get_card_token_id_offset() + self.tensor_env.hole_indices[idxs, player, 1]
        )
        result.card_ranks[:, HOLE1_INDEX] = (
            self.tensor_env.hole_indices[idxs, player, 1] % 13
        )
        result.card_suits[:, HOLE1_INDEX] = (
            self.tensor_env.hole_indices[idxs, player, 1] // 13
        )

        # Reverse all CONTEXT bet_to_call values. Will be 0 for non-context tokens, so this is fine.
        result.context_features[:, :, Context.BET_TO_CALL.value] = (
            -result.context_features[:, :, Context.BET_TO_CALL.value]
        )

        return result

    def reset_envs(self, idxs: torch.Tensor) -> None:
        if idxs.numel() == 0:
            return
        self.token_ids[idxs] = -1
        self.card_ranks[idxs] = 0
        self.card_suits[idxs] = 0
        self.card_streets[idxs] = 0
        self.action_actors[idxs] = -1
        self.action_streets[idxs] = -1
        self.action_legal_masks[idxs] = False
        self.context_features[idxs] = 0
        self.lengths[idxs] = 0
