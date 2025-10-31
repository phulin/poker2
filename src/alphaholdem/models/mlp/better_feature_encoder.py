from __future__ import annotations

import torch

from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_features import (
    PlayerContext,
    ScalarContext,
)
from alphaholdem.models.mlp.mlp_features import MLPFeatures


class BetterFeatureEncoder:
    """Construct better feature vectors from tensor env states."""

    belief_dim: int = 1326

    def __init__(
        self,
        env: HUNLTensorEnv,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.env = env
        self.device = device or env.device
        self.dtype = dtype or torch.float32

    def encode(
        self,
        beliefs: torch.Tensor,
        pre_chance_node: bool = False,
    ) -> MLPFeatures:
        """
        Build Better PBS features for a batch of env indices and agent ids.

        Args:
            beliefs: Tensor [B, 2, 1326] for beliefs (about p0 and p1).
        Returns:
            MLPFeatures with structured data.
        """

        N = beliefs.shape[0]
        num_players = beliefs.shape[1]
        scalar_context = torch.zeros(
            N,
            ScalarContext.NUM_SCALAR_CONTEXT.value,
            device=self.device,
            dtype=self.dtype,
        )
        scalar_context[:, ScalarContext.ACTOR.value] = self.env.to_act
        scalar_context[:, ScalarContext.POSITION.value] = (
            self.env.to_act - self.env.button
        ) % num_players
        scalar_context[:, ScalarContext.ACTIONS_ROUND.value] = (
            self.env.actions_this_round
        )
        scalar_context[:, ScalarContext.POT.value] = self.env.pot
        scalar_context[:, ScalarContext.MIN_RAISE.value] = self.env.min_raise

        stacks = self.env.stacks
        committed = self.env.committed
        pot = self.env.pot
        player_context = torch.zeros(
            N,
            PlayerContext.NUM_PLAYER_CONTEXT.value,
            num_players,
            device=self.device,
            dtype=self.dtype,
        )
        player_context[:, PlayerContext.STACK.value] = stacks
        player_context[:, PlayerContext.COMMITTED.value] = committed
        player_context[:, PlayerContext.SPR.value] = stacks / pot[:, None]

        street = (
            self.env.street
            if not pre_chance_node
            else torch.where(
                (self.env.street > 0)
                & (self.env.street < 4)
                & (self.env.actions_this_round == 0),
                self.env.street - 1,
                self.env.street,
            )
        )

        return MLPFeatures(
            context=torch.cat([scalar_context, player_context.view(N, -1)], dim=-1),
            street=street,
            board=(
                self.env.last_board_indices
                if pre_chance_node
                else self.env.board_indices
            ),
            beliefs=beliefs.view(-1, 2 * NUM_HANDS) * 2 - 1,
        )
