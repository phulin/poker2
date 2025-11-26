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
        pre_chance_node: torch.Tensor | bool | None = None,
        indices: torch.Tensor | None = None,
    ) -> MLPFeatures:
        """
        Build Better PBS features for a batch of env indices and agent ids.

        Args:
            beliefs: Tensor [B, 2, 1326] for beliefs (about p0 and p1).
            pre_chance_node: Optional mask for pre-chance nodes.
            indices: Optional indices to slice the environment and beliefs.
        Returns:
            MLPFeatures with structured data.
        """
        if indices is not None:
            beliefs = beliefs[indices]
            if isinstance(pre_chance_node, torch.Tensor):
                pre_chance_node = pre_chance_node[indices]

        N = beliefs.shape[0]
        num_players = beliefs.shape[1]
        scalar_context = torch.zeros(
            N,
            ScalarContext.NUM_SCALAR_CONTEXT.value,
            device=self.device,
            dtype=self.dtype,
        )

        if pre_chance_node is None:
            pre_chance_node = torch.zeros(N, dtype=torch.bool, device=self.device)
        elif isinstance(pre_chance_node, bool):
            pre_chance_node = torch.full(
                (N,), pre_chance_node, dtype=torch.bool, device=self.device
            )
        else:
            pre_chance_node = pre_chance_node.to(self.device)

        # Helper to get env tensor
        def get_env_tensor(attr_name: str) -> torch.Tensor:
            val = getattr(self.env, attr_name)
            return val[indices] if indices is not None else val

        actions_last_round = get_env_tensor("actions_last_round")
        actions_this_round = get_env_tensor("actions_this_round")
        actions_round = torch.where(
            pre_chance_node, actions_last_round, actions_this_round
        )
        # Keep to_act for actor, as that's the player perspective the model should take,
        # even in the pre-chance node context.
        to_act = get_env_tensor("to_act")
        scalar_context[:, ScalarContext.ACTOR.value] = to_act
        scalar_context[:, ScalarContext.POSITION.value] = (
            to_act - get_env_tensor("button")
        ) % num_players
        scalar_context[:, ScalarContext.ACTIONS_ROUND.value] = actions_round
        pot = get_env_tensor("pot")
        scalar_context[:, ScalarContext.POT.value] = pot
        scalar_context[:, ScalarContext.MIN_RAISE.value] = get_env_tensor("min_raise")

        stacks = get_env_tensor("stacks").to(self.dtype)
        committed = get_env_tensor("committed").to(self.dtype)
        pot = pot.to(self.dtype)
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

        street_tensor = get_env_tensor("street")
        street = torch.where(
            pre_chance_node & (actions_this_round == 0),
            torch.clamp(street_tensor - 1, min=0),
            street_tensor,
        )

        return MLPFeatures(
            context=torch.cat([scalar_context, player_context.view(N, -1)], dim=-1),
            street=street,
            to_act=to_act,
            board=torch.where(
                pre_chance_node[:, None],
                get_env_tensor("last_board_indices"),
                get_env_tensor("board_indices"),
            ),
            beliefs=beliefs.view(-1, 2 * NUM_HANDS),
        )
