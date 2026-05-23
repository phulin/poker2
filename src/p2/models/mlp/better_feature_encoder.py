from __future__ import annotations

import torch

from p2.env.card_utils import NUM_HANDS
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.better_features import (
    ChancePhase,
    PlayerContext,
    ScalarContext,
    ValueScalarContext,
)
from p2.models.mlp.mlp_features import MLPFeatures


class _BetterFeatureEncoderBase:
    """Shared tensor-env context construction for Better MLP feature encoders."""

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

    def _pre_chance_mask(
        self, N: int, pre_chance_node: torch.Tensor | bool | None
    ) -> torch.Tensor:
        if pre_chance_node is None:
            return torch.zeros(N, dtype=torch.bool, device=self.device)
        if isinstance(pre_chance_node, bool):
            return torch.full(
                (N,), pre_chance_node, dtype=torch.bool, device=self.device
            )
        return pre_chance_node.to(self.device)

    def _env_tensor(self, attr_name: str, indices: torch.Tensor | None) -> torch.Tensor:
        val = getattr(self.env, attr_name)
        return val[indices] if indices is not None else val

    def _street_for_phase(
        self,
        pre_chance_node: torch.Tensor,
        actions_this_round: torch.Tensor,
        street_tensor: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            pre_chance_node & (actions_this_round == 0),
            torch.clamp(street_tensor - 1, min=0),
            street_tensor,
        )

    def _player_context(
        self,
        N: int,
        num_players: int,
        scale: torch.Tensor,
        pot_float: torch.Tensor,
        bb: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        stacks = self._env_tensor("stacks", indices).to(self.dtype)
        committed = self._env_tensor("committed", indices).to(self.dtype)
        player_context = torch.zeros(
            N,
            PlayerContext.NUM_PLAYER_CONTEXT.value,
            num_players,
            device=self.device,
            dtype=self.dtype,
        )
        player_context[:, PlayerContext.STACK.value] = stacks / scale[:, None]
        player_context[:, PlayerContext.COMMITTED.value] = committed / scale[:, None]
        player_context[:, PlayerContext.SPR.value] = (
            stacks / pot_float.clamp_min(1.0)[:, None]
        )
        player_context[:, PlayerContext.LOG_COMMITTED_BB.value] = torch.log1p(
            committed / bb
        )
        return player_context.flatten(1)


class BetterPolicyFeatureEncoder(_BetterFeatureEncoderBase):
    """Construct policy features with full same-street betting context."""

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

        pre_chance_node = self._pre_chance_mask(N, pre_chance_node)
        actions_last_round = self._env_tensor("actions_last_round", indices)
        actions_this_round = self._env_tensor("actions_this_round", indices)
        actions_round = torch.where(
            pre_chance_node, actions_last_round, actions_this_round
        )
        # Keep to_act for actor, as that's the player perspective the model should take,
        # even in the pre-chance node context.
        to_act = self._env_tensor("to_act", indices)
        scalar_context[:, ScalarContext.ACTOR.value] = to_act
        scalar_context[:, ScalarContext.POSITION.value] = (
            to_act - self._env_tensor("button", indices)
        ) % num_players
        scalar_context[:, ScalarContext.ACTIONS_ROUND.value] = actions_round
        pot = self._env_tensor("pot", indices)
        scale = self._env_tensor("scale", indices).to(self.dtype).clamp_min(1.0)
        bb = torch.as_tensor(float(self.env.bb), device=self.device, dtype=self.dtype)
        bb = bb.clamp_min(1.0)
        pot_float = pot.to(self.dtype)
        stack_depth_bb = scale / bb
        pot_bb = pot_float / bb
        scalar_context[:, ScalarContext.POT.value] = pot_float / scale
        scalar_context[:, ScalarContext.MIN_RAISE.value] = (
            self._env_tensor("min_raise", indices).to(self.dtype) / scale
        )
        scalar_context[:, ScalarContext.LOG_STACK_DEPTH_BB.value] = torch.log(
            stack_depth_bb.clamp_min(1.0)
        ) / torch.log(
            torch.as_tensor(
                float(max(self.env.max_stack_bb, 2)),
                device=self.device,
                dtype=self.dtype,
            )
        )
        scalar_context[:, ScalarContext.LOG_POT_BB.value] = torch.log1p(pot_bb)

        player_context = self._player_context(
            N, num_players, scale, pot_float, bb, indices
        )

        street_tensor = self._env_tensor("street", indices)
        street = self._street_for_phase(
            pre_chance_node, actions_this_round, street_tensor
        )

        return MLPFeatures(
            context=torch.cat([scalar_context, player_context], dim=-1),
            street=street,
            to_act=to_act,
            board=torch.where(
                pre_chance_node[:, None],
                self._env_tensor("last_board_indices", indices),
                self._env_tensor("board_indices", indices),
            ),
            beliefs=beliefs.view(-1, 2 * NUM_HANDS),
        )


class BetterStreetValueFeatureEncoder(_BetterFeatureEncoderBase):
    """Construct boundary value features with chance-phase context."""

    def encode(
        self,
        beliefs: torch.Tensor,
        pre_chance_node: torch.Tensor | bool | None = None,
        indices: torch.Tensor | None = None,
    ) -> MLPFeatures:
        if indices is not None:
            beliefs = beliefs[indices]
            if isinstance(pre_chance_node, torch.Tensor):
                pre_chance_node = pre_chance_node[indices]

        N = beliefs.shape[0]
        num_players = beliefs.shape[1]
        scalar_context = torch.zeros(
            N,
            ValueScalarContext.NUM_SCALAR_CONTEXT.value,
            device=self.device,
            dtype=self.dtype,
        )

        pre_chance_node = self._pre_chance_mask(N, pre_chance_node)
        to_act = self._env_tensor("to_act", indices)
        scalar_context[:, ValueScalarContext.ACTOR.value] = to_act
        scalar_context[:, ValueScalarContext.POSITION.value] = (
            to_act - self._env_tensor("button", indices)
        ) % num_players
        scalar_context[:, ValueScalarContext.CHANCE_PHASE.value] = torch.where(
            pre_chance_node,
            torch.full(
                (N,),
                float(ChancePhase.PRE_CHANCE.value),
                device=self.device,
                dtype=self.dtype,
            ),
            torch.full(
                (N,),
                float(ChancePhase.POST_CHANCE.value),
                device=self.device,
                dtype=self.dtype,
            ),
        )

        pot = self._env_tensor("pot", indices)
        scale = self._env_tensor("scale", indices).to(self.dtype).clamp_min(1.0)
        bb = torch.as_tensor(float(self.env.bb), device=self.device, dtype=self.dtype)
        bb = bb.clamp_min(1.0)
        pot_float = pot.to(self.dtype)
        stack_depth_bb = scale / bb
        pot_bb = pot_float / bb
        scalar_context[:, ValueScalarContext.POT.value] = pot_float / scale
        scalar_context[:, ValueScalarContext.MIN_RAISE.value] = (
            self._env_tensor("min_raise", indices).to(self.dtype) / scale
        )
        scalar_context[:, ValueScalarContext.LOG_STACK_DEPTH_BB.value] = torch.log(
            stack_depth_bb.clamp_min(1.0)
        ) / torch.log(
            torch.as_tensor(
                float(max(self.env.max_stack_bb, 2)),
                device=self.device,
                dtype=self.dtype,
            )
        )
        scalar_context[:, ValueScalarContext.LOG_POT_BB.value] = torch.log1p(pot_bb)

        player_context = self._player_context(
            N, num_players, scale, pot_float, bb, indices
        )

        actions_this_round = self._env_tensor("actions_this_round", indices)
        street_tensor = self._env_tensor("street", indices)
        street = self._street_for_phase(
            pre_chance_node, actions_this_round, street_tensor
        )

        return MLPFeatures(
            context=torch.cat([scalar_context, player_context], dim=-1),
            street=street,
            to_act=to_act,
            board=torch.where(
                pre_chance_node[:, None],
                self._env_tensor("last_board_indices", indices),
                self._env_tensor("board_indices", indices),
            ),
            beliefs=beliefs.view(-1, 2 * NUM_HANDS),
        )


BetterFeatureEncoder = BetterPolicyFeatureEncoder
