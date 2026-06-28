from __future__ import annotations

import torch

from p2.env.card_utils import PREFLOP_HANDS
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
        self._bb_tensor = torch.tensor(
            float(self.env.bb), device=self.device, dtype=self.dtype
        ).clamp_min(1.0)
        self._log_max_stack_bb_tensor = torch.log(
            torch.tensor(
                float(max(self.env.max_stack_bb, 2)),
                device=self.device,
                dtype=self.dtype,
            )
        )

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

    def _optional_env_tensor(
        self,
        attr_name: str,
        indices: torch.Tensor | None,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not hasattr(self.env, attr_name):
            return torch.zeros(shape, dtype=dtype, device=self.device)
        val = getattr(self.env, attr_name)
        out = val[indices] if indices is not None else val
        return out.to(device=self.device)

    def _legal_bins_mask(self, indices: torch.Tensor | None) -> torch.Tensor:
        legal = self.env.legal_bins_mask()
        return legal[indices] if indices is not None else legal

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
        to_act: torch.Tensor,
        button: torch.Tensor,
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

        max_committed = committed.amax(dim=1)
        to_call = (max_committed[:, None] - committed).clamp_min(0.0)
        call_amount = torch.minimum(to_call, stacks)
        players = torch.arange(num_players, device=self.device)
        denom = float(max(num_players - 1, 1))
        folded = self._optional_env_tensor(
            "has_folded", indices, (N, num_players), torch.bool
        ).to(torch.bool)
        all_in = self._optional_env_tensor(
            "is_allin", indices, (N, num_players), torch.bool
        ).to(torch.bool)
        acted = self._optional_env_tensor(
            "acted_this_round", indices, (N, num_players), torch.bool
        ).to(torch.bool)
        is_actor = players[None, :] == to_act[:, None]
        rel_pos_to_actor = ((players[None, :] - to_act[:, None]) % num_players).to(
            self.dtype
        ) / denom
        rel_pos_to_button = ((players[None, :] - button[:, None]) % num_players).to(
            self.dtype
        ) / denom

        player_context[:, PlayerContext.FOLDED.value] = folded.to(self.dtype)
        player_context[:, PlayerContext.ALL_IN.value] = all_in.to(self.dtype)
        player_context[:, PlayerContext.ACTED_THIS_ROUND.value] = acted.to(self.dtype)
        player_context[:, PlayerContext.IS_ACTOR.value] = is_actor.to(self.dtype)
        player_context[:, PlayerContext.REL_POS_TO_ACTOR.value] = rel_pos_to_actor
        player_context[:, PlayerContext.REL_POS_TO_BUTTON.value] = rel_pos_to_button
        player_context[:, PlayerContext.TO_CALL_SCALE.value] = to_call / scale[:, None]
        player_context[:, PlayerContext.TO_CALL_POT.value] = (
            to_call / pot_float.clamp_min(1.0)[:, None]
        )
        player_context[:, PlayerContext.STACK_AFTER_CALL_SCALE.value] = (
            (stacks - call_amount).clamp_min(0.0) / scale[:, None]
        )
        return player_context.flatten(1)

    def _fill_policy_betting_context(
        self,
        scalar_context: torch.Tensor,
        to_act: torch.Tensor,
        pot_float: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> None:
        committed = self._env_tensor("committed", indices).to(self.dtype)
        actor_committed = committed.gather(1, to_act[:, None]).squeeze(1)
        max_committed = committed.amax(dim=1)
        actor_to_call = (max_committed - actor_committed).clamp_min(0.0)
        last_aggressive = self._optional_env_tensor(
            "last_aggressive_amount", indices, (to_act.shape[0],), torch.long
        ).to(self.dtype)
        legal = self._legal_bins_mask(indices)
        num_actions = legal.shape[1]
        scalar_context[:, ScalarContext.MAX_COMMITTED.value] = max_committed / scale
        scalar_context[:, ScalarContext.LAST_AGGRESSIVE_AMOUNT.value] = (
            last_aggressive / scale
        )
        scalar_context[:, ScalarContext.UNOPENED_OR_CHECKED_TO_ACTOR.value] = (
            actor_to_call <= 0
        ).to(self.dtype)
        scalar_context[:, ScalarContext.NUM_LEGAL_ACTIONS.value] = (
            legal.to(self.dtype).sum(dim=1) / float(max(num_actions, 1))
        )
        scalar_context[:, ScalarContext.CAN_FOLD.value] = legal[:, 0].to(self.dtype)
        scalar_context[:, ScalarContext.CAN_CALL.value] = legal[:, 1].to(self.dtype)
        scalar_context[:, ScalarContext.CAN_RAISE.value] = legal[:, 2:-1].any(
            dim=1
        ).to(self.dtype)
        scalar_context[:, ScalarContext.CAN_ALLIN.value] = legal[:, -1].to(self.dtype)

    def _fill_value_betting_context(
        self,
        scalar_context: torch.Tensor,
        to_act: torch.Tensor,
        pot_float: torch.Tensor,
        scale: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> None:
        committed = self._env_tensor("committed", indices).to(self.dtype)
        actor_committed = committed.gather(1, to_act[:, None]).squeeze(1)
        max_committed = committed.amax(dim=1)
        actor_to_call = (max_committed - actor_committed).clamp_min(0.0)
        last_aggressive = self._optional_env_tensor(
            "last_aggressive_amount", indices, (to_act.shape[0],), torch.long
        ).to(self.dtype)
        legal = self._legal_bins_mask(indices)
        num_actions = legal.shape[1]
        scalar_context[:, ValueScalarContext.MAX_COMMITTED.value] = (
            max_committed / scale
        )
        scalar_context[:, ValueScalarContext.LAST_AGGRESSIVE_AMOUNT.value] = (
            last_aggressive / scale
        )
        scalar_context[
            :, ValueScalarContext.UNOPENED_OR_CHECKED_TO_ACTOR.value
        ] = (actor_to_call <= 0).to(self.dtype)
        scalar_context[:, ValueScalarContext.NUM_LEGAL_ACTIONS.value] = (
            legal.to(self.dtype).sum(dim=1) / float(max(num_actions, 1))
        )
        scalar_context[:, ValueScalarContext.CAN_FOLD.value] = legal[:, 0].to(
            self.dtype
        )
        scalar_context[:, ValueScalarContext.CAN_CALL.value] = legal[:, 1].to(
            self.dtype
        )
        scalar_context[:, ValueScalarContext.CAN_RAISE.value] = legal[:, 2:-1].any(
            dim=1
        ).to(self.dtype)
        scalar_context[:, ValueScalarContext.CAN_ALLIN.value] = legal[:, -1].to(
            self.dtype
        )


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
        button = self._env_tensor("button", indices)
        scalar_context[:, ScalarContext.ACTOR.value] = to_act
        scalar_context[:, ScalarContext.POSITION.value] = (
            to_act - button
        ) % num_players
        scalar_context[:, ScalarContext.ACTIONS_ROUND.value] = actions_round
        pot = self._env_tensor("pot", indices)
        scale = self._env_tensor("scale", indices).to(self.dtype).clamp_min(1.0)
        bb = self._bb_tensor
        pot_float = pot.to(self.dtype)
        stack_depth_bb = scale / bb
        pot_bb = pot_float / bb
        scalar_context[:, ScalarContext.POT.value] = pot_float / scale
        scalar_context[:, ScalarContext.MIN_RAISE.value] = (
            self._env_tensor("min_raise", indices).to(self.dtype) / scale
        )
        scalar_context[:, ScalarContext.LOG_STACK_DEPTH_BB.value] = torch.log(
            stack_depth_bb.clamp_min(1.0)
        ) / self._log_max_stack_bb_tensor
        scalar_context[:, ScalarContext.LOG_POT_BB.value] = torch.log1p(pot_bb)
        self._fill_policy_betting_context(
            scalar_context,
            to_act,
            pot_float,
            scale,
            indices,
        )

        player_context = self._player_context(
            N, num_players, scale, pot_float, bb, indices, to_act, button
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
            beliefs=beliefs.reshape(N, num_players * self.belief_dim),
            hand_dim=self.belief_dim,
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
        button = self._env_tensor("button", indices)
        scalar_context[:, ValueScalarContext.ACTOR.value] = to_act
        scalar_context[:, ValueScalarContext.POSITION.value] = (
            to_act - button
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
        bb = self._bb_tensor
        pot_float = pot.to(self.dtype)
        stack_depth_bb = scale / bb
        pot_bb = pot_float / bb
        scalar_context[:, ValueScalarContext.POT.value] = pot_float / scale
        scalar_context[:, ValueScalarContext.MIN_RAISE.value] = (
            self._env_tensor("min_raise", indices).to(self.dtype) / scale
        )
        scalar_context[:, ValueScalarContext.LOG_STACK_DEPTH_BB.value] = torch.log(
            stack_depth_bb.clamp_min(1.0)
        ) / self._log_max_stack_bb_tensor
        scalar_context[:, ValueScalarContext.LOG_POT_BB.value] = torch.log1p(pot_bb)
        self._fill_value_betting_context(
            scalar_context,
            to_act,
            pot_float,
            scale,
            indices,
        )

        player_context = self._player_context(
            N, num_players, scale, pot_float, bb, indices, to_act, button
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
            beliefs=beliefs.reshape(N, num_players * self.belief_dim),
            hand_dim=self.belief_dim,
        )


BetterFeatureEncoder = BetterPolicyFeatureEncoder


class BetterPreflopPolicyFeatureEncoder(BetterPolicyFeatureEncoder):
    """Construct compact preflop policy features with 169 rank classes."""

    belief_dim = PREFLOP_HANDS


class BetterPreflopValueFeatureEncoder(BetterPolicyFeatureEncoder):
    """Construct compact preflop value features with betting context.

    Preflop value targets are same-street continuation roots, not postflop
    street-boundary chance phases. Keep the policy-context scalar layout so the
    compact value model sees actions_this_round instead of a constant phase bit.
    """

    belief_dim = PREFLOP_HANDS
