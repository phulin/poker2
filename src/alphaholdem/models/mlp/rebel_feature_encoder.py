from __future__ import annotations

from typing import Optional

import torch

from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_lookup_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.utils.profiling import profile


class RebelFeatureEncoder:
    """Construct flat ReBeL-style feature vectors from tensor env states."""

    feature_dim: int = 2661
    belief_dim: int = 1326

    def __init__(
        self,
        env: HUNLTensorEnv,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.env = env
        self.device = device or env.device
        self.dtype = dtype or torch.float32

        # Cache combo tensors on the target device for repeated encoding.
        self._combos = hand_combos_tensor(device=self.device)
        self._combo_lookup = combo_lookup_tensor(device=self.device)

    def _pot_fraction(self) -> torch.Tensor:
        denom = float(self.env.starting_stack * 2)
        pot = self.env.pot.to(torch.float32)
        return (pot / denom).clamp_(0.0, 10.0)

    def _has_bet_flag(self) -> torch.Tensor:
        """Return [B] float flag: 1.0 if a bet/raise has occurred this round, else 0.0.

        Approximation based on unequal committed chips and at least one action.
        """
        committed = self.env.committed  # [B, 2]
        unequal = committed[:, 0] != committed[:, 1]
        acted = self.env.actions_this_round > 0
        flag = (unequal & acted).to(torch.float32)
        return flag

    @profile
    def encode(
        self,
        beliefs: torch.Tensor,
        pre_chance_node: bool = False,
    ) -> MLPFeatures:
        """
        Build ReBeL flat features for all envs.

        Args:
            agents: Tensor of agent ids (0 or 1), shape [B].
            beliefs: Tensor [B, 2, 1326] for beliefs (about p0 and p1).
        Returns:
            MLPFeatures with structured fields:
            - context: agent, to_act, pot_fraction, has_bet_flag (indices 0,1,2,8)
            - board: board features (indices 3:8)
            - beliefs: beliefs (indices 9:)
            - street: unused (empty)
        """
        M = beliefs.shape[0]
        num_players = beliefs.shape[1]

        # Context features: to_act, position, pot_fraction, has_bet_flag
        context_features = torch.zeros(M, 4, device=self.device, dtype=self.dtype)
        context_features[:, 0] = self.env.to_act.to(self.dtype)
        context_features[:, 1] = (self.env.to_act - self.env.button) % num_players
        context_features[:, 2] = self._pot_fraction()
        context_features[:, 3] = self._has_bet_flag()

        return MLPFeatures(
            context=context_features,
            street=self.env.street,
            to_act=self.env.to_act,
            board=(
                self.env.last_board_indices
                if pre_chance_node
                else self.env.board_indices
            ),
            beliefs=beliefs.view(-1, 2 * NUM_HANDS),
        )
