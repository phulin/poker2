from __future__ import annotations

from typing import Optional, Tuple

import torch

from alphaholdem.env.card_utils import (
    combo_lookup_tensor,
    hand_combos_tensor,
    mask_conflicting_combos,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS
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

    def _board_features(self) -> torch.Tensor:
        board = self.env.board_indices  # [B, 5]
        board = board.to(torch.float32)
        board /= 51.0
        board.masked_fill_(board < 0, -1.0)
        return board

    def _has_bet_flag(self) -> torch.Tensor:
        """Return [B] float flag: 1.0 if a bet/raise has occurred this round, else 0.0.

        Approximation based on unequal committed chips and at least one action.
        """
        committed = self.env.committed  # [B, 2]
        unequal = committed[:, 0] != committed[:, 1]
        acted = self.env.actions_this_round > 0
        flag = (unequal & acted).to(torch.float32)
        return flag

    def aggregate_beliefs_from_samples(
        self,
        cards: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert sampled hole-card combinations with weights into belief vectors.

        Args:
            cards: Tensor [S, B, 2] of combo card indices per sample.
            weights: Tensor [S, B] of non-negative sample weights.
        Returns:
            Aggregated belief tensor [B, 1326] normalized per row.
        """
        if cards.numel() == 0:
            return torch.empty(0, self.belief_dim, device=self.device, dtype=self.dtype)

        samples, batch_size, _ = cards.shape
        belief = torch.zeros(
            batch_size, self.belief_dim, dtype=self.dtype, device=self.device
        )
        if samples == 0 or batch_size == 0:
            return belief

        row_indices = torch.arange(batch_size, device=self.device)
        for sample_idx in range(samples):
            weight_row = weights[sample_idx]
            card_row = cards[sample_idx]
            valid = (weight_row > 0) & (card_row[:, 0] >= 0) & (card_row[:, 1] >= 0)
            if not torch.any(valid):
                continue
            combos_idx = self._combo_lookup[card_row[valid, 0], card_row[valid, 1]].to(
                torch.long
            )
            belief[row_indices[valid], combos_idx] += weight_row[valid]

        sums = belief.sum(dim=1, keepdim=True)
        nonzero = sums.squeeze(1) > 0
        belief[nonzero] = belief[nonzero] / sums[nonzero]
        return belief

    @profile
    def encode(
        self,
        agents: torch.Tensor,
        beliefs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build ReBeL flat features for a batch of env indices and agent ids.

        Args:
            idxs: Tensor of environment indices, shape [B].
            agents: Tensor of agent ids (0 or 1), shape [B].
            beliefs: Optional tensor [B, 2, 1326] for beliefs (about p0 and p1).
        Returns:
            Tensor [B, 2660] of float32 features.
        """
        M = agents.shape[0]

        features = torch.zeros(
            M, self.feature_dim, device=self.device, dtype=self.dtype
        )

        features[:, 0] = agents.to(self.dtype)
        features[:, 1] = self.env.to_act.to(self.dtype)
        features[:, 2] = self._pot_fraction()
        features[:, 3:8] = self._board_features()
        features[:, 8] = self._has_bet_flag()

        features[:, 9:] = beliefs.reshape(-1, 2 * NUM_HANDS)
        return features
