from __future__ import annotations

from typing import Optional, Tuple

import torch

from alphaholdem.env.card_utils import (
    combo_lookup_tensor,
    hand_combos_tensor,
    mask_conflicting_combos,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv


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

    def _pot_fraction(self, idxs: torch.Tensor) -> torch.Tensor:
        denom = float(self.env.starting_stack * 2)
        pot = self.env.pot[idxs].to(torch.float32)
        return (pot / denom).clamp_(0.0, 10.0)

    def _board_features(self, idxs: torch.Tensor) -> torch.Tensor:
        board = self.env.board_indices[idxs]  # [B, 5]
        board = board.to(torch.float32)
        board[board < 0] = -1.0
        board[board >= 0] /= 51.0
        return board

    def _has_bet_flag(self, idxs: torch.Tensor) -> torch.Tensor:
        """Return [B] float flag: 1.0 if a bet/raise has occurred this round, else 0.0.

        Approximation based on unequal committed chips and at least one action.
        """
        committed = self.env.committed[idxs]  # [B, 2]
        unequal = committed[:, 0] != committed[:, 1]
        acted = self.env.actions_this_round[idxs] > 0
        flag = (unequal & acted).to(torch.float32)
        return flag

    def _hero_cards(self, idxs: torch.Tensor, agents: torch.Tensor) -> torch.Tensor:
        return self.env.hole_indices[idxs, agents]

    def _belief_vectors(
        self,
        hero_cards: torch.Tensor,
        board_cards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (hero_belief, opp_belief) tensors of shape [B, 1326].
        Hero belief is delta on actual cards when known; opponent belief uniform over
        combos consistent with hero's private cards and board.
        """
        B = hero_cards.shape[0]
        hero_beliefs = torch.zeros(
            B, self.belief_dim, dtype=self.dtype, device=self.device
        )
        opp_beliefs = torch.zeros_like(hero_beliefs)

        empty_long = torch.empty(0, dtype=torch.long, device=self.device)
        for i in range(B):
            hole = hero_cards[i]
            board = board_cards[i]
            known_mask = hole >= 0
            if known_mask.all():
                c1, c2 = sorted(hole.tolist())
                combo_idx = int(self._combo_lookup[c1, c2].item())
                hero_beliefs[i, combo_idx] = 1.0
            else:
                occupied_parts = []
                hero_known = hole[known_mask]
                if hero_known.numel() > 0:
                    occupied_parts.append(hero_known.to(torch.long))
                board_known = board[board >= 0]
                if board_known.numel() > 0:
                    occupied_parts.append(board_known.to(torch.long))
                occupied_tensor = (
                    torch.cat(occupied_parts) if occupied_parts else empty_long
                )
                available_mask = mask_conflicting_combos(
                    occupied_tensor,
                    device=self.device,
                )
                count = available_mask.sum()
                if count > 0:
                    hero_beliefs[i, available_mask] = 1.0 / count

            occupied_parts = []
            if known_mask.any():
                occupied_parts.append(hole[known_mask].to(torch.long))
            board_known = board[board >= 0]
            if board_known.numel() > 0:
                occupied_parts.append(board_known.to(torch.long))
            occupied = torch.cat(occupied_parts) if occupied_parts else empty_long
            opp_mask = mask_conflicting_combos(occupied, device=self.device)
            count_opp = opp_mask.sum()
            if count_opp > 0:
                opp_beliefs[i, opp_mask] = 1.0 / count_opp

        return hero_beliefs, opp_beliefs

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

    def encode(
        self,
        idxs: torch.Tensor,
        agents: torch.Tensor,
        hero_beliefs: Optional[torch.Tensor] = None,
        opp_beliefs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build ReBeL flat features for a batch of env indices and agent ids.

        Args:
            idxs: Tensor of environment indices, shape [B].
            agents: Tensor of agent ids (0 or 1), shape [B].
            hero_beliefs: Optional tensor [B, 1326] for acting player's belief.
            opp_beliefs: Optional tensor [B, 1326] for opponent's belief.
        Returns:
            Tensor [B, 2660] of float32 features.
        """
        if idxs.numel() == 0:
            return torch.empty(
                0, self.feature_dim, device=self.device, dtype=self.dtype
            )

        idxs = idxs.to(torch.long)
        agents = agents.to(torch.long)
        B = idxs.shape[0]

        features = torch.zeros(
            B, self.feature_dim, device=self.device, dtype=self.dtype
        )

        features[:, 0] = agents.to(self.dtype)
        features[:, 1] = self.env.to_act[idxs].to(self.dtype)
        features[:, 2] = self._pot_fraction(idxs)
        features[:, 3:8] = self._board_features(idxs)
        features[:, 8] = self._has_bet_flag(idxs)

        if hero_beliefs is not None and opp_beliefs is not None:
            hero_vec = hero_beliefs.to(device=self.device, dtype=self.dtype)
            opp_vec = opp_beliefs.to(device=self.device, dtype=self.dtype)
        else:
            hero_cards = self._hero_cards(idxs, agents)
            board_cards = self.env.board_indices[idxs]
            hero_vec, opp_vec = self._belief_vectors(hero_cards, board_cards)

        features[:, 9 : 9 + self.belief_dim] = hero_vec
        features[:, 9 + self.belief_dim :] = opp_vec
        return features
