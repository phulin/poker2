"""Base CFR evaluator class with shared methods."""

from __future__ import annotations

from abc import ABC

import torch
import torch.nn.functional as F

from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_to_onehot_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rules import rank_hands
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.utils.model_utils import compute_masked_logits


class CFREvaluator(ABC):
    """Base class for CFR evaluators with shared methods."""

    model: RebelFFN | BetterFFN
    device: torch.device
    env: HUNLTensorEnv
    feature_encoder: object
    beliefs: torch.Tensor
    legal_mask: torch.Tensor | None

    @torch.no_grad()
    def _get_model_policy_probs(self, indices: torch.Tensor) -> torch.Tensor:
        """Get policy probabilities from model for given indices."""
        features = self.feature_encoder.encode(self.beliefs)
        model_output = self.model(features[indices])
        logits = model_output.policy_logits
        legal_masks = self.legal_mask[indices]
        masked_logits = compute_masked_logits(logits, legal_masks[:, None, :])
        return F.softmax(masked_logits, dim=-1)

    def _showdown_value(self, hero: int, indices: torch.Tensor) -> torch.Tensor:
        """
        Exact river showdown EV using rank-CDF + blocker correction.
        Returns per-hand EV [N, 1326] (unsorted/original hand order) per env.
        Result is from hero perspective.

        Args:
            hero: Index of hero player (0 or 1).
            indices: Indices of nodes to compute showdown values for.

        Returns:
            Per-hand EV [N, 1326] (unsorted/original hand order) per env.
        """
        M = indices.numel()
        device = self.device
        dtype = torch.float32  # or match belief dtype
        villain = 1 - hero

        if M == 0:
            return torch.zeros(0, NUM_HANDS, device=device, dtype=dtype)

        # --- Beliefs & boards ---
        beliefs = self.beliefs[indices]  # (M,2,1326)
        b_opp = beliefs[:, villain, :].to(dtype)  # (M,1326)
        board = self.env.board_indices[indices].int()  # (M,5)

        # Sorted position k (0..1325) replicated across batch
        k = torch.arange(NUM_HANDS, device=device).expand(M, -1)  # (M,1326)

        # Check if hand_rank_data exists (for caching)
        if hasattr(self, "hand_rank_data") and self.hand_rank_data is not None:
            sorted_indices = self.hand_rank_data.sorted_indices
            inv_sorted = self.hand_rank_data.inv_sorted
            H = self.hand_rank_data.H
            card_ok = self.hand_rank_data.card_ok
            hand_ok_mask = self.hand_rank_data.hand_ok_mask
            hand_ok_mask_sorted = self.hand_rank_data.hand_ok_mask_sorted
            hands_c1c2_sorted = self.hand_rank_data.hands_c1c2_sorted
            L_idx = self.hand_rank_data.L_idx
            R_idx = self.hand_rank_data.R_idx
        else:
            # --- Ranks & sorted order per env (river deterministic strength) ---
            # hand_ranks: (M,1326) any integer/monotone rank key s.t. equal => tie
            # sorted_indices: argsort by (rank, tiebreak) ascending (weaker -> stronger)
            hand_ranks, sorted_indices = rank_hands(board)  # both (M,1326)

            # Ranks in sorted order
            ranks_sorted = torch.gather(hand_ranks, 1, sorted_indices)  # (M,1326)
            assert torch.all(
                ranks_sorted[:, 1:] >= ranks_sorted[:, :-1]
            ), "rank_hands order is descending; flip or fix rank_hands"

            # --- Tie groups: start flags, group ids, [L,R] spans per sorted position ---
            is_start = torch.ones_like(ranks_sorted, dtype=torch.bool)  # (M,1326)
            is_start[:, 1:] = ranks_sorted[:, 1:] != ranks_sorted[:, :-1]
            group_id = is_start.cumsum(dim=1, dtype=torch.int) - 1  # (M,1326), 0..G-1

            # For each group id, store first/last index in sorted order
            starts = torch.full(
                (M, NUM_HANDS), NUM_HANDS, dtype=torch.int, device=device
            )
            ends = torch.full((M, NUM_HANDS), -1, dtype=torch.int, device=device)
            starts.scatter_reduce_(
                1, group_id, k.int(), reduce="amin", include_self=True
            )
            ends.scatter_reduce_(1, group_id, k.int(), reduce="amax", include_self=True)

            # L,R per sorted position
            L = torch.gather(starts, 1, group_id)  # (M,1326)
            R = torch.gather(ends, 1, group_id)  # (M,1326)
            L_idx = L
            R_idx = (R + 1).clamp(max=NUM_HANDS)
            assert (L <= R).all(), "L must be <= R"
            assert torch.all(
                torch.gather(ranks_sorted, 1, L) == torch.gather(ranks_sorted, 1, R)
            ), "L/R must have same rank"

            # Inverse permutation (sorted->original) for mapping EV back
            inv_sorted = torch.argsort(sorted_indices, dim=1)  # (M,1326)

            # --- Hand/card incidence & board masking ---
            combo_to_onehot = combo_to_onehot_tensor(device=device)  # (1326,52)
            hands_c1c2 = hand_combos_tensor(device=device)  # (1326,2)

            # Per-env mask for cards not on the board: True = usable card
            card_ok = torch.ones((M, 52), dtype=torch.bool, device=device)
            card_ok.scatter_(1, board, False)  # False for board cards

            # Hand usable mask (unsorted): hand must use only ok cards
            H = combo_to_onehot.unsqueeze(0).expand(M, -1, -1)  # (M,1326,52)
            if hasattr(self, "allowed_hands"):
                hand_ok_mask = self.allowed_hands[indices]
            else:
                # Default: all hands are allowed
                hand_ok_mask = torch.ones(M, NUM_HANDS, dtype=torch.bool, device=device)
            hand_ok_mask_sorted = torch.gather(hand_ok_mask, 1, sorted_indices)

            # Cards (c1,c2) of each *sorted* hand per env
            hands_c1c2_sorted = torch.gather(
                hands_c1c2.unsqueeze(0).expand(M, -1, -1),  # (M,1326,2)
                1,
                sorted_indices.unsqueeze(-1).expand(-1, -1, 2),
            )  # (M,1326,2)

            # Cache hand_rank_data if the class supports it
            if hasattr(self, "hand_rank_data"):
                # Import HandRankData from rebel_cfr_evaluator if needed
                from alphaholdem.search.rebel_cfr_evaluator import HandRankData

                self.hand_rank_data = HandRankData(
                    sorted_indices=sorted_indices,
                    inv_sorted=inv_sorted,
                    H=H,
                    card_ok=card_ok,
                    hand_ok_mask=hand_ok_mask,
                    hand_ok_mask_sorted=hand_ok_mask_sorted,
                    hands_c1c2_sorted=hands_c1c2_sorted,
                    L_idx=L_idx,
                    R_idx=R_idx,
                )

        c1 = hands_c1c2_sorted[..., 0]  # (M,1326)
        c2 = hands_c1c2_sorted[..., 1]  # (M,1326)

        # Sort opponent marginal by strength order
        b_opp_sorted = b_opp.gather(1, sorted_indices)  # (M,1326)

        # Hand->card incidence in sorted order with board columns zeroed
        H_sorted = torch.gather(H, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 52))
        H_sorted = H_sorted & card_ok.unsqueeze(1)  # (M,1326,52)

        # --- Prefix sums over opponent mass (global and per-card), left-padded ---
        P = torch.cumsum(b_opp_sorted, dim=1)  # (M,1326)
        P = torch.cat(
            [torch.zeros(M, 1, device=device, dtype=dtype), P], dim=1
        )  # (M,1327)

        per_card_mass = H_sorted.to(dtype) * b_opp_sorted.unsqueeze(-1)  # (M,1326,52)
        Pcards = torch.cumsum(per_card_mass, dim=1)  # (M,1326,52)
        # -- Prefix sums over opponent mass, per card --
        Pcards = torch.cat(
            [torch.zeros(M, 1, 52, device=device, dtype=dtype), Pcards], dim=1
        )  # (M,1327,52)

        # --- Win/tie masses for each sorted position ---

        # Gather needed prefixes
        P_before = torch.gather(P, 1, L_idx)  # (M,1326)
        Pcards_before = torch.gather(
            Pcards, 1, L_idx.unsqueeze(-1).expand(-1, -1, 52)
        )  # (M,1326,52)

        # Win mass: all strictly weaker, excluding blockers
        Pcards_k_c1 = Pcards_before.gather(2, c1.unsqueeze(-1)).squeeze(-1)
        Pcards_k_c2 = Pcards_before.gather(2, c2.unsqueeze(-1)).squeeze(-1)
        win_mass = P_before - Pcards_k_c1 - Pcards_k_c2

        # Tie mass over [L,R] inclusive, excluding blockers
        P_R = torch.gather(P, 1, R_idx)
        P_L = torch.gather(P, 1, L_idx)
        seg_sum = P_R - P_L  # (M,1326)

        gL = L_idx.unsqueeze(-1).expand(-1, -1, 52)
        gR = R_idx.unsqueeze(-1).expand(-1, -1, 52)
        Pcards_R = torch.gather(Pcards, 1, gR)  # (M,1326,52)
        Pcards_L = torch.gather(Pcards, 1, gL)  # (M,1326,52)
        seg_c1 = (Pcards_R - Pcards_L).gather(2, c1.unsqueeze(-1)).squeeze(-1)
        seg_c2 = (Pcards_R - Pcards_L).gather(2, c2.unsqueeze(-1)).squeeze(-1)
        # Re-add hero combo mass (present in both seg_c1 and seg_c2)
        tie_mass = seg_sum - seg_c1 - seg_c2 + b_opp_sorted

        # --- Denominator: compatible opp mass for each hero hand (unsorted belief) ---
        Pc_last = Pcards[:, -1, :]  # (M, 52) totals per card
        denom = (
            1.0 - Pc_last.gather(1, c1) - Pc_last.gather(1, c2) + b_opp_sorted
        ).clamp(min=1e-12)
        valid_denom = denom > 1e-12
        assert ((valid_denom) | ((win_mass < 1e-5) & (tie_mass < 1e-5))).all()

        # Probabilities & EV (in sorted order)
        win_prob = torch.where(valid_denom, win_mass / denom, 0.0)
        tie_prob = torch.where(valid_denom, tie_mass / denom, 0.0)
        loss_prob = torch.where(valid_denom, 1.0 - win_prob - tie_prob, 0.0)

        EV_hand_sorted = win_prob - loss_prob

        # Map per-hand EV back to original hand order
        EV_hand = torch.gather(EV_hand_sorted, 1, inv_sorted)  # (M,1326)
        EV_hand = EV_hand * hand_ok_mask.to(dtype)  # zero impossible hands

        # Range EV for the player
        potential = (
            self.env.stacks[indices, hero]
            + self.env.pot[indices]
            - self.env.starting_stack
        )

        return EV_hand * potential[:, None] / self.env.scale
