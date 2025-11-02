from __future__ import annotations

from typing import Any

import torch

from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_to_onehot_tensor,
)
from alphaholdem.models.mlp.mlp_features import MLPFeatures


class ChanceNodeHelper:
    """Utilities for enumerating chance nodes when generating value targets.

    The helper assumes the value model is suit-symmetric. This constraint is
    enforced by the suit-symmetry loss in `alphaholdem/rl/losses.py`, allowing
    canonical-flop evaluation without explicitly remapping hand permutations.
    """

    FLOP_CHUNK_SIZE = 128

    device: torch.device
    float_dtype: torch.dtype
    num_players: int
    model: Any
    combo_onehot_float: torch.Tensor
    board_to_flop_id: torch.Tensor
    board_to_canonical: torch.Tensor
    flop_id_to_canonical: torch.Tensor
    flop_id_to_allowed_mask: torch.Tensor
    flop_id_to_count: torch.Tensor
    total_flop_count: int

    def __init__(
        self,
        device: torch.device,
        float_dtype: torch.dtype,
        num_players: int,
        model: Any,
    ) -> None:
        self.device = device
        self.float_dtype = float_dtype
        self.num_players = num_players
        self.model = model
        self.combo_onehot_float = combo_to_onehot_tensor(device=device).float()

        # Initialize cache as instance variables
        self._build_canonical_flops_cache()

    def _build_canonical_flops_cache(self) -> None:
        # Use vectorized approach to build canonical flop mapping
        device = self.device

        # 1) all 3-card combinations from 52
        cards = torch.arange(52, device=device, dtype=torch.long)
        flops = torch.combinations(cards, r=3, with_replacement=False)  # (22100, 3)

        # 2) Split cards into rank/suit
        # Card encoding: card = suit * 13 + rank
        # So: rank = card % 13, suit = card // 13
        ranks = flops % 13  # (22100, 3), values 0..12
        suits = flops // 13  # (22100, 3), values 0..3

        # 3) Sort by rank descending (deterministic). This fixes card order per flop.
        ranks_sorted, sort_idx = torch.sort(ranks, dim=1, descending=True)  # (22100, 3)

        # 4) All 24 permutations of 4 suits, as a (24, 4) tensor
        perms = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 3, 2],
                [0, 2, 1, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
                [0, 3, 2, 1],
                [1, 0, 2, 3],
                [1, 0, 3, 2],
                [1, 2, 0, 3],
                [1, 2, 3, 0],
                [1, 3, 0, 2],
                [1, 3, 2, 0],
                [2, 0, 1, 3],
                [2, 0, 3, 1],
                [2, 1, 0, 3],
                [2, 1, 3, 0],
                [2, 3, 0, 1],
                [2, 3, 1, 0],
                [3, 0, 1, 2],
                [3, 0, 2, 1],
                [3, 1, 0, 2],
                [3, 1, 2, 0],
                [3, 2, 0, 1],
                [3, 2, 1, 0],
            ],
            device=device,
            dtype=torch.long,
        )  # (24, 4)

        N = flops.size(0)  # 22100

        # 5) Apply every suit permutation to every flop, all at once.
        # suits: (N, 3) with values in 0..3
        # perms: (24, 4)
        # perms[:, suits] → (24, N, 3): for each perm, for each flop, map each suit index
        suits_perm = perms[:, suits]  # (24, N, 3)

        # 6) Re-order the permuted suits according to the rank-sort order we fixed in step 3
        sort_idx_exp = sort_idx.unsqueeze(0).expand(24, -1, -1)  # (24, N, 3)
        suits_perm_sorted = torch.gather(suits_perm, 2, sort_idx_exp)  # (24, N, 3)

        # 7) Canonicalize suits within equal-rank groups
        # Build masks on (N,) then broadcast
        eq01 = ranks_sorted[:, 0] == ranks_sorted[:, 1]  # (N,)
        eq12 = ranks_sorted[:, 1] == ranks_sorted[:, 2]  # (N,)
        all_eq = eq01 & eq12  # ranks like (x,x,x)
        first2_eq = eq01 & ~eq12  # (x,x,y)
        last2_eq = eq12 & ~eq01  # (x,y,y)

        # Case 1: all three equal → sort all 3 suits
        if all_eq.any():
            tmp = suits_perm_sorted[:, all_eq, :]  # (24, M, 3)
            tmp, _ = torch.sort(tmp, dim=2)  # sort all 3
            suits_perm_sorted[:, all_eq, :] = tmp

        # Case 2: first two equal → sort positions 0 and 1
        if first2_eq.any():
            tmp = suits_perm_sorted[:, first2_eq, :]  # (24, M, 3)
            first2, _ = torch.sort(tmp[:, :, :2], dim=2)  # sort the first 2
            tmp[:, :, :2] = first2
            suits_perm_sorted[:, first2_eq, :] = tmp

        # Case 3: last two equal → sort positions 1 and 2
        if last2_eq.any():
            tmp = suits_perm_sorted[:, last2_eq, :]  # (24, M, 3)
            last2, _ = torch.sort(tmp[:, :, 1:], dim=2)  # sort positions 1,2
            tmp[:, :, 1:] = last2
            suits_perm_sorted[:, last2_eq, :] = tmp

        # 8) Make the ranks broadcast to (24, N, 3) so we can form "permuted cards"
        ranks_sorted_b = ranks_sorted.unsqueeze(0).expand(24, -1, -1)  # (24, N, 3)

        # 9) Encode each card as suit*13 + rank → value in 0..51
        card_codes = suits_perm_sorted * 13 + ranks_sorted_b  # (24, N, 3)

        # 10) Turn each 3-card flop into a single scalar so we can pick the min over 24 perms
        # Use base-52 numbering: c0 * 52^2 + c1 * 52 + c2
        scalar_codes = (
            card_codes[:, :, 0] * (52 * 52)
            + card_codes[:, :, 1] * 52
            + card_codes[:, :, 2]
        )  # (24, N)

        # 11) Canonical representative = lexicographically smallest (i.e. smallest scalar) over 24 perms
        canonical_scalar, _ = scalar_codes.min(dim=0)  # (N,)

        # 12) Map each canonical scalar to a dense id 0..(num_flops-1)
        unique_vals, flop_to_canon = torch.unique(
            canonical_scalar, sorted=True, return_inverse=True
        )
        num_flops = unique_vals.shape[0]

        # Build canonical cards from unique scalars
        canonical_cards = torch.zeros(num_flops, 3, dtype=torch.long, device=device)
        canonical_cards[:, 0] = unique_vals // (52 * 52)
        canonical_cards[:, 1] = (unique_vals // 52) % 52
        canonical_cards[:, 2] = unique_vals % 52

        # Build board_to_flop_id tensor (vectorized)
        board_to_flop_id_tensor = torch.full(
            (52, 52, 52), -1, dtype=torch.long, device=device
        )
        # Vectorized indexing: flops[i] gives us [c0, c1, c2], use these as indices
        board_to_flop_id_tensor[flops[:, 0], flops[:, 1], flops[:, 2]] = flop_to_canon

        # Build count tensor - count how many flops map to each canonical id
        flop_id_to_count = torch.zeros(num_flops, dtype=torch.long, device=device)
        flop_id_to_count.index_add_(
            0, flop_to_canon, torch.ones(N, dtype=torch.long, device=device)
        )

        # Build board_to_canonical tensor (52, 52, 52, 3) mapping board indices to canonical cards
        board_to_canonical_tensor = torch.full(
            (52, 52, 52, 3), -1, dtype=torch.long, device=device
        )
        # Vectorized indexing: for each flop, store its canonical representation
        canonical_for_flops = canonical_cards[flop_to_canon]  # (N, 3)
        board_to_canonical_tensor[flops[:, 0], flops[:, 1], flops[:, 2]] = (
            canonical_for_flops
        )

        # Build flop_id_to_allowed_mask: [num_flops, NUM_HANDS]
        # For each canonical flop, which hands are allowed (don't conflict with board)?
        board_onehot = torch.zeros(num_flops, 52, dtype=self.float_dtype, device=device)
        board_onehot.scatter_(
            1,
            canonical_cards,
            torch.ones(num_flops, 3, dtype=self.float_dtype, device=device),
        )
        # Matrix multiply: [1326, 52] @ [52, num_flops] = [1326, num_flops]
        # If > 0, combo shares a card with board (conflict)
        conflict_matrix = self.combo_onehot_float @ board_onehot.T  # [1326, num_flops]
        flop_id_to_allowed_mask = (
            conflict_matrix == 0
        ).T  # [num_flops, 1326] -> [num_flops, NUM_HANDS]

        self.board_to_flop_id = board_to_flop_id_tensor
        self.board_to_canonical = board_to_canonical_tensor
        self.flop_id_to_canonical = canonical_cards
        self.flop_id_to_allowed_mask = flop_id_to_allowed_mask
        self.flop_id_to_count = flop_id_to_count
        self.total_flop_count = self.flop_id_to_count.sum().item()

    @torch.no_grad()
    def flop_chance_values(
        self,
        root_indices: torch.Tensor,
        root_features: MLPFeatures,
        pre_chance_beliefs: torch.Tensor,
    ) -> torch.Tensor:
        """Expected CFVs over three-card flop chance using canonical flops."""

        if root_indices.numel() == 0:
            return torch.zeros(
                0,
                self.num_players,
                NUM_HANDS,
                device=self.device,
                dtype=self.float_dtype,
            )

        dtype = self.float_dtype
        device = self.device
        B = root_indices.numel()

        pre_beliefs = pre_chance_beliefs[root_indices].to(dtype=dtype)
        context_root = root_features.context[root_indices]
        street_root = root_features.street[root_indices]
        to_act_root = root_features.to_act[root_indices]

        values_sum = torch.zeros(
            B, self.num_players, NUM_HANDS, device=device, dtype=dtype
        )

        model = self.model
        num_flops = self.flop_id_to_canonical.shape[0]

        pre_beliefs_broadcast = pre_beliefs.expand(
            B, self.num_players, NUM_HANDS
        )  # [B, 2, NUM_HANDS]

        chunk_size = self.FLOP_CHUNK_SIZE
        model.eval()
        for start in range(0, num_flops, chunk_size):
            end = min(start + chunk_size, num_flops)
            chunk_len = end - start

            canonical_chunk = self.flop_id_to_canonical[start:end]
            counts_chunk = self.flop_id_to_count[start:end]
            allowed_chunk = self.flop_id_to_allowed_mask[start:end]

            canonical_chunk_5 = torch.cat(
                [
                    canonical_chunk,
                    torch.full((chunk_len, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            )

            allowed_broadcast = (
                allowed_chunk.unsqueeze(0)
                .unsqueeze(2)
                .expand(B, chunk_len, self.num_players, NUM_HANDS)
            )

            post_beliefs = (
                pre_beliefs_broadcast.unsqueeze(1).expand(-1, chunk_len, -1, -1).clone()
            )
            post_beliefs[..., ~allowed_broadcast] = 0.0

            sums = post_beliefs.sum(dim=-1, keepdim=True)
            uniform = allowed_broadcast.to(dtype)
            uniform_sum = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = uniform / uniform_sum

            normalized_beliefs = torch.where(
                sums > 1e-12, post_beliefs / sums.clamp(min=1e-12), uniform
            )

            belief_features = normalized_beliefs.reshape(B * chunk_len, -1)
            board_samples_flat = (
                canonical_chunk_5.unsqueeze(0).expand(B, -1, -1).reshape(-1, 5)
            )

            context_expand = (
                context_root.unsqueeze(1)
                .expand(-1, chunk_len, -1)
                .reshape(-1, context_root.shape[1])
            )
            street_expand = street_root.unsqueeze(1).expand(-1, chunk_len).reshape(-1)
            to_act_expand = to_act_root.unsqueeze(1).expand(-1, chunk_len).reshape(-1)

            synthetic_features = MLPFeatures(
                context=context_expand,
                street=street_expand,
                to_act=to_act_expand,
                board=board_samples_flat,
                beliefs=belief_features,
            )
            hand_values = model(synthetic_features).hand_values.to(dtype=dtype)
            hand_values = hand_values.view(B, chunk_len, self.num_players, NUM_HANDS)

            weight = counts_chunk.view(1, chunk_len, 1, 1).to(dtype)
            values_sum += (hand_values * weight).sum(dim=1)

        expected = values_sum / self.total_flop_count
        return expected

    @torch.no_grad()
    def single_card_chance_values(
        self,
        root_indices: torch.Tensor,
        root_features: MLPFeatures,
        pre_chance_beliefs: torch.Tensor,
        board_pre: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expected CFVs over a single-card chance node (turn or river)."""

        if root_indices.numel() == 0:
            return torch.zeros(
                0,
                self.num_players,
                NUM_HANDS,
                device=self.device,
                dtype=self.float_dtype,
            )

        device = self.device
        dtype = self.float_dtype
        B = root_indices.numel()

        pre_beliefs = pre_chance_beliefs[root_indices].to(dtype=dtype)
        board_prev = board_pre[root_indices].clone()
        context_root = root_features.context[root_indices].clone()
        street_root = root_features.street[root_indices].clone()
        to_act_root = root_features.to_act[root_indices].clone()

        available_mask = torch.ones(B, 52, dtype=torch.bool, device=device)
        for slot in range(board_prev.shape[1]):
            cards = board_prev[:, slot]
            valid = cards >= 0
            if valid.any():
                available_mask[valid, cards[valid]] = False

        cards = torch.arange(52, device=device, dtype=torch.long)
        cards_expand = cards.unsqueeze(0).expand(B, -1)
        flat_mask = available_mask.view(-1)

        if flat_mask.sum().item() == 0:
            return torch.zeros(
                B, self.num_players, NUM_HANDS, device=device, dtype=dtype
            )

        flat_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(1)
        root_lookup = (
            torch.arange(B, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, 52)
            .reshape(-1)[flat_indices]
        )
        card_values = cards_expand.reshape(-1)[flat_indices]

        num_samples = flat_indices.numel()

        board_samples = board_prev[root_lookup].clone()
        num_cards = (board_samples >= 0).sum(dim=1)
        board_samples[torch.arange(num_samples, device=device), num_cards] = card_values

        board_onehot = torch.zeros(num_samples, 52, dtype=torch.bool, device=device)
        # Vectorized implementation: ignore -1 slots, set corresponding board_onehot.
        # board_samples: [num_samples, board_len], -1 means empty slot
        # We want: for every [i,slot], if card>=0 then set board_onehot[i,card]=True
        valid_mask = board_samples >= 0
        # mask out invalid slots
        idx_sample, idx_slot = torch.nonzero(valid_mask, as_tuple=True)
        cards = board_samples[idx_sample, idx_slot]
        board_onehot[idx_sample, cards] = True

        # board_onehot: [num_samples, 52]
        # allowed_mask: [num_samples, 1326]
        # Disallow any combo that collides with the board
        # A collision: (combo_onehot @ board_onehot.T) > 0
        # But want [num_samples, 1326], so transpose result
        allowed_mask = (self.combo_onehot_float @ board_onehot.T.float() < 0.5).T

        post_beliefs = pre_beliefs[root_lookup].clone()
        post_beliefs.masked_fill_(~allowed_mask.unsqueeze(1), 0.0)
        sums = post_beliefs.sum(dim=-1, keepdim=True)
        uniform = allowed_mask.unsqueeze(1).float()
        uniform_sum = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
        uniform = uniform / uniform_sum
        post_beliefs = torch.where(
            sums > 1e-12, post_beliefs / sums.clamp(min=1e-12), uniform
        )

        context_samples = context_root[root_lookup]
        street_samples = street_root[root_lookup]
        to_act_samples = to_act_root[root_lookup]
        belief_features = post_beliefs.reshape(num_samples, -1)

        synthetic_features = MLPFeatures(
            context=context_samples,
            street=street_samples,
            to_act=to_act_samples,
            board=board_samples,
            beliefs=belief_features,
        )

        model = self.model
        model.eval()

        # Note: Notionally these are EVs from the model, not CFVs.
        # But reach-weight only changes evenly across the chance node, so we ignore it.
        hand_values = model(synthetic_features).hand_values.to(dtype=dtype)

        values_sum = torch.zeros(
            B, self.num_players, NUM_HANDS, device=device, dtype=dtype
        )
        counts = torch.zeros(B, device=device, dtype=dtype)
        values_sum.index_add_(0, root_lookup, hand_values)
        counts.index_add_(
            0, root_lookup, torch.ones(num_samples, device=device, dtype=dtype)
        )
        counts = counts.clamp(min=1.0)
        expected = values_sum / counts.view(-1, 1, 1)

        return expected
