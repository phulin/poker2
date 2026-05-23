from __future__ import annotations

from typing import Any

import torch

from p2.env.card_utils import (
    NUM_HANDS,
    calculate_unblocked_mass,
    combo_to_onehot_tensor,
)
from p2.models.mlp.mlp_features import MLPFeatures


class ChanceNodeHelper:
    """Utilities for enumerating chance nodes when generating value targets."""

    FLOP_CHUNK_SIZE = 128
    # Number of raw flops to sample per call. 0 = exhaustive enumeration of all
    # 22,100 raw flops in chunks of FLOP_CHUNK_SIZE.
    FLOP_SAMPLE_SIZE = 256

    device: torch.device
    float_dtype: torch.dtype
    num_players: int
    model: Any
    combo_onehot_float: torch.Tensor
    all_flops: torch.Tensor

    def __init__(
        self,
        device: torch.device,
        float_dtype: torch.dtype,
        num_players: int,
        model: Any,
        generator: torch.Generator | None = None,
    ) -> None:
        self.device = device
        self.float_dtype = float_dtype
        self.num_players = num_players
        self.model = model
        self.generator = generator
        self.combo_onehot_float = combo_to_onehot_tensor(device=device).float()
        cards = torch.arange(52, device=device, dtype=torch.long)
        self.all_flops = torch.combinations(cards, r=3, with_replacement=False)

    @torch.no_grad()
    def flop_chance_values(
        self,
        root_indices: torch.Tensor,
        root_features: MLPFeatures,
        pre_chance_beliefs: torch.Tensor,
    ) -> torch.Tensor:
        """Expected CFVs over three-card flop chance using raw flop samples."""

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
        weight_sum = torch.zeros_like(values_sum)

        model = self.model
        all_flops = self.all_flops
        num_flops = all_flops.shape[0]
        static_feature_prefix = getattr(model, "static_feature_prefix", None)
        static_feature_base_from_prefix = getattr(
            model, "static_feature_base_from_prefix", None
        )

        pre_beliefs_broadcast = pre_beliefs.expand(
            B, self.num_players, NUM_HANDS
        )  # [B, 2, NUM_HANDS]

        model.eval()
        static_prefix_root = None
        if callable(static_feature_prefix) and callable(
            static_feature_base_from_prefix
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                static_prefix_root = static_feature_prefix(context_root, street_root)

        def eval_chunk(flop_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            chunk_len = flop_chunk.shape[0]
            board_chunk = torch.cat(
                [
                    flop_chunk,
                    torch.full((chunk_len, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            )

            board_onehot = torch.zeros(
                chunk_len, 52, dtype=self.combo_onehot_float.dtype, device=device
            )
            board_onehot.scatter_(
                1,
                flop_chunk,
                torch.ones(
                    chunk_len, 3, dtype=self.combo_onehot_float.dtype, device=device
                ),
            )
            allowed_chunk = (self.combo_onehot_float @ board_onehot.T < 0.5).T

            allowed_broadcast = (
                allowed_chunk.unsqueeze(0)
                .unsqueeze(2)
                .expand(B, chunk_len, self.num_players, NUM_HANDS)
            )

            post_unnorm = (
                pre_beliefs_broadcast.unsqueeze(1).expand(-1, chunk_len, -1, -1).clone()
            )
            post_unnorm.masked_fill_(~allowed_broadcast, 0.0)

            sums = post_unnorm.sum(dim=-1, keepdim=True)
            uniform = allowed_broadcast.to(dtype)
            uniform_sum = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = uniform / uniform_sum

            normalized_beliefs = torch.where(
                sums > 1e-12, post_unnorm / sums.clamp(min=1e-12), uniform
            )

            belief_features = normalized_beliefs.reshape(B * chunk_len, -1)
            board_samples_flat = (
                board_chunk.unsqueeze(0).expand(B, -1, -1).reshape(-1, 5)
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                static_base_features = None
                if static_prefix_root is not None:
                    static_prefix = (
                        static_prefix_root.unsqueeze(1)
                        .expand(-1, chunk_len, -1)
                        .reshape(-1, static_prefix_root.shape[-1])
                    )
                    static_base_features = static_feature_base_from_prefix(
                        static_prefix, board_samples_flat
                    )
                if hasattr(model, "forward_post"):
                    hand_values = model.forward_post(
                        synthetic_features,
                        static_base_features=static_base_features,
                    )
                else:
                    hand_values = model(
                        synthetic_features,
                        include_policy=False,
                        static_base_features=static_base_features,
                    ).hand_values
            hand_values = hand_values.to(dtype=dtype).view(
                B, chunk_len, self.num_players, NUM_HANDS
            )

            weights = calculate_unblocked_mass(post_unnorm).flip(dims=[-2])
            weights = weights * allowed_chunk.unsqueeze(0).unsqueeze(2).to(
                dtype=weights.dtype
            )
            return hand_values * weights.to(dtype=dtype), weights.to(dtype=dtype)

        S = self.FLOP_SAMPLE_SIZE
        if S > 0 and S < num_flops:
            sample_idx = torch.randperm(
                num_flops, device=device, generator=self.generator
            )[:S]
            weighted_values, weights = eval_chunk(all_flops[sample_idx])
            values_sum = weighted_values.sum(dim=1)
            weight_sum = weights.sum(dim=1)
            return torch.where(
                weight_sum > 1e-12,
                values_sum / weight_sum.clamp(min=1e-12),
                torch.zeros_like(values_sum),
            )

        chunk_size = self.FLOP_CHUNK_SIZE
        for start in range(0, num_flops, chunk_size):
            end = min(start + chunk_size, num_flops)
            weighted_values, weights = eval_chunk(all_flops[start:end])
            values_sum += weighted_values.sum(dim=1)
            weight_sum += weights.sum(dim=1)

        return torch.where(
            weight_sum > 1e-12,
            values_sum / weight_sum.clamp(min=1e-12),
            torch.zeros_like(values_sum),
        )

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

        post_unnorm = pre_beliefs[root_lookup].clone()
        post_unnorm.masked_fill_(~allowed_mask.unsqueeze(1), 0.0)
        sums = post_unnorm.sum(dim=-1, keepdim=True)
        uniform = allowed_mask.unsqueeze(1).float()
        uniform_sum = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
        uniform = uniform / uniform_sum
        post_beliefs = torch.where(
            sums > 1e-12, post_unnorm / sums.clamp(min=1e-12), uniform
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
        static_feature_prefix = getattr(model, "static_feature_prefix", None)
        static_feature_base_from_prefix = getattr(
            model, "static_feature_base_from_prefix", None
        )
        static_base_features = None

        # The model returns per-hand EVs; convert the chance expectation into a
        # hand-conditional average with opponent compatible-mass weights.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if callable(static_feature_prefix) and callable(
                static_feature_base_from_prefix
            ):
                static_prefix_root = static_feature_prefix(context_root, street_root)
                static_base_features = static_feature_base_from_prefix(
                    static_prefix_root[root_lookup], board_samples
                )
            if hasattr(model, "forward_post"):
                hand_values = model.forward_post(
                    synthetic_features,
                    static_base_features=static_base_features,
                )
            else:
                hand_values = model(
                    synthetic_features,
                    include_policy=False,
                    static_base_features=static_base_features,
                ).hand_values
        hand_values = hand_values.to(dtype=dtype)

        weights = calculate_unblocked_mass(post_unnorm).flip(dims=[-2])
        weights = weights * allowed_mask.unsqueeze(1).to(dtype=weights.dtype)

        values_sum = torch.zeros(
            B, self.num_players, NUM_HANDS, device=device, dtype=dtype
        )
        weight_sum = torch.zeros_like(values_sum)
        values_sum.index_add_(0, root_lookup, hand_values * weights.to(dtype=dtype))
        weight_sum.index_add_(0, root_lookup, weights.to(dtype=dtype))
        expected = torch.where(
            weight_sum > 1e-12,
            values_sum / weight_sum.clamp(min=1e-12),
            torch.zeros_like(values_sum),
        )

        return expected
