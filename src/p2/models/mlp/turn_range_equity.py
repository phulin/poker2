from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import torch
try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA optimization
    triton = None
    tl = None

from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.env.rules import rank_hands as rank_hands_torch
from p2.env.rules_triton import rank_hands_triton, triton_is_available
from p2.models.mlp.better_features import PlayerContext, ValueScalarContext
from p2.models.mlp.mlp_features import MLPFeatures


RankGroupsFn = Callable[[torch.Tensor], torch.Tensor]


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


if triton is not None:

    @triton.jit
    def _turn_pair_payoff_precompute_kernel(
        rank_groups_ptr,
        hand_runout_ok_ptr,
        legal_hands_ptr,
        pair_payoff_ptr,
        hand_count: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        row = tl.program_id(0)
        block_i = tl.program_id(1)
        block_j = tl.program_id(2)
        offs_i = block_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_j = block_j * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_i = offs_i < hand_count
        mask_j = offs_j < hand_count
        hand_i = tl.load(
            legal_hands_ptr + row * hand_count + offs_i,
            mask=mask_i,
            other=0,
        ).to(tl.int32)
        hand_j = tl.load(
            legal_hands_ptr + row * hand_count + offs_j,
            mask=mask_j,
            other=0,
        ).to(tl.int32)
        payoff = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        for river in tl.static_range(0, 48):
            base = (row * 48 + river) * 1326
            rank_i = tl.load(
                rank_groups_ptr + base + hand_i,
                mask=mask_i,
                other=0,
            ).to(tl.int32)
            rank_j = tl.load(
                rank_groups_ptr + base + hand_j,
                mask=mask_j,
                other=0,
            ).to(tl.int32)
            ok_i = tl.load(
                hand_runout_ok_ptr + base + hand_i,
                mask=mask_i,
                other=0,
            ) != 0
            ok_j = tl.load(
                hand_runout_ok_ptr + base + hand_j,
                mask=mask_j,
                other=0,
            ) != 0
            diff = rank_i[:, None] - rank_j[None, :]
            pair_ok = ok_i[:, None] & ok_j[None, :]
            score = tl.where(diff > 0, 1.0, tl.where(diff < 0, -1.0, 0.0))
            payoff += tl.where(pair_ok, score, 0.0)
        out_offsets = (
            row * hand_count * hand_count
            + offs_i[:, None] * hand_count
            + offs_j[None, :]
        )
        out_mask = mask_i[:, None] & mask_j[None, :]
        tl.store(pair_payoff_ptr + out_offsets, payoff, mask=out_mask)

    @triton.jit
    def _turn_pair_operator_store_kernel(
        context_ptr,
        grouped_leaf_indices_ptr,
        root_leaf_offsets_ptr,
        bucket_roots_ptr,
        legal_hands_ptr,
        legal_hand_cards_ptr,
        num0_ptr,
        num1_ptr,
        card_mass0_ptr,
        card_mass1_ptr,
        total0_ptr,
        total1_ptr,
        out_ptr,
        root_count: tl.constexpr,
        leaves_per_root: tl.constexpr,
        hand_count: tl.constexpr,
        context_stride: tl.constexpr,
        pot_index: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_player: tl.constexpr,
        pos_scale: tl.constexpr,
        neg_scale: tl.constexpr,
        intercept: tl.constexpr,
        baseline_scale: tl.constexpr,
        use_pos_neg: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        root = tl.program_id(0)
        leaf_local = tl.program_id(1)
        hand_block = tl.program_id(2)
        root_cache = tl.load(bucket_roots_ptr + root, mask=root < root_count, other=0).to(
            tl.int32
        )
        root_start = tl.load(
            root_leaf_offsets_ptr + root_cache,
            mask=root < root_count,
            other=0,
        ).to(tl.int32)
        root_end = tl.load(
            root_leaf_offsets_ptr + root_cache + 1,
            mask=root < root_count,
            other=0,
        ).to(tl.int32)
        valid_leaf = (root < root_count) & (leaf_local < (root_end - root_start))
        leaf_pos = root_start + leaf_local
        leaf = tl.load(grouped_leaf_indices_ptr + leaf_pos, mask=valid_leaf, other=0).to(
            tl.int32
        )
        offs = hand_block * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = valid_leaf & (offs < hand_count)
        hand = tl.load(
            legal_hands_ptr + root * hand_count + offs,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        card0 = tl.load(
            legal_hand_cards_ptr + (root * hand_count + offs) * 2,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        card1 = tl.load(
            legal_hand_cards_ptr + (root * hand_count + offs) * 2 + 1,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        num_base = (root * hand_count + offs) * leaves_per_root + leaf_local
        num0 = tl.load(num0_ptr + num_base, mask=hand_mask, other=0.0).to(tl.float32)
        num1 = tl.load(num1_ptr + num_base, mask=hand_mask, other=0.0).to(tl.float32)
        total0 = tl.load(
            total0_ptr + root * leaves_per_root + leaf_local,
            mask=valid_leaf,
            other=0.0,
        ).to(tl.float32)
        total1 = tl.load(
            total1_ptr + root * leaves_per_root + leaf_local,
            mask=valid_leaf,
            other=0.0,
        ).to(tl.float32)
        cm_base0 = (root * leaves_per_root + leaf_local) * 52
        cm0a = tl.load(card_mass0_ptr + cm_base0 + card0, mask=hand_mask, other=0.0).to(
            tl.float32
        )
        cm0b = tl.load(card_mass0_ptr + cm_base0 + card1, mask=hand_mask, other=0.0).to(
            tl.float32
        )
        cm1a = tl.load(card_mass1_ptr + cm_base0 + card0, mask=hand_mask, other=0.0).to(
            tl.float32
        )
        cm1b = tl.load(card_mass1_ptr + cm_base0 + card1, mask=hand_mask, other=0.0).to(
            tl.float32
        )
        den0 = 44.0 * total0 + cm0a + cm0b
        den1 = 44.0 * total1 + cm1a + cm1b
        pot = tl.load(
            context_ptr + leaf * context_stride + pot_index,
            mask=valid_leaf,
            other=0.0,
        ).to(tl.float32)
        sdv0 = (num0 / tl.maximum(den0, 1.0e-8)) * pot
        sdv1 = (num1 / tl.maximum(den1, 1.0e-8)) * pot
        if use_pos_neg:
            val0 = (
                tl.maximum(sdv0, 0.0) * pos_scale
                + tl.minimum(sdv0, 0.0) * neg_scale
                + intercept
            )
            val1 = (
                tl.maximum(sdv1, 0.0) * pos_scale
                + tl.minimum(sdv1, 0.0) * neg_scale
                + intercept
            )
        else:
            val0 = sdv0 * baseline_scale
            val1 = sdv1 * baseline_scale
        out_base = leaf * out_stride_row + hand
        tl.store(out_ptr + out_base, val0, mask=hand_mask)
        tl.store(out_ptr + out_base + out_stride_player, val1, mask=hand_mask)

    @triton.jit
    def _turn_rank_cumulative_prefix_hu_kernel(
        beliefs_ptr,
        cache_index_ptr,
        sorted_hands_ptr,
        bin_offsets_ptr,
        rank_cumsum_ptr,
        total_rows: tl.constexpr,
        rank_bins: tl.constexpr,
        hand_count: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_BINS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        river = pid % 48
        player = (pid // 48) & 1
        row = pid // 96
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        hand_offs = tl.arange(0, BLOCK_H)
        base_offsets = (cache_row * 48 + river) * (rank_bins + 1)
        legal_total = tl.load(
            bin_offsets_ptr + base_offsets + rank_bins,
            mask=row < total_rows,
            other=0,
        )
        hand_mask = (row < total_rows) & (hand_offs < legal_total)
        hands = tl.load(
            sorted_hands_ptr + (cache_row * 48 + river) * hand_count + hand_offs,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        vals = tl.load(
            beliefs_ptr + (row * 2 + opp_player) * hand_count + hands,
            mask=hand_mask,
            other=0.0,
        ).to(tl.float32)
        prefix = tl.cumsum(vals, 0)
        bin_offs = tl.arange(0, BLOCK_BINS)
        bin_mask = bin_offs < rank_bins
        ends = tl.load(
            bin_offsets_ptr + base_offsets + bin_offs + 1,
            mask=(row < total_rows) & bin_mask,
            other=0,
        )
        end_idx = ends - 1
        end_idx = tl.maximum(end_idx, 0)
        cumulative = tl.gather(
            prefix,
            end_idx,
            0,
        )
        cumulative = tl.where((row < total_rows) & bin_mask & (ends > 0), cumulative, 0.0)
        out = ((row * 2 + player) * 48 + river) * rank_bins + bin_offs
        tl.store(rank_cumsum_ptr + out, cumulative, mask=(row < total_rows) & bin_mask)

    @triton.jit
    def _turn_rank_cumulative_prefix_hu_grouped_kernel(
        beliefs_ptr,
        grouped_leaf_indices_ptr,
        root_leaf_offsets_ptr,
        root_block_cache_rows_ptr,
        root_block_leaf_starts_ptr,
        sorted_hands_ptr,
        bin_offsets_ptr,
        rank_cumsum_ptr,
        total_leaf_blocks: tl.constexpr,
        rank_bins: tl.constexpr,
        hand_count: tl.constexpr,
        BLOCK_L: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_BINS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        river = pid % 48
        player = (pid // 48) & 1
        leaf_block = pid // 96
        cache_row = tl.load(
            root_block_cache_rows_ptr + leaf_block,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        ).to(tl.int32)
        leaf_start = tl.load(
            root_block_leaf_starts_ptr + leaf_block,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        ).to(tl.int32)
        leaf_end = tl.load(
            root_leaf_offsets_ptr + cache_row + 1,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        ).to(tl.int32)
        leaf_offs = tl.arange(0, BLOCK_L)
        leaf_pos = leaf_start + leaf_offs
        leaf_mask = (leaf_block < total_leaf_blocks) & (leaf_pos < leaf_end)
        leaf_rows = tl.load(
            grouped_leaf_indices_ptr + leaf_pos,
            mask=leaf_mask,
            other=0,
        ).to(tl.int32)

        hand_offs = tl.arange(0, BLOCK_H)
        base_offsets = (cache_row * 48 + river) * (rank_bins + 1)
        legal_total = tl.load(
            bin_offsets_ptr + base_offsets + rank_bins,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        )
        hand_mask = hand_offs < legal_total
        hands = tl.load(
            sorted_hands_ptr + (cache_row * 48 + river) * hand_count + hand_offs,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        opp_player = 1 - player
        vals = tl.load(
            beliefs_ptr
            + (leaf_rows[:, None] * 2 + opp_player) * hand_count
            + hands[None, :],
            mask=leaf_mask[:, None] & hand_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        prefix = tl.cumsum(vals, 1)

        bin_offs = tl.arange(0, BLOCK_BINS)
        bin_mask = bin_offs < rank_bins
        ends = tl.load(
            bin_offsets_ptr + base_offsets + bin_offs + 1,
            mask=(leaf_block < total_leaf_blocks) & bin_mask,
            other=0,
        )
        end_idx = tl.maximum(ends - 1, 0)
        gather_idx = end_idx[None, :] + tl.zeros((BLOCK_L, BLOCK_BINS), dtype=tl.int32)
        cumulative = tl.gather(prefix, gather_idx, 1)
        cumulative = tl.where(
            leaf_mask[:, None] & bin_mask[None, :] & (ends[None, :] > 0),
            cumulative,
            0.0,
        )
        out = (
            ((leaf_rows[:, None] * 2 + player) * 48 + river) * rank_bins
            + bin_offs[None, :]
        )
        tl.store(
            rank_cumsum_ptr + out,
            cumulative,
            mask=leaf_mask[:, None] & bin_mask[None, :],
        )

    @triton.jit
    def _turn_rank_cumulative_prefix_hu_grouped_l2_kernel(
        beliefs_ptr,
        grouped_leaf_indices_ptr,
        root_leaf_offsets_ptr,
        root_block_cache_rows_ptr,
        root_block_leaf_starts_ptr,
        sorted_hands_ptr,
        bin_offsets_ptr,
        rank_cumsum_ptr,
        total_leaf_blocks: tl.constexpr,
        rank_bins: tl.constexpr,
        hand_count: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_BINS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        river = pid % 48
        player = (pid // 48) & 1
        leaf_block = pid // 96
        cache_row = tl.load(
            root_block_cache_rows_ptr + leaf_block,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        ).to(tl.int32)
        leaf_start = tl.load(
            root_block_leaf_starts_ptr + leaf_block,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        ).to(tl.int32)
        leaf_end = tl.load(
            root_leaf_offsets_ptr + cache_row + 1,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        ).to(tl.int32)
        leaf_pos0 = leaf_start
        leaf_pos1 = leaf_start + 1
        leaf_mask0 = (leaf_block < total_leaf_blocks) & (leaf_pos0 < leaf_end)
        leaf_mask1 = (leaf_block < total_leaf_blocks) & (leaf_pos1 < leaf_end)
        leaf0 = tl.load(grouped_leaf_indices_ptr + leaf_pos0, mask=leaf_mask0, other=0).to(tl.int32)
        leaf1 = tl.load(grouped_leaf_indices_ptr + leaf_pos1, mask=leaf_mask1, other=0).to(tl.int32)

        hand_offs = tl.arange(0, BLOCK_H)
        base_offsets = (cache_row * 48 + river) * (rank_bins + 1)
        legal_total = tl.load(
            bin_offsets_ptr + base_offsets + rank_bins,
            mask=leaf_block < total_leaf_blocks,
            other=0,
        )
        hand_mask = hand_offs < legal_total
        hands = tl.load(
            sorted_hands_ptr + (cache_row * 48 + river) * hand_count + hand_offs,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        opp_player = 1 - player
        vals0 = tl.load(
            beliefs_ptr + (leaf0 * 2 + opp_player) * hand_count + hands,
            mask=leaf_mask0 & hand_mask,
            other=0.0,
        ).to(tl.float32)
        vals1 = tl.load(
            beliefs_ptr + (leaf1 * 2 + opp_player) * hand_count + hands,
            mask=leaf_mask1 & hand_mask,
            other=0.0,
        ).to(tl.float32)
        prefix0 = tl.cumsum(vals0, 0)
        prefix1 = tl.cumsum(vals1, 0)

        bin_offs = tl.arange(0, BLOCK_BINS)
        bin_mask = bin_offs < rank_bins
        ends = tl.load(
            bin_offsets_ptr + base_offsets + bin_offs + 1,
            mask=(leaf_block < total_leaf_blocks) & bin_mask,
            other=0,
        )
        end_idx = tl.maximum(ends - 1, 0)
        cumulative0 = tl.gather(prefix0, end_idx, 0)
        cumulative1 = tl.gather(prefix1, end_idx, 0)
        valid_bins = bin_mask & (ends > 0)
        cumulative0 = tl.where(leaf_mask0 & valid_bins, cumulative0, 0.0)
        cumulative1 = tl.where(leaf_mask1 & valid_bins, cumulative1, 0.0)
        out0 = ((leaf0 * 2 + player) * 48 + river) * rank_bins + bin_offs
        out1 = ((leaf1 * 2 + player) * 48 + river) * rank_bins + bin_offs
        tl.store(rank_cumsum_ptr + out0, cumulative0, mask=leaf_mask0 & bin_mask)
        tl.store(rank_cumsum_ptr + out1, cumulative1, mask=leaf_mask1 & bin_mask)

    @triton.jit
    def _turn_rank_score_prefix_hu_kernel(
        beliefs_ptr,
        cache_index_ptr,
        sorted_hands_ptr,
        bin_offsets_ptr,
        rank_score_ptr,
        river_total_ptr,
        total_rows: tl.constexpr,
        rank_bins: tl.constexpr,
        hand_count: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_BINS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        river = pid % 48
        player = (pid // 48) & 1
        row = pid // 96
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        hand_offs = tl.arange(0, BLOCK_H)
        base_offsets = (cache_row * 48 + river) * (rank_bins + 1)
        legal_total = tl.load(
            bin_offsets_ptr + base_offsets + rank_bins,
            mask=row < total_rows,
            other=0,
        )
        hand_mask = (row < total_rows) & (hand_offs < legal_total)
        hands = tl.load(
            sorted_hands_ptr + (cache_row * 48 + river) * hand_count + hand_offs,
            mask=hand_mask,
            other=0,
        ).to(tl.int32)
        vals = tl.load(
            beliefs_ptr + (row * 2 + opp_player) * hand_count + hands,
            mask=hand_mask,
            other=0.0,
        ).to(tl.float32)
        prefix = tl.cumsum(vals, 0)
        total = tl.sum(vals, axis=0)
        tl.store(
            river_total_ptr + (row * 2 + player) * 48 + river,
            total,
            mask=row < total_rows,
        )

        bin_offs = tl.arange(0, BLOCK_BINS)
        bin_mask = bin_offs < rank_bins
        starts = tl.load(
            bin_offsets_ptr + base_offsets + bin_offs,
            mask=(row < total_rows) & bin_mask,
            other=0,
        )
        ends = tl.load(
            bin_offsets_ptr + base_offsets + bin_offs + 1,
            mask=(row < total_rows) & bin_mask,
            other=0,
        )
        end_idx = tl.maximum(ends - 1, 0)
        start_idx = tl.maximum(starts - 1, 0)
        cumulative = tl.gather(prefix, end_idx, 0)
        prev = tl.gather(prefix, start_idx, 0)
        cumulative = tl.where((row < total_rows) & bin_mask & (ends > 0), cumulative, 0.0)
        prev = tl.where((row < total_rows) & bin_mask & (starts > 0), prev, 0.0)
        score = cumulative + prev - total
        out = ((row * 2 + player) * 48 + river) * rank_bins + bin_offs
        tl.store(rank_score_ptr + out, score, mask=(row < total_rows) & bin_mask)

    @triton.jit
    def _turn_rank_mass_bins_hu_kernel(
        beliefs_ptr,
        cache_index_ptr,
        sorted_hands_ptr,
        bin_offsets_ptr,
        rank_mass_ptr,
        total_rows: tl.constexpr,
        rank_bins: tl.constexpr,
        hand_count: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        bin_id = pid % rank_bins
        river = (pid // rank_bins) % 48
        player = (pid // (rank_bins * 48)) & 1
        row = pid // (rank_bins * 96)
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        base_offset = (cache_row * 48 + river) * (rank_bins + 1)
        start = tl.load(
            bin_offsets_ptr + base_offset + bin_id,
            mask=row < total_rows,
            other=0,
        )
        end = tl.load(
            bin_offsets_ptr + base_offset + bin_id + 1,
            mask=row < total_rows,
            other=0,
        )
        count = end - start
        acc = tl.zeros((), dtype=tl.float32)
        for block_start in tl.static_range(0, hand_count, BLOCK_K):
            offs = block_start + tl.arange(0, BLOCK_K)
            mask = (row < total_rows) & (offs < count)
            hands = tl.load(
                sorted_hands_ptr
                + (cache_row * 48 + river) * hand_count
                + start
                + offs,
                mask=mask,
                other=0,
            ).to(tl.int32)
            vals = tl.load(
                beliefs_ptr + (row * 2 + opp_player) * hand_count + hands,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(vals, axis=0)
        tl.store(
            rank_mass_ptr + ((row * 2 + player) * 48 + river) * rank_bins + bin_id,
            acc,
            mask=row < total_rows,
        )

    @triton.jit
    def _turn_rank_mass_hu_kernel(
        beliefs_ptr,
        cache_index_ptr,
        board_ok_ptr,
        rivers_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_mass_ptr,
        total_rows: tl.constexpr,
        hand_count: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        river = pid_rp % 48
        player = (pid_rp // 48) & 1
        row = pid_rp // 96
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (row < total_rows) & (offs < hand_count)
        card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=-1)
        card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=-2)
        river_card = tl.load(
            rivers_ptr + cache_row * 48 + river,
            mask=row < total_rows,
            other=-3,
        )
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=mask,
            other=0,
        )
        ok = board_ok & (card_a != river_card) & (card_b != river_card)
        opp = tl.load(
            beliefs_ptr + (row * 2 + opp_player) * hand_count + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        ranks = tl.load(
            rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
            mask=mask,
            other=0,
        ).to(tl.int32)
        out = ((row * 2 + player) * 48 + river) * rank_bins + ranks
        tl.atomic_add(rank_mass_ptr + out, opp, sem="relaxed", mask=mask & ok)

    @triton.jit
    def _turn_rank_mass_hu_all_rivers_kernel(
        beliefs_ptr,
        cache_index_ptr,
        board_ok_ptr,
        hand_runout_ok_ptr,
        rank_groups_ptr,
        rank_mass_ptr,
        total_rows: tl.constexpr,
        hand_count: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        player = pid_rp & 1
        row = pid_rp // 2
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        opp_player = 1 - player
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = (row < total_rows) & (offs < hand_count)
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=hand_mask,
            other=0,
        )
        opp = tl.load(
            beliefs_ptr + (row * 2 + opp_player) * hand_count + offs,
            mask=hand_mask & board_ok,
            other=0.0,
        ).to(tl.float32)
        for river in tl.static_range(0, 48):
            ok = tl.load(
                hand_runout_ok_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            )
            ranks = tl.load(
                rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            ).to(tl.int32)
            out = ((row * 2 + player) * 48 + river) * rank_bins + ranks
            tl.atomic_add(rank_mass_ptr + out, opp, sem="relaxed", mask=hand_mask & ok)

    @triton.jit
    def _turn_rank_mass_kernel(
        beliefs_ptr,
        cache_index_ptr,
        board_ok_ptr,
        rivers_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_mass_ptr,
        total_rows: tl.constexpr,
        num_players: tl.constexpr,
        hand_count: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        river = pid_rp % 48
        player = (pid_rp // 48) % num_players
        row = pid_rp // (48 * num_players)
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (row < total_rows) & (offs < hand_count)
        card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=-1)
        card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=-2)
        river_card = tl.load(
            rivers_ptr + cache_row * 48 + river,
            mask=row < total_rows,
            other=-3,
        )
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=mask,
            other=0,
        )
        ok = board_ok & (card_a != river_card) & (card_b != river_card)
        own = tl.load(
            beliefs_ptr + (row * num_players + player) * hand_count + offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        total = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for p in tl.static_range(0, num_players):
            total += tl.load(
                beliefs_ptr + (row * num_players + p) * hand_count + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        opp = total - own
        ranks = tl.load(
            rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
            mask=mask,
            other=0,
        ).to(tl.int32)
        out = ((row * num_players + player) * 48 + river) * rank_bins + ranks
        tl.atomic_add(
            rank_mass_ptr + out,
            opp,
            sem="relaxed",
            mask=mask & ok,
        )

    @triton.jit
    def _turn_rank_cumsum_kernel(
        rank_mass_ptr,
        total_groups: tl.constexpr,
        rank_bins: tl.constexpr,
        BLOCK_BINS: tl.constexpr,
    ):
        group = tl.program_id(0)
        offs = tl.arange(0, BLOCK_BINS)
        mask = offs < rank_bins
        ptrs = rank_mass_ptr + group * rank_bins + offs
        mass = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        cumulative = tl.cumsum(mass, 0)
        tl.store(ptrs, cumulative, mask=(group < total_groups) & mask)

    @triton.jit
    def _turn_baseline_from_cumsum_kernel(
        beliefs_ptr,
        context_ptr,
        cache_index_ptr,
        board_ok_ptr,
        hand_runout_ok_ptr,
        rivers_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_cumsum_ptr,
        baseline_ptr,
        total_rows: tl.constexpr,
        num_players: tl.constexpr,
        hand_count: tl.constexpr,
        context_stride: tl.constexpr,
        pot_index: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        has_hand_runout_ok: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_player: tl.constexpr,
        pos_scale: tl.constexpr,
        neg_scale: tl.constexpr,
        intercept: tl.constexpr,
        baseline_scale: tl.constexpr,
        use_pos_neg: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        player = pid_rp % num_players
        row = pid_rp // num_players
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = (row < total_rows) & (offs < hand_count)
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=hand_mask,
            other=0,
        )
        if not has_hand_runout_ok:
            card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=-1)
            card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=-2)
        lower_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        tie_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for river in tl.static_range(0, 48):
            ok = board_ok
            if has_hand_runout_ok:
                ok = tl.load(
                    hand_runout_ok_ptr + (cache_row * 48 + river) * hand_count + offs,
                    mask=hand_mask,
                    other=0,
                )
            else:
                river_card = tl.load(
                    rivers_ptr + cache_row * 48 + river,
                    mask=row < total_rows,
                    other=-3,
                )
                ok = board_ok & (card_a != river_card) & (card_b != river_card)
            ranks = tl.load(
                rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            ).to(tl.int32)
            base = ((row * num_players + player) * 48 + river) * rank_bins
            cum = tl.load(
                rank_cumsum_ptr + base + ranks,
                mask=hand_mask,
                other=0.0,
            ).to(tl.float32)
            prev_rank = ranks - 1
            prev = tl.load(
                rank_cumsum_ptr + base + prev_rank,
                mask=hand_mask & (ranks > 0),
                other=0.0,
            ).to(tl.float32)
            total = tl.load(
                rank_cumsum_ptr + base + (rank_bins - 1),
                mask=row < total_rows,
                other=0.0,
            ).to(tl.float32)
            lower_sum += tl.where(ok, prev, 0.0)
            tie_sum += tl.where(ok, cum - prev, 0.0)
            total_sum += tl.where(ok, total, 0.0)
        safe_total = tl.maximum(total_sum, 1.0e-8)
        equity = (2.0 * lower_sum + tie_sum - total_sum) / safe_total
        equity = tl.where(total_sum > 0.0, equity, 0.0)
        pot = tl.load(
            context_ptr + row * context_stride + pot_index,
            mask=row < total_rows,
            other=0.0,
        ).to(tl.float32)
        sdv = equity * pot
        if use_pos_neg:
            value = (
                tl.maximum(sdv, 0.0) * pos_scale
                + tl.minimum(sdv, 0.0) * neg_scale
                + intercept
            )
        else:
            value = sdv * baseline_scale
        tl.store(
            baseline_ptr + row * out_stride_row + player * out_stride_player + offs,
            value,
            mask=hand_mask,
        )

    @triton.jit
    def _turn_baseline_from_cumsum_hu_both_kernel(
        context_ptr,
        cache_index_ptr,
        board_ok_ptr,
        hand_runout_ok_ptr,
        rank_groups_ptr,
        rank_cumsum_ptr,
        baseline_ptr,
        total_rows: tl.constexpr,
        hand_count: tl.constexpr,
        context_stride: tl.constexpr,
        pot_index: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_player: tl.constexpr,
        pos_scale: tl.constexpr,
        neg_scale: tl.constexpr,
        intercept: tl.constexpr,
        baseline_scale: tl.constexpr,
        use_pos_neg: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_h = tl.program_id(1)
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = (row < total_rows) & (offs < hand_count)
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=hand_mask,
            other=0,
        )
        num0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        den0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        num1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        den1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for river in tl.static_range(0, 48):
            ok = tl.load(
                hand_runout_ok_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            )
            ranks = tl.load(
                rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            ).to(tl.int32)
            base0 = ((row * 2) * 48 + river) * rank_bins
            cum0 = tl.load(
                rank_cumsum_ptr + base0 + ranks,
                mask=hand_mask,
                other=0.0,
            ).to(tl.float32)
            prev_rank = ranks - 1
            prev0 = tl.load(
                rank_cumsum_ptr + base0 + prev_rank,
                mask=hand_mask & (ranks > 0),
                other=0.0,
            ).to(tl.float32)
            total0 = tl.load(
                rank_cumsum_ptr + base0 + (rank_bins - 1),
                mask=row < total_rows,
                other=0.0,
            ).to(tl.float32)
            base1 = ((row * 2 + 1) * 48 + river) * rank_bins
            cum1 = tl.load(
                rank_cumsum_ptr + base1 + ranks,
                mask=hand_mask,
                other=0.0,
            ).to(tl.float32)
            prev1 = tl.load(
                rank_cumsum_ptr + base1 + prev_rank,
                mask=hand_mask & (ranks > 0),
                other=0.0,
            ).to(tl.float32)
            total1 = tl.load(
                rank_cumsum_ptr + base1 + (rank_bins - 1),
                mask=row < total_rows,
                other=0.0,
            ).to(tl.float32)
            num0 += tl.where(ok, cum0 + prev0 - total0, 0.0)
            den0 += tl.where(ok, total0, 0.0)
            num1 += tl.where(ok, cum1 + prev1 - total1, 0.0)
            den1 += tl.where(ok, total1, 0.0)
        den0 = tl.where(board_ok, den0, 0.0)
        den1 = tl.where(board_ok, den1, 0.0)
        eq0 = num0 / tl.maximum(den0, 1.0e-8)
        eq1 = num1 / tl.maximum(den1, 1.0e-8)
        eq0 = tl.where(den0 > 0.0, eq0, 0.0)
        eq1 = tl.where(den1 > 0.0, eq1, 0.0)
        pot = tl.load(
            context_ptr + row * context_stride + pot_index,
            mask=row < total_rows,
            other=0.0,
        ).to(tl.float32)
        sdv0 = eq0 * pot
        sdv1 = eq1 * pot
        if use_pos_neg:
            value0 = (
                tl.maximum(sdv0, 0.0) * pos_scale
                + tl.minimum(sdv0, 0.0) * neg_scale
                + intercept
            )
            value1 = (
                tl.maximum(sdv1, 0.0) * pos_scale
                + tl.minimum(sdv1, 0.0) * neg_scale
                + intercept
            )
        else:
            value0 = sdv0 * baseline_scale
            value1 = sdv1 * baseline_scale
        out_base = row * out_stride_row + offs
        tl.store(
            baseline_ptr + out_base,
            value0,
            mask=hand_mask,
        )
        tl.store(
            baseline_ptr + out_base + out_stride_player,
            value1,
            mask=hand_mask,
        )

    @triton.jit
    def _turn_baseline_from_cumsum_mask_kernel(
        context_ptr,
        cache_index_ptr,
        hand_runout_mask_ptr,
        rank_groups_ptr,
        rank_cumsum_ptr,
        baseline_ptr,
        total_rows: tl.constexpr,
        num_players: tl.constexpr,
        hand_count: tl.constexpr,
        context_stride: tl.constexpr,
        pot_index: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_player: tl.constexpr,
        pos_scale: tl.constexpr,
        neg_scale: tl.constexpr,
        intercept: tl.constexpr,
        baseline_scale: tl.constexpr,
        use_pos_neg: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        player = pid_rp % num_players
        row = pid_rp // num_players
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = (row < total_rows) & (offs < hand_count)
        runout_bits = tl.load(
            hand_runout_mask_ptr + cache_row * hand_count + offs,
            mask=hand_mask,
            other=0,
        ).to(tl.int64)
        numerator = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for river in tl.static_range(0, 48):
            ranks = tl.load(
                rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            ).to(tl.int32)
            base = ((row * num_players + player) * 48 + river) * rank_bins
            cum = tl.load(
                rank_cumsum_ptr + base + ranks,
                mask=hand_mask,
                other=0.0,
            ).to(tl.float32)
            prev_rank = ranks - 1
            prev = tl.load(
                rank_cumsum_ptr + base + prev_rank,
                mask=hand_mask & (ranks > 0),
                other=0.0,
            ).to(tl.float32)
            total = tl.load(
                rank_cumsum_ptr + base + (rank_bins - 1),
                mask=row < total_rows,
                other=0.0,
            ).to(tl.float32)
            ok = ((runout_bits >> river) & 1) != 0
            numerator += tl.where(ok, cum + prev - total, 0.0)
            total_sum += tl.where(ok, total, 0.0)
        safe_total = tl.maximum(total_sum, 1.0e-8)
        equity = numerator / safe_total
        equity = tl.where(total_sum > 0.0, equity, 0.0)
        pot = tl.load(
            context_ptr + row * context_stride + pot_index,
            mask=row < total_rows,
            other=0.0,
        ).to(tl.float32)
        sdv = equity * pot
        if use_pos_neg:
            value = (
                tl.maximum(sdv, 0.0) * pos_scale
                + tl.minimum(sdv, 0.0) * neg_scale
                + intercept
            )
        else:
            value = sdv * baseline_scale
        tl.store(
            baseline_ptr + row * out_stride_row + player * out_stride_player + offs,
            value,
            mask=hand_mask,
        )

    @triton.jit
    def _turn_baseline_from_cumsum_slots_kernel(
        context_ptr,
        cache_index_ptr,
        board_ok_ptr,
        card_river_slots_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_cumsum_ptr,
        baseline_ptr,
        total_rows: tl.constexpr,
        num_players: tl.constexpr,
        hand_count: tl.constexpr,
        context_stride: tl.constexpr,
        pot_index: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_player: tl.constexpr,
        pos_scale: tl.constexpr,
        neg_scale: tl.constexpr,
        intercept: tl.constexpr,
        baseline_scale: tl.constexpr,
        use_pos_neg: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        player = pid_rp % num_players
        row = pid_rp // num_players
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = (row < total_rows) & (offs < hand_count)
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=hand_mask,
            other=0,
        )
        card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=0)
        card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=0)
        slot_a = tl.load(
            card_river_slots_ptr + cache_row * 52 + card_a,
            mask=hand_mask,
            other=-1,
        ).to(tl.int32)
        slot_b = tl.load(
            card_river_slots_ptr + cache_row * 52 + card_b,
            mask=hand_mask,
            other=-1,
        ).to(tl.int32)

        numerator = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_all = tl.zeros((), dtype=tl.float32)
        total_a = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_b = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for river in tl.static_range(0, 48):
            ranks = tl.load(
                rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            ).to(tl.int32)
            base = ((row * num_players + player) * 48 + river) * rank_bins
            cum = tl.load(
                rank_cumsum_ptr + base + ranks,
                mask=hand_mask,
                other=0.0,
            ).to(tl.float32)
            prev_rank = ranks - 1
            prev = tl.load(
                rank_cumsum_ptr + base + prev_rank,
                mask=hand_mask & (ranks > 0),
                other=0.0,
            ).to(tl.float32)
            total = tl.load(
                rank_cumsum_ptr + base + (rank_bins - 1),
                mask=row < total_rows,
                other=0.0,
            ).to(tl.float32)
            ok = board_ok & (slot_a != river) & (slot_b != river)
            numerator += tl.where(ok, cum + prev - total, 0.0)
            total_all += total
            total_a += tl.where(slot_a == river, total, 0.0)
            total_b += tl.where(slot_b == river, total, 0.0)
        total_sum = total_all - total_a - total_b
        total_sum = tl.where(board_ok, total_sum, 0.0)
        safe_total = tl.maximum(total_sum, 1.0e-8)
        equity = numerator / safe_total
        equity = tl.where(total_sum > 0.0, equity, 0.0)
        pot = tl.load(
            context_ptr + row * context_stride + pot_index,
            mask=row < total_rows,
            other=0.0,
        ).to(tl.float32)
        sdv = equity * pot
        if use_pos_neg:
            value = (
                tl.maximum(sdv, 0.0) * pos_scale
                + tl.minimum(sdv, 0.0) * neg_scale
                + intercept
            )
        else:
            value = sdv * baseline_scale
        tl.store(
            baseline_ptr + row * out_stride_row + player * out_stride_player + offs,
            value,
            mask=hand_mask,
        )

    @triton.jit
    def _turn_baseline_from_rank_score_kernel(
        context_ptr,
        cache_index_ptr,
        board_ok_ptr,
        card_river_slots_ptr,
        card_a_ptr,
        card_b_ptr,
        rank_groups_ptr,
        rank_score_ptr,
        river_total_ptr,
        baseline_ptr,
        total_rows: tl.constexpr,
        num_players: tl.constexpr,
        hand_count: tl.constexpr,
        context_stride: tl.constexpr,
        pot_index: tl.constexpr,
        rank_bins: tl.constexpr,
        has_cache_index: tl.constexpr,
        out_stride_row: tl.constexpr,
        out_stride_player: tl.constexpr,
        pos_scale: tl.constexpr,
        neg_scale: tl.constexpr,
        intercept: tl.constexpr,
        baseline_scale: tl.constexpr,
        use_pos_neg: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        pid_rp = tl.program_id(0)
        pid_h = tl.program_id(1)
        player = pid_rp % num_players
        row = pid_rp // num_players
        cache_row = row
        if has_cache_index:
            cache_row = tl.load(cache_index_ptr + row, mask=row < total_rows, other=0)
        offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        hand_mask = (row < total_rows) & (offs < hand_count)
        board_ok = tl.load(
            board_ok_ptr + cache_row * hand_count + offs,
            mask=hand_mask,
            other=0,
        )
        card_a = tl.load(card_a_ptr + offs, mask=offs < hand_count, other=0)
        card_b = tl.load(card_b_ptr + offs, mask=offs < hand_count, other=0)
        slot_a = tl.load(
            card_river_slots_ptr + cache_row * 52 + card_a,
            mask=hand_mask,
            other=-1,
        ).to(tl.int32)
        slot_b = tl.load(
            card_river_slots_ptr + cache_row * 52 + card_b,
            mask=hand_mask,
            other=-1,
        ).to(tl.int32)

        numerator = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_all = tl.zeros((), dtype=tl.float32)
        total_a = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_b = tl.zeros((BLOCK_H,), dtype=tl.float32)
        total_base = (row * num_players + player) * 48
        for river in tl.static_range(0, 48):
            total = tl.load(
                river_total_ptr + total_base + river,
                mask=row < total_rows,
                other=0.0,
            ).to(tl.float32)
            total_all += total
            total_a += tl.where(slot_a == river, total, 0.0)
            total_b += tl.where(slot_b == river, total, 0.0)
            ranks = tl.load(
                rank_groups_ptr + (cache_row * 48 + river) * hand_count + offs,
                mask=hand_mask,
                other=0,
            ).to(tl.int32)
            score = tl.load(
                rank_score_ptr + ((row * num_players + player) * 48 + river) * rank_bins + ranks,
                mask=hand_mask,
                other=0.0,
            ).to(tl.float32)
            ok = board_ok & (slot_a != river) & (slot_b != river)
            numerator += tl.where(ok, score, 0.0)
        total_sum = total_all - total_a - total_b
        total_sum = tl.where(board_ok, total_sum, 0.0)
        safe_total = tl.maximum(total_sum, 1.0e-8)
        equity = numerator / safe_total
        equity = tl.where(total_sum > 0.0, equity, 0.0)
        pot = tl.load(
            context_ptr + row * context_stride + pot_index,
            mask=row < total_rows,
            other=0.0,
        ).to(tl.float32)
        sdv = equity * pot
        if use_pos_neg:
            value = (
                tl.maximum(sdv, 0.0) * pos_scale
                + tl.minimum(sdv, 0.0) * neg_scale
                + intercept
            )
        else:
            value = sdv * baseline_scale
        tl.store(
            baseline_ptr + row * out_stride_row + player * out_stride_player + offs,
            value,
            mask=hand_mask,
        )


@dataclass(frozen=True)
class TurnRangeEquityConfig:
    rank_bins: int
    chunk_size: int
    blockers: bool
    baseline_scale: float
    pot_power: float
    pos_scale: float
    neg_scale: float
    intercept: float
    runout_std: bool = False
    decomposition: bool = False


def _leaf_grouping_tensors(
    leaf_to_cache: torch.Tensor | None,
    *,
    leaf_count: int,
    cache_rows: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if block_size <= 0:
        return None, None, None, None
    if leaf_to_cache is not None:
        grouped_leaf_indices = torch.argsort(leaf_to_cache).to(torch.int32).contiguous()
        root_counts = torch.bincount(
            leaf_to_cache.long(),
            minlength=cache_rows,
        ).to(torch.int32)
    else:
        grouped_leaf_indices = torch.arange(leaf_count, dtype=torch.int32, device=device)
        root_counts = torch.ones(cache_rows, dtype=torch.int32, device=device)
    root_leaf_offsets = torch.empty(cache_rows + 1, dtype=torch.int32, device=device)
    root_leaf_offsets[0] = 0
    root_leaf_offsets[1:] = root_counts.cumsum(dim=0)
    blocks_per_root = torch.div(
        root_counts + int(block_size) - 1,
        int(block_size),
        rounding_mode="floor",
    )
    block_offsets = torch.empty(cache_rows + 1, dtype=torch.int32, device=device)
    block_offsets[0] = 0
    block_offsets[1:] = blocks_per_root.cumsum(dim=0)
    total_blocks = int(block_offsets[-1].item())
    if total_blocks <= 0:
        return grouped_leaf_indices, root_leaf_offsets, None, None
    root_ids = torch.arange(cache_rows, dtype=torch.int32, device=device)
    root_block_cache_rows = torch.repeat_interleave(root_ids, blocks_per_root).contiguous()
    global_blocks = torch.arange(total_blocks, dtype=torch.int32, device=device)
    repeated_block_offsets = torch.repeat_interleave(block_offsets[:-1], blocks_per_root)
    local_blocks = global_blocks - repeated_block_offsets
    root_block_leaf_starts = (
        root_leaf_offsets.index_select(0, root_block_cache_rows.long())
        + local_blocks * int(block_size)
    ).contiguous()
    return (
        grouped_leaf_indices,
        root_leaf_offsets,
        root_block_cache_rows,
        root_block_leaf_starts,
    )


def _uniform_leaf_count(root_leaf_offsets: torch.Tensor | None) -> int:
    if root_leaf_offsets is None or int(root_leaf_offsets.numel()) <= 1:
        return 0
    counts = root_leaf_offsets[1:] - root_leaf_offsets[:-1]
    first = int(counts[0].item())
    if first <= 0:
        return 0
    if bool((counts == first).all().item()):
        return first
    return 0


def _leaf_count_bucket_tensors(
    root_leaf_offsets: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if root_leaf_offsets is None or int(root_leaf_offsets.numel()) <= 1:
        return None, None, None
    counts = root_leaf_offsets[1:] - root_leaf_offsets[:-1]
    if int(counts.numel()) == 0:
        return None, None, None
    max_count = int(counts.max().item())
    bucket_limits_list = []
    limit = 8
    while limit < max_count:
        bucket_limits_list.append(limit)
        limit *= 2
    bucket_limits_list.append(limit)
    bucket_limits = torch.tensor(
        bucket_limits_list,
        dtype=torch.int32,
        device=counts.device,
    )
    bucket_ids = torch.bucketize(counts, bucket_limits)
    used_ids, inverse = torch.unique(bucket_ids, sorted=True, return_inverse=True)
    values = bucket_limits.index_select(0, used_ids.long())
    order = torch.argsort(inverse)
    bucket_sizes = torch.bincount(inverse, minlength=int(values.numel())).to(torch.int32)
    offsets = torch.empty(int(values.numel()) + 1, dtype=torch.int32, device=counts.device)
    offsets[0] = 0
    offsets[1:] = bucket_sizes.cumsum(dim=0)
    return values.to(torch.int32).contiguous(), order.to(torch.int32).contiguous(), offsets


def _leaf_count_bucket_specs(
    counts: torch.Tensor | None,
    offsets: torch.Tensor | None,
) -> tuple[tuple[int, int, int], ...]:
    if counts is None or offsets is None:
        return ()
    counts_cpu = counts.detach().cpu().tolist()
    offsets_cpu = offsets.detach().cpu().tolist()
    return tuple(
        (int(count), int(offsets_cpu[idx]), int(offsets_cpu[idx + 1]))
        for idx, count in enumerate(counts_cpu)
    )


@dataclass(frozen=True)
class TurnRangeEquityBoardCache:
    """Board-only turn equity data reusable while CFR beliefs change.

    Large tensors are cache-row aligned, usually one row per unique turn board.
    ``leaf_to_cache`` maps model leaf rows to cache rows when those differ.
    ``rank_groups`` is already binned to the configured rank-bin count. It is
    uint8 when possible to reduce final-kernel cache bandwidth.
    """

    board4: torch.Tensor
    rivers: torch.Tensor
    rank_groups: torch.Tensor
    board_ok: torch.Tensor
    hand_runout_ok: torch.Tensor | None
    leaf_to_cache: torch.Tensor | None = None
    sorted_hands: torch.Tensor | None = None
    bin_offsets: torch.Tensor | None = None
    card_river_slots: torch.Tensor | None = None
    hand_runout_mask: torch.Tensor | None = None
    legal_hands: torch.Tensor | None = None
    pair_payoff: torch.Tensor | None = None
    card_incidence: torch.Tensor | None = None
    legal_hand_cards: torch.Tensor | None = None
    pair_operator_bucketed: bool = False
    grouped_leaf_indices: torch.Tensor | None = None
    root_leaf_offsets: torch.Tensor | None = None
    root_block_cache_rows: torch.Tensor | None = None
    root_block_leaf_starts: torch.Tensor | None = None
    root_block_leaf_size: int = 0
    uniform_leaf_count: int = 0
    leaf_count_bucket_counts: torch.Tensor | None = None
    leaf_count_bucket_roots: torch.Tensor | None = None
    leaf_count_bucket_offsets: torch.Tensor | None = None
    leaf_count_bucket_specs: tuple[tuple[int, int, int], ...] = ()

    def slice(self, rows: torch.Tensor) -> "TurnRangeEquityBoardCache":
        leaf_count = (
            int(self.leaf_to_cache.numel())
            if self.leaf_to_cache is not None
            else int(self.board4.shape[0])
        )
        if int(rows.numel()) == leaf_count:
            identity = torch.arange(
                leaf_count,
                dtype=rows.dtype,
                device=rows.device,
            )
            if bool((rows == identity).all().item()):
                return self
        leaf_to_cache = (
            rows.to(torch.int32).contiguous()
            if self.leaf_to_cache is None
            else self.leaf_to_cache.index_select(0, rows).to(torch.int32).contiguous()
        )
        (
            grouped_leaf_indices,
            root_leaf_offsets,
            root_block_cache_rows,
            root_block_leaf_starts,
        ) = _leaf_grouping_tensors(
            leaf_to_cache,
            leaf_count=int(rows.numel()),
            cache_rows=int(self.board4.shape[0]),
            block_size=int(self.root_block_leaf_size),
            device=rows.device,
        )
        uniform_leaf_count = _uniform_leaf_count(root_leaf_offsets)
        (
            leaf_count_bucket_counts,
            leaf_count_bucket_roots,
            leaf_count_bucket_offsets,
        ) = _leaf_count_bucket_tensors(root_leaf_offsets)
        leaf_count_bucket_specs = _leaf_count_bucket_specs(
            leaf_count_bucket_counts,
            leaf_count_bucket_offsets,
        )
        return TurnRangeEquityBoardCache(
            board4=self.board4,
            rivers=self.rivers,
            rank_groups=self.rank_groups,
            board_ok=self.board_ok,
            hand_runout_ok=self.hand_runout_ok,
            leaf_to_cache=leaf_to_cache,
            sorted_hands=self.sorted_hands,
            bin_offsets=self.bin_offsets,
            card_river_slots=self.card_river_slots,
            hand_runout_mask=self.hand_runout_mask,
            legal_hands=None,
            pair_payoff=None,
            card_incidence=None,
            legal_hand_cards=None,
            pair_operator_bucketed=False,
            grouped_leaf_indices=grouped_leaf_indices,
            root_leaf_offsets=root_leaf_offsets,
            root_block_cache_rows=root_block_cache_rows,
            root_block_leaf_starts=root_block_leaf_starts,
            root_block_leaf_size=self.root_block_leaf_size,
            uniform_leaf_count=uniform_leaf_count,
            leaf_count_bucket_counts=leaf_count_bucket_counts,
            leaf_count_bucket_roots=leaf_count_bucket_roots,
            leaf_count_bucket_offsets=leaf_count_bucket_offsets,
            leaf_count_bucket_specs=leaf_count_bucket_specs,
        )


def river_rank_groups(board: torch.Tensor) -> torch.Tensor:
    if board.device.type == "cuda" and triton_is_available():
        try:
            hand_ranks, sorted_indices = rank_hands_triton(board.int())
        except Exception:
            hand_ranks, sorted_indices = rank_hands_torch(board.int())
    else:
        hand_ranks, sorted_indices = rank_hands_torch(board.int())
    sorted_ranks = hand_ranks.gather(1, sorted_indices.long())
    group_start = sorted_ranks[:, 1:] != sorted_ranks[:, :-1]
    group_start = torch.cat(
        (
            torch.ones(
                sorted_ranks.shape[0],
                1,
                device=sorted_ranks.device,
                dtype=torch.bool,
            ),
            group_start,
        ),
        dim=1,
    )
    sorted_groups = group_start.to(dtype=torch.long).cumsum(dim=1) - 1
    rank_groups = torch.empty_like(sorted_groups)
    rank_groups.scatter_(1, sorted_indices.long(), sorted_groups)
    return rank_groups


def turn_runout_boards(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    board4 = board[:, :4].long()
    cards = torch.arange(52, dtype=torch.long, device=board.device)
    cards_expanded = cards.view(1, 52).expand(board4.shape[0], -1)
    river_ok = (cards_expanded[:, :, None] != board4[:, None, :]).all(dim=2)
    removed_before = (board4[:, None, :] < cards_expanded[:, :, None]).sum(dim=2)
    river_slots = (cards_expanded - removed_before).clamp(min=0, max=47)
    river_src = torch.where(
        river_ok,
        cards_expanded + 1,
        torch.zeros_like(cards_expanded),
    )
    rivers = torch.zeros(board4.shape[0], 48, dtype=torch.long, device=board.device)
    rivers.scatter_reduce_(
        1,
        river_slots,
        river_src,
        reduce="amax",
        include_self=True,
    )
    rivers = (rivers - 1).clamp_min(0)
    full = torch.cat(
        (
            board4[:, None, :].expand(-1, 48, -1),
            rivers[:, :, None],
        ),
        dim=2,
    ).reshape(-1, 5)
    return rivers, full


def _legal_hands_from_board_ok(board_ok: torch.Tensor) -> torch.Tensor:
    rows = int(board_ok.shape[0])
    legal_count = int(board_ok.sum(dim=1).min().item())
    hand_ids = torch.arange(NUM_HANDS, dtype=torch.int32, device=board_ok.device)
    key = torch.where(
        board_ok,
        hand_ids.view(1, NUM_HANDS),
        torch.full((rows, NUM_HANDS), NUM_HANDS, dtype=torch.int32, device=board_ok.device),
    )
    return torch.argsort(key, dim=1)[:, :legal_count].to(torch.int64).contiguous()


def _build_pair_payoff_triton(
    *,
    rank_groups: torch.Tensor,
    hand_runout_ok: torch.Tensor,
    legal_hands: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if (
        triton is None
        or rank_groups.device.type != "cuda"
        or not triton_is_available()
        or _env_int("P2_TURN_EQUITY_PAIR_PRECOMPUTE_TRITON", 1) == 0
    ):
        return None
    rows = int(rank_groups.shape[0])
    hand_count = int(legal_hands.shape[1])
    pair_payoff = torch.empty(
        rows,
        hand_count,
        hand_count,
        dtype=dtype,
        device=rank_groups.device,
    )
    block_i = int(_env_int("P2_TURN_EQUITY_PAIR_PRECOMPUTE_BLOCK_I", 8))
    block_j = int(_env_int("P2_TURN_EQUITY_PAIR_PRECOMPUTE_BLOCK_J", 32))
    grid = (
        rows,
        triton.cdiv(hand_count, block_i),
        triton.cdiv(hand_count, block_j),
    )
    _turn_pair_payoff_precompute_kernel[grid](
        rank_groups,
        hand_runout_ok,
        legal_hands,
        pair_payoff,
        hand_count,
        BLOCK_I=block_i,
        BLOCK_J=block_j,
    )
    return pair_payoff


def _build_pair_payoff_torch(
    *,
    rank_groups: torch.Tensor,
    hand_runout_ok: torch.Tensor,
    legal_hands: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows = int(rank_groups.shape[0])
    hand_count = int(legal_hands.shape[1])
    gather_idx = legal_hands[:, None, :].expand(-1, 48, -1)
    compact_ranks = rank_groups.long().gather(2, gather_idx)
    compact_ok = hand_runout_ok.gather(2, gather_idx)
    pair_payoff = torch.empty(
        rows,
        hand_count,
        hand_count,
        dtype=dtype,
        device=rank_groups.device,
    )
    for row in range(rows):
        payoff = torch.zeros(
            hand_count,
            hand_count,
            dtype=torch.float32,
            device=rank_groups.device,
        )
        for river in range(48):
            ok = compact_ok[row, river]
            ranks = compact_ranks[row, river].to(torch.int16)
            pair_ok = ok[:, None] & ok[None, :]
            diff = ranks[:, None] - ranks[None, :]
            payoff += torch.sign(diff).to(torch.float32) * pair_ok.to(torch.float32)
        pair_payoff[row] = payoff.to(dtype)
    return pair_payoff


def _build_turn_pair_operator_cache(
    *,
    rank_groups: torch.Tensor,
    hand_runout_ok: torch.Tensor,
    board_ok: torch.Tensor,
    dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    legal_hands = _legal_hands_from_board_ok(board_ok)
    rows = int(rank_groups.shape[0])
    hand_count = int(legal_hands.shape[1])
    pair_payoff = _build_pair_payoff_triton(
        rank_groups=rank_groups,
        hand_runout_ok=hand_runout_ok,
        legal_hands=legal_hands,
        dtype=dtype,
    )
    if pair_payoff is None:
        pair_payoff = _build_pair_payoff_torch(
            rank_groups=rank_groups,
            hand_runout_ok=hand_runout_ok,
            legal_hands=legal_hands,
            dtype=dtype,
        )
    combos = hand_combos_tensor(device=rank_groups.device)
    legal_hand_cards = combos[legal_hands.long()].contiguous()
    card_incidence = torch.zeros(
        rows,
        hand_count,
        52,
        dtype=dtype,
        device=rank_groups.device,
    )
    card_incidence.scatter_(
        2,
        legal_hand_cards,
        torch.ones_like(legal_hand_cards, dtype=dtype),
    )
    return (
        legal_hands.contiguous(),
        pair_payoff.contiguous(),
        card_incidence.contiguous(),
        legal_hand_cards.contiguous(),
    )


def build_turn_range_equity_board_cache(
    board4: torch.Tensor,
    *,
    rank_bins: int,
    rank_groups_fn: RankGroupsFn = river_rank_groups,
    rank_chunk_size: int = 256,
    include_hand_runout_ok: bool = True,
    dedupe_boards: bool = False,
    balanced_bins: bool = True,
    include_sorted_bins: bool = False,
    include_card_river_slots: bool = True,
    include_hand_runout_mask: bool = False,
    include_pair_operator: bool = False,
    pair_operator_dtype: torch.dtype = torch.float16,
    grouped_leaf_block_size: int = 2,
) -> TurnRangeEquityBoardCache:
    leaf_board = board4[:, :4].long()
    leaf_count = int(leaf_board.shape[0])
    if dedupe_boards and leaf_board.shape[0] > 0:
        board, leaf_to_cache = torch.unique(
            leaf_board,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        leaf_to_cache = leaf_to_cache.to(torch.int32).contiguous()
    else:
        board = leaf_board
        leaf_to_cache = None
    device = board.device
    combos = hand_combos_tensor(device=device)
    card_a = combos[:, 0]
    card_b = combos[:, 1]
    rows = int(board.shape[0])
    rivers = torch.empty(rows, 48, dtype=torch.long, device=device)
    rank_group_dtype = torch.uint8 if rank_bins <= 256 else torch.int16
    rank_groups = torch.empty(rows, 48, NUM_HANDS, dtype=rank_group_dtype, device=device)
    for start in range(0, rows, int(rank_chunk_size)):
        end = min(rows, start + int(rank_chunk_size))
        chunk_rivers, full_boards = turn_runout_boards(board[start:end])
        rivers[start:end] = chunk_rivers
        chunk_groups = rank_groups_fn(full_boards).clamp_min(0)
        if balanced_bins:
            num_groups = chunk_groups.amax(dim=1, keepdim=True).add_(1).clamp_min_(1)
            chunk_groups = (chunk_groups * rank_bins) // num_groups
        chunk_groups = chunk_groups.clamp(max=rank_bins - 1)
        rank_groups[start:end] = chunk_groups.view(
            end - start,
            48,
            NUM_HANDS,
        ).to(rank_group_dtype)
    board_ok = (
        (card_a[None, :, None] != board[:, None, :])
        & (card_b[None, :, None] != board[:, None, :])
    ).all(dim=2)
    hand_runout_ok = None
    sorted_hands = None
    bin_offsets = None
    card_river_slots = None
    hand_runout_mask = None
    legal_hands = None
    pair_payoff = None
    card_incidence = None
    legal_hand_cards = None
    grouped_leaf_indices = None
    root_leaf_offsets = None
    root_block_cache_rows = None
    root_block_leaf_starts = None
    river_blocked = None
    (
        grouped_leaf_indices,
        root_leaf_offsets,
        root_block_cache_rows,
        root_block_leaf_starts,
    ) = _leaf_grouping_tensors(
        leaf_to_cache,
        leaf_count=leaf_count,
        cache_rows=rows,
        block_size=int(grouped_leaf_block_size),
        device=device,
    )
    uniform_leaf_count = _uniform_leaf_count(root_leaf_offsets)
    allow_ragged_pair_operator = _env_int("P2_TURN_EQUITY_PAIR_OPERATOR_RAGGED", 1) != 0
    build_pair_operator = include_pair_operator and (
        uniform_leaf_count > 0 or allow_ragged_pair_operator
    )
    (
        leaf_count_bucket_counts,
        leaf_count_bucket_roots,
        leaf_count_bucket_offsets,
    ) = _leaf_count_bucket_tensors(root_leaf_offsets)
    leaf_count_bucket_specs = _leaf_count_bucket_specs(
        leaf_count_bucket_counts,
        leaf_count_bucket_offsets,
    )
    if include_card_river_slots:
        card_river_slots = torch.full(
            (rows, 52),
            -1,
            dtype=torch.int16,
            device=device,
        )
        river_slots = torch.arange(48, dtype=torch.int16, device=device)
        card_river_slots.scatter_(
            1,
            rivers,
            river_slots.view(1, 48).expand(rows, -1),
        )
    if include_hand_runout_ok:
        river_blocked = (card_a[None, None, :] == rivers[:, :, None]) | (
            card_b[None, None, :] == rivers[:, :, None]
        )
        hand_runout_ok = board_ok[:, None, :] & ~river_blocked
    if include_hand_runout_mask:
        if river_blocked is None:
            river_blocked = (card_a[None, None, :] == rivers[:, :, None]) | (
                card_b[None, None, :] == rivers[:, :, None]
            )
        legal_runouts = board_ok[:, None, :] & ~river_blocked
        river_bits = (
            torch.ones((), dtype=torch.int64, device=device)
            << torch.arange(48, dtype=torch.int64, device=device)
        )
        hand_runout_mask = (
            legal_runouts.to(torch.int64) * river_bits.view(1, 48, 1)
        ).sum(dim=1)
    if include_sorted_bins:
        if river_blocked is None:
            river_blocked = (card_a[None, None, :] == rivers[:, :, None]) | (
                card_b[None, None, :] == rivers[:, :, None]
            )
        legal = board_ok[:, None, :] & ~river_blocked
        hand_ids = torch.arange(NUM_HANDS, dtype=torch.int32, device=device)
        bin_counts = torch.zeros(rows, 48, rank_bins, dtype=torch.int32, device=device)
        bin_counts.scatter_add_(
            2,
            rank_groups.long(),
            legal.to(torch.int32),
        )
        bin_offsets = torch.empty(
            rows,
            48,
            rank_bins + 1,
            dtype=torch.int32,
            device=device,
        )
        bin_offsets[:, :, 0] = 0
        bin_offsets[:, :, 1:] = bin_counts.cumsum(dim=2)
        rank_major_key = torch.where(
            legal,
            rank_groups.long() * NUM_HANDS + hand_ids.view(1, 1, NUM_HANDS),
            torch.full(
                (),
                rank_bins * NUM_HANDS + NUM_HANDS,
                dtype=torch.long,
                device=device,
            ),
        )
        sorted_hands = torch.argsort(rank_major_key, dim=2).to(torch.int16)
    if build_pair_operator:
        if hand_runout_ok is None:
            if river_blocked is None:
                river_blocked = (card_a[None, None, :] == rivers[:, :, None]) | (
                    card_b[None, None, :] == rivers[:, :, None]
                )
            hand_runout_ok = board_ok[:, None, :] & ~river_blocked
        legal_hands, pair_payoff, card_incidence, legal_hand_cards = (
            _build_turn_pair_operator_cache(
                rank_groups=rank_groups,
                hand_runout_ok=hand_runout_ok,
                board_ok=board_ok,
                dtype=pair_operator_dtype,
            )
        )
        if leaf_count_bucket_roots is not None:
            bucket_order = leaf_count_bucket_roots.long()
            legal_hands = legal_hands.index_select(0, bucket_order).contiguous()
            pair_payoff = pair_payoff.index_select(0, bucket_order).contiguous()
            card_incidence = card_incidence.index_select(0, bucket_order).contiguous()
            legal_hand_cards = legal_hand_cards.index_select(0, bucket_order).contiguous()
    return TurnRangeEquityBoardCache(
        board4=board,
        rivers=rivers,
        rank_groups=rank_groups,
        board_ok=board_ok,
        hand_runout_ok=hand_runout_ok,
        leaf_to_cache=leaf_to_cache,
        sorted_hands=sorted_hands,
        bin_offsets=bin_offsets,
        card_river_slots=card_river_slots,
        hand_runout_mask=hand_runout_mask,
        legal_hands=legal_hands,
        pair_payoff=pair_payoff,
        card_incidence=card_incidence,
        legal_hand_cards=legal_hand_cards,
        pair_operator_bucketed=build_pair_operator and leaf_count_bucket_roots is not None,
        grouped_leaf_indices=grouped_leaf_indices,
        root_leaf_offsets=root_leaf_offsets,
        root_block_cache_rows=root_block_cache_rows,
        root_block_leaf_starts=root_block_leaf_starts,
        root_block_leaf_size=int(grouped_leaf_block_size),
        uniform_leaf_count=uniform_leaf_count,
        leaf_count_bucket_counts=leaf_count_bucket_counts,
        leaf_count_bucket_roots=leaf_count_bucket_roots,
        leaf_count_bucket_offsets=leaf_count_bucket_offsets,
        leaf_count_bucket_specs=leaf_count_bucket_specs,
    )


def player_spr_context(context: torch.Tensor, num_players: int) -> torch.Tensor:
    base = ValueScalarContext.NUM_SCALAR_CONTEXT.value
    stride = PlayerContext.NUM_PLAYER_CONTEXT.value
    spr_idx = base + torch.arange(
        num_players,
        device=context.device,
        dtype=torch.long,
    ) * stride + PlayerContext.SPR.value
    return context.index_select(1, spr_idx)


def _can_use_triton_turn_baseline(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    config: TurnRangeEquityConfig,
    board_cache: TurnRangeEquityBoardCache | None,
) -> bool:
    return (
        triton is not None
        and player_beliefs.device.type == "cuda"
        and features.context.device.type == "cuda"
        and board_cache is not None
        and board_cache.rank_groups.is_contiguous()
        and board_cache.rivers.is_contiguous()
        and board_cache.board_ok.is_contiguous()
        and (
            board_cache.leaf_to_cache is None
            or board_cache.leaf_to_cache.is_contiguous()
        )
        and (
            board_cache.sorted_hands is None
            or board_cache.sorted_hands.is_contiguous()
        )
        and (
            board_cache.bin_offsets is None
            or board_cache.bin_offsets.is_contiguous()
        )
        and (
            board_cache.hand_runout_ok is None
            or board_cache.hand_runout_ok.is_contiguous()
        )
        and (
            board_cache.card_river_slots is None
            or board_cache.card_river_slots.is_contiguous()
        )
        and (
            board_cache.hand_runout_mask is None
            or board_cache.hand_runout_mask.is_contiguous()
        )
        and (
            board_cache.grouped_leaf_indices is None
            or board_cache.grouped_leaf_indices.is_contiguous()
        )
        and (
            board_cache.root_leaf_offsets is None
            or board_cache.root_leaf_offsets.is_contiguous()
        )
        and (
            board_cache.root_block_cache_rows is None
            or board_cache.root_block_cache_rows.is_contiguous()
        )
        and (
            board_cache.root_block_leaf_starts is None
            or board_cache.root_block_leaf_starts.is_contiguous()
        )
        and not config.blockers
        and config.pot_power == 1.0
        and config.rank_bins <= 256
    )


def _can_use_turn_pair_operator_baseline(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    config: TurnRangeEquityConfig,
    board_cache: TurnRangeEquityBoardCache | None,
) -> bool:
    return (
        player_beliefs.device.type == "cuda"
        and features.context.device.type == "cuda"
        and board_cache is not None
        and not config.blockers
        and config.pot_power == 1.0
        and int(player_beliefs.shape[1]) == 2
        and (
            int(board_cache.uniform_leaf_count) > 0
            or _env_int("P2_TURN_EQUITY_PAIR_OPERATOR_RAGGED", 1) != 0
        )
        and board_cache.grouped_leaf_indices is not None
        and board_cache.root_leaf_offsets is not None
        and board_cache.leaf_count_bucket_counts is not None
        and board_cache.leaf_count_bucket_roots is not None
        and board_cache.leaf_count_bucket_offsets is not None
        and len(board_cache.leaf_count_bucket_specs) > 0
        and board_cache.legal_hands is not None
        and board_cache.pair_payoff is not None
        and board_cache.card_incidence is not None
        and board_cache.legal_hand_cards is not None
        and board_cache.grouped_leaf_indices.is_contiguous()
        and board_cache.root_leaf_offsets.is_contiguous()
        and board_cache.leaf_count_bucket_counts.is_contiguous()
        and board_cache.leaf_count_bucket_roots.is_contiguous()
        and board_cache.leaf_count_bucket_offsets.is_contiguous()
        and board_cache.legal_hands.is_contiguous()
        and board_cache.pair_payoff.is_contiguous()
        and board_cache.card_incidence.is_contiguous()
        and board_cache.legal_hand_cards.is_contiguous()
    )


def _turn_pair_operator_baseline(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    *,
    config: TurnRangeEquityConfig,
    dtype: torch.dtype,
    board_cache: TurnRangeEquityBoardCache,
) -> torch.Tensor:
    assert board_cache.grouped_leaf_indices is not None
    assert board_cache.root_leaf_offsets is not None
    assert board_cache.leaf_count_bucket_counts is not None
    assert board_cache.leaf_count_bucket_roots is not None
    assert board_cache.leaf_count_bucket_offsets is not None
    assert board_cache.legal_hands is not None
    assert board_cache.pair_payoff is not None
    assert board_cache.card_incidence is not None
    assert board_cache.legal_hand_cards is not None
    hand_count = int(board_cache.legal_hands.shape[1])
    # The payoff/incidence tables are integer-like and may be stored compactly,
    # but the belief reductions must accumulate in fp32. With fp16 BMMs, small
    # range-mass errors are amplified by CFR backup into materially different
    # root value targets.
    matmul_dtype = torch.float32
    use_store_kernel = (
        triton is not None
        and player_beliefs.device.type == "cuda"
        and _env_int("P2_TURN_EQUITY_PAIR_STORE_KERNEL", 1) != 0
    )
    baseline = (
        player_beliefs.new_zeros(
            player_beliefs.shape[0],
            2,
            NUM_HANDS,
            dtype=dtype,
        )
        if use_store_kernel
        else player_beliefs.new_empty(
            player_beliefs.shape[0],
            2,
            NUM_HANDS,
            dtype=dtype,
        )
    )
    grouped_leaf_indices = board_cache.grouped_leaf_indices.long()
    root_leaf_offsets = board_cache.root_leaf_offsets.long()
    bucket_roots = board_cache.leaf_count_bucket_roots.long()
    for leaves_per_root, start, end in board_cache.leaf_count_bucket_specs:
        if leaves_per_root <= 0:
            continue
        roots = bucket_roots[start:end]
        root_count = int(roots.numel())
        root_starts = root_leaf_offsets.index_select(0, roots)
        root_ends = root_leaf_offsets.index_select(0, roots + 1)
        root_counts = root_ends - root_starts
        leaf_offsets = torch.arange(
            leaves_per_root,
            dtype=torch.long,
            device=player_beliefs.device,
        )
        leaf_mask = leaf_offsets[None, :] < root_counts[:, None]
        safe_leaf_offsets = torch.minimum(
            leaf_offsets[None, :],
            (root_counts[:, None] - 1).clamp_min(0),
        )
        leaf_positions = root_starts[:, None] + safe_leaf_offsets
        leaf_indices = grouped_leaf_indices.index_select(0, leaf_positions.reshape(-1))
        grouped_beliefs = player_beliefs.index_select(0, leaf_indices).view(
            root_count,
            leaves_per_root,
            2,
            NUM_HANDS,
        )
        grouped_beliefs = grouped_beliefs * leaf_mask[:, :, None, None].to(
            dtype=grouped_beliefs.dtype,
        )
        if board_cache.pair_operator_bucketed:
            legal_hands = board_cache.legal_hands[start:end]
        else:
            legal_hands = board_cache.legal_hands.index_select(0, roots)
        compact_idx = legal_hands[:, None, None, :].expand(
            root_count,
            leaves_per_root,
            2,
            hand_count,
        )
        compact_beliefs = torch.gather(grouped_beliefs, 3, compact_idx)
        if board_cache.pair_operator_bucketed:
            pair_payoff = board_cache.pair_payoff[start:end]
        else:
            pair_payoff = board_cache.pair_payoff.index_select(0, roots)
        if board_cache.pair_operator_bucketed:
            card_incidence = board_cache.card_incidence[start:end]
        else:
            card_incidence = board_cache.card_incidence.index_select(0, roots)
        with torch.autocast(device_type="cuda", enabled=False):
            opp_for_p0 = compact_beliefs[:, :, 1, :].transpose(1, 2).to(
                matmul_dtype
            )
            opp_for_p1 = compact_beliefs[:, :, 0, :].transpose(1, 2).to(
                matmul_dtype
            )
            pair_payoff = pair_payoff.to(matmul_dtype)
            card_incidence = card_incidence.to(matmul_dtype)
            num0 = torch.bmm(pair_payoff, opp_for_p0)
            num1 = torch.bmm(pair_payoff, opp_for_p1)
            card_mass0 = torch.bmm(opp_for_p0.transpose(1, 2), card_incidence)
            card_mass1 = torch.bmm(opp_for_p1.transpose(1, 2), card_incidence)
            total0 = opp_for_p0.sum(dim=1).float()
            total1 = opp_for_p1.sum(dim=1).float()
        if board_cache.pair_operator_bucketed:
            legal_hand_cards = board_cache.legal_hand_cards[start:end]
        else:
            legal_hand_cards = board_cache.legal_hand_cards.index_select(0, roots)
        if use_store_kernel:
            _turn_pair_operator_store_kernel[
                (
                    root_count,
                    leaves_per_root,
                    triton.cdiv(hand_count, 128),
                )
            ](
                features.context,
                board_cache.grouped_leaf_indices,
                board_cache.root_leaf_offsets,
                bucket_roots[start:end],
                legal_hands,
                legal_hand_cards,
                num0.contiguous(),
                num1.contiguous(),
                card_mass0.contiguous(),
                card_mass1.contiguous(),
                total0.contiguous(),
                total1.contiguous(),
                baseline,
                root_count,
                leaves_per_root,
                hand_count,
                int(features.context.stride(0)),
                ValueScalarContext.POT.value,
                int(baseline.stride(0)),
                int(baseline.stride(1)),
                float(config.pos_scale),
                float(config.neg_scale),
                float(config.intercept),
                float(config.baseline_scale),
                bool(config.pos_scale >= 0.0),
                BLOCK_H=128,
                num_warps=4,
            )
            continue
        gather_idx = legal_hand_cards[:, None, :, :].expand(
            root_count,
            leaves_per_root,
            hand_count,
            2,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            den0 = 44.0 * total0[:, :, None]
            den0 = den0 + card_mass0.float().gather(2, gather_idx[..., 0])
            den0 = den0 + card_mass0.float().gather(2, gather_idx[..., 1])
            den1 = 44.0 * total1[:, :, None]
            den1 = den1 + card_mass1.float().gather(2, gather_idx[..., 0])
            den1 = den1 + card_mass1.float().gather(2, gather_idx[..., 1])
            val0 = num0.float().transpose(1, 2) / den0.clamp_min(1e-8)
            val1 = num1.float().transpose(1, 2) / den1.clamp_min(1e-8)
            pot = features.context.index_select(0, leaf_indices)[
                :,
                ValueScalarContext.POT.value,
            ].view(root_count, leaves_per_root)
            sdv0 = val0 * pot[:, :, None]
            sdv1 = val1 * pot[:, :, None]
            if config.pos_scale >= 0.0:
                compact_values = torch.stack(
                    (
                        sdv0.clamp_min(0.0) * config.pos_scale
                        + sdv0.clamp_max(0.0) * config.neg_scale
                        + config.intercept,
                        sdv1.clamp_min(0.0) * config.pos_scale
                        + sdv1.clamp_max(0.0) * config.neg_scale
                        + config.intercept,
                    ),
                    dim=2,
                )
            else:
                compact_values = (
                    torch.stack((sdv0, sdv1), dim=2) * config.baseline_scale
                )
        dense = player_beliefs.new_zeros(
            root_count,
            leaves_per_root,
            2,
            NUM_HANDS,
            dtype=dtype,
        )
        dense.scatter_(3, compact_idx, compact_values.to(dtype=dtype))
        valid_leaf_mask = leaf_mask.reshape(-1)
        baseline.index_copy_(
            0,
            leaf_indices[valid_leaf_mask],
            dense.view(root_count * leaves_per_root, 2, NUM_HANDS)[valid_leaf_mask],
        )
    return baseline


def apply_turn_pair_operator_baseline_value(
    hand_values: torch.Tensor,
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    *,
    config: TurnRangeEquityConfig,
    board_cache: TurnRangeEquityBoardCache | None,
) -> torch.Tensor | None:
    if not _can_use_turn_pair_operator_baseline(
        player_beliefs,
        features,
        config,
        board_cache,
    ):
        return None
    assert board_cache is not None
    assert board_cache.grouped_leaf_indices is not None
    assert board_cache.root_leaf_offsets is not None
    assert board_cache.legal_hands is not None
    assert board_cache.pair_payoff is not None
    assert board_cache.card_incidence is not None
    assert board_cache.legal_hand_cards is not None
    out = hand_values.clone()
    hand_count = int(board_cache.legal_hands.shape[1])
    # Keep the cached integer-like tables compact, but do live range-mass
    # reductions in fp32 for consistency with the prefix/cumsum baseline.
    matmul_dtype = torch.float32
    beliefs = player_beliefs.contiguous()
    grouped_leaf_indices = board_cache.grouped_leaf_indices.long()
    root_leaf_offsets = board_cache.root_leaf_offsets.long()
    bucket_roots = board_cache.leaf_count_bucket_roots.long()
    for leaves_per_root, start, end in board_cache.leaf_count_bucket_specs:
        roots = bucket_roots[start:end]
        root_count = int(roots.numel())
        root_starts = root_leaf_offsets.index_select(0, roots)
        root_ends = root_leaf_offsets.index_select(0, roots + 1)
        root_counts = root_ends - root_starts
        leaf_offsets = torch.arange(
            leaves_per_root,
            dtype=torch.long,
            device=beliefs.device,
        )
        leaf_mask = leaf_offsets[None, :] < root_counts[:, None]
        safe_leaf_offsets = torch.minimum(
            leaf_offsets[None, :],
            (root_counts[:, None] - 1).clamp_min(0),
        )
        leaf_positions = root_starts[:, None] + safe_leaf_offsets
        leaf_indices = grouped_leaf_indices.index_select(0, leaf_positions.reshape(-1))
        grouped_beliefs = beliefs.index_select(0, leaf_indices).view(
            root_count,
            leaves_per_root,
            2,
            NUM_HANDS,
        )
        grouped_beliefs = grouped_beliefs * leaf_mask[:, :, None, None].to(
            dtype=grouped_beliefs.dtype,
        )
        if board_cache.pair_operator_bucketed:
            legal_hands = board_cache.legal_hands[start:end]
            pair_payoff = board_cache.pair_payoff[start:end]
            card_incidence = board_cache.card_incidence[start:end]
            legal_hand_cards = board_cache.legal_hand_cards[start:end]
        else:
            legal_hands = board_cache.legal_hands.index_select(0, roots)
            pair_payoff = board_cache.pair_payoff.index_select(0, roots)
            card_incidence = board_cache.card_incidence.index_select(0, roots)
            legal_hand_cards = board_cache.legal_hand_cards.index_select(0, roots)
        compact_idx = legal_hands[:, None, None, :].expand(
            root_count,
            leaves_per_root,
            2,
            hand_count,
        )
        compact_beliefs = torch.gather(grouped_beliefs, 3, compact_idx)
        with torch.autocast(device_type="cuda", enabled=False):
            pair_payoff = pair_payoff.to(matmul_dtype)
            card_incidence = card_incidence.to(matmul_dtype)
            opp_for_p0 = compact_beliefs[:, :, 1, :].transpose(1, 2).to(
                matmul_dtype
            )
            opp_for_p1 = compact_beliefs[:, :, 0, :].transpose(1, 2).to(
                matmul_dtype
            )
            num0 = torch.bmm(pair_payoff, opp_for_p0)
            num1 = torch.bmm(pair_payoff, opp_for_p1)
            card_mass0 = torch.bmm(opp_for_p0.transpose(1, 2), card_incidence)
            card_mass1 = torch.bmm(opp_for_p1.transpose(1, 2), card_incidence)
            total0 = opp_for_p0.sum(dim=1).float()
            total1 = opp_for_p1.sum(dim=1).float()
        gather_idx = legal_hand_cards[:, None, :, :].expand(
            root_count,
            leaves_per_root,
            hand_count,
            2,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            den0 = 44.0 * total0[:, :, None]
            den0 = den0 + card_mass0.float().gather(2, gather_idx[..., 0])
            den0 = den0 + card_mass0.float().gather(2, gather_idx[..., 1])
            den1 = 44.0 * total1[:, :, None]
            den1 = den1 + card_mass1.float().gather(2, gather_idx[..., 0])
            den1 = den1 + card_mass1.float().gather(2, gather_idx[..., 1])
            pot = features.context.index_select(0, leaf_indices)[
                :,
                ValueScalarContext.POT.value,
            ].view(root_count, leaves_per_root)
            sdv0 = (num0.float().transpose(1, 2) / den0.clamp_min(1e-8)) * pot[
                :,
                :,
                None,
            ]
            sdv1 = (num1.float().transpose(1, 2) / den1.clamp_min(1e-8)) * pot[
                :,
                :,
                None,
            ]
            if config.pos_scale >= 0.0:
                compact_values = torch.stack(
                    (
                        sdv0.clamp_min(0.0) * config.pos_scale
                        + sdv0.clamp_max(0.0) * config.neg_scale
                        + config.intercept,
                        sdv1.clamp_min(0.0) * config.pos_scale
                        + sdv1.clamp_max(0.0) * config.neg_scale
                        + config.intercept,
                    ),
                    dim=2,
                )
            else:
                compact_values = (
                    torch.stack((sdv0, sdv1), dim=2) * config.baseline_scale
                )
        bucket_values = out.index_select(0, leaf_indices).view(
            root_count,
            leaves_per_root,
            2,
            NUM_HANDS,
        )
        bucket_values.scatter_add_(3, compact_idx, compact_values.to(dtype=out.dtype))
        valid_leaf_mask = leaf_mask.reshape(-1)
        out.index_copy_(
            0,
            leaf_indices[valid_leaf_mask],
            bucket_values.view(root_count * leaves_per_root, 2, NUM_HANDS)[
                valid_leaf_mask
            ],
        )
    return out


def turn_range_equity_baseline(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    *,
    config: TurnRangeEquityConfig,
    dtype: torch.dtype,
    board_cache: TurnRangeEquityBoardCache | None = None,
    rank_groups_fn: RankGroupsFn = river_rank_groups,
) -> torch.Tensor:
    if (
        _env_int("P2_TURN_EQUITY_PAIR_OPERATOR_BASELINE", 1) != 0
        and _can_use_turn_pair_operator_baseline(
            player_beliefs,
            features,
            config,
            board_cache,
        )
    ):
        assert board_cache is not None
        return _turn_pair_operator_baseline(
            player_beliefs.contiguous(),
            features,
            config=config,
            dtype=dtype,
            board_cache=board_cache,
        )
    if _can_use_triton_turn_baseline(
        player_beliefs,
        features,
        config,
        board_cache,
    ):
        assert board_cache is not None
        kernel_beliefs = player_beliefs.contiguous()
        kernel_context = features.context.contiguous()
        kernel_board_ok = board_cache.board_ok.contiguous()
        kernel_hand_runout_ok = (
            kernel_board_ok
            if board_cache.hand_runout_ok is None
            else board_cache.hand_runout_ok.contiguous()
        )
        has_hand_runout_ok = board_cache.hand_runout_ok is not None
        kernel_cache_index = (
            kernel_beliefs
            if board_cache.leaf_to_cache is None
            else board_cache.leaf_to_cache.contiguous()
        )
        has_cache_index = board_cache.leaf_to_cache is not None
        batch_size = int(kernel_beliefs.shape[0])
        num_players = int(kernel_beliefs.shape[1])
        rank_bins = int(config.rank_bins)
        baseline = kernel_beliefs.new_empty(
            batch_size,
            num_players,
            NUM_HANDS,
            dtype=dtype,
        )
        combos = hand_combos_tensor(device=kernel_beliefs.device)
        card_a = combos[:, 0].contiguous()
        card_b = combos[:, 1].contiguous()
        block_h_mass = _env_int("P2_TURN_EQUITY_BLOCK_H_MASS", 128)
        use_sorted_bins = (
            _env_int("P2_TURN_EQUITY_SORTED_BIN_KERNEL", 0) != 0
            and
            num_players == 2
            and board_cache.sorted_hands is not None
            and board_cache.bin_offsets is not None
        )
        use_prefix_bins = (
            _env_int("P2_TURN_EQUITY_PREFIX_KERNEL", 1) != 0
            and num_players == 2
            and board_cache.sorted_hands is not None
            and board_cache.bin_offsets is not None
        )
        use_score_bins = (
            _env_int("P2_TURN_EQUITY_SCORE_KERNEL", 0) != 0
            and use_prefix_bins
            and board_cache.card_river_slots is not None
        )
        use_all_river_atomic = (
            _env_int("P2_TURN_EQUITY_ALL_RIVER_ATOMIC_KERNEL", 0) != 0
            and num_players == 2
            and has_hand_runout_ok
        )
        if use_score_bins:
            rank_score = kernel_beliefs.new_empty(
                batch_size,
                num_players,
                48,
                rank_bins,
                dtype=torch.float32,
            )
            river_total = kernel_beliefs.new_empty(
                batch_size,
                num_players,
                48,
                dtype=torch.float32,
            )
            _turn_rank_score_prefix_hu_kernel[(batch_size * 2 * 48,)](
                kernel_beliefs,
                kernel_cache_index,
                board_cache.sorted_hands,
                board_cache.bin_offsets,
                rank_score,
                river_total,
                batch_size,
                rank_bins,
                NUM_HANDS,
                has_cache_index,
                BLOCK_H=triton.next_power_of_2(NUM_HANDS),
                BLOCK_BINS=triton.next_power_of_2(rank_bins),
                num_warps=8,
            )
            block_h_out = _env_int("P2_TURN_EQUITY_BLOCK_H_OUT", 128)
            _turn_baseline_from_rank_score_kernel[
                (
                    batch_size * num_players,
                    triton.cdiv(NUM_HANDS, block_h_out),
                )
            ](
                kernel_context,
                kernel_cache_index,
                kernel_board_ok,
                board_cache.card_river_slots,
                card_a,
                card_b,
                board_cache.rank_groups,
                rank_score,
                river_total,
                baseline,
                batch_size,
                num_players,
                NUM_HANDS,
                int(kernel_context.stride(0)),
                ValueScalarContext.POT.value,
                rank_bins,
                has_cache_index,
                int(baseline.stride(0)),
                int(baseline.stride(1)),
                float(config.pos_scale),
                float(config.neg_scale),
                float(config.intercept),
                float(config.baseline_scale),
                bool(config.pos_scale >= 0.0),
                BLOCK_H=block_h_out,
                num_warps=4,
            )
            return baseline
        rank_mass = kernel_beliefs.new_empty(
            batch_size,
            num_players,
            48,
            rank_bins,
            dtype=torch.float32,
        )
        rank_mass_is_cumulative = False
        if use_all_river_atomic:
            rank_mass.zero_()
            _turn_rank_mass_hu_all_rivers_kernel[
                (
                    batch_size * 2,
                    triton.cdiv(NUM_HANDS, block_h_mass),
                )
            ](
                kernel_beliefs,
                kernel_cache_index,
                kernel_board_ok,
                kernel_hand_runout_ok,
                board_cache.rank_groups,
                rank_mass,
                batch_size,
                NUM_HANDS,
                rank_bins,
                has_cache_index,
                BLOCK_H=block_h_mass,
                num_warps=4,
            )
        elif use_prefix_bins:
            grouped_prefix_leaf_block = _env_int(
                "P2_TURN_EQUITY_GROUPED_PREFIX_BLOCK_L",
                2,
            )
            use_grouped_prefix = (
                _env_int("P2_TURN_EQUITY_GROUPED_PREFIX_KERNEL", 1) != 0
                and board_cache.grouped_leaf_indices is not None
                and board_cache.root_leaf_offsets is not None
                and board_cache.root_block_cache_rows is not None
                and board_cache.root_block_leaf_starts is not None
                and int(board_cache.root_block_leaf_size) == grouped_prefix_leaf_block
            )
            if use_grouped_prefix:
                total_leaf_blocks = int(board_cache.root_block_cache_rows.numel())
                if grouped_prefix_leaf_block == 2:
                    _turn_rank_cumulative_prefix_hu_grouped_l2_kernel[
                        (total_leaf_blocks * 2 * 48,)
                    ](
                        kernel_beliefs,
                        board_cache.grouped_leaf_indices,
                        board_cache.root_leaf_offsets,
                        board_cache.root_block_cache_rows,
                        board_cache.root_block_leaf_starts,
                        board_cache.sorted_hands,
                        board_cache.bin_offsets,
                        rank_mass,
                        total_leaf_blocks,
                        rank_bins,
                        NUM_HANDS,
                        BLOCK_H=triton.next_power_of_2(NUM_HANDS),
                        BLOCK_BINS=triton.next_power_of_2(rank_bins),
                        num_warps=8,
                    )
                else:
                    _turn_rank_cumulative_prefix_hu_grouped_kernel[
                        (total_leaf_blocks * 2 * 48,)
                    ](
                        kernel_beliefs,
                        board_cache.grouped_leaf_indices,
                        board_cache.root_leaf_offsets,
                        board_cache.root_block_cache_rows,
                        board_cache.root_block_leaf_starts,
                        board_cache.sorted_hands,
                        board_cache.bin_offsets,
                        rank_mass,
                        total_leaf_blocks,
                        rank_bins,
                        NUM_HANDS,
                        BLOCK_L=grouped_prefix_leaf_block,
                        BLOCK_H=triton.next_power_of_2(NUM_HANDS),
                        BLOCK_BINS=triton.next_power_of_2(rank_bins),
                        num_warps=8,
                    )
            else:
                _turn_rank_cumulative_prefix_hu_kernel[(batch_size * 2 * 48,)](
                    kernel_beliefs,
                    kernel_cache_index,
                    board_cache.sorted_hands,
                    board_cache.bin_offsets,
                    rank_mass,
                    batch_size,
                    rank_bins,
                    NUM_HANDS,
                    has_cache_index,
                    BLOCK_H=triton.next_power_of_2(NUM_HANDS),
                    BLOCK_BINS=triton.next_power_of_2(rank_bins),
                    num_warps=8,
                )
            rank_mass_is_cumulative = True
        elif use_sorted_bins:
            _turn_rank_mass_bins_hu_kernel[(batch_size * 2 * 48 * rank_bins,)](
                kernel_beliefs,
                kernel_cache_index,
                board_cache.sorted_hands,
                board_cache.bin_offsets,
                rank_mass,
                batch_size,
                rank_bins,
                NUM_HANDS,
                has_cache_index,
                BLOCK_K=_env_int("P2_TURN_EQUITY_BIN_BLOCK_K", 128),
                num_warps=4,
            )
        elif num_players == 2:
            rank_mass.zero_()
            _turn_rank_mass_hu_kernel[
                (
                    batch_size * 2 * 48,
                    triton.cdiv(NUM_HANDS, block_h_mass),
                )
            ](
                kernel_beliefs,
                kernel_cache_index,
                kernel_board_ok,
                board_cache.rivers,
                card_a,
                card_b,
                board_cache.rank_groups,
                rank_mass,
                batch_size,
                NUM_HANDS,
                rank_bins,
                has_cache_index,
                BLOCK_H=block_h_mass,
                num_warps=4,
            )
        else:
            rank_mass.zero_()
            _turn_rank_mass_kernel[
                (
                    batch_size * num_players * 48,
                    triton.cdiv(NUM_HANDS, block_h_mass),
                )
            ](
                kernel_beliefs,
                kernel_cache_index,
                kernel_board_ok,
                board_cache.rivers,
                card_a,
                card_b,
                board_cache.rank_groups,
                rank_mass,
                batch_size,
                num_players,
                NUM_HANDS,
                rank_bins,
                has_cache_index,
                BLOCK_H=block_h_mass,
                num_warps=4,
            )
        if not rank_mass_is_cumulative:
            block_bins = triton.next_power_of_2(rank_bins)
            _turn_rank_cumsum_kernel[(batch_size * num_players * 48,)](
                rank_mass,
                batch_size * num_players * 48,
                rank_bins,
                BLOCK_BINS=block_bins,
                num_warps=4,
            )
        block_h_out = _env_int("P2_TURN_EQUITY_BLOCK_H_OUT", 128)
        use_mask_final = (
            _env_int("P2_TURN_EQUITY_MASK_FINAL_KERNEL", 0) != 0
            and board_cache.hand_runout_mask is not None
        )
        if use_mask_final:
            _turn_baseline_from_cumsum_mask_kernel[
                (
                    batch_size * num_players,
                    triton.cdiv(NUM_HANDS, block_h_out),
                )
            ](
                kernel_context,
                kernel_cache_index,
                board_cache.hand_runout_mask,
                board_cache.rank_groups,
                rank_mass,
                baseline,
                batch_size,
                num_players,
                NUM_HANDS,
                int(kernel_context.stride(0)),
                ValueScalarContext.POT.value,
                rank_bins,
                has_cache_index,
                int(baseline.stride(0)),
                int(baseline.stride(1)),
                float(config.pos_scale),
                float(config.neg_scale),
                float(config.intercept),
                float(config.baseline_scale),
                bool(config.pos_scale >= 0.0),
                BLOCK_H=block_h_out,
                num_warps=4,
            )
            return baseline
        use_hu_both_final = (
            _env_int("P2_TURN_EQUITY_HU_BOTH_FINAL_KERNEL", 0) != 0
            and num_players == 2
            and has_hand_runout_ok
        )
        if use_hu_both_final:
            _turn_baseline_from_cumsum_hu_both_kernel[
                (
                    batch_size,
                    triton.cdiv(NUM_HANDS, block_h_out),
                )
            ](
                kernel_context,
                kernel_cache_index,
                kernel_board_ok,
                kernel_hand_runout_ok,
                board_cache.rank_groups,
                rank_mass,
                baseline,
                batch_size,
                NUM_HANDS,
                int(kernel_context.stride(0)),
                ValueScalarContext.POT.value,
                rank_bins,
                has_cache_index,
                int(baseline.stride(0)),
                int(baseline.stride(1)),
                float(config.pos_scale),
                float(config.neg_scale),
                float(config.intercept),
                float(config.baseline_scale),
                bool(config.pos_scale >= 0.0),
                BLOCK_H=block_h_out,
                num_warps=4,
            )
            return baseline
        use_slot_final = (
            _env_int("P2_TURN_EQUITY_SLOT_FINAL_KERNEL", 0) != 0
            and board_cache.card_river_slots is not None
        )
        if use_slot_final:
            _turn_baseline_from_cumsum_slots_kernel[
                (
                    batch_size * num_players,
                    triton.cdiv(NUM_HANDS, block_h_out),
                )
            ](
                kernel_context,
                kernel_cache_index,
                kernel_board_ok,
                board_cache.card_river_slots,
                card_a,
                card_b,
                board_cache.rank_groups,
                rank_mass,
                baseline,
                batch_size,
                num_players,
                NUM_HANDS,
                int(kernel_context.stride(0)),
                ValueScalarContext.POT.value,
                rank_bins,
                has_cache_index,
                int(baseline.stride(0)),
                int(baseline.stride(1)),
                float(config.pos_scale),
                float(config.neg_scale),
                float(config.intercept),
                float(config.baseline_scale),
                bool(config.pos_scale >= 0.0),
                BLOCK_H=block_h_out,
                num_warps=4,
            )
        else:
            _turn_baseline_from_cumsum_kernel[
                (
                    batch_size * num_players,
                    triton.cdiv(NUM_HANDS, block_h_out),
                )
            ](
                kernel_beliefs,
                kernel_context,
                kernel_cache_index,
                kernel_board_ok,
                kernel_hand_runout_ok,
                board_cache.rivers,
                card_a,
                card_b,
                board_cache.rank_groups,
                rank_mass,
                baseline,
                batch_size,
                num_players,
                NUM_HANDS,
                int(kernel_context.stride(0)),
                ValueScalarContext.POT.value,
                rank_bins,
                has_cache_index,
                has_hand_runout_ok,
                int(baseline.stride(0)),
                int(baseline.stride(1)),
                float(config.pos_scale),
                float(config.neg_scale),
                float(config.intercept),
                float(config.baseline_scale),
                bool(config.pos_scale >= 0.0),
                BLOCK_H=block_h_out,
                num_warps=4,
            )
        return baseline

    if board_cache is not None and board_cache.hand_runout_ok is None:
        checks = {
            "triton": triton is not None,
            "beliefs_cuda": player_beliefs.device.type == "cuda",
            "context_cuda": features.context.device.type == "cuda",
            "rank_groups_contiguous": board_cache.rank_groups.is_contiguous(),
            "rivers_contiguous": board_cache.rivers.is_contiguous(),
            "board_ok_contiguous": board_cache.board_ok.is_contiguous(),
            "leaf_to_cache_contiguous": (
                board_cache.leaf_to_cache is None
                or board_cache.leaf_to_cache.is_contiguous()
            ),
            "sorted_hands_contiguous": (
                board_cache.sorted_hands is None
                or board_cache.sorted_hands.is_contiguous()
            ),
            "bin_offsets_contiguous": (
                board_cache.bin_offsets is None
                or board_cache.bin_offsets.is_contiguous()
            ),
            "hand_runout_ok_contiguous": (
                board_cache.hand_runout_ok is None
                or board_cache.hand_runout_ok.is_contiguous()
            ),
            "card_river_slots_contiguous": (
                board_cache.card_river_slots is None
                or board_cache.card_river_slots.is_contiguous()
            ),
            "hand_runout_mask_contiguous": (
                board_cache.hand_runout_mask is None
                or board_cache.hand_runout_mask.is_contiguous()
            ),
            "not_blockers": not config.blockers,
            "pot_power_one": config.pot_power == 1.0,
            "rank_bins": config.rank_bins <= 256,
        }
        failed = ",".join(name for name, ok in checks.items() if not ok)
        raise ValueError(f"Compact turn equity cache requires Triton path: {failed}")

    baseline, _ = turn_range_equity_features(
        player_beliefs,
        features,
        config=config,
        dtype=dtype,
        board_cache=board_cache,
        rank_groups_fn=rank_groups_fn,
    )
    return baseline


def _turn_range_equity_chunk(
    *,
    beliefs: torch.Tensor,
    context: torch.Tensor,
    cache: TurnRangeEquityBoardCache,
    config: TurnRangeEquityConfig,
    num_players: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = int(beliefs.shape[0])
    rank_bins = int(config.rank_bins)
    opponents = beliefs.sum(dim=1, keepdim=True) - beliefs
    hand_runout_ok = cache.hand_runout_ok
    if hand_runout_ok is None:
        raise ValueError("PyTorch turn equity path requires hand_runout_ok in cache")
    if cache.leaf_to_cache is not None:
        cache_rows = cache.leaf_to_cache.to(device=beliefs.device, dtype=torch.long)
        rank_groups = cache.rank_groups.index_select(0, cache_rows).long()
        hand_runout_ok = hand_runout_ok.index_select(0, cache_rows)
    else:
        rank_groups = cache.rank_groups.long()
    rank_idx = rank_groups[:, None, :, :].expand(-1, num_players, -1, -1)
    opp_weights = opponents[:, :, None, :] * hand_runout_ok[:, None, :, :]
    flat_rank_idx = rank_idx.reshape(-1, NUM_HANDS)
    flat_opp_weights = opp_weights.reshape(-1, NUM_HANDS)
    rank_mass = beliefs.new_zeros(flat_opp_weights.shape[0], rank_bins)
    rank_mass.scatter_add_(1, flat_rank_idx, flat_opp_weights)
    cumulative = rank_mass.cumsum(dim=1)
    tie = rank_mass.gather(1, flat_rank_idx)
    lower = cumulative.gather(1, flat_rank_idx) - tie
    per_river_total = rank_mass.sum(dim=1)
    raw_total = per_river_total.view(chunk, num_players, 48, 1)
    raw_tie = tie
    raw_lower = lower

    if config.blockers:
        combos = hand_combos_tensor(device=beliefs.device)
        card_a = combos[:, 0]
        card_b = combos[:, 1]
        card_rank_bins = 52 * rank_bins
        card_rank_mass = beliefs.new_zeros(
            flat_opp_weights.shape[0],
            card_rank_bins,
        )
        card_a_idx = card_a.view(1, NUM_HANDS).expand_as(flat_rank_idx)
        card_b_idx = card_b.view(1, NUM_HANDS).expand_as(flat_rank_idx)
        flat_idx_a = card_a_idx * rank_bins + flat_rank_idx
        flat_idx_b = card_b_idx * rank_bins + flat_rank_idx
        card_rank_mass.scatter_add_(1, flat_idx_a, flat_opp_weights)
        card_rank_mass.scatter_add_(1, flat_idx_b, flat_opp_weights)
        card_rank_view = card_rank_mass.view(
            flat_opp_weights.shape[0],
            52,
            rank_bins,
        )
        card_mass = card_rank_view.sum(dim=2)
        card_rank_cumulative = card_rank_view.cumsum(dim=2).reshape(
            flat_opp_weights.shape[0],
            card_rank_bins,
        )
        card_tie_a = card_rank_mass.gather(1, flat_idx_a)
        card_tie_b = card_rank_mass.gather(1, flat_idx_b)
        card_lower_a = card_rank_cumulative.gather(1, flat_idx_a) - card_tie_a
        card_lower_b = card_rank_cumulative.gather(1, flat_idx_b) - card_tie_b
        same_combo_mass = flat_opp_weights
        blocked_tie = card_tie_a + card_tie_b - same_combo_mass
        blocked_lower = card_lower_a + card_lower_b
        blocked_total = (
            card_mass.gather(1, card_a_idx)
            + card_mass.gather(1, card_b_idx)
            - same_combo_mass
        )
        tie = (tie - blocked_tie).clamp_min(0.0)
        lower = (lower - blocked_lower).clamp_min(0.0)
        total = (per_river_total[:, None] - blocked_total).clamp_min(0.0).view(
            chunk,
            num_players,
            48,
            NUM_HANDS,
        )
    else:
        total = raw_total

    hero_ok = hand_runout_ok[:, None, :, :].to(dtype=beliefs.dtype)
    lower_sum = (lower.view(chunk, num_players, 48, NUM_HANDS) * hero_ok).sum(dim=2)
    tie_sum = (tie.view(chunk, num_players, 48, NUM_HANDS) * hero_ok).sum(dim=2)
    total_sum = (total * hero_ok).sum(dim=2)
    raw_total_sum = (raw_total * hero_ok).sum(dim=2)
    blocked_fraction = torch.where(
        raw_total_sum > 0.0,
        (raw_total_sum - total_sum).clamp_min(0.0) / raw_total_sum.clamp_min(1e-8),
        torch.zeros_like(raw_total_sum),
    )
    safe_total = total_sum.clamp_min(1e-8)
    equity_score = (2.0 * lower_sum + tie_sum - total_sum) / safe_total
    equity_score = torch.where(
        total_sum > 0.0,
        equity_score,
        torch.zeros_like(equity_score),
    )
    if config.decomposition:
        raw_lower_sum = (
            raw_lower.view(chunk, num_players, 48, NUM_HANDS) * hero_ok
        ).sum(dim=2)
        raw_tie_sum = (
            raw_tie.view(chunk, num_players, 48, NUM_HANDS) * hero_ok
        ).sum(dim=2)
        raw_equity_score = torch.where(
            raw_total_sum > 0.0,
            (2.0 * raw_lower_sum + raw_tie_sum - raw_total_sum)
            / raw_total_sum.clamp_min(1e-8),
            torch.zeros_like(raw_total_sum),
        )
    else:
        raw_equity_score = torch.zeros_like(equity_score)

    if config.runout_std:
        river_lower = lower.view(chunk, num_players, 48, NUM_HANDS)
        river_tie = tie.view(chunk, num_players, 48, NUM_HANDS)
        river_total = total.expand_as(river_lower)
        river_valid = (river_total > 0.0) & (hero_ok > 0.0)
        river_equity = torch.where(
            river_valid,
            (2.0 * river_lower + river_tie - river_total)
            / river_total.clamp_min(1e-8),
            torch.zeros_like(river_total),
        )
        river_valid_float = river_valid.to(beliefs.dtype)
        river_count = river_valid_float.sum(dim=2).clamp_min(1.0)
        river_mean = (river_equity * river_valid_float).sum(dim=2) / river_count
        river_variance = (
            (river_equity - river_mean[:, :, None, :]).square()
            * river_valid_float
        ).sum(dim=2) / river_count
    else:
        river_variance = torch.zeros_like(equity_score)

    pot_scale = context[:, ValueScalarContext.POT.value].float()
    if config.pot_power != 1.0:
        pot_scale = pot_scale.clamp_min(0.0).pow(config.pot_power)
    sdv = equity_score * pot_scale[:, None, None]
    raw_sdv = raw_equity_score * pot_scale[:, None, None]
    blocker_sdv = sdv - raw_sdv
    runout_sdv_std = river_variance.clamp_min(0.0).sqrt() * pot_scale[:, None, None]
    if config.pos_scale >= 0.0:
        value = (
            sdv.clamp_min(0.0) * config.pos_scale
            + sdv.clamp_max(0.0) * config.neg_scale
            + config.intercept
        )
    else:
        value = sdv * config.baseline_scale

    valid_rivers = hand_runout_ok.sum(dim=1).clamp_min(1)
    avg_total_mass = total_sum / valid_rivers[:, None, :].to(dtype=total_sum.dtype)
    avg_blocked_mass = (raw_total_sum - total_sum).clamp_min(0.0) / valid_rivers[
        :, None, :
    ].to(dtype=total_sum.dtype)
    spr = player_spr_context(context, num_players).float()
    feature_values = torch.stack(
        (
            sdv,
            beliefs,
            avg_total_mass,
            blocked_fraction,
            pot_scale[:, None, None].expand_as(equity_score),
            spr[:, :, None].expand_as(equity_score),
            raw_sdv,
            blocker_sdv,
            runout_sdv_std,
            avg_blocked_mass,
            blocked_fraction * sdv,
            blocked_fraction.square(),
        ),
        dim=-1,
    )
    return value, feature_values


def turn_range_equity_features(
    player_beliefs: torch.Tensor,
    features: MLPFeatures,
    *,
    config: TurnRangeEquityConfig,
    dtype: torch.dtype,
    board_cache: TurnRangeEquityBoardCache | None = None,
    rank_groups_fn: RankGroupsFn = river_rank_groups,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline = player_beliefs.new_zeros(
        player_beliefs.shape[0],
        player_beliefs.shape[1],
        NUM_HANDS,
        dtype=dtype,
    )
    feature_values = player_beliefs.new_zeros(
        player_beliefs.shape[0],
            player_beliefs.shape[1],
            NUM_HANDS,
            12,
        dtype=dtype,
    )
    rows = torch.arange(player_beliefs.shape[0], device=player_beliefs.device)
    board4_all = features.board[:, :4].long()

    for start in range(0, int(rows.shape[0]), int(config.chunk_size)):
        chunk_rows = rows[start : start + int(config.chunk_size)]
        beliefs = player_beliefs[chunk_rows].float()
        if board_cache is None:
            chunk_cache = build_turn_range_equity_board_cache(
                board4_all[chunk_rows],
                rank_bins=config.rank_bins,
                rank_groups_fn=rank_groups_fn,
            )
        else:
            chunk_cache = board_cache.slice(chunk_rows)
        value, stacked_features = _turn_range_equity_chunk(
            beliefs=beliefs,
            context=features.context[chunk_rows],
            cache=chunk_cache,
            config=config,
            num_players=player_beliefs.shape[1],
        )
        baseline[chunk_rows] = value.to(dtype=dtype)
        feature_values[chunk_rows] = stacked_features.to(dtype=dtype)
    return baseline, feature_values
