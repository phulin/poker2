from __future__ import annotations

from typing import Final

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None

HAND_VECTOR_SIZE: Final[int] = 52
RANK_LANES: Final[int] = 16
CARD_LANES: Final[int] = 8


def triton_is_available() -> bool:
    return triton is not None


def _validate_input(ab_batch: torch.Tensor) -> torch.Tensor:
    if not triton_is_available():
        raise RuntimeError(
            "Triton is not installed. Install `triton` in a CUDA environment to use "
            "the Triton hand evaluator."
        )
    if ab_batch.dim() != 4 or ab_batch.shape[1:] != (2, 4, 13):
        raise ValueError(f"Expected [N, 2, 4, 13] input, got {tuple(ab_batch.shape)}")
    if ab_batch.device.type != "cuda":
        raise ValueError("Triton hand evaluator requires a CUDA tensor.")
    return ab_batch.contiguous().to(torch.int8)


def compare_7_single_batch_triton(ab_batch: torch.Tensor) -> torch.Tensor:
    """CUDA Triton implementation of ``compare_7_single_batch``.

    This mirrors the current ``rules.py`` comparison semantics, including the
    existing straight-selection behavior for hands with multiple straights.
    """
    ab_batch_i8 = _validate_input(ab_batch)
    out = torch.empty((ab_batch_i8.shape[0],), device=ab_batch_i8.device, dtype=torch.int32)
    grid = (ab_batch_i8.shape[0],)
    _compare_7_single_batch_kernel[grid](
        ab_batch_i8,
        out,
        RANK_LANES=RANK_LANES,
        num_warps=1,
    )
    return out


def compare_7_batches_triton(a_batch: torch.Tensor, b_batch: torch.Tensor) -> torch.Tensor:
    if a_batch.shape != b_batch.shape:
        raise ValueError("a_batch and b_batch must have the same shape.")
    if a_batch.dim() != 3 or a_batch.shape[1:] != (4, 13):
        raise ValueError(f"Expected [N, 4, 13] inputs, got {tuple(a_batch.shape)}")
    ab_batch = torch.stack([a_batch, b_batch], dim=1).bool()
    return compare_7_single_batch_triton(ab_batch)


def _validate_cards_input(cards_batch: torch.Tensor) -> torch.Tensor:
    if not triton_is_available():
        raise RuntimeError(
            "Triton is not installed. Install `triton` in a CUDA environment to use "
            "the Triton hand evaluator."
        )
    if cards_batch.dim() != 3 or cards_batch.shape[1:] != (2, 7):
        raise ValueError(f"Expected [N, 2, 7] input, got {tuple(cards_batch.shape)}")
    if cards_batch.device.type != "cuda":
        raise ValueError("Triton hand evaluator requires a CUDA tensor.")
    if cards_batch.dtype == torch.int16 and cards_batch.is_contiguous():
        return cards_batch
    return cards_batch.contiguous().to(torch.int16)


def compare_7_cards_single_batch_triton(cards_batch: torch.Tensor) -> torch.Tensor:
    """CUDA Triton hand comparison from compact card IDs.

    ``cards_batch`` has shape ``[N, 2, 7]`` with card IDs in ``0..51``.
    """
    cards_batch_i16 = _validate_cards_input(cards_batch)
    out = torch.empty((cards_batch_i16.shape[0],), device=cards_batch_i16.device, dtype=torch.int32)
    grid = (cards_batch_i16.shape[0],)
    _compare_7_cards_single_batch_kernel[grid](
        cards_batch_i16,
        out,
        CARD_LANES=CARD_LANES,
        RANK_LANES=RANK_LANES,
        num_warps=1,
    )
    return out


if triton is not None:

    @triton.jit
    def _to_i32(value):
        return (value + tl.full((), 0, tl.int32)).to(tl.int32)


    @triton.jit
    def _pack_fields(
        f0,
        f1,
        f2,
        f3,
        f4,
        f5,
    ):
        return (
            (_to_i32(f0) << 20)
            | (_to_i32(f1) << 16)
            | (_to_i32(f2) << 12)
            | (_to_i32(f3) << 8)
            | (_to_i32(f4) << 4)
            | _to_i32(f5)
        )


    @triton.jit
    def _select_max_index(scores, valid_mask, ranks):
        masked_scores = tl.where(valid_mask, scores, -1)
        max_score = tl.max(masked_scores, axis=0)
        is_best = masked_scores == max_score
        best_idx = tl.max(tl.where(is_best, ranks, 0), axis=0)
        return max_score, best_idx


    @triton.jit
    def _evaluate_player(base_ptr, RANK_LANES: tl.constexpr):
        ranks = tl.arange(0, RANK_LANES)
        valid_ranks = ranks < 13

        suit0 = tl.load(base_ptr + ranks, mask=valid_ranks, other=0)
        suit1 = tl.load(base_ptr + 13 + ranks, mask=valid_ranks, other=0)
        suit2 = tl.load(base_ptr + 26 + ranks, mask=valid_ranks, other=0)
        suit3 = tl.load(base_ptr + 39 + ranks, mask=valid_ranks, other=0)

        suit0 = suit0.to(tl.int32)
        suit1 = suit1.to(tl.int32)
        suit2 = suit2.to(tl.int32)
        suit3 = suit3.to(tl.int32)

        rank_counts = suit0 + suit1 + suit2 + suit3
        rank_presence = rank_counts > 0
        rank_bits = tl.where(valid_ranks, tl.full([RANK_LANES], 1, tl.int32) << ranks, 0)
        rank_mask = tl.sum(tl.where(rank_presence, rank_bits, 0), axis=0)
        quads = tl.sum(tl.where(rank_counts == 4, rank_bits, 0), axis=0)
        trips = tl.sum(tl.where(rank_counts == 3, rank_bits, 0), axis=0)
        pairs = tl.sum(tl.where(rank_counts == 2, rank_bits, 0), axis=0)
        singles = tl.sum(tl.where(rank_counts == 1, rank_bits, 0), axis=0)
        zeros = ((tl.full((), 1, tl.int32) << 13) - 1) & ~rank_mask

        top_scores_0, top_idx_0, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_1, top_idx_1, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_2, top_idx_2, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_3, top_idx_3, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_4, top_idx_4, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)

        top_val_0 = top_scores_0 // 16
        top_val_1 = top_scores_1 // 16

        suit_count_0 = tl.sum(suit0, axis=0)
        suit_count_1 = tl.sum(suit1, axis=0)
        suit_count_2 = tl.sum(suit2, axis=0)
        suit_count_3 = tl.sum(suit3, axis=0)
        suit_mask_0 = tl.sum(tl.where(suit0 > 0, rank_bits, 0), axis=0)
        suit_mask_1 = tl.sum(tl.where(suit1 > 0, rank_bits, 0), axis=0)
        suit_mask_2 = tl.sum(tl.where(suit2 > 0, rank_bits, 0), axis=0)
        suit_mask_3 = tl.sum(tl.where(suit3 > 0, rank_bits, 0), axis=0)

        flush_score_0 = tl.where(suit_count_0 >= 5, 0, -1)
        flush_score_1 = tl.where(suit_count_1 >= 5, 1, -1)
        flush_score_2 = tl.where(suit_count_2 >= 5, 2, -1)
        flush_score_3 = tl.where(suit_count_3 >= 5, 3, -1)
        has_flush = tl.maximum(
            tl.maximum(flush_score_0, flush_score_1),
            tl.maximum(flush_score_2, flush_score_3),
        ) >= 0

        flush_suit = tl.where(
            flush_score_0 >= 0,
            0,
            tl.where(flush_score_1 >= 0, 1, tl.where(flush_score_2 >= 0, 2, 3)),
        )
        flush_mask = tl.where(
            flush_suit == 0,
            suit_mask_0,
            tl.where(flush_suit == 1, suit_mask_1, tl.where(flush_suit == 2, suit_mask_2, suit_mask_3)),
        )
        flush_idx_0, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_1, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_2, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_3, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_4, flush_mask = _pop_highest_rank(flush_mask)

        straight_found = tl.full((), 0, tl.int32)
        straight_low = tl.full((), 0, tl.int32)
        for low in range(10):
            if low == 0:
                straight_mask = (tl.full((), 1, tl.int32) << 12) | 15
            else:
                straight_mask = 31 << (low - 1)
            hit = (rank_mask & straight_mask) == straight_mask
            straight_low = tl.where((straight_found == 0) & hit, low, straight_low)
            straight_found = straight_found | hit.to(tl.int32)

        sf_found = tl.full((), 0, tl.int32)
        sf_high = tl.full((), 0, tl.int32)
        for low in range(10):
            if low == 0:
                sf_mask = (tl.full((), 1, tl.int32) << 12) | 15
            else:
                sf_mask = 31 << (low - 1)
            hit0 = (suit_mask_0 & sf_mask) == sf_mask
            hit1 = (suit_mask_1 & sf_mask) == sf_mask
            hit2 = (suit_mask_2 & sf_mask) == sf_mask
            hit3 = (suit_mask_3 & sf_mask) == sf_mask
            hit = hit0 | hit1 | hit2 | hit3
            sf_high = tl.where(hit, low, sf_high)
            sf_found = sf_found | hit.to(tl.int32)

        straight_flush_score = _pack_fields(
            sf_found * 9,
            tl.where(sf_found > 0, sf_high, 0),
            0,
            0,
            0,
            0,
        )
        four_kind_score = _pack_fields(
            (top_val_0 >= 4) * 8,
            tl.where(top_val_0 >= 4, top_idx_0, 0),
            tl.where(top_val_0 >= 4, top_idx_1, 0),
            0,
            0,
            0,
        )
        full_house_mask = (top_val_0 >= 3) & (top_val_1 >= 2)
        full_house_score = _pack_fields(
            full_house_mask * 7,
            tl.where(full_house_mask, top_idx_0, 0),
            tl.where(full_house_mask, top_idx_1, 0),
            0,
            0,
            0,
        )
        flush_score = _pack_fields(
            has_flush * 6,
            tl.where(has_flush, flush_idx_0, 0),
            tl.where(has_flush, flush_idx_1, 0),
            tl.where(has_flush, flush_idx_2, 0),
            tl.where(has_flush, flush_idx_3, 0),
            tl.where(has_flush, flush_idx_4, 0),
        )
        straight_score = _pack_fields(
            straight_found * 5,
            tl.where(straight_found > 0, straight_low, 0),
            0,
            0,
            0,
            0,
        )
        three_kind_mask = top_val_0 >= 3
        three_kind_score = _pack_fields(
            three_kind_mask * 4,
            tl.where(three_kind_mask, top_idx_0, 0),
            tl.where(three_kind_mask, top_idx_1, 0),
            tl.where(three_kind_mask, top_idx_2, 0),
            0,
            0,
        )
        two_pair_mask = (top_val_0 >= 2) & (top_val_1 >= 2)
        two_pair_score = _pack_fields(
            two_pair_mask * 3,
            tl.where(two_pair_mask, top_idx_0, 0),
            tl.where(two_pair_mask, top_idx_1, 0),
            tl.where(two_pair_mask, top_idx_2, 0),
            0,
            0,
        )
        one_pair_mask = top_val_0 >= 2
        one_pair_score = _pack_fields(
            one_pair_mask * 2,
            tl.where(one_pair_mask, top_idx_0, 0),
            tl.where(one_pair_mask, top_idx_1, 0),
            tl.where(one_pair_mask, top_idx_2, 0),
            tl.where(one_pair_mask, top_idx_3, 0),
            0,
        )
        high_card_score = _pack_fields(1, top_idx_0, top_idx_1, top_idx_2, top_idx_3, top_idx_4)

        score = high_card_score
        score = tl.maximum(score, one_pair_score)
        score = tl.maximum(score, two_pair_score)
        score = tl.maximum(score, three_kind_score)
        score = tl.maximum(score, straight_score)
        score = tl.maximum(score, flush_score)
        score = tl.maximum(score, full_house_score)
        score = tl.maximum(score, four_kind_score)
        score = tl.maximum(score, straight_flush_score)
        return score


    @triton.jit
    def _pop_highest_rank(rank_mask):
        best_idx = tl.inline_asm_elementwise(
            asm="bfind.u32 $0, $1;",
            constraints="=r,r",
            args=[rank_mask],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        best_idx = tl.maximum(best_idx, 0)
        return best_idx, rank_mask & ~(tl.full((), 1, tl.int32) << best_idx)


    @triton.jit
    def _popcount(mask):
        return tl.inline_asm_elementwise(
            asm="popc.b32 $0, $1;",
            constraints="=r,r",
            args=[mask],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _pop_top_rank(quads, trips, pairs, singles, zeros):
        has_quads = quads != 0
        has_trips = trips != 0
        has_pairs = pairs != 0
        has_singles = singles != 0
        selected = tl.where(
            has_quads,
            quads,
            tl.where(has_trips, trips, tl.where(has_pairs, pairs, tl.where(has_singles, singles, zeros))),
        )
        count = tl.where(has_quads, 4, tl.where(has_trips, 3, tl.where(has_pairs, 2, tl.where(has_singles, 1, 0))))
        idx, selected_next = _pop_highest_rank(selected)
        take_quads = has_quads
        take_trips = (has_quads == 0) & has_trips
        take_pairs = (has_quads == 0) & (has_trips == 0) & has_pairs
        take_singles = (has_quads == 0) & (has_trips == 0) & (has_pairs == 0) & has_singles
        take_zeros = (has_quads == 0) & (has_trips == 0) & (has_pairs == 0) & (has_singles == 0)
        quads = tl.where(take_quads, selected_next, quads)
        trips = tl.where(take_trips, selected_next, trips)
        pairs = tl.where(take_pairs, selected_next, pairs)
        singles = tl.where(take_singles, selected_next, singles)
        zeros = tl.where(take_zeros, selected_next, zeros)
        return count * 16 + idx, idx, quads, trips, pairs, singles, zeros


    @triton.jit
    def _evaluate_cards_player(base_ptr, CARD_LANES: tl.constexpr, RANK_LANES: tl.constexpr):
        card_lanes = tl.arange(0, CARD_LANES)
        valid_cards = card_lanes < 7
        cards = tl.load(base_ptr + card_lanes, mask=valid_cards, other=0).to(tl.int32)
        card_ranks = cards % 13
        card_suits = cards // 13
        card_bits = tl.where(valid_cards, tl.full([CARD_LANES], 1, tl.int32) << card_ranks, 0)
        suit_mask_0 = tl.sum(tl.where(card_suits == 0, card_bits, 0), axis=0)
        suit_mask_1 = tl.sum(tl.where(card_suits == 1, card_bits, 0), axis=0)
        suit_mask_2 = tl.sum(tl.where(card_suits == 2, card_bits, 0), axis=0)
        suit_mask_3 = tl.sum(tl.where(card_suits == 3, card_bits, 0), axis=0)
        suit_count_0 = _popcount(suit_mask_0)
        suit_count_1 = _popcount(suit_mask_1)
        suit_count_2 = _popcount(suit_mask_2)
        suit_count_3 = _popcount(suit_mask_3)

        seen = tl.full((), 0, tl.int32)
        pairs_or_better = tl.full((), 0, tl.int32)
        trips_or_better = tl.full((), 0, tl.int32)
        quads = tl.full((), 0, tl.int32)
        for card_idx in range(7):
            bit = tl.full((), 1, tl.int32) << tl.load(base_ptr + card_idx).to(tl.int32) % 13
            quads = quads | (trips_or_better & bit)
            trips_or_better = trips_or_better | (pairs_or_better & bit)
            pairs_or_better = pairs_or_better | (seen & bit)
            seen = seen | bit

        rank_mask = seen
        trips = trips_or_better & ~quads
        pairs = pairs_or_better & ~trips_or_better
        singles = seen & ~pairs_or_better
        zeros = ((tl.full((), 1, tl.int32) << 13) - 1) & ~seen

        top_scores_0, top_idx_0, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_1, top_idx_1, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_2, top_idx_2, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_3, top_idx_3, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)
        top_scores_4, top_idx_4, quads, trips, pairs, singles, zeros = _pop_top_rank(quads, trips, pairs, singles, zeros)

        top_val_0 = top_scores_0 // 16
        top_val_1 = top_scores_1 // 16

        flush_score_0 = tl.where(suit_count_0 >= 5, 0, -1)
        flush_score_1 = tl.where(suit_count_1 >= 5, 1, -1)
        flush_score_2 = tl.where(suit_count_2 >= 5, 2, -1)
        flush_score_3 = tl.where(suit_count_3 >= 5, 3, -1)
        has_flush = tl.maximum(
            tl.maximum(flush_score_0, flush_score_1),
            tl.maximum(flush_score_2, flush_score_3),
        ) >= 0
        flush_suit = tl.where(
            flush_score_0 >= 0,
            0,
            tl.where(flush_score_1 >= 0, 1, tl.where(flush_score_2 >= 0, 2, 3)),
        )
        flush_mask = tl.where(
            flush_suit == 0,
            suit_mask_0,
            tl.where(flush_suit == 1, suit_mask_1, tl.where(flush_suit == 2, suit_mask_2, suit_mask_3)),
        )
        flush_idx_0, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_1, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_2, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_3, flush_mask = _pop_highest_rank(flush_mask)
        flush_idx_4, flush_mask = _pop_highest_rank(flush_mask)

        straight_found = tl.full((), 0, tl.int32)
        straight_low = tl.full((), 0, tl.int32)
        sf_found = tl.full((), 0, tl.int32)
        sf_high = tl.full((), 0, tl.int32)
        for low in range(10):
            if low == 0:
                straight_mask = (tl.full((), 1, tl.int32) << 12) | 15
            else:
                straight_mask = 31 << (low - 1)
            hit = (rank_mask & straight_mask) == straight_mask
            straight_low = tl.where((straight_found == 0) & hit, low, straight_low)
            straight_found = straight_found | hit.to(tl.int32)

            hit0 = (suit_mask_0 & straight_mask) == straight_mask
            hit1 = (suit_mask_1 & straight_mask) == straight_mask
            hit2 = (suit_mask_2 & straight_mask) == straight_mask
            hit3 = (suit_mask_3 & straight_mask) == straight_mask
            sf_hit = hit0 | hit1 | hit2 | hit3
            sf_high = tl.where(sf_hit, low, sf_high)
            sf_found = sf_found | sf_hit.to(tl.int32)

        straight_flush_score = _pack_fields(
            sf_found * 9,
            tl.where(sf_found > 0, sf_high, 0),
            0,
            0,
            0,
            0,
        )
        four_kind_score = _pack_fields(
            (top_val_0 >= 4) * 8,
            tl.where(top_val_0 >= 4, top_idx_0, 0),
            tl.where(top_val_0 >= 4, top_idx_1, 0),
            0,
            0,
            0,
        )
        full_house_mask = (top_val_0 >= 3) & (top_val_1 >= 2)
        full_house_score = _pack_fields(
            full_house_mask * 7,
            tl.where(full_house_mask, top_idx_0, 0),
            tl.where(full_house_mask, top_idx_1, 0),
            0,
            0,
            0,
        )
        flush_score = _pack_fields(
            has_flush * 6,
            tl.where(has_flush, flush_idx_0, 0),
            tl.where(has_flush, flush_idx_1, 0),
            tl.where(has_flush, flush_idx_2, 0),
            tl.where(has_flush, flush_idx_3, 0),
            tl.where(has_flush, flush_idx_4, 0),
        )
        straight_score = _pack_fields(
            straight_found * 5,
            tl.where(straight_found > 0, straight_low, 0),
            0,
            0,
            0,
            0,
        )
        three_kind_mask = top_val_0 >= 3
        three_kind_score = _pack_fields(
            three_kind_mask * 4,
            tl.where(three_kind_mask, top_idx_0, 0),
            tl.where(three_kind_mask, top_idx_1, 0),
            tl.where(three_kind_mask, top_idx_2, 0),
            0,
            0,
        )
        two_pair_mask = (top_val_0 >= 2) & (top_val_1 >= 2)
        two_pair_score = _pack_fields(
            two_pair_mask * 3,
            tl.where(two_pair_mask, top_idx_0, 0),
            tl.where(two_pair_mask, top_idx_1, 0),
            tl.where(two_pair_mask, top_idx_2, 0),
            0,
            0,
        )
        one_pair_mask = top_val_0 >= 2
        one_pair_score = _pack_fields(
            one_pair_mask * 2,
            tl.where(one_pair_mask, top_idx_0, 0),
            tl.where(one_pair_mask, top_idx_1, 0),
            tl.where(one_pair_mask, top_idx_2, 0),
            tl.where(one_pair_mask, top_idx_3, 0),
            0,
        )
        high_card_score = _pack_fields(1, top_idx_0, top_idx_1, top_idx_2, top_idx_3, top_idx_4)

        score = high_card_score
        score = tl.maximum(score, one_pair_score)
        score = tl.maximum(score, two_pair_score)
        score = tl.maximum(score, three_kind_score)
        score = tl.maximum(score, straight_score)
        score = tl.maximum(score, flush_score)
        score = tl.maximum(score, full_house_score)
        score = tl.maximum(score, four_kind_score)
        score = tl.maximum(score, straight_flush_score)
        return score


    @triton.jit
    def _compare_7_single_batch_kernel(
        ab_batch_ptr,
        out_ptr,
        RANK_LANES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base_ptr = ab_batch_ptr + pid * 104

        score0 = _evaluate_player(base_ptr, RANK_LANES)
        score1 = _evaluate_player(base_ptr + 52, RANK_LANES)

        result = tl.where(score0 > score1, 1, tl.where(score0 < score1, -1, 0))
        tl.store(out_ptr + pid, result)


    @triton.jit
    def _compare_7_cards_single_batch_kernel(
        cards_batch_ptr,
        out_ptr,
        CARD_LANES: tl.constexpr,
        RANK_LANES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base_ptr = cards_batch_ptr + pid * 14

        score0 = _evaluate_cards_player(base_ptr, CARD_LANES, RANK_LANES)
        score1 = _evaluate_cards_player(base_ptr + 7, CARD_LANES, RANK_LANES)

        result = tl.where(score0 > score1, 1, tl.where(score0 < score1, -1, 0))
        tl.store(out_ptr + pid, result)
