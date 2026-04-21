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
        ab_batch_i8.stride(0),
        ab_batch_i8.stride(1),
        ab_batch_i8.stride(2),
        ab_batch_i8.stride(3),
        out.stride(0),
        RANK_LANES=RANK_LANES,
    )
    return out


def compare_7_batches_triton(a_batch: torch.Tensor, b_batch: torch.Tensor) -> torch.Tensor:
    if a_batch.shape != b_batch.shape:
        raise ValueError("a_batch and b_batch must have the same shape.")
    if a_batch.dim() != 3 or a_batch.shape[1:] != (4, 13):
        raise ValueError(f"Expected [N, 4, 13] inputs, got {tuple(a_batch.shape)}")
    ab_batch = torch.stack([a_batch, b_batch], dim=1).bool()
    return compare_7_single_batch_triton(ab_batch)


if triton is not None:

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
            (f0.to(tl.int32) << 20)
            | (f1.to(tl.int32) << 16)
            | (f2.to(tl.int32) << 12)
            | (f3.to(tl.int32) << 8)
            | (f4.to(tl.int32) << 4)
            | f5.to(tl.int32)
        )


    @triton.jit
    def _select_max_index(scores, valid_mask, ranks):
        masked_scores = tl.where(valid_mask, scores, -1)
        max_score = tl.max(masked_scores, axis=0)
        is_best = masked_scores == max_score
        best_idx = tl.max(tl.where(is_best, ranks, 0), axis=0)
        return max_score, best_idx


    @triton.jit
    def _evaluate_player(base_ptr, stride_suit, stride_rank, RANK_LANES: tl.constexpr):
        ranks = tl.arange(0, RANK_LANES)
        valid_ranks = ranks < 13

        suit0 = tl.load(base_ptr + 0 * stride_suit + ranks * stride_rank, mask=valid_ranks, other=0)
        suit1 = tl.load(base_ptr + 1 * stride_suit + ranks * stride_rank, mask=valid_ranks, other=0)
        suit2 = tl.load(base_ptr + 2 * stride_suit + ranks * stride_rank, mask=valid_ranks, other=0)
        suit3 = tl.load(base_ptr + 3 * stride_suit + ranks * stride_rank, mask=valid_ranks, other=0)

        suit0 = suit0.to(tl.int32)
        suit1 = suit1.to(tl.int32)
        suit2 = suit2.to(tl.int32)
        suit3 = suit3.to(tl.int32)

        rank_counts = suit0 + suit1 + suit2 + suit3
        rank_scores = rank_counts * 16 + ranks
        rank_presence = rank_counts > 0

        used = tl.zeros([RANK_LANES], dtype=tl.int32)
        top_scores_0, top_idx_0 = _select_max_index(rank_scores, valid_ranks & (used == 0), ranks)
        used = used + (ranks == top_idx_0).to(tl.int32)
        top_scores_1, top_idx_1 = _select_max_index(rank_scores, valid_ranks & (used == 0), ranks)
        used = used + (ranks == top_idx_1).to(tl.int32)
        top_scores_2, top_idx_2 = _select_max_index(rank_scores, valid_ranks & (used == 0), ranks)
        used = used + (ranks == top_idx_2).to(tl.int32)
        top_scores_3, top_idx_3 = _select_max_index(rank_scores, valid_ranks & (used == 0), ranks)
        used = used + (ranks == top_idx_3).to(tl.int32)
        top_scores_4, top_idx_4 = _select_max_index(rank_scores, valid_ranks & (used == 0), ranks)

        top_val_0 = top_scores_0 // 16
        top_val_1 = top_scores_1 // 16

        suit_count_0 = tl.sum(suit0, axis=0)
        suit_count_1 = tl.sum(suit1, axis=0)
        suit_count_2 = tl.sum(suit2, axis=0)
        suit_count_3 = tl.sum(suit3, axis=0)

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
        flush_row = tl.where(
            flush_suit == 0,
            suit0,
            tl.where(flush_suit == 1, suit1, tl.where(flush_suit == 2, suit2, suit3)),
        )

        flush_valid = valid_ranks & (flush_row > 0)
        flush_scores = tl.where(flush_valid, ranks, -1)
        flush_used = tl.zeros([RANK_LANES], dtype=tl.int32)
        _, flush_idx_0 = _select_max_index(flush_scores, flush_valid & (flush_used == 0), ranks)
        flush_used = flush_used + (ranks == flush_idx_0).to(tl.int32)
        _, flush_idx_1 = _select_max_index(flush_scores, flush_valid & (flush_used == 0), ranks)
        flush_used = flush_used + (ranks == flush_idx_1).to(tl.int32)
        _, flush_idx_2 = _select_max_index(flush_scores, flush_valid & (flush_used == 0), ranks)
        flush_used = flush_used + (ranks == flush_idx_2).to(tl.int32)
        _, flush_idx_3 = _select_max_index(flush_scores, flush_valid & (flush_used == 0), ranks)
        flush_used = flush_used + (ranks == flush_idx_3).to(tl.int32)
        _, flush_idx_4 = _select_max_index(flush_scores, flush_valid & (flush_used == 0), ranks)

        straight_found = tl.full((), 0, tl.int32)
        straight_low = tl.full((), 0, tl.int32)
        for low in range(10):
            if low == 0:
                hit = (
                    rank_presence[12]
                    & rank_presence[0]
                    & rank_presence[1]
                    & rank_presence[2]
                    & rank_presence[3]
                )
            else:
                hit = (
                    rank_presence[low]
                    & rank_presence[low + 1]
                    & rank_presence[low + 2]
                    & rank_presence[low + 3]
                    & rank_presence[low + 4]
                )
            straight_low = tl.where((straight_found == 0) & hit, low, straight_low)
            straight_found = straight_found | hit.to(tl.int32)

        sf_found = tl.full((), 0, tl.int32)
        sf_high = tl.full((), 0, tl.int32)
        for low in range(10):
            hit0 = (
                (suit0[12] > 0) & (suit0[0] > 0) & (suit0[1] > 0) & (suit0[2] > 0) & (suit0[3] > 0)
                if low == 0
                else (suit0[low] > 0)
                & (suit0[low + 1] > 0)
                & (suit0[low + 2] > 0)
                & (suit0[low + 3] > 0)
                & (suit0[low + 4] > 0)
            )
            hit1 = (
                (suit1[12] > 0) & (suit1[0] > 0) & (suit1[1] > 0) & (suit1[2] > 0) & (suit1[3] > 0)
                if low == 0
                else (suit1[low] > 0)
                & (suit1[low + 1] > 0)
                & (suit1[low + 2] > 0)
                & (suit1[low + 3] > 0)
                & (suit1[low + 4] > 0)
            )
            hit2 = (
                (suit2[12] > 0) & (suit2[0] > 0) & (suit2[1] > 0) & (suit2[2] > 0) & (suit2[3] > 0)
                if low == 0
                else (suit2[low] > 0)
                & (suit2[low + 1] > 0)
                & (suit2[low + 2] > 0)
                & (suit2[low + 3] > 0)
                & (suit2[low + 4] > 0)
            )
            hit3 = (
                (suit3[12] > 0) & (suit3[0] > 0) & (suit3[1] > 0) & (suit3[2] > 0) & (suit3[3] > 0)
                if low == 0
                else (suit3[low] > 0)
                & (suit3[low + 1] > 0)
                & (suit3[low + 2] > 0)
                & (suit3[low + 3] > 0)
                & (suit3[low + 4] > 0)
            )
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
    def _compare_7_single_batch_kernel(
        ab_batch_ptr,
        out_ptr,
        stride_batch,
        stride_player,
        stride_suit,
        stride_rank,
        out_stride,
        RANK_LANES: tl.constexpr,
    ):
        pid = tl.program_id(0)
        base_ptr = ab_batch_ptr + pid * stride_batch

        score0 = _evaluate_player(base_ptr + 0 * stride_player, stride_suit, stride_rank, RANK_LANES)
        score1 = _evaluate_player(base_ptr + 1 * stride_player, stride_suit, stride_rank, RANK_LANES)

        result = tl.where(score0 > score1, 1, tl.where(score0 < score1, -1, 0))
        tl.store(out_ptr + pid * out_stride, result)
