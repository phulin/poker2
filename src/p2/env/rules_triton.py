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

    This mirrors ``rules.py`` comparison semantics.
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


def rank_7_cards_single_batch_triton(cards_batch: torch.Tensor) -> torch.Tensor:
    """Score each 7-card hand in ``cards_batch`` (shape ``[N, 7]``) and return
    an int32 ``[N]`` tensor of packed rank integers (higher = stronger).
    Produces the same scores as the comparator kernel for any hand whose seven
    cards are distinct.

    .. note::
        Behaviour is **undefined** for hands containing a duplicate card (it
        differs from the comparator path, which double-counts the dup in its
        suit mask). Callers must only pass legal 7-distinct-card hands.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if cards_batch.dim() != 2 or cards_batch.shape[1] != 7:
        raise ValueError(f"Expected [N, 7] input, got {tuple(cards_batch.shape)}")
    if cards_batch.device.type != "cuda":
        raise ValueError("Triton hand evaluator requires a CUDA tensor.")
    cards_i16 = cards_batch.contiguous().to(torch.int16)
    n = cards_i16.shape[0]
    out = torch.empty((n,), device=cards_i16.device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]),)  # noqa: E731
    _rank_7_cards_batched_kernel[grid](cards_i16, out, n)
    return out


def rank_hand_scores_triton(board: torch.Tensor) -> torch.Tensor:
    """Score every private-hand combo on each board without sorting.

    Args:
      board: [M, 5] int tensor of board card indices (0..51).

    Returns:
      hand_ranks: [M, 1326] int32 packed rank integer per (env, combo).

    This is the scores-only part of :func:`rank_hands_triton`; it avoids both
    the ``[M, 1326, 7]`` card tensor and the argsort needed by callers that want
    rank order.
    """
    from p2.env.card_utils import NUM_HANDS, hand_combos_tensor

    if board.dim() != 2 or board.shape[1] != 5:
        raise ValueError(f"board must be [M, 5], got {tuple(board.shape)}")
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if board.device.type != "cuda":
        raise ValueError("Triton hand evaluator requires a CUDA tensor.")
    device = board.device
    rows = board.shape[0]
    combos = hand_combos_tensor(device=device).to(torch.int16).contiguous()
    board_i16 = board.to(torch.int16).contiguous()

    n = rows * NUM_HANDS
    scores_flat = torch.empty((n,), device=device, dtype=torch.int32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_M"]),)  # noqa: E731
    _rank_hands_fused_kernel[grid](board_i16, combos, scores_flat, n, NUM_HANDS=NUM_HANDS)
    return scores_flat.view(rows, NUM_HANDS)


def rank_hands_triton(board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated drop-in for ``p2.env.rules.rank_hands``.

    Args:
      board: [M, 5] int tensor of board card indices (0..51).

    Returns:
      hand_ranks:     [M, 1326] int32 — packed rank integer per (env, combo).
      sorted_indices: [M, 1326] int64 — argsort by (hand_ranks) ascending
                      (weaker → stronger), stable; matches ``rank_hands``.
    """
    from p2.env.card_utils import NUM_HANDS

    hand_ranks = rank_hand_scores_triton(board)
    sorted_indices = torch.argsort(hand_ranks, dim=1, stable=True)
    return hand_ranks, sorted_indices


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
        trips_or_better = tl.sum(tl.where(rank_counts >= 3, rank_bits, 0), axis=0)
        pairs_or_better = tl.sum(tl.where(rank_counts >= 2, rank_bits, 0), axis=0)

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

        straight_found, straight_high = _straight_high(rank_mask)
        sf_found_0, sf_high_0 = _straight_high(suit_mask_0)
        sf_found_1, sf_high_1 = _straight_high(suit_mask_1)
        sf_found_2, sf_high_2 = _straight_high(suit_mask_2)
        sf_found_3, sf_high_3 = _straight_high(suit_mask_3)
        sf_found = sf_found_0 | sf_found_1 | sf_found_2 | sf_found_3
        sf_high = tl.maximum(
            tl.maximum(tl.where(sf_found_0 > 0, sf_high_0, 0), tl.where(sf_found_1 > 0, sf_high_1, 0)),
            tl.maximum(tl.where(sf_found_2 > 0, sf_high_2, 0), tl.where(sf_found_3 > 0, sf_high_3, 0)),
        )

        quad_rank, _ = _pop_highest_rank(quads)
        quad_kicker, _ = _pop_highest_rank(_clear_rank(rank_mask, quad_rank))
        trip_rank, _ = _pop_highest_rank(trips_or_better)
        pair_for_boat, _ = _pop_highest_rank(_clear_rank(pairs_or_better, trip_rank))
        trip_kicker_0, trip_kickers = _pop_highest_rank(_clear_rank(rank_mask, trip_rank))
        trip_kicker_1, _ = _pop_highest_rank(trip_kickers)
        pair_rank_0, pair_ranks_rest = _pop_highest_rank(pairs_or_better)
        pair_rank_1, _ = _pop_highest_rank(pair_ranks_rest)
        two_pair_kicker, _ = _pop_highest_rank(_clear_rank(_clear_rank(rank_mask, pair_rank_0), pair_rank_1))
        one_pair_kicker_0, one_pair_kickers = _pop_highest_rank(_clear_rank(rank_mask, pair_rank_0))
        one_pair_kicker_1, one_pair_kickers = _pop_highest_rank(one_pair_kickers)
        one_pair_kicker_2, _ = _pop_highest_rank(one_pair_kickers)
        high_idx_0, high_rest = _pop_highest_rank(rank_mask)
        high_idx_1, high_rest = _pop_highest_rank(high_rest)
        high_idx_2, high_rest = _pop_highest_rank(high_rest)
        high_idx_3, high_rest = _pop_highest_rank(high_rest)
        high_idx_4, _ = _pop_highest_rank(high_rest)

        straight_flush_score = _pack_fields(
            sf_found * 9,
            tl.where(sf_found > 0, sf_high, 0),
            0,
            0,
            0,
            0,
        )
        four_kind_score = _pack_fields(
            (quads != 0) * 8,
            tl.where(quads != 0, quad_rank, 0),
            tl.where(quads != 0, quad_kicker, 0),
            0,
            0,
            0,
        )
        full_house_mask = (trips_or_better != 0) & (_clear_rank(pairs_or_better, trip_rank) != 0)
        full_house_score = _pack_fields(
            full_house_mask * 7,
            tl.where(full_house_mask, trip_rank, 0),
            tl.where(full_house_mask, pair_for_boat, 0),
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
            tl.where(straight_found > 0, straight_high, 0),
            0,
            0,
            0,
            0,
        )
        three_kind_mask = trips_or_better != 0
        three_kind_score = _pack_fields(
            three_kind_mask * 4,
            tl.where(three_kind_mask, trip_rank, 0),
            tl.where(three_kind_mask, trip_kicker_0, 0),
            tl.where(three_kind_mask, trip_kicker_1, 0),
            0,
            0,
        )
        two_pair_mask = pair_ranks_rest != 0
        two_pair_score = _pack_fields(
            two_pair_mask * 3,
            tl.where(two_pair_mask, pair_rank_0, 0),
            tl.where(two_pair_mask, pair_rank_1, 0),
            tl.where(two_pair_mask, two_pair_kicker, 0),
            0,
            0,
        )
        one_pair_mask = pairs_or_better != 0
        one_pair_score = _pack_fields(
            one_pair_mask * 2,
            tl.where(one_pair_mask, pair_rank_0, 0),
            tl.where(one_pair_mask, one_pair_kicker_0, 0),
            tl.where(one_pair_mask, one_pair_kicker_1, 0),
            tl.where(one_pair_mask, one_pair_kicker_2, 0),
            0,
        )
        high_card_score = _pack_fields(1, high_idx_0, high_idx_1, high_idx_2, high_idx_3, high_idx_4)

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
    def _clear_rank(rank_mask, rank):
        return rank_mask & ~(tl.full((), 1, tl.int32) << rank)


    @triton.jit
    def _straight_high(rank_mask):
        found = tl.full((), 0, tl.int32)
        high = tl.full((), 0, tl.int32)
        wheel_mask = (tl.full((), 1, tl.int32) << 12) | 15
        wheel = (rank_mask & wheel_mask) == wheel_mask
        found = found | wheel.to(tl.int32)
        high = tl.where(wheel, 3, high)
        for low in range(1, 10):
            straight_mask = 31 << (low - 1)
            hit = (rank_mask & straight_mask) == straight_mask
            found = found | hit.to(tl.int32)
            high = tl.where(hit, low + 3, high)
        return found, high


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

        straight_found, straight_high = _straight_high(rank_mask)
        sf_found_0, sf_high_0 = _straight_high(suit_mask_0)
        sf_found_1, sf_high_1 = _straight_high(suit_mask_1)
        sf_found_2, sf_high_2 = _straight_high(suit_mask_2)
        sf_found_3, sf_high_3 = _straight_high(suit_mask_3)
        sf_found = sf_found_0 | sf_found_1 | sf_found_2 | sf_found_3
        sf_high = tl.maximum(
            tl.maximum(tl.where(sf_found_0 > 0, sf_high_0, 0), tl.where(sf_found_1 > 0, sf_high_1, 0)),
            tl.maximum(tl.where(sf_found_2 > 0, sf_high_2, 0), tl.where(sf_found_3 > 0, sf_high_3, 0)),
        )

        quad_rank, _ = _pop_highest_rank(quads)
        quad_kicker, _ = _pop_highest_rank(_clear_rank(rank_mask, quad_rank))
        trip_rank, _ = _pop_highest_rank(trips_or_better)
        pair_for_boat, _ = _pop_highest_rank(_clear_rank(pairs_or_better, trip_rank))
        trip_kicker_0, trip_kickers = _pop_highest_rank(_clear_rank(rank_mask, trip_rank))
        trip_kicker_1, _ = _pop_highest_rank(trip_kickers)
        pair_rank_0, pair_ranks_rest = _pop_highest_rank(pairs_or_better)
        pair_rank_1, _ = _pop_highest_rank(pair_ranks_rest)
        two_pair_kicker, _ = _pop_highest_rank(_clear_rank(_clear_rank(rank_mask, pair_rank_0), pair_rank_1))
        one_pair_kicker_0, one_pair_kickers = _pop_highest_rank(_clear_rank(rank_mask, pair_rank_0))
        one_pair_kicker_1, one_pair_kickers = _pop_highest_rank(one_pair_kickers)
        one_pair_kicker_2, _ = _pop_highest_rank(one_pair_kickers)
        high_idx_0, high_rest = _pop_highest_rank(rank_mask)
        high_idx_1, high_rest = _pop_highest_rank(high_rest)
        high_idx_2, high_rest = _pop_highest_rank(high_rest)
        high_idx_3, high_rest = _pop_highest_rank(high_rest)
        high_idx_4, _ = _pop_highest_rank(high_rest)

        straight_flush_score = _pack_fields(
            sf_found * 9,
            tl.where(sf_found > 0, sf_high, 0),
            0,
            0,
            0,
            0,
        )
        four_kind_score = _pack_fields(
            (quads != 0) * 8,
            tl.where(quads != 0, quad_rank, 0),
            tl.where(quads != 0, quad_kicker, 0),
            0,
            0,
            0,
        )
        full_house_mask = (trips_or_better != 0) & (_clear_rank(pairs_or_better, trip_rank) != 0)
        full_house_score = _pack_fields(
            full_house_mask * 7,
            tl.where(full_house_mask, trip_rank, 0),
            tl.where(full_house_mask, pair_for_boat, 0),
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
            tl.where(straight_found > 0, straight_high, 0),
            0,
            0,
            0,
            0,
        )
        three_kind_mask = trips_or_better != 0
        three_kind_score = _pack_fields(
            three_kind_mask * 4,
            tl.where(three_kind_mask, trip_rank, 0),
            tl.where(three_kind_mask, trip_kicker_0, 0),
            tl.where(three_kind_mask, trip_kicker_1, 0),
            0,
            0,
        )
        two_pair_mask = pair_ranks_rest != 0
        two_pair_score = _pack_fields(
            two_pair_mask * 3,
            tl.where(two_pair_mask, pair_rank_0, 0),
            tl.where(two_pair_mask, pair_rank_1, 0),
            tl.where(two_pair_mask, two_pair_kicker, 0),
            0,
            0,
        )
        one_pair_mask = pairs_or_better != 0
        one_pair_score = _pack_fields(
            one_pair_mask * 2,
            tl.where(one_pair_mask, pair_rank_0, 0),
            tl.where(one_pair_mask, one_pair_kicker_0, 0),
            tl.where(one_pair_mask, one_pair_kicker_1, 0),
            tl.where(one_pair_mask, one_pair_kicker_2, 0),
            0,
        )
        high_card_score = _pack_fields(1, high_idx_0, high_idx_1, high_idx_2, high_idx_3, high_idx_4)

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


    @triton.jit
    def _score_from_cards(c0, c1, c2, c3, c4, c5, c6):
        """Vectorized 7-card scorer. Each argument is a tile of card IDs
        (0..51); lane ``i`` is an independent hand. Returns the packed rank
        score per lane, matching ``_evaluate_cards_player`` for distinct cards.

        Unlike ``_evaluate_cards_player`` this builds the suit masks by OR
        (not sum), so a duplicated card collapses instead of corrupting the
        mask — dup-card hands are undefined and intentionally left this way.
        """
        # Rank presence/multiplicity masks via the incremental seen-bit trick.
        one = tl.full((), 1, tl.int32)
        b0 = one << (c0 % 13)
        b1 = one << (c1 % 13)
        b2 = one << (c2 % 13)
        b3 = one << (c3 % 13)
        b4 = one << (c4 % 13)
        b5 = one << (c5 % 13)
        b6 = one << (c6 % 13)

        seen = b0
        pairs_or_better = b0 & 0
        trips_or_better = b0 & 0
        quads = b0 & 0
        quads = quads | (trips_or_better & b1); trips_or_better = trips_or_better | (pairs_or_better & b1); pairs_or_better = pairs_or_better | (seen & b1); seen = seen | b1  # noqa: E702
        quads = quads | (trips_or_better & b2); trips_or_better = trips_or_better | (pairs_or_better & b2); pairs_or_better = pairs_or_better | (seen & b2); seen = seen | b2  # noqa: E702
        quads = quads | (trips_or_better & b3); trips_or_better = trips_or_better | (pairs_or_better & b3); pairs_or_better = pairs_or_better | (seen & b3); seen = seen | b3  # noqa: E702
        quads = quads | (trips_or_better & b4); trips_or_better = trips_or_better | (pairs_or_better & b4); pairs_or_better = pairs_or_better | (seen & b4); seen = seen | b4  # noqa: E702
        quads = quads | (trips_or_better & b5); trips_or_better = trips_or_better | (pairs_or_better & b5); pairs_or_better = pairs_or_better | (seen & b5); seen = seen | b5  # noqa: E702
        quads = quads | (trips_or_better & b6); trips_or_better = trips_or_better | (pairs_or_better & b6); pairs_or_better = pairs_or_better | (seen & b6); seen = seen | b6  # noqa: E702

        rank_mask = seen

        s0 = c0 // 13
        s1 = c1 // 13
        s2 = c2 // 13
        s3 = c3 // 13
        s4 = c4 // 13
        s5 = c5 // 13
        s6 = c6 // 13
        suit_mask_0 = (tl.where(s0 == 0, b0, 0) | tl.where(s1 == 0, b1, 0) | tl.where(s2 == 0, b2, 0)
                       | tl.where(s3 == 0, b3, 0) | tl.where(s4 == 0, b4, 0) | tl.where(s5 == 0, b5, 0)
                       | tl.where(s6 == 0, b6, 0))
        suit_mask_1 = (tl.where(s0 == 1, b0, 0) | tl.where(s1 == 1, b1, 0) | tl.where(s2 == 1, b2, 0)
                       | tl.where(s3 == 1, b3, 0) | tl.where(s4 == 1, b4, 0) | tl.where(s5 == 1, b5, 0)
                       | tl.where(s6 == 1, b6, 0))
        suit_mask_2 = (tl.where(s0 == 2, b0, 0) | tl.where(s1 == 2, b1, 0) | tl.where(s2 == 2, b2, 0)
                       | tl.where(s3 == 2, b3, 0) | tl.where(s4 == 2, b4, 0) | tl.where(s5 == 2, b5, 0)
                       | tl.where(s6 == 2, b6, 0))
        suit_mask_3 = (tl.where(s0 == 3, b0, 0) | tl.where(s1 == 3, b1, 0) | tl.where(s2 == 3, b2, 0)
                       | tl.where(s3 == 3, b3, 0) | tl.where(s4 == 3, b4, 0) | tl.where(s5 == 3, b5, 0)
                       | tl.where(s6 == 3, b6, 0))
        suit_count_0 = _popcount(suit_mask_0)
        suit_count_1 = _popcount(suit_mask_1)
        suit_count_2 = _popcount(suit_mask_2)
        suit_count_3 = _popcount(suit_mask_3)

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

        straight_found, straight_high = _straight_high(rank_mask)
        sf_found_0, sf_high_0 = _straight_high(suit_mask_0)
        sf_found_1, sf_high_1 = _straight_high(suit_mask_1)
        sf_found_2, sf_high_2 = _straight_high(suit_mask_2)
        sf_found_3, sf_high_3 = _straight_high(suit_mask_3)
        sf_found = sf_found_0 | sf_found_1 | sf_found_2 | sf_found_3
        sf_high = tl.maximum(
            tl.maximum(tl.where(sf_found_0 > 0, sf_high_0, 0), tl.where(sf_found_1 > 0, sf_high_1, 0)),
            tl.maximum(tl.where(sf_found_2 > 0, sf_high_2, 0), tl.where(sf_found_3 > 0, sf_high_3, 0)),
        )

        quad_rank, _ = _pop_highest_rank(quads)
        quad_kicker, _ = _pop_highest_rank(_clear_rank(rank_mask, quad_rank))
        trip_rank, _ = _pop_highest_rank(trips_or_better)
        pair_for_boat, _ = _pop_highest_rank(_clear_rank(pairs_or_better, trip_rank))
        trip_kicker_0, trip_kickers = _pop_highest_rank(_clear_rank(rank_mask, trip_rank))
        trip_kicker_1, _ = _pop_highest_rank(trip_kickers)
        pair_rank_0, pair_ranks_rest = _pop_highest_rank(pairs_or_better)
        pair_rank_1, _ = _pop_highest_rank(pair_ranks_rest)
        two_pair_kicker, _ = _pop_highest_rank(_clear_rank(_clear_rank(rank_mask, pair_rank_0), pair_rank_1))
        one_pair_kicker_0, one_pair_kickers = _pop_highest_rank(_clear_rank(rank_mask, pair_rank_0))
        one_pair_kicker_1, one_pair_kickers = _pop_highest_rank(one_pair_kickers)
        one_pair_kicker_2, _ = _pop_highest_rank(one_pair_kickers)
        high_idx_0, high_rest = _pop_highest_rank(rank_mask)
        high_idx_1, high_rest = _pop_highest_rank(high_rest)
        high_idx_2, high_rest = _pop_highest_rank(high_rest)
        high_idx_3, high_rest = _pop_highest_rank(high_rest)
        high_idx_4, _ = _pop_highest_rank(high_rest)

        straight_flush_score = _pack_fields(
            sf_found * 9, tl.where(sf_found > 0, sf_high, 0), 0, 0, 0, 0
        )
        four_kind_score = _pack_fields(
            (quads != 0) * 8,
            tl.where(quads != 0, quad_rank, 0),
            tl.where(quads != 0, quad_kicker, 0),
            0, 0, 0,
        )
        full_house_mask = (trips_or_better != 0) & (_clear_rank(pairs_or_better, trip_rank) != 0)
        full_house_score = _pack_fields(
            full_house_mask * 7,
            tl.where(full_house_mask, trip_rank, 0),
            tl.where(full_house_mask, pair_for_boat, 0),
            0, 0, 0,
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
            straight_found * 5, tl.where(straight_found > 0, straight_high, 0), 0, 0, 0, 0
        )
        three_kind_mask = trips_or_better != 0
        three_kind_score = _pack_fields(
            three_kind_mask * 4,
            tl.where(three_kind_mask, trip_rank, 0),
            tl.where(three_kind_mask, trip_kicker_0, 0),
            tl.where(three_kind_mask, trip_kicker_1, 0),
            0, 0,
        )
        two_pair_mask = pair_ranks_rest != 0
        two_pair_score = _pack_fields(
            two_pair_mask * 3,
            tl.where(two_pair_mask, pair_rank_0, 0),
            tl.where(two_pair_mask, pair_rank_1, 0),
            tl.where(two_pair_mask, two_pair_kicker, 0),
            0, 0,
        )
        one_pair_mask = pairs_or_better != 0
        one_pair_score = _pack_fields(
            one_pair_mask * 2,
            tl.where(one_pair_mask, pair_rank_0, 0),
            tl.where(one_pair_mask, one_pair_kicker_0, 0),
            tl.where(one_pair_mask, one_pair_kicker_1, 0),
            tl.where(one_pair_mask, one_pair_kicker_2, 0),
            0,
        )
        high_card_score = _pack_fields(1, high_idx_0, high_idx_1, high_idx_2, high_idx_3, high_idx_4)

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


    _RANK_AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_M": bm}, num_warps=nw)
        for bm in (64, 128, 256)
        for nw in (1, 2, 4)
    ]


    @triton.autotune(configs=_RANK_AUTOTUNE_CONFIGS, key=[])
    @triton.jit
    def _rank_7_cards_batched_kernel(
        cards_ptr,           # [N, 7] int16 — 7 cards per hand
        out_ptr,             # [N] int32 — packed rank score
        N,
        BLOCK_M: tl.constexpr,
    ):
        """One thread per hand, ``BLOCK_M`` hands per program. Vectorizes the
        scoring across the hand dimension (vs. the old scalar one-hand-per-warp
        kernel). Undefined for dup-card hands; see ``_score_from_cards``.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < N
        base = rows * 7
        c0 = tl.load(cards_ptr + base + 0, mask=valid, other=0).to(tl.int32)
        c1 = tl.load(cards_ptr + base + 1, mask=valid, other=0).to(tl.int32)
        c2 = tl.load(cards_ptr + base + 2, mask=valid, other=0).to(tl.int32)
        c3 = tl.load(cards_ptr + base + 3, mask=valid, other=0).to(tl.int32)
        c4 = tl.load(cards_ptr + base + 4, mask=valid, other=0).to(tl.int32)
        c5 = tl.load(cards_ptr + base + 5, mask=valid, other=0).to(tl.int32)
        c6 = tl.load(cards_ptr + base + 6, mask=valid, other=0).to(tl.int32)
        score = _score_from_cards(c0, c1, c2, c3, c4, c5, c6)
        tl.store(out_ptr + rows, score, mask=valid)


    @triton.autotune(configs=_RANK_AUTOTUNE_CONFIGS, key=[])
    @triton.jit
    def _rank_hands_fused_kernel(
        board_ptr,           # [M, 5] int16 — board card IDs
        combos_ptr,          # [NUM_HANDS, 2] int16 — hole-card combos
        out_ptr,             # [M * NUM_HANDS] int32 — packed rank score
        N,                   # M * NUM_HANDS
        NUM_HANDS: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Like ``_rank_7_cards_batched_kernel`` but gathers each hand's seven
        cards straight from ``board`` + ``combos`` — avoids materializing the
        [M, NUM_HANDS, 7] tensor. Row ``r`` maps to board ``r // NUM_HANDS``
        and combo ``r % NUM_HANDS``.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        valid = rows < N
        m = rows // NUM_HANDS
        c = rows % NUM_HANDS
        c0 = tl.load(combos_ptr + c * 2 + 0, mask=valid, other=0).to(tl.int32)
        c1 = tl.load(combos_ptr + c * 2 + 1, mask=valid, other=0).to(tl.int32)
        bb = m * 5
        c2 = tl.load(board_ptr + bb + 0, mask=valid, other=0).to(tl.int32)
        c3 = tl.load(board_ptr + bb + 1, mask=valid, other=0).to(tl.int32)
        c4 = tl.load(board_ptr + bb + 2, mask=valid, other=0).to(tl.int32)
        c5 = tl.load(board_ptr + bb + 3, mask=valid, other=0).to(tl.int32)
        c6 = tl.load(board_ptr + bb + 4, mask=valid, other=0).to(tl.int32)
        score = _score_from_cards(c0, c1, c2, c3, c4, c5, c6)
        tl.store(out_ptr + rows, score, mask=valid)
