"""Microbench for the rank-7-cards single-batch Triton kernel.

Skips the argsort in ``rank_hands_triton`` and isolates the kernel cost, then
tests three candidate improvements:

  V0  baseline            current ``_rank_7_cards_single_batch_kernel``
                          (one hand per program, num_warps=1, scalar body,
                          double-loads the 7 cards).
  V2  batched + dedup     BLOCK_M hands per program, vectorized across hands,
                          cards loaded once (covers ideas #1 and #2).
  V3  batched + fused     like V2 but gathers the 7 cards from board[M,5] +
                          combos[1326,2] inside the kernel, so the big
                          [M,1326,7] intermediate is never materialized
                          (covers ideas #1 + #2 + #3).

Run: .venv/bin/python .codex/bench_rank_kernel.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.env.rules_triton import (
    CARD_LANES,
    RANK_LANES,
    _clear_rank,
    _pack_fields,
    _pop_highest_rank,
    _popcount,
    _straight_high,
    rank_7_cards_single_batch_triton,
)

DEVICE = torch.device("cuda")


# --------------------------------------------------------------------------
# V2 / V3 shared vectorized evaluator: scores BLOCK_M hands at once.
# Each lane of the [BLOCK_M] vectors is one independent hand.
# --------------------------------------------------------------------------
@triton.jit
def _score_from_cards(c0, c1, c2, c3, c4, c5, c6):
    # rank multiplicity masks via the incremental seen-bit trick, vectorized.
    one = tl.full((), 1, tl.int32)
    b0 = one << (c0 % 13)
    b1 = one << (c1 % 13)
    b2 = one << (c2 % 13)
    b3 = one << (c3 % 13)
    b4 = one << (c4 % 13)
    b5 = one << (c5 % 13)
    b6 = one << (c6 % 13)

    seen = b0
    pairs = b0 & 0
    trips = b0 & 0
    quads = b0 & 0
    # card 1
    quads = quads | (trips & b1); trips = trips | (pairs & b1); pairs = pairs | (seen & b1); seen = seen | b1
    quads = quads | (trips & b2); trips = trips | (pairs & b2); pairs = pairs | (seen & b2); seen = seen | b2
    quads = quads | (trips & b3); trips = trips | (pairs & b3); pairs = pairs | (seen & b3); seen = seen | b3
    quads = quads | (trips & b4); trips = trips | (pairs & b4); pairs = pairs | (seen & b4); seen = seen | b4
    quads = quads | (trips & b5); trips = trips | (pairs & b5); pairs = pairs | (seen & b5); seen = seen | b5
    quads = quads | (trips & b6); trips = trips | (pairs & b6); pairs = pairs | (seen & b6); seen = seen | b6

    suit0 = (tl.where(c0 // 13 == 0, b0, 0) | tl.where(c1 // 13 == 0, b1, 0) | tl.where(c2 // 13 == 0, b2, 0)
             | tl.where(c3 // 13 == 0, b3, 0) | tl.where(c4 // 13 == 0, b4, 0) | tl.where(c5 // 13 == 0, b5, 0)
             | tl.where(c6 // 13 == 0, b6, 0))
    suit1 = (tl.where(c0 // 13 == 1, b0, 0) | tl.where(c1 // 13 == 1, b1, 0) | tl.where(c2 // 13 == 1, b2, 0)
             | tl.where(c3 // 13 == 1, b3, 0) | tl.where(c4 // 13 == 1, b4, 0) | tl.where(c5 // 13 == 1, b5, 0)
             | tl.where(c6 // 13 == 1, b6, 0))
    suit2 = (tl.where(c0 // 13 == 2, b0, 0) | tl.where(c1 // 13 == 2, b1, 0) | tl.where(c2 // 13 == 2, b2, 0)
             | tl.where(c3 // 13 == 2, b3, 0) | tl.where(c4 // 13 == 2, b4, 0) | tl.where(c5 // 13 == 2, b5, 0)
             | tl.where(c6 // 13 == 2, b6, 0))
    suit3 = (tl.where(c0 // 13 == 3, b0, 0) | tl.where(c1 // 13 == 3, b1, 0) | tl.where(c2 // 13 == 3, b2, 0)
             | tl.where(c3 // 13 == 3, b3, 0) | tl.where(c4 // 13 == 3, b4, 0) | tl.where(c5 // 13 == 3, b5, 0)
             | tl.where(c6 // 13 == 3, b6, 0))

    rank_mask = seen
    pairs_or_better = pairs
    trips_or_better = trips

    suit_count_0 = _popcount(suit0)
    suit_count_1 = _popcount(suit1)
    suit_count_2 = _popcount(suit2)
    suit_count_3 = _popcount(suit3)

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
        suit0,
        tl.where(flush_suit == 1, suit1, tl.where(flush_suit == 2, suit2, suit3)),
    )
    flush_idx_0, flush_mask = _pop_highest_rank(flush_mask)
    flush_idx_1, flush_mask = _pop_highest_rank(flush_mask)
    flush_idx_2, flush_mask = _pop_highest_rank(flush_mask)
    flush_idx_3, flush_mask = _pop_highest_rank(flush_mask)
    flush_idx_4, flush_mask = _pop_highest_rank(flush_mask)

    straight_found, straight_high = _straight_high(rank_mask)
    sf_found_0, sf_high_0 = _straight_high(suit0)
    sf_found_1, sf_high_1 = _straight_high(suit1)
    sf_found_2, sf_high_2 = _straight_high(suit2)
    sf_found_3, sf_high_3 = _straight_high(suit3)
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

    straight_flush_score = _pack_fields(sf_found * 9, tl.where(sf_found > 0, sf_high, 0), 0, 0, 0, 0)
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


@triton.jit
def _rank_batched_kernel(cards_ptr, out_ptr, N, BLOCK_M: tl.constexpr):
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


def rank_batched(cards_flat: torch.Tensor, block_m: int, num_warps: int) -> torch.Tensor:
    cards_i16 = cards_flat.contiguous().to(torch.int16)
    n = cards_i16.shape[0]
    out = torch.empty((n,), device=cards_i16.device, dtype=torch.int32)
    grid = (triton.cdiv(n, block_m),)
    _rank_batched_kernel[grid](cards_i16, out, n, BLOCK_M=block_m, num_warps=num_warps)
    return out


@triton.jit
def _rank_fused_kernel(board_ptr, combos_ptr, out_ptr, M, NUM_HANDS, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n = M * NUM_HANDS
    valid = rows < n
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


def rank_fused(board: torch.Tensor, block_m: int, num_warps: int) -> torch.Tensor:
    m = board.shape[0]
    combos = hand_combos_tensor(device=board.device).to(torch.int16).contiguous()
    board_i16 = board.to(torch.int16).contiguous()
    n = m * NUM_HANDS
    out = torch.empty((n,), device=board.device, dtype=torch.int32)
    grid = (triton.cdiv(n, block_m),)
    _rank_fused_kernel[grid](board_i16, combos, out, m, NUM_HANDS, BLOCK_M=block_m, num_warps=num_warps)
    return out


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------
def build_cards_flat(board: torch.Tensor) -> torch.Tensor:
    """Replicates rank_hands_triton's materialization of [M,1326,7]."""
    m = board.shape[0]
    combos = hand_combos_tensor(device=board.device)
    holes = combos[None, :, :].expand(m, NUM_HANDS, 2)
    boards_b = board[:, None, :].expand(m, NUM_HANDS, 5)
    cards = torch.cat([holes, boards_b], dim=-1).to(torch.int16)
    return cards.reshape(-1, 7).contiguous()


def make_boards(m: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(0)
    boards = torch.empty((m, 5), dtype=torch.long)
    for i in range(m):
        boards[i] = torch.randperm(52, generator=g)[:5]
    return boards.to(DEVICE)


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    M = 512
    board = make_boards(M)
    cards_flat = build_cards_flat(board)
    N = cards_flat.shape[0]
    print(f"M={M} boards, N={N} hands ({N/1e6:.2f}M)\n")

    # ---- correctness (legal hands only) ----
    # Combos that share a card with the board are blocked/illegal; their score
    # is undefined (the baseline's tl.sum suit-mask double-counts the dup card,
    # the batched kernels OR it). We only require agreement on distinct-card hands.
    ref = rank_7_cards_single_batch_triton(cards_flat)
    v2 = rank_batched(cards_flat, block_m=128, num_warps=4)
    v3 = rank_fused(board, block_m=128, num_warps=4)
    sorted_cards, _ = cards_flat.sort(dim=1)
    legal = (sorted_cards[:, 1:] != sorted_cards[:, :-1]).all(dim=1)
    ok2 = torch.equal(ref[legal], v2[legal])
    ok3 = torch.equal(ref[legal], v3[legal])
    print(f"legal hands: {int(legal.sum())}/{N}  (dup-card hands undefined, excluded)")
    print(f"correctness on legal: V2=={'OK' if ok2 else 'FAIL'}  V3=={'OK' if ok3 else 'FAIL'}")
    print()

    # ---- kernel-only timings ----
    print("=== kernel only (cards_flat already materialized) ===")
    t0 = timed(lambda: rank_7_cards_single_batch_triton(cards_flat))
    print(f"V0 baseline (1 hand/prog, nw=1)        {t0:8.3f} ms   1.00x")

    best = None
    for bm in (32, 64, 128, 256, 512):
        for nw in (1, 2, 4, 8):
            t = timed(lambda: rank_batched(cards_flat, bm, nw))
            tag = f"V2 batched+dedup BLOCK_M={bm:<4d} nw={nw}"
            print(f"{tag:38s} {t:8.3f} ms   {t0/t:5.2f}x")
            if best is None or t < best[0]:
                best = (t, bm, nw)
    print(f"  -> best V2: BLOCK_M={best[1]} nw={best[2]}  {best[0]:.3f} ms  ({t0/best[0]:.2f}x)\n")
    bm_b, nw_b = best[1], best[2]

    # ---- end-to-end (skip argsort), includes materialization cost ----
    print("=== end-to-end minus argsort (materialize + kernel) ===")
    te0 = timed(lambda: rank_7_cards_single_batch_triton(build_cards_flat(board)))
    print(f"V0 materialize+baseline                {te0:8.3f} ms   1.00x")
    te2 = timed(lambda: rank_batched(build_cards_flat(board), bm_b, nw_b))
    print(f"V2 materialize+batched                 {te2:8.3f} ms   {te0/te2:5.2f}x")

    best3 = None
    for bm in (64, 128, 256, 512):
        for nw in (2, 4, 8):
            t = timed(lambda: rank_fused(board, bm, nw))
            tag = f"V3 fused (no materialize) BLOCK_M={bm:<4d} nw={nw}"
            print(f"{tag:44s} {t:8.3f} ms   {te0/t:5.2f}x")
            if best3 is None or t < best3[0]:
                best3 = (t, bm, nw)
    print(f"  -> best V3: BLOCK_M={best3[1]} nw={best3[2]}  {best3[0]:.3f} ms  ({te0/best3[0]:.2f}x vs V0 e2e)")


if __name__ == "__main__":
    main()
