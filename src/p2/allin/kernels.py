from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from p2.env.card_utils import NUM_HANDS

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _pcg_hash_u32(x):
        state = x * 747796405 + 2891336453
        shift = (state >> 28) + 4
        word = ((state >> shift) ^ state) * 277803737
        return (word >> 22) ^ word

    @triton.jit
    def _allin_build_board_cdf_kernel(
        beliefs_ptr,
        board_allowed_ptr,
        cdf_ptr,
        rows: tl.constexpr,
        players: tl.constexpr,
        hands: tl.constexpr,
        boards_per_root: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        flat_row = tl.program_id(0)
        row = flat_row // players
        player = flat_row - row * players
        root = row // boards_per_root
        offs = tl.arange(0, BLOCK_H)
        active = offs < hands
        allowed = tl.load(
            board_allowed_ptr + row * hands + offs,
            mask=active,
            other=0,
        ) != 0
        belief = tl.load(
            beliefs_ptr + (root * players + player) * hands + offs,
            mask=active,
            other=0.0,
        )
        masked = tl.where(allowed, belief, 0.0)
        total = tl.sum(masked, axis=0)
        cdf = tl.cumsum(masked, 0) / tl.maximum(total, 1.0e-30)
        cdf = tl.where(active, cdf, 1.0)
        cdf = tl.where(offs == hands - 1, 1.0, cdf)
        tl.store(cdf_ptr + flat_row * hands + offs, cdf, mask=active)

    @triton.jit
    def _allin_cdf_sample_candidates_kernel(
        cdf_ptr,
        candidates_ptr,
        seed_ptr,
        sample_count: tl.constexpr,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        TUPLE_TRIES: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        s_block = tl.program_id(1)
        sample = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        active = sample < sample_count
        seed_value = tl.load(seed_ptr).to(tl.uint32)
        seed_mix = seed_value * tl.full((), 0x9E3779B9, tl.uint32)
        base_sample = row * sample_count + sample

        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                row_offset = (row * PLAYERS + player) * active_hands
                out_base = ((row * PLAYERS + player) * TUPLE_TRIES) * sample_count
                for attempt in tl.static_range(0, TUPLE_TRIES):
                    random_offset = (
                        base_sample * (PLAYERS * TUPLE_TRIES)
                        + attempt * PLAYERS
                        + player
                    )
                    x = _pcg_hash_u32(random_offset.to(tl.uint32) + seed_mix)
                    # Keep u in (0, 1) so zero-mass CDF prefixes cannot be selected.
                    u = (x.to(tl.float32) + 0.5) * 2.3283064365386963e-10
                    lo = tl.full((BLOCK_S,), 0, tl.int32)
                    hi = tl.full((BLOCK_S,), active_hands - 1, tl.int32)
                    for _ in tl.static_range(0, SEARCH_STEPS):
                        mid = (lo + hi) // 2
                        mid_cdf = tl.load(cdf_ptr + row_offset + mid, mask=active, other=1.0)
                        go_left = u <= mid_cdf
                        hi = tl.where(go_left, mid, hi)
                        lo = tl.where(go_left, lo, mid + 1)
                    candidate = tl.minimum(lo, active_hands - 1)
                    tl.store(
                        candidates_ptr + out_base + attempt * sample_count + sample,
                        candidate,
                        mask=active,
                    )

    @triton.jit
    def _allin_select_hero_samples_kernel(
        candidates_ptr,
        hand_masks_ptr,
        hand_ranks_ptr,
        live_ptr,
        used_out_ptr,
        ranks_out_ptr,
        alive_out_ptr,
        sample_count: tl.constexpr,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        TUPLE_TRIES: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        hero = tl.program_id(1)
        s_block = tl.program_id(2)
        sample = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = sample < sample_count
        hero_live = tl.load(live_ptr + row * PLAYERS + hero) != 0

        alive = tl.full((BLOCK_S,), False, tl.int1)
        final_used = tl.full((BLOCK_S,), 0, tl.int64)
        r0 = tl.full((BLOCK_S,), -1, tl.int32)
        r1 = tl.full((BLOCK_S,), -1, tl.int32)
        r2 = tl.full((BLOCK_S,), -1, tl.int32)
        r3 = tl.full((BLOCK_S,), -1, tl.int32)
        r4 = tl.full((BLOCK_S,), -1, tl.int32)
        r5 = tl.full((BLOCK_S,), -1, tl.int32)

        for attempt in tl.static_range(0, TUPLE_TRIES):
            active = s_mask & ~alive & hero_live
            valid = active
            used = tl.full((BLOCK_S,), 0, tl.int64)
            ar0 = tl.full((BLOCK_S,), -1, tl.int32)
            ar1 = tl.full((BLOCK_S,), -1, tl.int32)
            ar2 = tl.full((BLOCK_S,), -1, tl.int32)
            ar3 = tl.full((BLOCK_S,), -1, tl.int32)
            ar4 = tl.full((BLOCK_S,), -1, tl.int32)
            ar5 = tl.full((BLOCK_S,), -1, tl.int32)

            for player in tl.static_range(0, 6):
                if player < PLAYERS:
                    is_opp = player != hero
                    player_live = tl.load(live_ptr + row * PLAYERS + player) != 0
                    sampling = active & valid & is_opp & player_live
                    candidate = tl.load(
                        candidates_ptr
                        + ((row * PLAYERS + player) * TUPLE_TRIES + attempt) * sample_count
                        + sample,
                        mask=sampling,
                        other=0,
                    ).to(tl.int32)
                    candidate_mask = tl.load(hand_masks_ptr + candidate, mask=sampling, other=0)
                    compatible = (candidate_mask & used) == 0
                    valid = valid & (~(is_opp & player_live) | compatible)
                    used = tl.where(sampling, used | candidate_mask, used)
                    candidate_rank = tl.load(
                        hand_ranks_ptr + row * active_hands + candidate,
                        mask=sampling,
                        other=-1,
                    )
                    if player == 0:
                        ar0 = tl.where(sampling, candidate_rank, ar0)
                    if player == 1:
                        ar1 = tl.where(sampling, candidate_rank, ar1)
                    if player == 2:
                        ar2 = tl.where(sampling, candidate_rank, ar2)
                    if player == 3:
                        ar3 = tl.where(sampling, candidate_rank, ar3)
                    if player == 4:
                        ar4 = tl.where(sampling, candidate_rank, ar4)
                    if player == 5:
                        ar5 = tl.where(sampling, candidate_rank, ar5)

            take = active & valid
            alive = alive | take
            final_used = tl.where(take, used, final_used)
            r0 = tl.where(take, ar0, r0)
            r1 = tl.where(take, ar1, r1)
            r2 = tl.where(take, ar2, r2)
            r3 = tl.where(take, ar3, r3)
            r4 = tl.where(take, ar4, r4)
            r5 = tl.where(take, ar5, r5)

        sample_base = (row * PLAYERS + hero) * sample_count + sample
        tl.store(used_out_ptr + sample_base, final_used, mask=s_mask)
        tl.store(alive_out_ptr + sample_base, alive.to(tl.int8), mask=s_mask)
        rank_base = ((row * PLAYERS + hero) * PLAYERS) * sample_count + sample
        tl.store(ranks_out_ptr + rank_base, r0, mask=s_mask)
        if PLAYERS > 1:
            tl.store(ranks_out_ptr + rank_base + sample_count, r1, mask=s_mask)
        if PLAYERS > 2:
            tl.store(ranks_out_ptr + rank_base + 2 * sample_count, r2, mask=s_mask)
        if PLAYERS > 3:
            tl.store(ranks_out_ptr + rank_base + 3 * sample_count, r3, mask=s_mask)
        if PLAYERS > 4:
            tl.store(ranks_out_ptr + rank_base + 4 * sample_count, r4, mask=s_mask)
        if PLAYERS > 5:
            tl.store(ranks_out_ptr + rank_base + 5 * sample_count, r5, mask=s_mask)

    @triton.jit
    def _allin_precompute_layer_results_kernel(
        rank_samples_ptr,
        eligible_ptr,
        layer_amount_ptr,
        best_rank_out_ptr,
        best_count_out_ptr,
        sample_count: tl.constexpr,
        PLAYERS: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        hero_layer = tl.program_id(1)
        s_block = tl.program_id(2)
        hero = hero_layer // PLAYERS
        layer = hero_layer - hero * PLAYERS
        sample = s_block * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = sample < sample_count
        hero_eligible = tl.load(
            eligible_ptr + (row * PLAYERS + layer) * PLAYERS + hero
        ) != 0
        layer_amount = tl.load(layer_amount_ptr + row * PLAYERS + layer)
        active = s_mask & hero_eligible & (layer_amount != 0.0)
        rank_base = ((row * PLAYERS + hero) * PLAYERS) * sample_count + sample

        best = tl.full((BLOCK_S,), -1, tl.int32)
        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                is_opp = player != hero
                player_eligible = tl.load(
                    eligible_ptr + (row * PLAYERS + layer) * PLAYERS + player
                ) != 0
                opp_rank = tl.load(
                    rank_samples_ptr + rank_base + player * sample_count,
                    mask=active & is_opp,
                    other=-1,
                )
                use_opp = is_opp & player_eligible
                best = tl.maximum(best, tl.where(use_opp, opp_rank, -1))

        count = tl.full((BLOCK_S,), 0, tl.int8)
        for player in tl.static_range(0, 6):
            if player < PLAYERS:
                is_opp = player != hero
                player_eligible = tl.load(
                    eligible_ptr + (row * PLAYERS + layer) * PLAYERS + player
                ) != 0
                opp_rank = tl.load(
                    rank_samples_ptr + rank_base + player * sample_count,
                    mask=active & is_opp,
                    other=-1,
                )
                use_opp = is_opp & player_eligible & (opp_rank == best)
                count += use_opp.to(tl.int8)

        out_base = ((row * PLAYERS + hero) * PLAYERS + layer) * sample_count + sample
        tl.store(best_rank_out_ptr + out_base, best, mask=s_mask)
        tl.store(best_count_out_ptr + out_base, count, mask=s_mask)

    @triton.jit
    def _allin_score_hero_hands_from_samples_kernel(
        hand_masks_ptr,
        hand_ranks_ptr,
        board_allowed_ptr,
        used_samples_ptr,
        alive_samples_ptr,
        best_rank_samples_ptr,
        best_count_samples_ptr,
        layer_amount_ptr,
        eligible_ptr,
        payout_ptr,
        denom_ptr,
        sample_count: tl.constexpr,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        hero = tl.program_id(1)
        h_block = tl.program_id(2)
        hand = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = hand < active_hands
        hero_mask = tl.load(hand_masks_ptr + hand, mask=h_mask, other=0)
        hero_rank = tl.load(
            hand_ranks_ptr + row * active_hands + hand,
            mask=h_mask,
            other=-1,
        )
        board_ok = tl.load(board_allowed_ptr + row * active_hands + hand, mask=h_mask, other=0) != 0
        numerator = tl.zeros((BLOCK_H,), dtype=tl.float32)
        denominator = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for sample_start in tl.range(0, sample_count, BLOCK_S):
            sample = sample_start + tl.arange(0, BLOCK_S)
            s_mask = sample < sample_count
            sample_base = (row * PLAYERS + hero) * sample_count + sample
            used = tl.load(used_samples_ptr + sample_base, mask=s_mask, other=0)
            alive = tl.load(alive_samples_ptr + sample_base, mask=s_mask, other=0) != 0
            compatible = (
                h_mask[:, None]
                & s_mask[None, :]
                & alive[None, :]
                & board_ok[:, None]
                & ((hero_mask[:, None] & used[None, :]) == 0)
            )
            payout = tl.zeros((BLOCK_H, BLOCK_S), dtype=tl.float32)

            for layer in tl.static_range(0, 6):
                if layer < PLAYERS:
                    hero_eligible = tl.load(
                        eligible_ptr + (row * PLAYERS + layer) * PLAYERS + hero
                    ) != 0
                    layer_amount = tl.load(layer_amount_ptr + row * PLAYERS + layer)
                    if hero_eligible & (layer_amount != 0.0):
                        best_base = (
                            ((row * PLAYERS + hero) * PLAYERS + layer)
                            * sample_count
                            + sample
                        )
                        best_opp = tl.load(
                            best_rank_samples_ptr + best_base,
                            mask=s_mask,
                            other=-1,
                        )
                        best_count = tl.load(
                            best_count_samples_ptr + best_base,
                            mask=s_mask,
                            other=0,
                        ).to(tl.float32)
                        wins = hero_rank[:, None] > best_opp[None, :]
                        ties = hero_rank[:, None] == best_opp[None, :]
                        share = tl.where(
                            wins,
                            1.0,
                            tl.where(ties, 1.0 / (1.0 + best_count[None, :]), 0.0),
                        )
                        payout += layer_amount * share

            comp_f = compatible.to(tl.float32)
            numerator += tl.sum(comp_f * payout, axis=1)
            denominator += tl.sum(comp_f, axis=1)

        out_base = (row * PLAYERS + hero) * active_hands + hand
        tl.store(payout_ptr + out_base, numerator, mask=h_mask)
        tl.store(denom_ptr + out_base, denominator, mask=h_mask)

    @triton.jit
    def _allin_score_hero_hands_accumulate_kernel(
        hand_masks_ptr,
        hand_ranks_ptr,
        board_allowed_ptr,
        used_samples_ptr,
        alive_samples_ptr,
        best_rank_samples_ptr,
        best_count_samples_ptr,
        layer_amount_ptr,
        eligible_ptr,
        payout_sum_ptr,
        denom_sum_ptr,
        sample_count: tl.constexpr,
        active_hands: tl.constexpr,
        PLAYERS: tl.constexpr,
        BOARDS_PER_ROOT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        row = tl.program_id(0)
        hero = tl.program_id(1)
        h_block = tl.program_id(2)
        hand = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = hand < active_hands
        hero_mask = tl.load(hand_masks_ptr + hand, mask=h_mask, other=0)
        hero_rank = tl.load(
            hand_ranks_ptr + row * active_hands + hand,
            mask=h_mask,
            other=-1,
        )
        board_ok = tl.load(board_allowed_ptr + row * active_hands + hand, mask=h_mask, other=0) != 0
        numerator = tl.zeros((BLOCK_H,), dtype=tl.float32)
        denominator = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for sample_start in tl.range(0, sample_count, BLOCK_S):
            sample = sample_start + tl.arange(0, BLOCK_S)
            s_mask = sample < sample_count
            sample_base = (row * PLAYERS + hero) * sample_count + sample
            used = tl.load(used_samples_ptr + sample_base, mask=s_mask, other=0)
            alive = tl.load(alive_samples_ptr + sample_base, mask=s_mask, other=0) != 0
            compatible = (
                h_mask[:, None]
                & s_mask[None, :]
                & alive[None, :]
                & board_ok[:, None]
                & ((hero_mask[:, None] & used[None, :]) == 0)
            )
            payout = tl.zeros((BLOCK_H, BLOCK_S), dtype=tl.float32)

            for layer in tl.static_range(0, 6):
                if layer < PLAYERS:
                    hero_eligible = tl.load(
                        eligible_ptr + (row * PLAYERS + layer) * PLAYERS + hero
                    ) != 0
                    layer_amount = tl.load(layer_amount_ptr + row * PLAYERS + layer)
                    if hero_eligible & (layer_amount != 0.0):
                        best_base = (
                            ((row * PLAYERS + hero) * PLAYERS + layer)
                            * sample_count
                            + sample
                        )
                        best_opp = tl.load(
                            best_rank_samples_ptr + best_base,
                            mask=s_mask,
                            other=-1,
                        )
                        best_count = tl.load(
                            best_count_samples_ptr + best_base,
                            mask=s_mask,
                            other=0,
                        ).to(tl.float32)
                        wins = hero_rank[:, None] > best_opp[None, :]
                        ties = hero_rank[:, None] == best_opp[None, :]
                        share = tl.where(
                            wins,
                            1.0,
                            tl.where(ties, 1.0 / (1.0 + best_count[None, :]), 0.0),
                        )
                        payout += layer_amount * share

            comp_f = compatible.to(tl.float32)
            numerator += tl.sum(comp_f * payout, axis=1)
            denominator += tl.sum(comp_f, axis=1)

        root = row // BOARDS_PER_ROOT
        out_base = (root * PLAYERS + hero) * active_hands + hand
        tl.atomic_add(payout_sum_ptr + out_base, numerator, sem="relaxed", mask=h_mask)
        tl.atomic_add(denom_sum_ptr + out_base, denominator, sem="relaxed", mask=h_mask)


def triton_available() -> bool:
    return triton is not None


@dataclass
class AllinCdfWorkspace:
    """Reusable GPU buffers for repeated all-in CDF tuple-reject scoring.

    The buffers are sized for ``max_rows`` board rows; chunks with fewer rows use
    contiguous prefix views. The inner dims (``players``, ``tuple_tries``,
    ``sample_count``, ``hands``) are fixed at creation time because the Triton
    kernels index these buffers with manual flat offsets derived from them.
    """

    cdf: torch.Tensor
    candidates: torch.Tensor
    used_samples: torch.Tensor
    rank_samples: torch.Tensor
    alive_samples: torch.Tensor
    best_rank_samples: torch.Tensor
    best_count_samples: torch.Tensor
    payout: torch.Tensor
    denom: torch.Tensor
    seed: torch.Tensor
    max_rows: int
    players: int
    tuple_tries: int
    sample_count: int
    hands: int


def make_allin_cdf_workspace(
    *,
    max_rows: int,
    players: int,
    sample_count: int,
    tuple_tries: int,
    device: torch.device,
    hands: int = NUM_HANDS,
) -> AllinCdfWorkspace:
    """Preallocate reusable buffers for ``allin_cdf_tuple_score_cuda_into``."""
    if triton is None:
        raise RuntimeError("triton is not available")
    if max_rows <= 0 or players <= 0 or sample_count <= 0 or tuple_tries <= 0:
        raise ValueError("workspace dimensions must be positive")
    flat_rows = max_rows * players
    return AllinCdfWorkspace(
        cdf=torch.empty(flat_rows, hands, dtype=torch.float32, device=device),
        candidates=torch.empty(
            max_rows, players, tuple_tries, sample_count, dtype=torch.int16, device=device
        ),
        used_samples=torch.empty(
            max_rows, players, sample_count, dtype=torch.int64, device=device
        ),
        rank_samples=torch.empty(
            max_rows, players, players, sample_count, dtype=torch.int32, device=device
        ),
        alive_samples=torch.empty(
            max_rows, players, sample_count, dtype=torch.int8, device=device
        ),
        best_rank_samples=torch.empty(
            max_rows, players, players, sample_count, dtype=torch.int32, device=device
        ),
        best_count_samples=torch.empty(
            max_rows, players, players, sample_count, dtype=torch.int8, device=device
        ),
        payout=torch.empty(max_rows, players, hands, dtype=torch.float32, device=device),
        denom=torch.empty(max_rows, players, hands, dtype=torch.float32, device=device),
        seed=torch.zeros((), dtype=torch.int64, device=device),
        max_rows=max_rows,
        players=players,
        tuple_tries=tuple_tries,
        sample_count=sample_count,
        hands=hands,
    )


def _build_cdf_tables_into(
    workspace: AllinCdfWorkspace,
    flat_beliefs: torch.Tensor,
) -> torch.Tensor:
    rows_flat, active_hands = flat_beliefs.shape
    cdf = workspace.cdf[:rows_flat]
    torch.cumsum(flat_beliefs, dim=1, out=cdf)
    cdf[:, -1].fill_(1.0)
    return cdf


def _build_board_cdf_tables_into(
    workspace: AllinCdfWorkspace,
    beliefs: torch.Tensor,
    board_allowed: torch.Tensor,
    *,
    rows: int,
    players: int,
    hands: int,
    boards_per_root: int,
) -> torch.Tensor:
    rows_flat = rows * players
    cdf = workspace.cdf[:rows_flat]
    block_h = triton.next_power_of_2(hands)
    _allin_build_board_cdf_kernel[(rows_flat,)](
        beliefs,
        board_allowed,
        cdf,
        rows=rows,
        players=players,
        hands=hands,
        boards_per_root=boards_per_root,
        BLOCK_H=block_h,
        num_warps=2,
    )
    return cdf


def allin_cdf_tuple_score_cuda_into(
    workspace: AllinCdfWorkspace,
    *,
    board_beliefs: torch.Tensor,
    hand_masks: torch.Tensor,
    hand_ranks: torch.Tensor,
    board_allowed: torch.Tensor,
    live_mask: torch.Tensor,
    layer_amount: torch.Tensor,
    eligible: torch.Tensor,
    seed: int,
    block_s: int | None = None,
    block_h: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one board chunk into ``workspace``; return prefix views of the sums.

    ``board_beliefs`` is ``[rows, players, 1326]`` and must already be
    board-masked and normalized. The side tensors must already carry the dtypes
    the kernels expect (``hand_masks`` int64, ``hand_ranks`` int32,
    ``board_allowed``/``live_mask``/``eligible`` int8) and be contiguous. The
    returned ``payout``/``denom`` are views into ``workspace`` and are
    overwritten by the next call.

    ``block_s`` tiles the per-row samples. The kernels key their RNG offsets on
    the absolute sample index, so the value only affects tiling/occupancy, not
    the result. When ``None`` it defaults to ``min(32, next_pow2(sample_count))``
    so small sample counts (e.g. 16) don't waste half the lanes of a 32-wide tile.
    """
    if triton is None:
        raise RuntimeError("triton is not available")
    if board_beliefs.device.type != "cuda":
        raise ValueError("all-in tuple scorer requires CUDA tensors")
    rows, players, hands = board_beliefs.shape
    if players != workspace.players or hands != workspace.hands:
        raise ValueError("workspace players/hands mismatch")
    if rows > workspace.max_rows:
        raise ValueError("chunk has more rows than the workspace was sized for")
    sample_count = workspace.sample_count
    tuple_tries = workspace.tuple_tries
    if block_s is None:
        block_s = min(32, triton.next_power_of_2(sample_count))

    flat_beliefs = board_beliefs.reshape(rows * players, hands).contiguous()
    cdf = _build_cdf_tables_into(workspace, flat_beliefs)

    candidates = workspace.candidates[:rows]
    used_samples = workspace.used_samples[:rows]
    rank_samples = workspace.rank_samples[:rows]
    alive_samples = workspace.alive_samples[:rows]
    best_rank_samples = workspace.best_rank_samples[:rows]
    best_count_samples = workspace.best_count_samples[:rows]
    payout = workspace.payout[:rows]
    denom = workspace.denom[:rows]
    workspace.seed.fill_(int(seed))

    sample_grid = (rows, triton.cdiv(sample_count, block_s))
    _allin_cdf_sample_candidates_kernel[sample_grid](
        cdf,
        candidates,
        workspace.seed,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        TUPLE_TRIES=tuple_tries,
        SEARCH_STEPS=(hands - 1).bit_length(),
        BLOCK_S=block_s,
        num_warps=4,
    )
    num_h_blocks = triton.cdiv(hands, block_h)
    num_s_blocks = triton.cdiv(sample_count, block_s)
    select_grid = (rows, players, num_s_blocks)
    _allin_select_hero_samples_kernel[select_grid](
        candidates,
        hand_masks,
        hand_ranks,
        live_mask,
        used_samples,
        rank_samples,
        alive_samples,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        TUPLE_TRIES=tuple_tries,
        BLOCK_S=block_s,
        num_warps=4,
    )
    precompute_grid = (rows, players * players, num_s_blocks)
    _allin_precompute_layer_results_kernel[precompute_grid](
        rank_samples,
        eligible,
        layer_amount,
        best_rank_samples,
        best_count_samples,
        sample_count,
        PLAYERS=players,
        BLOCK_S=block_s,
        num_warps=4,
    )
    score_grid = (rows, players, num_h_blocks)
    _allin_score_hero_hands_from_samples_kernel[score_grid](
        hand_masks,
        hand_ranks,
        board_allowed,
        used_samples,
        alive_samples,
        best_rank_samples,
        best_count_samples,
        layer_amount,
        eligible,
        payout,
        denom,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        BLOCK_H=block_h,
        BLOCK_S=block_s,
        num_warps=4,
    )
    return payout, denom


def allin_cdf_tuple_score_from_roots_cuda_into(
    workspace: AllinCdfWorkspace,
    *,
    beliefs: torch.Tensor,
    board_allowed: torch.Tensor,
    boards_per_root: int,
    hand_masks: torch.Tensor,
    hand_ranks: torch.Tensor,
    live_mask: torch.Tensor,
    layer_amount: torch.Tensor,
    eligible: torch.Tensor,
    seed: int,
    block_s: int | None = None,
    block_h: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score one board chunk while building board-conditioned CDFs in Triton.

    ``beliefs`` is the root tensor ``[B, players, 1326]`` and rows in
    ``board_allowed``/``hand_ranks`` must be ordered as
    ``arange(B).repeat_interleave(boards_per_root)``.
    """
    if triton is None:
        raise RuntimeError("triton is not available")
    if beliefs.device.type != "cuda":
        raise ValueError("all-in tuple scorer requires CUDA tensors")
    rows, hands = board_allowed.shape
    root_count, players, belief_hands = beliefs.shape
    if belief_hands != hands or hands != workspace.hands or players != workspace.players:
        raise ValueError("workspace/belief shape mismatch")
    if boards_per_root <= 0 or rows != root_count * boards_per_root:
        raise ValueError("rows must equal roots * boards_per_root")
    if rows > workspace.max_rows:
        raise ValueError("chunk has more rows than the workspace was sized for")
    sample_count = workspace.sample_count
    tuple_tries = workspace.tuple_tries
    if block_s is None:
        block_s = min(32, triton.next_power_of_2(sample_count))

    cdf = _build_board_cdf_tables_into(
        workspace,
        beliefs,
        board_allowed,
        rows=rows,
        players=players,
        hands=hands,
        boards_per_root=boards_per_root,
    )

    candidates = workspace.candidates[:rows]
    used_samples = workspace.used_samples[:rows]
    rank_samples = workspace.rank_samples[:rows]
    alive_samples = workspace.alive_samples[:rows]
    best_rank_samples = workspace.best_rank_samples[:rows]
    best_count_samples = workspace.best_count_samples[:rows]
    payout = workspace.payout[:rows]
    denom = workspace.denom[:rows]
    workspace.seed.fill_(int(seed))

    sample_grid = (rows, triton.cdiv(sample_count, block_s))
    _allin_cdf_sample_candidates_kernel[sample_grid](
        cdf,
        candidates,
        workspace.seed,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        TUPLE_TRIES=tuple_tries,
        SEARCH_STEPS=(hands - 1).bit_length(),
        BLOCK_S=block_s,
        num_warps=4,
    )
    num_h_blocks = triton.cdiv(hands, block_h)
    num_s_blocks = triton.cdiv(sample_count, block_s)
    select_grid = (rows, players, num_s_blocks)
    _allin_select_hero_samples_kernel[select_grid](
        candidates,
        hand_masks,
        hand_ranks,
        live_mask,
        used_samples,
        rank_samples,
        alive_samples,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        TUPLE_TRIES=tuple_tries,
        BLOCK_S=block_s,
        num_warps=4,
    )
    precompute_grid = (rows, players * players, num_s_blocks)
    _allin_precompute_layer_results_kernel[precompute_grid](
        rank_samples,
        eligible,
        layer_amount,
        best_rank_samples,
        best_count_samples,
        sample_count,
        PLAYERS=players,
        BLOCK_S=block_s,
        num_warps=4,
    )
    score_grid = (rows, players, num_h_blocks)
    _allin_score_hero_hands_from_samples_kernel[score_grid](
        hand_masks,
        hand_ranks,
        board_allowed,
        used_samples,
        alive_samples,
        best_rank_samples,
        best_count_samples,
        layer_amount,
        eligible,
        payout,
        denom,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        BLOCK_H=block_h,
        BLOCK_S=block_s,
        num_warps=4,
    )
    return payout, denom


def allin_cdf_tuple_score_from_roots_accumulate_cuda_(
    workspace: AllinCdfWorkspace,
    *,
    beliefs: torch.Tensor,
    board_allowed: torch.Tensor,
    boards_per_root: int,
    hand_masks: torch.Tensor,
    hand_ranks: torch.Tensor,
    live_mask: torch.Tensor,
    layer_amount: torch.Tensor,
    eligible: torch.Tensor,
    payout_sum: torch.Tensor,
    denom_sum: torch.Tensor,
    seed: int,
    block_s: int | None = None,
    block_h: int = 64,
) -> None:
    """Score one board chunk and atomically accumulate into root-level sums."""
    if triton is None:
        raise RuntimeError("triton is not available")
    if beliefs.device.type != "cuda":
        raise ValueError("all-in tuple scorer requires CUDA tensors")
    rows, hands = board_allowed.shape
    root_count, players, belief_hands = beliefs.shape
    if belief_hands != hands or hands != workspace.hands or players != workspace.players:
        raise ValueError("workspace/belief shape mismatch")
    if boards_per_root <= 0 or rows != root_count * boards_per_root:
        raise ValueError("rows must equal roots * boards_per_root")
    if payout_sum.shape != (root_count, players, hands):
        raise ValueError("payout_sum shape mismatch")
    if denom_sum.shape != (root_count, players, hands):
        raise ValueError("denom_sum shape mismatch")
    if rows > workspace.max_rows:
        raise ValueError("chunk has more rows than the workspace was sized for")
    sample_count = workspace.sample_count
    tuple_tries = workspace.tuple_tries
    if block_s is None:
        block_s = min(32, triton.next_power_of_2(sample_count))

    cdf = _build_board_cdf_tables_into(
        workspace,
        beliefs,
        board_allowed,
        rows=rows,
        players=players,
        hands=hands,
        boards_per_root=boards_per_root,
    )

    candidates = workspace.candidates[:rows]
    used_samples = workspace.used_samples[:rows]
    rank_samples = workspace.rank_samples[:rows]
    alive_samples = workspace.alive_samples[:rows]
    best_rank_samples = workspace.best_rank_samples[:rows]
    best_count_samples = workspace.best_count_samples[:rows]
    workspace.seed.fill_(int(seed))

    sample_grid = (rows, triton.cdiv(sample_count, block_s))
    _allin_cdf_sample_candidates_kernel[sample_grid](
        cdf,
        candidates,
        workspace.seed,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        TUPLE_TRIES=tuple_tries,
        SEARCH_STEPS=(hands - 1).bit_length(),
        BLOCK_S=block_s,
        num_warps=4,
    )
    num_h_blocks = triton.cdiv(hands, block_h)
    num_s_blocks = triton.cdiv(sample_count, block_s)
    select_grid = (rows, players, num_s_blocks)
    _allin_select_hero_samples_kernel[select_grid](
        candidates,
        hand_masks,
        hand_ranks,
        live_mask,
        used_samples,
        rank_samples,
        alive_samples,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        TUPLE_TRIES=tuple_tries,
        BLOCK_S=block_s,
        num_warps=4,
    )
    precompute_grid = (rows, players * players, num_s_blocks)
    _allin_precompute_layer_results_kernel[precompute_grid](
        rank_samples,
        eligible,
        layer_amount,
        best_rank_samples,
        best_count_samples,
        sample_count,
        PLAYERS=players,
        BLOCK_S=block_s,
        num_warps=4,
    )
    score_grid = (rows, players, num_h_blocks)
    _allin_score_hero_hands_accumulate_kernel[score_grid](
        hand_masks,
        hand_ranks,
        board_allowed,
        used_samples,
        alive_samples,
        best_rank_samples,
        best_count_samples,
        layer_amount,
        eligible,
        payout_sum,
        denom_sum,
        sample_count,
        active_hands=hands,
        PLAYERS=players,
        BOARDS_PER_ROOT=boards_per_root,
        BLOCK_H=block_h,
        BLOCK_S=block_s,
        num_warps=4,
    )


def allin_cdf_tuple_score_cuda(
    *,
    board_beliefs: torch.Tensor,
    hand_masks: torch.Tensor,
    hand_ranks: torch.Tensor,
    board_allowed: torch.Tensor,
    live_mask: torch.Tensor,
    layer_amount: torch.Tensor,
    eligible: torch.Tensor,
    sample_count: int,
    tuple_tries: int,
    seed: int,
    block_s: int | None = None,
    block_h: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Score full-board all-in samples with CDF tuple-reject sampling.

    ``board_beliefs`` is ``[rows, players, 1326]`` and is expected to already be
    board-masked and normalized. Outputs are unnormalized payout and denominator
    sums with shape ``[rows, players, 1326]``.

    This is a convenience wrapper that allocates a one-shot workspace; callers
    that score many chunks should reuse a :func:`make_allin_cdf_workspace`
    workspace via :func:`allin_cdf_tuple_score_cuda_into` instead.
    """
    if triton is None:
        raise RuntimeError("triton is not available")
    if board_beliefs.device.type != "cuda":
        raise ValueError("all-in tuple scorer requires CUDA tensors")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if tuple_tries <= 0:
        raise ValueError("tuple_tries must be positive")

    start = time.perf_counter()
    rows, players, hands = board_beliefs.shape
    if hands != NUM_HANDS:
        raise ValueError(f"expected {NUM_HANDS} hands, got {hands}")
    workspace = make_allin_cdf_workspace(
        max_rows=rows,
        players=players,
        sample_count=sample_count,
        tuple_tries=tuple_tries,
        device=board_beliefs.device,
        hands=hands,
    )
    payout, denom = allin_cdf_tuple_score_cuda_into(
        workspace,
        board_beliefs=board_beliefs,
        hand_masks=hand_masks.to(torch.int64).contiguous(),
        hand_ranks=hand_ranks.to(torch.int32).contiguous(),
        board_allowed=board_allowed.to(torch.int8).contiguous(),
        live_mask=live_mask.to(torch.int8).contiguous(),
        layer_amount=layer_amount.to(torch.float32).contiguous(),
        eligible=eligible.to(torch.int8).contiguous(),
        seed=seed,
        block_s=block_s,
        block_h=block_h,
    )
    return payout, denom, {"kernel_launch_seconds": time.perf_counter() - start}


# Backward-compatible names for callers that have not been updated yet.
AllinAliasWorkspace = AllinCdfWorkspace
make_allin_alias_workspace = make_allin_cdf_workspace
allin_alias_tuple_score_cuda_into = allin_cdf_tuple_score_cuda_into
allin_alias_tuple_score_cuda = allin_cdf_tuple_score_cuda
