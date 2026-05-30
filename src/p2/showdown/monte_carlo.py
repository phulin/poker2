from __future__ import annotations

import time

import torch

from p2.env.card_utils import NUM_HANDS, board_allowed_hands

from .multiway_showdown_estimators import (
    BatchedAliasTupleRejectWorkspace,
    PreparedBatchedFastSISBelief,
    PreparedFastSISBoard,
    PreparedShowdown,
    alias_triton_tuple_reject_batched_fixed_into,
    build_batched_alias_tables_triton_into,
    make_batched_active_belief_workspace,
    make_batched_alias_tuple_reject_workspace,
    prepare_batched_active_belief_triton_into,
    prepare_fast_sis_board,
)
from .results import (
    PerHandEquityResult,
    aggregate_all_players_from_num_denom,
    safe_divide_by_hand,
)

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
    def _sample_hero_opponents_alias_kernel(
        alias_prob,
        alias_idx,
        hand_masks,
        hand_ranks,
        used_out,
        ranks_out,
        alive_out,
        seed,
        active_hands: tl.constexpr,
        sample_count: tl.constexpr,
        players: tl.constexpr,
        tuple_tries: tl.constexpr,
        block_s: tl.constexpr,
    ):
        batch = tl.program_id(0)
        hero = tl.program_id(1)
        sample_block = tl.program_id(2)
        sample = sample_block * block_s + tl.arange(0, block_s)
        in_range = sample < sample_count
        seed_value = tl.load(seed).to(tl.uint32)
        seed_mix = seed_value * tl.full((), 0x9E3779B9, tl.uint32)
        base_sample = batch * sample_count + sample

        alive = tl.full((block_s,), False, tl.int1)
        final_used = tl.full((block_s,), 0, tl.int64)
        r0 = tl.full((block_s,), -1, tl.int32)
        r1 = tl.full((block_s,), -1, tl.int32)
        r2 = tl.full((block_s,), -1, tl.int32)
        r3 = tl.full((block_s,), -1, tl.int32)
        r4 = tl.full((block_s,), -1, tl.int32)
        r5 = tl.full((block_s,), -1, tl.int32)

        for attempt in tl.static_range(0, tuple_tries):
            active = in_range & ~alive
            valid = active
            used = tl.full((block_s,), 0, tl.int64)
            ar0 = tl.full((block_s,), -1, tl.int32)
            ar1 = tl.full((block_s,), -1, tl.int32)
            ar2 = tl.full((block_s,), -1, tl.int32)
            ar3 = tl.full((block_s,), -1, tl.int32)
            ar4 = tl.full((block_s,), -1, tl.int32)
            ar5 = tl.full((block_s,), -1, tl.int32)

            for player in tl.static_range(0, 6):
                if player < players:
                    is_opp = player != hero
                    row_offset = (batch * players + player) * active_hands
                    random_offset = (
                        base_sample * (players * tuple_tries * 2)
                        + attempt * players * 2
                        + player * 2
                        + hero * players * tuple_tries * sample_count * 2
                    )
                    x0 = _pcg_hash_u32(random_offset.to(tl.uint32) + seed_mix)
                    x1 = _pcg_hash_u32((random_offset + 1).to(tl.uint32) + seed_mix)
                    u_col = x0.to(tl.float32) * 2.3283064365386963e-10
                    u_pick = x1.to(tl.float32) * 2.3283064365386963e-10
                    column = tl.minimum((u_col * active_hands).to(tl.int32), active_hands - 1)
                    sampling = active & valid & is_opp
                    cutoff = tl.load(alias_prob + row_offset + column, mask=sampling, other=1.0)
                    alias = tl.load(alias_idx + row_offset + column, mask=sampling, other=0)
                    candidate = tl.where(u_pick < cutoff, column, alias)
                    candidate_mask = tl.load(hand_masks + candidate, mask=sampling, other=0)
                    compatible = (candidate_mask & used) == 0
                    valid = valid & (~is_opp | compatible)
                    used = tl.where(sampling, used | candidate_mask, used)
                    candidate_rank = tl.load(hand_ranks + candidate, mask=sampling, other=-1)
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

        sample_base = (batch * players + hero) * sample_count + sample
        tl.store(used_out + sample_base, final_used, mask=in_range)
        tl.store(alive_out + sample_base, alive.to(tl.int8), mask=in_range)
        rank_base = ((batch * players + hero) * players) * sample_count + sample
        tl.store(ranks_out + rank_base, r0, mask=in_range)
        if players > 1:
            tl.store(ranks_out + rank_base + sample_count, r1, mask=in_range)
        if players > 2:
            tl.store(ranks_out + rank_base + 2 * sample_count, r2, mask=in_range)
        if players > 3:
            tl.store(ranks_out + rank_base + 3 * sample_count, r3, mask=in_range)
        if players > 4:
            tl.store(ranks_out + rank_base + 4 * sample_count, r4, mask=in_range)
        if players > 5:
            tl.store(ranks_out + rank_base + 5 * sample_count, r5, mask=in_range)

    @triton.jit
    def _score_hero_hands_from_samples_kernel(
        hand_masks,
        hand_ranks,
        used_samples,
        rank_samples,
        alive_samples,
        numerator_out,
        denominator_out,
        active_hands: tl.constexpr,
        sample_count: tl.constexpr,
        players: tl.constexpr,
        block_h: tl.constexpr,
        block_s: tl.constexpr,
    ):
        batch = tl.program_id(0)
        hero = tl.program_id(1)
        h_block = tl.program_id(2)
        h = h_block * block_h + tl.arange(0, block_h)
        h_mask = h < active_hands
        hero_mask = tl.load(hand_masks + h, mask=h_mask, other=0)
        hero_rank = tl.load(hand_ranks + h, mask=h_mask, other=-1)
        numerator = tl.zeros((block_h,), dtype=tl.float32)
        denominator = tl.zeros((block_h,), dtype=tl.float32)

        for sample_start in tl.range(0, sample_count, block_s):
            sample = sample_start + tl.arange(0, block_s)
            s_mask = sample < sample_count
            sample_base = (batch * players + hero) * sample_count + sample
            used = tl.load(used_samples + sample_base, mask=s_mask, other=0)
            alive = tl.load(alive_samples + sample_base, mask=s_mask, other=0) != 0
            compatible = (
                h_mask[:, None]
                & s_mask[None, :]
                & alive[None, :]
                & ((hero_mask[:, None] & used[None, :]) == 0)
            )
            best_opp = tl.full((block_s,), -1, tl.int32)
            tie_count = tl.zeros((block_h, block_s), dtype=tl.float32)
            rank_base = ((batch * players + hero) * players) * sample_count + sample

            for player in tl.static_range(0, 6):
                if player < players:
                    is_opp = player != hero
                    opp_rank = tl.load(
                        rank_samples + rank_base + player * sample_count,
                        mask=s_mask & is_opp,
                        other=-1,
                    )
                    best_opp = tl.maximum(best_opp, tl.where(is_opp, opp_rank, -1))
                    tie_count += tl.where(
                        is_opp & (opp_rank[None, :] == hero_rank[:, None]),
                        1.0,
                        0.0,
                    )

            wins = hero_rank[:, None] > best_opp[None, :]
            ties = hero_rank[:, None] == best_opp[None, :]
            share = tl.where(
                wins,
                1.0,
                tl.where(ties, 1.0 / (tie_count + 1.0), 0.0),
            )
            comp_f = compatible.to(tl.float32)
            denominator += tl.sum(comp_f, axis=1)
            numerator += tl.sum(comp_f * share, axis=1)

        scale = 1.0 / sample_count
        out_base = (batch * players + hero) * active_hands + h
        tl.store(denominator_out + out_base, denominator * scale, mask=h_mask)
        tl.store(numerator_out + out_base, numerator * scale, mask=h_mask)


def alias_tuple_reject_aggregate(*args, **kwargs) -> None:
    """Compatibility name for the existing Triton aggregate tuple-reject runner."""
    return alias_triton_tuple_reject_batched_fixed_into(*args, **kwargs)


def conditional_tuple_reject_by_hand(
    prepared: PreparedShowdown,
    *,
    sample_count: int,
    generator: torch.Generator,
) -> PerHandEquityResult:
    """Reference conditional per-hand MC estimator.

    For each hero hand, opponents are sampled sequentially from their ranges
    restricted by already-used cards. The product of proposal normalizers is the
    importance weight for the collision-free opponent tuple.
    """
    if prepared.beliefs.shape[0] != 1:
        raise ValueError("conditional_tuple_reject_by_hand expects one board")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")

    start = time.perf_counter()
    device = prepared.beliefs.device
    beliefs = prepared.beliefs[0].to(torch.float32)
    players = beliefs.shape[0]
    masks = prepared.hand_masks.reshape(NUM_HANDS)
    ranks = prepared.hand_ranks.reshape(NUM_HANDS)
    allowed = board_allowed_hands(prepared.board)[0]
    active_ids = torch.nonzero(allowed, as_tuple=False).flatten()

    numerator = torch.zeros(players, NUM_HANDS, dtype=torch.float32, device=device)
    denominator = torch.zeros_like(numerator)

    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        hero_belief = beliefs[hero]
        for hand_id in active_ids.detach().cpu().tolist():
            hero_mask = masks[hand_id]
            hero_rank = ranks[hand_id]
            used = hero_mask.expand(sample_count).clone()
            weights = torch.ones(sample_count, dtype=torch.float32, device=device)
            opp_ranks = torch.empty(
                sample_count,
                len(opponents),
                dtype=ranks.dtype,
                device=device,
            )
            alive = torch.ones(sample_count, dtype=torch.bool, device=device)

            for opp_slot, player in enumerate(opponents):
                compatible = (masks[None, :] & used[:, None]) == 0
                proposal = beliefs[player][None, :] * compatible.to(torch.float32)
                normalizer = proposal.sum(dim=1)
                alive = alive & (normalizer > 0)
                safe_normalizer = normalizer.clamp_min(1.0e-30)
                weights = weights * normalizer
                sampled = torch.multinomial(
                    proposal / safe_normalizer[:, None],
                    1,
                    replacement=True,
                    generator=generator,
                ).flatten()
                sampled = torch.where(alive, sampled, torch.zeros_like(sampled))
                used = used | masks.index_select(0, sampled)
                opp_ranks[:, opp_slot] = ranks.index_select(0, sampled)

            best_opp = opp_ranks.max(dim=1).values if opponents else hero_rank
            hero_wins = hero_rank > best_opp
            hero_ties = hero_rank == best_opp
            tie_count = (opp_ranks == hero_rank).sum(dim=1).to(torch.float32) + 1.0
            share = torch.where(
                hero_wins,
                torch.ones_like(weights),
                torch.where(hero_ties, 1.0 / tie_count, torch.zeros_like(weights)),
            )
            weights = torch.where(alive, weights, torch.zeros_like(weights))
            denominator[hero, hand_id] = weights.mean()
            numerator[hero, hand_id] = (weights * share).mean()

        del hero_belief

    equity_by_hand = safe_divide_by_hand(numerator, denominator)
    aggregate = torch.stack(
        [
            (beliefs[player] * numerator[player]).sum()
            / (beliefs[player] * denominator[player]).sum().clamp_min(1.0e-30)
            for player in range(players)
        ],
    ).to(torch.float32)[None, :]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand,
        aggregate_equity=aggregate,
        denominator_by_hand=denominator,
        numerator_by_hand=numerator,
        seconds=time.perf_counter() - start,
    )


def _can_use_fast_tuple_reject_sis(prepared: PreparedShowdown, players: int) -> bool:
    if triton is None or prepared.beliefs.device.type != "cuda":
        return False
    if players > 6:
        return False
    if prepared.board.shape[0] == 1:
        return True
    return bool(torch.equal(prepared.board, prepared.board[:1].expand_as(prepared.board)))


def _tuple_reject_sis_by_hand_triton(
    prepared: PreparedShowdown,
    *,
    sample_count: int,
    seed: int,
    tuple_tries: int,
    sample_chunk: int,
    hand_chunk: int,
) -> PerHandEquityResult:
    if triton is None:
        raise RuntimeError("triton is not available")
    start = time.perf_counter()
    device = prepared.beliefs.device
    batch_size, players, _ = prepared.beliefs.shape
    if sample_chunk & (sample_chunk - 1):
        raise ValueError("sample_chunk must be a power of two for the Triton path")
    if hand_chunk & (hand_chunk - 1):
        raise ValueError("hand_chunk must be a power of two for the Triton path")

    single_board = PreparedShowdown(
        board=prepared.board[:1],
        beliefs=prepared.beliefs[:1],
        hand_ranks=prepared.hand_ranks[:1],
        combos=prepared.combos,
        hand_masks=prepared.hand_masks,
        setup_seconds=prepared.setup_seconds,
    )
    fast_board = prepare_fast_sis_board(single_board)
    belief_workspace = make_batched_active_belief_workspace(
        fast_board,
        batch_size=batch_size,
        players=players,
    )
    prepare_batched_active_belief_triton_into(
        belief_workspace,
        prepared.beliefs,
        normalize=True,
        synchronize=False,
    )
    alias_workspace = make_batched_alias_tuple_reject_workspace(
        belief_workspace,
        sample_count=sample_count,
        tuple_tries=tuple_tries,
    )
    build_batched_alias_tables_triton_into(
        belief_workspace,
        alias_workspace,
        synchronize=False,
    )

    active_hands = int(fast_board.active_ids.numel())
    used_samples = torch.empty(
        batch_size,
        players,
        sample_count,
        dtype=torch.int64,
        device=device,
    )
    rank_samples = torch.empty(
        batch_size,
        players,
        players,
        sample_count,
        dtype=torch.int32,
        device=device,
    )
    alive_samples = torch.empty(
        batch_size,
        players,
        sample_count,
        dtype=torch.int8,
        device=device,
    )
    numerator_active = torch.empty(
        batch_size,
        players,
        active_hands,
        dtype=torch.float32,
        device=device,
    )
    denominator_active = torch.empty_like(numerator_active)
    alias_workspace.seed.fill_(seed)

    sample_grid = (batch_size, players, triton.cdiv(sample_count, sample_chunk))
    _sample_hero_opponents_alias_kernel[sample_grid](
        alias_workspace.alias_prob,
        alias_workspace.alias_idx,
        fast_board.hand_masks,
        fast_board.hand_ranks,
        used_samples,
        rank_samples,
        alive_samples,
        alias_workspace.seed,
        active_hands=active_hands,
        sample_count=sample_count,
        players=players,
        tuple_tries=tuple_tries,
        block_s=sample_chunk,
        num_warps=4,
    )

    score_grid = (batch_size, players, triton.cdiv(active_hands, hand_chunk))
    _score_hero_hands_from_samples_kernel[score_grid](
        fast_board.hand_masks,
        fast_board.hand_ranks,
        used_samples,
        rank_samples,
        alive_samples,
        numerator_active,
        denominator_active,
        active_hands=active_hands,
        sample_count=sample_count,
        players=players,
        block_h=hand_chunk,
        block_s=sample_chunk,
        num_warps=4,
    )

    numerator = torch.zeros(
        batch_size,
        players,
        NUM_HANDS,
        dtype=torch.float32,
        device=device,
    )
    denominator = torch.zeros_like(numerator)
    numerator[..., fast_board.active_ids] = numerator_active
    denominator[..., fast_board.active_ids] = denominator_active
    equity_by_hand = safe_divide_by_hand(numerator, denominator)
    aggregate = aggregate_all_players_from_num_denom(
        prepared.beliefs.to(torch.float32),
        numerator,
        denominator,
    )
    torch.cuda.synchronize(device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand,
        aggregate_equity=aggregate,
        denominator_by_hand=denominator,
        numerator_by_hand=numerator,
        seconds=time.perf_counter() - start,
    )


def tuple_reject_sis_by_hand(
    prepared: PreparedShowdown,
    *,
    sample_count: int,
    generator: torch.Generator,
    tuple_tries: int = 3,
    sample_chunk: int = 32,
    hand_chunk: int = 32,
) -> PerHandEquityResult:
    """Batched SIS-style per-hand equity estimator.

    For each board and hero player, this samples collision-free opponent tuples
    from their ranges using tuple rejection. Every accepted opponent tuple is
    then scored against streamed chunks of hero hands, producing a full
    `[B, players, NUM_HANDS]` conditional equity vector.
    """
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if tuple_tries <= 0:
        raise ValueError("tuple_tries must be positive")
    if sample_chunk <= 0 or hand_chunk <= 0:
        raise ValueError("sample_chunk and hand_chunk must be positive")

    start = time.perf_counter()
    device = prepared.beliefs.device
    beliefs = prepared.beliefs.to(torch.float32)
    batch_size, players, hand_count = beliefs.shape
    if hand_count != NUM_HANDS:
        raise ValueError(f"expected {NUM_HANDS} hand beliefs, got {hand_count}")
    if _can_use_fast_tuple_reject_sis(prepared, players):
        seed_device = getattr(generator, "device", torch.device("cpu"))
        seed = int(
            torch.randint(
                0,
                2_147_483_647,
                (),
                device=seed_device,
                generator=generator,
            ).item()
        )
        return _tuple_reject_sis_by_hand_triton(
            prepared,
            sample_count=sample_count,
            seed=seed,
            tuple_tries=tuple_tries,
            sample_chunk=sample_chunk,
            hand_chunk=hand_chunk,
        )

    masks = prepared.hand_masks.reshape(NUM_HANDS).to(device=device)
    ranks = prepared.hand_ranks.reshape(batch_size, NUM_HANDS)
    allowed = board_allowed_hands(prepared.board).to(device=device)

    numerator = torch.zeros(
        batch_size,
        players,
        NUM_HANDS,
        dtype=torch.float32,
        device=device,
    )
    denominator = torch.zeros_like(numerator)
    flat_beliefs = beliefs.reshape(batch_size * players, NUM_HANDS)

    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        opp_count = len(opponents)
        if opp_count == 0:
            denominator[:, hero] = allowed.to(torch.float32)
            numerator[:, hero] = allowed.to(torch.float32)
            continue

        done = 0
        while done < sample_count:
            chunk = min(sample_chunk, sample_count - done)
            candidates = torch.multinomial(
                flat_beliefs,
                chunk * tuple_tries,
                replacement=True,
                generator=generator,
            ).reshape(batch_size, players, tuple_tries, chunk)

            alive = torch.zeros(batch_size, chunk, dtype=torch.bool, device=device)
            selected = torch.zeros(
                batch_size,
                opp_count,
                chunk,
                dtype=torch.long,
                device=device,
            )
            for attempt in range(tuple_tries):
                opp_hands = candidates[:, opponents, attempt, :]
                opp_masks = masks.index_select(0, opp_hands.reshape(-1)).reshape(
                    batch_size,
                    opp_count,
                    chunk,
                )
                valid = torch.ones(batch_size, chunk, dtype=torch.bool, device=device)
                for left in range(opp_count):
                    for right in range(left + 1, opp_count):
                        valid &= (opp_masks[:, left] & opp_masks[:, right]) == 0
                take = (~alive) & valid
                selected = torch.where(take[:, None, :], opp_hands, selected)
                alive |= take

            selected_flat = selected.reshape(batch_size, opp_count * chunk)
            selected_masks = masks.index_select(0, selected_flat.reshape(-1)).reshape(
                batch_size,
                opp_count,
                chunk,
            )
            used_mask = selected_masks[:, 0].clone()
            for opp_slot in range(1, opp_count):
                used_mask |= selected_masks[:, opp_slot]

            selected_ranks = ranks.gather(1, selected_flat).reshape(
                batch_size,
                opp_count,
                chunk,
            )
            best_opp = selected_ranks.max(dim=1).values

            for hand_start in range(0, NUM_HANDS, hand_chunk):
                hand_end = min(hand_start + hand_chunk, NUM_HANDS)
                hero_masks = masks[hand_start:hand_end]
                hero_ranks = ranks[:, hand_start:hand_end]
                compatible = (
                    alive[:, None, :]
                    & allowed[:, hand_start:hand_end, None]
                    & ((used_mask[:, None, :] & hero_masks[None, :, None]) == 0)
                )
                hero_wins = hero_ranks[:, :, None] > best_opp[:, None, :]
                hero_ties = hero_ranks[:, :, None] == best_opp[:, None, :]
                tie_count = (
                    selected_ranks[:, None, :, :]
                    == hero_ranks[:, :, None, None]
                ).sum(dim=2, dtype=torch.float32) + 1.0
                share = torch.where(
                    hero_wins,
                    torch.ones_like(tie_count),
                    torch.where(hero_ties, 1.0 / tie_count, torch.zeros_like(tie_count)),
                )
                comp_f = compatible.to(torch.float32)
                denominator[:, hero, hand_start:hand_end] += comp_f.sum(dim=2)
                numerator[:, hero, hand_start:hand_end] += (comp_f * share).sum(dim=2)

            done += chunk

    denominator /= float(sample_count)
    numerator /= float(sample_count)
    equity_by_hand = safe_divide_by_hand(numerator, denominator)
    aggregate = aggregate_all_players_from_num_denom(beliefs, numerator, denominator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand,
        aggregate_equity=aggregate,
        denominator_by_hand=denominator,
        numerator_by_hand=numerator,
        seconds=time.perf_counter() - start,
    )

__all__ = [
    "BatchedAliasTupleRejectWorkspace",
    "PreparedBatchedFastSISBelief",
    "PreparedFastSISBoard",
    "PreparedShowdown",
    "alias_tuple_reject_aggregate",
    "alias_triton_tuple_reject_batched_fixed_into",
    "build_batched_alias_tables_triton_into",
    "conditional_tuple_reject_by_hand",
    "make_batched_active_belief_workspace",
    "make_batched_alias_tuple_reject_workspace",
    "prepare_batched_active_belief_triton_into",
    "prepare_fast_sis_board",
    "tuple_reject_sis_by_hand",
]
