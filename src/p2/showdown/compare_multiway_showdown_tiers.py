from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from p2.env.card_utils import NUM_HANDS, board_allowed_hands, hand_combos_tensor
from p2.env.rules import rank_hands

from .multiway_showdown_estimators import (
    PreparedShowdown,
    _exact_pattern_specs,
    _pair_lookup,
    _set_partitions_with_mobius,
    exact_nway_ie,
)
from .results import PerHandEquityResult, safe_divide_by_hand

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


@dataclass
class TierResult:
    equity: torch.Tensor
    seconds: float
    min_denom: float
    negative_denom_count: int


@dataclass
class _ActiveTierContext:
    beliefs: torch.Tensor
    ranks: torch.Tensor
    masks: torch.Tensor
    combos: torch.Tensor
    active_ids: torch.Tensor
    active_cards: torch.Tensor
    local_c0: torch.Tensor
    local_c1: torch.Tensor
    local_pair_ids: torch.Tensor
    allowed: torch.Tensor


def random_full_board(
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    scores = torch.rand(52, device=device, generator=generator)
    return torch.topk(scores, 5).indices.view(1, 5)


def random_beliefs(
    board: torch.Tensor,
    players: int,
    *,
    generator: torch.Generator,
    concentration: float,
) -> torch.Tensor:
    batch_size = board.shape[0]
    weights = torch.empty(
        batch_size,
        players,
        NUM_HANDS,
        dtype=torch.float32,
        device=board.device,
    ).exponential_(generator=generator)
    if concentration != 1.0:
        weights = weights.pow(1.0 / concentration)
    allowed = board_allowed_hands(board)
    weights.masked_fill_(~allowed[:, None, :], 0.0)
    return weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-30)


def prepare_random_showdown(
    *,
    players: int,
    device: torch.device,
    generator: torch.Generator,
    concentration: float,
) -> PreparedShowdown:
    start = time.perf_counter()
    board = random_full_board(device=device, generator=generator)
    beliefs = random_beliefs(
        board,
        players,
        generator=generator,
        concentration=concentration,
    )
    hand_ranks, _ = rank_hands(board)
    combos = hand_combos_tensor(device=device).long()
    hand_masks = ((1 << combos[:, 0]) | (1 << combos[:, 1])).long().contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return PreparedShowdown(
        board=board,
        beliefs=beliefs,
        hand_ranks=hand_ranks,
        combos=combos,
        hand_masks=hand_masks,
        setup_seconds=time.perf_counter() - start,
    )


def _independent_share_numerators(lower: torch.Tensor, tied: torch.Tensor) -> torch.Tensor:
    """Return unnormalized hero share under independent opponent factors.

    ``lower`` and ``tied`` have shape ``[..., H, O]`` for one hero player. The result
    is ``sum_S prod tied[S] prod lower[~S] / (|S| + 1)`` for all tie subsets.
    """
    opponents = lower.shape[-1]
    if opponents == 3:
        l0, l1, l2 = lower.unbind(dim=-1)
        t0, t1, t2 = tied.unbind(dim=-1)
        return (
            l0 * l1 * l2
            + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
            + (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2) / 3.0
            + 0.25 * t0 * t1 * t2
        )
    if opponents == 2:
        l0, l1 = lower.unbind(dim=-1)
        t0, t1 = tied.unbind(dim=-1)
        return l0 * l1 + 0.5 * (t0 * l1 + l0 * t1) + (t0 * t1) / 3.0
    out = torch.zeros(lower.shape[:-1], dtype=lower.dtype, device=lower.device)
    for subset in range(1 << opponents):
        term = torch.ones_like(out)
        ties = 0
        for opp_idx in range(opponents):
            if (subset >> opp_idx) & 1:
                term = term * tied[..., opp_idx]
                ties += 1
            else:
                term = term * lower[..., opp_idx]
        out = out + term / float(ties + 1)
    return out


def _aggregate_from_num_denom(
    hero_belief: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return (hero_belief * numerator).sum(dim=-1) / (
        hero_belief * denominator
    ).sum(dim=-1).clamp_min(1.0e-30)


def _tier_result_from_by_hand(result: PerHandEquityResult) -> TierResult:
    return TierResult(
        equity=result.aggregate_equity,
        seconds=result.seconds,
        min_denom=float(result.denominator_by_hand.min().item()),
        negative_denom_count=int((result.denominator_by_hand <= 0).sum().item()),
    )


def _zero_blocked_hands(
    prepared: PreparedShowdown,
    numerator_by_hand: torch.Tensor,
    denominator_by_hand: torch.Tensor,
    equity_by_hand: torch.Tensor,
) -> None:
    allowed = board_allowed_hands(prepared.board).to(numerator_by_hand.dtype)
    numerator_by_hand *= allowed[:, None, :]
    denominator_by_hand *= allowed[:, None, :]
    equity_by_hand *= allowed[:, None, :]


def _active_context(
    prepared: PreparedShowdown,
    *,
    dtype: torch.dtype,
) -> _ActiveTierContext:
    allowed = board_allowed_hands(prepared.board)
    batch_size = prepared.beliefs.shape[0]
    active_count = 1081
    active_ids = torch.topk(allowed.to(torch.int8), active_count, dim=1).indices
    combos = prepared.combos.long()
    active_combos = combos[active_ids]
    ranks = prepared.hand_ranks.reshape(batch_size, NUM_HANDS).gather(1, active_ids)
    hand_masks = prepared.hand_masks.reshape(NUM_HANDS)
    masks = hand_masks[active_ids]
    beliefs = prepared.beliefs.to(dtype).gather(
        2,
        active_ids[:, None, :].expand(-1, prepared.beliefs.shape[1], -1),
    )

    board_mask = torch.ones(batch_size, 52, dtype=torch.bool, device=prepared.beliefs.device)
    board_mask.scatter_(1, prepared.board.long(), False)
    active_cards = torch.topk(board_mask.to(torch.int8), 47, dim=1).indices
    card_to_local = torch.full(
        (batch_size, 52),
        -1,
        dtype=torch.long,
        device=prepared.beliefs.device,
    )
    card_to_local.scatter_(
        1,
        active_cards,
        torch.arange(47, device=prepared.beliefs.device)[None, :].expand(batch_size, -1),
    )
    local_c0 = card_to_local.gather(1, active_combos[..., 0])
    local_c1 = card_to_local.gather(1, active_combos[..., 1])
    pair_lookup = _pair_lookup(prepared.beliefs.device)
    local_pair_ids = pair_lookup[active_cards[:, :, None], active_cards[:, None, :]]
    return _ActiveTierContext(
        beliefs=beliefs,
        ranks=ranks,
        masks=masks,
        combos=active_combos,
        active_ids=active_ids,
        active_cards=active_cards,
        local_c0=local_c0,
        local_c1=local_c1,
        local_pair_ids=local_pair_ids,
        allowed=allowed,
    )


def _hand_relations(
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    disjoint = (ctx.masks[:, :, None] & ctx.masks[:, None, :]) == 0
    lower_rel = disjoint & (ctx.ranks[:, None, :] < ctx.ranks[:, :, None])
    tie_rel = disjoint & (ctx.ranks[:, None, :] == ctx.ranks[:, :, None])
    total_rel = disjoint
    return lower_rel.to(dtype), tie_rel.to(dtype), total_rel.to(dtype)


def _contains_matrix(
    combos: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    contains = torch.zeros(NUM_HANDS, 52, dtype=dtype, device=device)
    hand_ids = torch.arange(NUM_HANDS, device=device)
    contains[hand_ids, combos[:, 0]] = 1.0
    contains[hand_ids, combos[:, 1]] = 1.0
    return contains


def _active_contains_matrix(
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, active_count = ctx.active_ids.shape
    contains = torch.zeros(
        batch_size,
        active_count,
        47,
        dtype=dtype,
        device=ctx.beliefs.device,
    )
    boards = torch.arange(batch_size, device=ctx.beliefs.device)[:, None]
    hands = torch.arange(active_count, device=ctx.beliefs.device)[None, :]
    contains[boards, hands, ctx.local_c0] = 1.0
    contains[boards, hands, ctx.local_c1] = 1.0
    return contains


def _scatter_active_outputs(
    ctx: _ActiveTierContext,
    numerator_active: torch.Tensor,
    denominator_active: torch.Tensor,
    equity_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (*numerator_active.shape[:-1], NUM_HANDS)
    numerator = torch.zeros(shape, dtype=numerator_active.dtype, device=numerator_active.device)
    denominator = torch.zeros_like(numerator)
    equity = torch.zeros_like(numerator)
    index = ctx.active_ids[:, None, :].expand(-1, numerator_active.shape[1], -1)
    numerator.scatter_(2, index, numerator_active)
    denominator.scatter_(2, index, denominator_active)
    equity.scatter_(2, index, equity_active)
    return numerator, denominator, equity


def _active_local_combo_cards(ctx: _ActiveTierContext) -> tuple[torch.Tensor, torch.Tensor]:
    return ctx.local_c0, ctx.local_c1


def _build_local_mats_from_indices(
    weights: torch.Tensor,
    board_ids: torch.Tensor,
    hand_ids: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
) -> torch.Tensor:
    batch_size = weights.shape[0]
    lead_shape = weights.shape[1:-1]
    lead_count = int(torch.tensor(lead_shape).prod().item()) if lead_shape else 1
    flat_weights = weights.reshape(batch_size, lead_count, weights.shape[-1])
    mats = torch.zeros(
        batch_size,
        lead_count,
        47,
        47,
        dtype=weights.dtype,
        device=weights.device,
    )
    values = flat_weights[board_ids, :, hand_ids]
    lead_ids = torch.arange(lead_count, device=weights.device)[None, :]
    mats[board_ids[:, None], lead_ids, local_c0[:, None], local_c1[:, None]] = values
    mats[board_ids[:, None], lead_ids, local_c1[:, None], local_c0[:, None]] = values
    return mats.reshape(batch_size, *lead_shape, 47, 47)


def _masked_scalar_candidates(
    mats: torch.Tensor,
    blocked0: torch.Tensor,
    blocked1: torch.Tensor,
) -> torch.Tensor:
    sample_count = mats.shape[0]
    lead_shape = mats.shape[1:-2]
    lead_count = int(torch.tensor(lead_shape).prod().item()) if lead_shape else 1
    flat = mats.reshape(sample_count, lead_count, 47, 47)
    marginal = flat.sum(dim=-1)
    total = flat.triu(1).sum(dim=(-2, -1))
    gather0 = blocked0[:, None, None].expand(sample_count, lead_count, 1)
    gather1 = blocked1[:, None, None].expand(sample_count, lead_count, 1)
    row0 = marginal.gather(2, gather0).squeeze(2)
    row1 = marginal.gather(2, gather1).squeeze(2)
    sample_ids = torch.arange(sample_count, device=mats.device)[:, None]
    lead_ids = torch.arange(lead_count, device=mats.device)[None, :]
    edge = flat[sample_ids, lead_ids, blocked0[:, None], blocked1[:, None]]
    return (total - row0 - row1 + edge).reshape(sample_count, *lead_shape)


def _masked_card_candidates(
    mats: torch.Tensor,
    blocked0: torch.Tensor,
    blocked1: torch.Tensor,
) -> torch.Tensor:
    sample_count = mats.shape[0]
    lead_shape = mats.shape[1:-2]
    lead_count = int(torch.tensor(lead_shape).prod().item()) if lead_shape else 1
    flat = mats.reshape(sample_count, lead_count, 47, 47)
    marginal = flat.sum(dim=-1)
    idx0 = blocked0[:, None, None, None].expand(sample_count, lead_count, 47, 1)
    idx1 = blocked1[:, None, None, None].expand(sample_count, lead_count, 47, 1)
    col0 = flat.gather(3, idx0).squeeze(3)
    col1 = flat.gather(3, idx1).squeeze(3)
    out = marginal - col0 - col1
    scatter0 = blocked0[:, None, None].expand(sample_count, lead_count, 1)
    scatter1 = blocked1[:, None, None].expand(sample_count, lead_count, 1)
    out = out.scatter(2, scatter0, 0.0)
    out = out.scatter(2, scatter1, 0.0)
    return out.reshape(sample_count, *lead_shape, 47)


def _masked_scalar_all_hands(
    mats: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
) -> torch.Tensor:
    batch_size, active_count = local_c0.shape
    lead_shape = mats.shape[1:-2]
    lead_count = int(torch.tensor(lead_shape).prod().item()) if lead_shape else 1
    flat = mats.reshape(batch_size, lead_count, 47, 47)
    marginal = flat.sum(dim=-1)
    total = flat.triu(1).sum(dim=(-2, -1))
    idx0 = local_c0[:, None, :].expand(batch_size, lead_count, active_count)
    idx1 = local_c1[:, None, :].expand(batch_size, lead_count, active_count)
    row0 = marginal.gather(2, idx0)
    row1 = marginal.gather(2, idx1)
    edge = flat[
        torch.arange(batch_size, device=mats.device)[:, None, None],
        torch.arange(lead_count, device=mats.device)[None, :, None],
        local_c0[:, None, :],
        local_c1[:, None, :],
    ]
    return (total[:, :, None] - row0 - row1 + edge).reshape(
        batch_size,
        *lead_shape,
        active_count,
    )


def _masked_card_all_hands(
    mats: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
) -> torch.Tensor:
    batch_size, active_count = local_c0.shape
    lead_shape = mats.shape[1:-2]
    lead_count = int(torch.tensor(lead_shape).prod().item()) if lead_shape else 1
    flat = mats.reshape(batch_size, lead_count, 47, 47)
    marginal = flat.sum(dim=-1)
    idx0 = local_c0[:, None, None, :].expand(batch_size, lead_count, 47, active_count)
    idx1 = local_c1[:, None, None, :].expand(batch_size, lead_count, 47, active_count)
    col0 = flat.gather(3, idx0).permute(0, 1, 3, 2)
    col1 = flat.gather(3, idx1).permute(0, 1, 3, 2)
    out = marginal[:, :, None, :] - col0 - col1
    scatter0 = local_c0[:, None, :, None].expand(batch_size, lead_count, active_count, 1)
    scatter1 = local_c1[:, None, :, None].expand(batch_size, lead_count, active_count, 1)
    out = out.scatter(3, scatter0, 0.0)
    out = out.scatter(3, scatter1, 0.0)
    return out.reshape(batch_size, *lead_shape, active_count, 47)


def _assign_player_mode(
    scalar_all: torch.Tensor,
    card_all: torch.Tensor,
    mode: int,
    board_ids: torch.Tensor,
    hand_ids: torch.Tensor,
    mats: torch.Tensor,
    blocked0: torch.Tensor,
    blocked1: torch.Tensor,
) -> None:
    scalar = _masked_scalar_candidates(mats, blocked0, blocked1)
    card = _masked_card_candidates(mats, blocked0, blocked1)
    scalar_all[:, mode, board_ids, hand_ids] = scalar.T
    card_all[:, mode, board_ids, hand_ids] = card.permute(1, 0, 2)


def _assign_pair_mode(
    same_all: torch.Tensor,
    mode: int,
    board_ids: torch.Tensor,
    hand_ids: torch.Tensor,
    mats: torch.Tensor,
    blocked0: torch.Tensor,
    blocked1: torch.Tensor,
) -> None:
    scalar = _masked_scalar_candidates(mats, blocked0, blocked1)
    same_all[:, :, mode, board_ids, hand_ids] = scalar.permute(1, 2, 0)


def _prefix_gather_scalar(prefix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return prefix.gather(2, index[:, None, :].expand(-1, prefix.shape[1], -1))


def _prefix_gather_card(prefix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return prefix.gather(
        2,
        index[:, None, :, None].expand(-1, prefix.shape[1], -1, prefix.shape[3]),
    )


def _prefix_gather_pair_scalar(prefix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return prefix.gather(
        3,
        index[:, None, None, :].expand(-1, prefix.shape[1], prefix.shape[2], -1),
    )


def _prefix_gather_pair_card(prefix: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return prefix.gather(
        3,
        index[:, None, None, :, None].expand(
            -1,
            prefix.shape[1],
            prefix.shape[2],
            -1,
            prefix.shape[4],
        ),
    )


def _interval_scalar(prefix: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    return _prefix_gather_scalar(prefix, end) - _prefix_gather_scalar(prefix, start)


def _interval_card(prefix: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    return _prefix_gather_card(prefix, end) - _prefix_gather_card(prefix, start)


def _interval_pair_scalar(
    prefix: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    return _prefix_gather_pair_scalar(prefix, end) - _prefix_gather_pair_scalar(prefix, start)


def _interval_pair_card(
    prefix: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> torch.Tensor:
    return _prefix_gather_pair_card(prefix, end) - _prefix_gather_pair_card(prefix, start)


def _gather_full_by_pair_ids(values: torch.Tensor, pair_ids: torch.Tensor) -> torch.Tensor:
    safe_ids = pair_ids.clamp_min(0).reshape(values.shape[0], -1)
    gathered = values.gather(2, safe_ids[:, None, :].expand(-1, values.shape[1], -1))
    return gathered.reshape(values.shape[0], values.shape[1], *pair_ids.shape[1:])


if triton is not None:

    @triton.jit
    def _tier2_prefix_scalar_card_kernel(
        scalar_prefix,
        card_prefix,
        beliefs,
        full_beliefs,
        hand_ranks,
        local_c0,
        local_c1,
        pair_p_ids,
        pair_q_ids,
        ranks,
        lower_end,
        tie_end,
        scalar_out,
        card_out,
        B: tl.constexpr,
        P: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        FULL_H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        pmode = tl.program_id(1)
        h = row % H
        b = row // H
        mode = pmode % 3
        player = pmode // 3

        lower = tl.load(lower_end + b * H + h)
        tie = tl.load(tie_end + b * H + h)
        start = tl.where(mode == 1, lower, 0)
        end = tl.where(mode == 0, lower, tl.where(mode == 1, tie, H))

        c0 = tl.load(local_c0 + b * H + h)
        c1 = tl.load(local_c1 + b * H + h)
        rank = tl.load(ranks + b * H + h)

        scalar_base = (b * P + player) * H1
        scalar = tl.load(scalar_prefix + scalar_base + end) - tl.load(
            scalar_prefix + scalar_base + start
        )
        card_base = ((b * P + player) * H1) * CARD_COUNT
        hero0 = tl.load(card_prefix + (card_base + end * CARD_COUNT + c0)) - tl.load(
            card_prefix + (card_base + start * CARD_COUNT + c0)
        )
        hero1 = tl.load(card_prefix + (card_base + end * CARD_COUNT + c1)) - tl.load(
            card_prefix + (card_base + start * CARD_COUNT + c1)
        )
        edge = tl.load(beliefs + (b * P + player) * H + h)
        edge = tl.where(mode == 0, 0.0, edge)
        scalar = scalar - hero0 - hero1 + edge
        tl.store(scalar_out + ((player * 3 + mode) * B + b) * H + h, scalar)

        card = tl.arange(0, BLOCK_C)
        valid_card = card < CARD_COUNT
        interval = tl.load(
            card_prefix + card_base + end * CARD_COUNT + card,
            mask=valid_card,
            other=0.0,
        ) - tl.load(
            card_prefix + card_base + start * CARD_COUNT + card,
            mask=valid_card,
            other=0.0,
        )

        pair_base = (b * H + h) * CARD_COUNT + card
        pair_p = tl.load(pair_p_ids + pair_base, mask=valid_card, other=-1)
        pair_q = tl.load(pair_q_ids + pair_base, mask=valid_card, other=-1)
        valid_p = valid_card & (pair_p >= 0)
        valid_q = valid_card & (pair_q >= 0)
        rank_p = tl.load(hand_ranks + b * FULL_H + pair_p, mask=valid_p, other=-1)
        rank_q = tl.load(hand_ranks + b * FULL_H + pair_q, mask=valid_q, other=-1)
        row_p = valid_p & tl.where(mode == 0, rank_p < rank, tl.where(mode == 1, rank_p == rank, True))
        row_q = valid_q & tl.where(mode == 0, rank_q < rank, tl.where(mode == 1, rank_q == rank, True))
        weight_base = (b * P + player) * FULL_H
        corr = tl.load(full_beliefs + weight_base + pair_p, mask=row_p, other=0.0)
        corr += tl.load(full_beliefs + weight_base + pair_q, mask=row_q, other=0.0)
        value = interval - corr
        value = tl.where((card == c0) | (card == c1), 0.0, value)
        card_out_base = (((player * 3 + mode) * B + b) * H + h) * CARD_COUNT
        tl.store(card_out + card_out_base + card, value, mask=valid_card)

    @triton.jit
    def _tier2_prefix_same_kernel(
        pair_prefix,
        pair_card_prefix,
        beliefs,
        local_c0,
        local_c1,
        lower_end,
        tie_end,
        same_out,
        B: tl.constexpr,
        P: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        CARD_COUNT: tl.constexpr,
    ):
        row = tl.program_id(0)
        pair_mode = tl.program_id(1)
        h = row % H
        b = row // H
        mode = pair_mode % 3
        pair = pair_mode // 3
        right = pair % P
        left = pair // P

        lower = tl.load(lower_end + b * H + h)
        tie = tl.load(tie_end + b * H + h)
        start = tl.where(mode == 1, lower, 0)
        end = tl.where(mode == 0, lower, tl.where(mode == 1, tie, H))

        c0 = tl.load(local_c0 + b * H + h)
        c1 = tl.load(local_c1 + b * H + h)
        pair_prefix_base = ((b * P + left) * P + right) * H1
        scalar = tl.load(pair_prefix + pair_prefix_base + end) - tl.load(
            pair_prefix + pair_prefix_base + start
        )
        pair_card_base = (((b * P + left) * P + right) * H1) * CARD_COUNT
        card0 = tl.load(pair_card_prefix + pair_card_base + end * CARD_COUNT + c0) - tl.load(
            pair_card_prefix + pair_card_base + start * CARD_COUNT + c0
        )
        card1 = tl.load(pair_card_prefix + pair_card_base + end * CARD_COUNT + c1) - tl.load(
            pair_card_prefix + pair_card_base + start * CARD_COUNT + c1
        )
        edge = tl.load(beliefs + (b * P + left) * H + h) * tl.load(
            beliefs + (b * P + right) * H + h
        )
        edge = tl.where(mode == 0, 0.0, edge)
        value = scalar - card0 - card1 + edge
        tl.store(same_out + (((left * P + right) * 3 + mode) * B + b) * H + h, value)

    @triton.jit
    def _tier3_wedge_p4_kernel(
        beliefs,
        ranks,
        local_c0,
        local_c1,
        card_all,
        wedge_num_out,
        wedge_den_out,
        B: tl.constexpr,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ):
        b = tl.program_id(0)
        hero = tl.program_id(1)
        tile = tl.program_id(2)
        h_block = tile // K_BLOCKS
        k_block = tile - h_block * K_BLOCKS
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        hk_mask = (h[:, None] < H) & (k[None, :] < H)
        h_mask = h < H
        hero_c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=-1)
        hero_c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=-1)
        k_c0 = tl.load(local_c0 + b * H + k, mask=k < H, other=-2)
        k_c1 = tl.load(local_c1 + b * H + k, mask=k < H, other=-3)
        rank_h = tl.load(ranks + b * H + h, mask=h_mask, other=0)
        rank_k = tl.load(ranks + b * H + k, mask=k < H, other=-1)
        disjoint = (
            (k_c0[None, :] != hero_c0[:, None])
            & (k_c0[None, :] != hero_c1[:, None])
            & (k_c1[None, :] != hero_c0[:, None])
            & (k_c1[None, :] != hero_c1[:, None])
            & hk_mask
        )
        lower = disjoint & (rank_k[None, :] < rank_h[:, None])
        tied = disjoint & (rank_k[None, :] == rank_h[:, None])

        den_total = tl.zeros((BLOCK_H,), dtype=tl.float32)
        num_total = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for center_local in tl.static_range(0, 3):
            center = tl.where(center_local < hero, center_local, center_local + 1)
            side_a_local = tl.where(center_local == 0, 1, 0)
            side_b_local = tl.where(center_local == 2, 1, 2)
            side_a = tl.where(side_a_local < hero, side_a_local, side_a_local + 1)
            side_b = tl.where(side_b_local < hero, side_b_local, side_b_local + 1)

            center_belief = tl.load(
                beliefs + (b * 4 + center) * H + k,
                mask=k < H,
                other=0.0,
            )
            w0 = tl.where(lower, center_belief[None, :], 0.0)
            w1 = tl.where(tied, center_belief[None, :], 0.0)
            w2 = tl.where(disjoint, center_belief[None, :], 0.0)

            side_a_belief = tl.load(
                beliefs + (b * 4 + side_a) * H + k,
                mask=k < H,
                other=0.0,
            )
            side_b_belief = tl.load(
                beliefs + (b * 4 + side_b) * H + k,
                mask=k < H,
                other=0.0,
            )
            side_a_w0 = tl.where(lower, side_a_belief[None, :], 0.0)
            side_a_w1 = tl.where(tied, side_a_belief[None, :], 0.0)
            side_a_w2 = tl.where(disjoint, side_a_belief[None, :], 0.0)
            side_b_w0 = tl.where(lower, side_b_belief[None, :], 0.0)
            side_b_w1 = tl.where(tied, side_b_belief[None, :], 0.0)
            side_b_w2 = tl.where(disjoint, side_b_belief[None, :], 0.0)

            ca0_base = (((side_a * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
            ca1_base = ca0_base + B * H * CARD_COUNT
            ca2_base = ca0_base + 2 * B * H * CARD_COUNT
            cb0_base = (((side_b * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
            cb1_base = cb0_base + B * H * CARD_COUNT
            cb2_base = cb0_base + 2 * B * H * CARD_COUNT

            a0 = tl.load(card_all + ca0_base + k_c0[None, :], mask=hk_mask, other=0.0)
            a0 += tl.load(card_all + ca0_base + k_c1[None, :], mask=hk_mask, other=0.0)
            a0 -= side_a_w0
            a1 = tl.load(card_all + ca1_base + k_c0[None, :], mask=hk_mask, other=0.0)
            a1 += tl.load(card_all + ca1_base + k_c1[None, :], mask=hk_mask, other=0.0)
            a1 -= side_a_w1
            a2 = tl.load(card_all + ca2_base + k_c0[None, :], mask=hk_mask, other=0.0)
            a2 += tl.load(card_all + ca2_base + k_c1[None, :], mask=hk_mask, other=0.0)
            a2 -= side_a_w2
            c0 = tl.load(card_all + cb0_base + k_c0[None, :], mask=hk_mask, other=0.0)
            c0 += tl.load(card_all + cb0_base + k_c1[None, :], mask=hk_mask, other=0.0)
            c0 -= side_b_w0
            c1 = tl.load(card_all + cb1_base + k_c0[None, :], mask=hk_mask, other=0.0)
            c1 += tl.load(card_all + cb1_base + k_c1[None, :], mask=hk_mask, other=0.0)
            c1 -= side_b_w1
            c2 = tl.load(card_all + cb2_base + k_c0[None, :], mask=hk_mask, other=0.0)
            c2 += tl.load(card_all + cb2_base + k_c1[None, :], mask=hk_mask, other=0.0)
            c2 -= side_b_w2

            ac00 = a0 * c0
            ac10 = a1 * c0
            ac01 = a0 * c1
            ac11 = a1 * c1
            combo0 = ac00 + 0.5 * (ac10 + ac01) + (1.0 / 3.0) * ac11
            combo1 = 0.5 * ac00 + (1.0 / 3.0) * (ac10 + ac01) + 0.25 * ac11
            den_total += tl.sum(w2 * a2 * c2, axis=1)
            num_total += tl.sum(w0 * combo0 + w1 * combo1, axis=1)

        out_base = ((b * 4 + hero) * K_BLOCKS + k_block) * H + h
        tl.store(wedge_den_out + out_base, den_total, mask=h_mask)
        tl.store(wedge_num_out + out_base, num_total, mask=h_mask)


def _tier2_prefix_factors_triton(
    *,
    scalar_prefix: torch.Tensor,
    card_prefix: torch.Tensor,
    pair_prefix: torch.Tensor,
    pair_card_prefix: torch.Tensor,
    beliefs: torch.Tensor,
    full_beliefs: torch.Tensor,
    hand_ranks: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    pair_p_ids: torch.Tensor,
    pair_q_ids: torch.Tensor,
    ranks: torch.Tensor,
    lower_end: torch.Tensor,
    tie_end: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, players, active_count = beliefs.shape
    device = beliefs.device
    scalar_all = torch.empty(players, 3, batch_size, active_count, dtype=dtype, device=device)
    card_all = torch.empty(
        players,
        3,
        batch_size,
        active_count,
        47,
        dtype=dtype,
        device=device,
    )
    same_all = torch.empty(
        players,
        players,
        3,
        batch_size,
        active_count,
        dtype=dtype,
        device=device,
    )
    grid_scalar = (batch_size * active_count, players * 3)
    _tier2_prefix_scalar_card_kernel[grid_scalar](
        scalar_prefix,
        card_prefix,
        beliefs,
        full_beliefs,
        hand_ranks,
        local_c0,
        local_c1,
        pair_p_ids,
        pair_q_ids,
        ranks,
        lower_end,
        tie_end,
        scalar_all,
        card_all,
        B=batch_size,
        P=players,
        H=active_count,
        H1=active_count + 1,
        FULL_H=NUM_HANDS,
        CARD_COUNT=47,
        BLOCK_C=64,
        num_warps=2,
    )
    grid_same = (batch_size * active_count, players * players * 3)
    _tier2_prefix_same_kernel[grid_same](
        pair_prefix,
        pair_card_prefix,
        beliefs,
        local_c0,
        local_c1,
        lower_end,
        tie_end,
        same_all,
        B=batch_size,
        P=players,
        H=active_count,
        H1=active_count + 1,
        CARD_COUNT=47,
        num_warps=1,
    )
    return scalar_all, card_all, same_all


def _tier3_wedge_p4_triton(
    beliefs: torch.Tensor,
    ranks: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    card_all: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or beliefs.device.type != "cuda" or beliefs.shape[1] != 4:
        return None
    batch_size, _, active_count = beliefs.shape
    block_h = 4
    block_k = 32
    k_blocks = triton.cdiv(active_count, block_k)
    partial_num = torch.empty(
        batch_size,
        4,
        k_blocks,
        active_count,
        dtype=beliefs.dtype,
        device=beliefs.device,
    )
    partial_den = torch.empty_like(partial_num)
    h_blocks = triton.cdiv(active_count, block_h)
    grid = (batch_size, 4, h_blocks * k_blocks)
    _tier3_wedge_p4_kernel[grid](
        beliefs.contiguous(),
        ranks.contiguous(),
        local_c0.contiguous(),
        local_c1.contiguous(),
        card_all.contiguous(),
        partial_num,
        partial_den,
        B=batch_size,
        H=active_count,
        CARD_COUNT=47,
        K_BLOCKS=k_blocks,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return partial_num.sum(dim=2), partial_den.sum(dim=2)


def _tier2_prefix_factors(
    prepared: PreparedShowdown,
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beliefs = ctx.beliefs
    batch_size, players, active_count = beliefs.shape
    device = beliefs.device
    local_c0, local_c1 = _active_local_combo_cards(ctx)
    order = torch.argsort(ctx.ranks, dim=1)
    sorted_ranks = ctx.ranks.gather(1, order)
    sorted_beliefs = beliefs.gather(2, order[:, None, :].expand(-1, players, -1))
    active_contains = _active_contains_matrix(ctx, dtype=dtype)
    sorted_contains = active_contains.gather(1, order[:, :, None].expand(-1, -1, 47))

    zero_scalar = torch.zeros(batch_size, players, 1, dtype=dtype, device=device)
    scalar_prefix = torch.cat([zero_scalar, sorted_beliefs.cumsum(dim=2)], dim=2)
    card_values = sorted_beliefs[:, :, :, None] * sorted_contains[:, None, :, :]
    zero_card = torch.zeros(batch_size, players, 1, 47, dtype=dtype, device=device)
    card_prefix = torch.cat([zero_card, card_values.cumsum(dim=2)], dim=2)

    pair_values = sorted_beliefs[:, :, None, :] * sorted_beliefs[:, None, :, :]
    zero_pair = torch.zeros(batch_size, players, players, 1, dtype=dtype, device=device)
    pair_prefix = torch.cat([zero_pair, pair_values.cumsum(dim=3)], dim=3)
    pair_card_values = pair_values[:, :, :, :, None] * sorted_contains[:, None, None, :, :]
    zero_pair_card = torch.zeros(
        batch_size,
        players,
        players,
        1,
        47,
        dtype=dtype,
        device=device,
    )
    pair_card_prefix = torch.cat([zero_pair_card, pair_card_values.cumsum(dim=3)], dim=3)

    lower_end = torch.searchsorted(sorted_ranks, ctx.ranks, right=False)
    tie_end = torch.searchsorted(sorted_ranks, ctx.ranks, right=True)
    pair_ids = ctx.local_pair_ids
    full_beliefs = prepared.beliefs.to(dtype).contiguous()
    pair_p_ids = pair_ids.gather(2, local_c0[:, None, :].expand(-1, 47, -1)).permute(0, 2, 1)
    pair_q_ids = pair_ids.gather(2, local_c1[:, None, :].expand(-1, 47, -1)).permute(0, 2, 1)
    if triton is not None and device.type == "cuda":
        return _tier2_prefix_factors_triton(
            scalar_prefix=scalar_prefix.contiguous(),
            card_prefix=card_prefix.contiguous(),
            pair_prefix=pair_prefix.contiguous(),
            pair_card_prefix=pair_card_prefix.contiguous(),
            beliefs=beliefs.contiguous(),
            full_beliefs=full_beliefs,
            hand_ranks=prepared.hand_ranks.reshape(batch_size, NUM_HANDS).contiguous(),
            local_c0=local_c0.contiguous(),
            local_c1=local_c1.contiguous(),
            pair_p_ids=pair_p_ids.contiguous(),
            pair_q_ids=pair_q_ids.contiguous(),
            ranks=ctx.ranks.contiguous(),
            lower_end=lower_end.contiguous(),
            tie_end=tie_end.contiguous(),
            dtype=dtype,
        )

    zero_index = torch.zeros_like(lower_end)
    total_end = torch.full_like(lower_end, active_count)
    starts = [zero_index, lower_end, zero_index]
    ends = [lower_end, tie_end, total_end]

    scalar_all = torch.empty(players, 3, batch_size, active_count, dtype=dtype, device=device)
    card_all = torch.empty(players, 3, batch_size, active_count, 47, dtype=dtype, device=device)
    same_all = torch.empty(
        players,
        players,
        3,
        batch_size,
        active_count,
        dtype=dtype,
        device=device,
    )

    valid_p = pair_p_ids >= 0
    valid_q = pair_q_ids >= 0
    pair_p_rank = prepared.hand_ranks.reshape(batch_size, NUM_HANDS).gather(
        1,
        pair_p_ids.clamp_min(0).reshape(batch_size, -1),
    ).reshape(batch_size, active_count, 47)
    pair_q_rank = prepared.hand_ranks.reshape(batch_size, NUM_HANDS).gather(
        1,
        pair_q_ids.clamp_min(0).reshape(batch_size, -1),
    ).reshape(batch_size, active_count, 47)
    pair_p_weight = _gather_full_by_pair_ids(full_beliefs, pair_p_ids)
    pair_q_weight = _gather_full_by_pair_ids(full_beliefs, pair_q_ids)

    for mode, (start, end) in enumerate(zip(starts, ends, strict=True)):
        interval_scalar = _interval_scalar(scalar_prefix, start, end)
        interval_card = _interval_card(card_prefix, start, end)
        hero_card0 = interval_card.gather(
            3,
            local_c0[:, None, :, None].expand(-1, players, -1, 1),
        ).squeeze(3)
        hero_card1 = interval_card.gather(
            3,
            local_c1[:, None, :, None].expand(-1, players, -1, 1),
        ).squeeze(3)
        if mode == 0:
            edge = torch.zeros_like(interval_scalar)
            row_mask_p = valid_p & (pair_p_rank < ctx.ranks[:, :, None])
            row_mask_q = valid_q & (pair_q_rank < ctx.ranks[:, :, None])
        elif mode == 1:
            edge = beliefs
            row_mask_p = valid_p & (pair_p_rank == ctx.ranks[:, :, None])
            row_mask_q = valid_q & (pair_q_rank == ctx.ranks[:, :, None])
        else:
            edge = beliefs
            row_mask_p = valid_p
            row_mask_q = valid_q
        scalar_all[:, mode] = (interval_scalar - hero_card0 - hero_card1 + edge).permute(
            1,
            0,
            2,
        )

        row_corr = (
            pair_p_weight * row_mask_p[:, None].to(dtype)
            + pair_q_weight * row_mask_q[:, None].to(dtype)
        )
        card_mode = interval_card - row_corr
        card_mode = card_mode.scatter(
            3,
            local_c0[:, None, :, None].expand(-1, players, -1, 1),
            0.0,
        )
        card_mode = card_mode.scatter(
            3,
            local_c1[:, None, :, None].expand(-1, players, -1, 1),
            0.0,
        )
        card_all[:, mode] = card_mode.permute(1, 0, 2, 3)

        interval_pair_scalar = _interval_pair_scalar(pair_prefix, start, end)
        interval_pair_card = _interval_pair_card(pair_card_prefix, start, end)
        pair_card0 = interval_pair_card.gather(
            4,
            local_c0[:, None, None, :, None].expand(-1, players, players, -1, 1),
        ).squeeze(4)
        pair_card1 = interval_pair_card.gather(
            4,
            local_c1[:, None, None, :, None].expand(-1, players, players, -1, 1),
        ).squeeze(4)
        if mode == 0:
            pair_edge = torch.zeros_like(interval_pair_scalar)
        else:
            pair_edge = beliefs[:, :, None] * beliefs[:, None, :]
        same_all[:, :, mode] = (
            interval_pair_scalar - pair_card0 - pair_card1 + pair_edge
        ).permute(1, 2, 0, 3)

    return scalar_all, card_all, same_all


def tier1_hero_removal_by_hand(prepared: PreparedShowdown) -> PerHandEquityResult:
    start = time.perf_counter()
    dtype = torch.float32
    ctx = _active_context(prepared, dtype=dtype)
    beliefs = ctx.beliefs
    players = beliefs.shape[1]
    lower_rel, tie_rel, total_rel = _hand_relations(ctx, dtype=dtype)
    rel_f = [lower_rel, tie_rel, total_rel]

    equities: list[torch.Tensor] = []
    numerator_active = torch.zeros(
        beliefs.shape[0],
        players,
        ctx.active_ids.shape[1],
        dtype=dtype,
        device=beliefs.device,
    )
    denominator_active = torch.zeros_like(numerator_active)
    equity_active = torch.zeros_like(numerator_active)
    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        opp_beliefs = beliefs[:, opponents].transpose(1, 2).contiguous()
        lower = rel_f[0].matmul(opp_beliefs)
        tied = rel_f[1].matmul(opp_beliefs)
        denom_terms = rel_f[2].matmul(opp_beliefs)
        denominator = denom_terms.prod(dim=-1)
        numerator = _independent_share_numerators(lower, tied)
        equities.append(_aggregate_from_num_denom(beliefs[:, hero], numerator, denominator))
        numerator_active[:, hero] = numerator
        denominator_active[:, hero] = denominator
        equity_active[:, hero] = safe_divide_by_hand(numerator, denominator)

    numerator_by_hand, denominator_by_hand, equity_by_hand = _scatter_active_outputs(
        ctx,
        numerator_active,
        denominator_active,
        equity_active,
    )
    result = torch.stack(equities, dim=1).to(torch.float32)
    if prepared.beliefs.device.type == "cuda":
        torch.cuda.synchronize(prepared.beliefs.device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand.to(torch.float32),
        aggregate_equity=result,
        denominator_by_hand=denominator_by_hand,
        numerator_by_hand=numerator_by_hand,
        seconds=time.perf_counter() - start,
    )


def tier1_hero_removal(prepared: PreparedShowdown) -> TierResult:
    return _tier_result_from_by_hand(tier1_hero_removal_by_hand(prepared))


def tier2_first_order_opp_collision_by_hand(
    prepared: PreparedShowdown,
) -> PerHandEquityResult:
    start = time.perf_counter()
    dtype = torch.float32
    ctx = _active_context(prepared, dtype=dtype)
    beliefs = ctx.beliefs
    players = beliefs.shape[1]
    device = beliefs.device

    active_count = ctx.active_ids.shape[1]
    scalar_all, card_all, same_all = _tier2_prefix_factors(prepared, ctx, dtype=dtype)
    pair_event_all = torch.einsum("pmbhc,qnbhc->pqmnbh", card_all, card_all)
    for mode in range(3):
        pair_event_all[:, :, mode, mode] -= same_all[:, :, mode]

    equities: list[torch.Tensor] = []
    numerator_active = torch.zeros(
        beliefs.shape[0],
        players,
        active_count,
        dtype=dtype,
        device=device,
    )
    denominator_active = torch.zeros_like(numerator_active)
    equity_active = torch.zeros_like(numerator_active)
    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        opp_count = len(opponents)
        scalar = scalar_all[opponents]
        pair_event = pair_event_all[opponents][:, opponents]

        if opp_count == 3:
            l0, l1, l2 = scalar[0, 0], scalar[1, 0], scalar[2, 0]
            t0, t1, t2 = scalar[0, 1], scalar[1, 1], scalar[2, 1]
            numerator = (
                l0 * l1 * l2
                + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
                + (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2) / 3.0
                + 0.25 * t0 * t1 * t2
            )
            denominator = scalar[0, 2] * scalar[1, 2] * scalar[2, 2]
        else:
            lower = scalar[:, 0].permute(1, 2, 0).contiguous()
            tied = scalar[:, 1].permute(1, 2, 0).contiguous()
            denom_terms = scalar[:, 2].permute(1, 2, 0).contiguous()
            numerator = _independent_share_numerators(lower, tied)
            denominator = denom_terms.prod(dim=-1)

        for left in range(opp_count):
            for right in range(left + 1, opp_count):
                other = [idx for idx in range(opp_count) if idx not in (left, right)]
                if opp_count == 3:
                    other_idx = other[0]
                    denominator = (
                        denominator - pair_event[left, right, 2, 2] * scalar[other_idx, 2]
                    )
                    pair_num = _tier2_pair_num_three_opponents(
                        pair_event[left, right],
                        scalar[other_idx],
                    )
                else:
                    pair_total = pair_event[left, right, 2, 2]
                    denom_other = torch.ones(
                        beliefs.shape[0],
                        active_count,
                        dtype=dtype,
                        device=device,
                    )
                    for opp_idx in other:
                        denom_other = denom_other * scalar[opp_idx, 2]
                    denominator = denominator - pair_total * denom_other

                    pair_num = torch.zeros(
                        beliefs.shape[0],
                        active_count,
                        dtype=dtype,
                        device=device,
                    )
                    for subset in range(1 << opp_count):
                        modes = []
                        ties = 0
                        for opp_idx in range(opp_count):
                            mode = 1 if (subset >> opp_idx) & 1 else 0
                            modes.append(mode)
                            ties += mode
                        left_mode = modes[left]
                        right_mode = modes[right]
                        pair_factor = pair_event[left, right, left_mode, right_mode]
                        other_factor = torch.ones(
                            beliefs.shape[0],
                            active_count,
                            dtype=dtype,
                            device=device,
                        )
                        for opp_idx in other:
                            other_factor = other_factor * scalar[opp_idx, modes[opp_idx]]
                        pair_num = pair_num + pair_factor * other_factor / float(ties + 1)
                numerator = numerator - pair_num

        equities.append(_aggregate_from_num_denom(beliefs[:, hero], numerator, denominator))
        numerator_active[:, hero] = numerator
        denominator_active[:, hero] = denominator
        equity_active[:, hero] = safe_divide_by_hand(numerator, denominator)

    numerator_by_hand, denominator_by_hand, equity_by_hand = _scatter_active_outputs(
        ctx,
        numerator_active,
        denominator_active,
        equity_active,
    )
    result = torch.stack(equities, dim=1).to(torch.float32)
    if prepared.beliefs.device.type == "cuda":
        torch.cuda.synchronize(prepared.beliefs.device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand.to(torch.float32),
        aggregate_equity=result,
        denominator_by_hand=denominator_by_hand,
        numerator_by_hand=numerator_by_hand,
        seconds=time.perf_counter() - start,
    )


def _tier2_pair_num_three_opponents(
    pair_event: torch.Tensor,
    other_scalar: torch.Tensor,
) -> torch.Tensor:
    pair00 = pair_event[0, 0]
    pair10_plus_01 = pair_event[1, 0] + pair_event[0, 1]
    pair11 = pair_event[1, 1]
    other0 = pair00 + 0.5 * pair10_plus_01 + pair11 / 3.0
    other1 = 0.5 * pair00 + pair10_plus_01 / 3.0 + 0.25 * pair11
    return other0 * other_scalar[0] + other1 * other_scalar[1]


def tier2_first_order_opp_collision(prepared: PreparedShowdown) -> TierResult:
    return _tier_result_from_by_hand(tier2_first_order_opp_collision_by_hand(prepared))


def _slice_prepared_boards(prepared: PreparedShowdown, start: int, end: int) -> PreparedShowdown:
    return PreparedShowdown(
        board=prepared.board[start:end],
        beliefs=prepared.beliefs[start:end],
        hand_ranks=prepared.hand_ranks[start:end],
        combos=prepared.combos,
        hand_masks=prepared.hand_masks,
        setup_seconds=prepared.setup_seconds,
    )


def tier3_second_order_opp_collision_by_hand(
    prepared: PreparedShowdown,
) -> PerHandEquityResult:
    chunk_size = 128
    batch_size = prepared.beliefs.shape[0]
    has_streaming_wedge = (
        triton is not None
        and prepared.beliefs.shape[1] == 4
        and prepared.beliefs.device.type == "cuda"
        and batch_size > chunk_size
    )
    if prepared.beliefs.device.type == "cuda" and batch_size > chunk_size and not has_streaming_wedge:
        start = time.perf_counter()
        chunks = [
            _tier3_second_order_opp_collision_by_hand_impl(
                _slice_prepared_boards(prepared, begin, min(begin + chunk_size, batch_size)),
            )
            for begin in range(0, batch_size, chunk_size)
        ]
        numerator_chunks = [chunk.numerator_by_hand for chunk in chunks]
        numerator = (
            torch.cat(numerator_chunks, dim=0)
            if all(chunk is not None for chunk in numerator_chunks)
            else None
        )
        if prepared.beliefs.device.type == "cuda":
            torch.cuda.synchronize(prepared.beliefs.device)
        return PerHandEquityResult(
            equity_by_hand=torch.cat([chunk.equity_by_hand for chunk in chunks], dim=0),
            aggregate_equity=torch.cat([chunk.aggregate_equity for chunk in chunks], dim=0),
            denominator_by_hand=torch.cat(
                [chunk.denominator_by_hand for chunk in chunks],
                dim=0,
            ),
            numerator_by_hand=numerator,
            seconds=time.perf_counter() - start,
        )
    return _tier3_second_order_opp_collision_by_hand_impl(prepared)


def _tier3_second_order_opp_collision_by_hand_impl(
    prepared: PreparedShowdown,
) -> PerHandEquityResult:
    start = time.perf_counter()
    dtype = torch.float32
    ctx = _active_context(prepared, dtype=dtype)
    beliefs = ctx.beliefs
    players = beliefs.shape[1]
    device = beliefs.device
    active_count = ctx.active_ids.shape[1]

    scalar_all, card_all, same_all = _tier2_prefix_factors(prepared, ctx, dtype=dtype)
    pair_event_all = torch.einsum("pmbhc,qnbhc->pqmnbh", card_all, card_all)
    for mode in range(3):
        pair_event_all[:, :, mode, mode] -= same_all[:, :, mode]

    local_c0, local_c1 = _active_local_combo_cards(ctx)
    use_streaming_wedge = (
        triton is not None
        and players == 4
        and device.type == "cuda"
        and beliefs.shape[0] > 128
    )
    wedge_p4 = (
        _tier3_wedge_p4_triton(beliefs, ctx.ranks, local_c0, local_c1, card_all)
        if use_streaming_wedge
        else None
    )
    weighted_rel_all = None
    conflict_response_all = None
    if wedge_p4 is None:
        relations = list(_hand_relations(ctx, dtype=dtype))
        weighted_rel_all = torch.empty(
            players,
            3,
            beliefs.shape[0],
            active_count,
            active_count,
            dtype=dtype,
            device=device,
        )
        for player in range(players):
            belief = beliefs[:, player]
            for mode, rel in enumerate(relations):
                rel_weight = rel * belief[:, None, :]
                weighted_rel_all[player, mode] = rel_weight
        card_idx0 = local_c0[:, None, :].expand(beliefs.shape[0], active_count, active_count)
        card_idx1 = local_c1[:, None, :].expand(beliefs.shape[0], active_count, active_count)
        card_idx0 = card_idx0[None, None].expand(players, 3, -1, -1, -1)
        card_idx1 = card_idx1[None, None].expand(players, 3, -1, -1, -1)
        conflict_response_all = card_all.gather(4, card_idx0) + card_all.gather(4, card_idx1)
        conflict_response_all = conflict_response_all - weighted_rel_all
    tie_weight_3 = torch.empty(2, 2, 2, dtype=dtype, device=device)
    for mode0 in range(2):
        for mode1 in range(2):
            for mode2 in range(2):
                tie_weight_3[mode0, mode1, mode2] = 1.0 / float(mode0 + mode1 + mode2 + 1)

    equities: list[torch.Tensor] = []
    numerator_active = torch.zeros(
        beliefs.shape[0],
        players,
        active_count,
        dtype=dtype,
        device=device,
    )
    denominator_active = torch.zeros_like(numerator_active)
    equity_active = torch.zeros_like(numerator_active)
    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        opp_count = len(opponents)
        edges = [
            (left, right)
            for left in range(opp_count)
            for right in range(left + 1, opp_count)
        ]
        scalar = scalar_all[opponents]
        pair_event = pair_event_all[opponents][:, opponents]
        if wedge_p4 is None:
            weighted_rel = weighted_rel_all[opponents]
            conflict_response = conflict_response_all[opponents]

        if opp_count == 3:
            l0, l1, l2 = scalar[0, 0], scalar[1, 0], scalar[2, 0]
            t0, t1, t2 = scalar[0, 1], scalar[1, 1], scalar[2, 1]
            numerator = (
                l0 * l1 * l2
                + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
                + (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2) / 3.0
                + 0.25 * t0 * t1 * t2
            )
            denominator = scalar[0, 2] * scalar[1, 2] * scalar[2, 2]
        else:
            lower = scalar[:, 0].permute(1, 2, 0).contiguous()
            tied = scalar[:, 1].permute(1, 2, 0).contiguous()
            denom_terms = scalar[:, 2].permute(1, 2, 0).contiguous()
            numerator = _independent_share_numerators(lower, tied)
            denominator = denom_terms.prod(dim=-1)

        for left, right in edges:
            other = [idx for idx in range(opp_count) if idx not in (left, right)]
            if opp_count == 3:
                other_idx = other[0]
                denominator = denominator - pair_event[left, right, 2, 2] * scalar[other_idx, 2]
                pair_num = _tier2_pair_num_three_opponents(
                    pair_event[left, right],
                    scalar[other_idx],
                )
            else:
                denom_other = torch.ones(beliefs.shape[0], active_count, dtype=dtype, device=device)
                for opp_idx in other:
                    denom_other = denom_other * scalar[opp_idx, 2]
                denominator = denominator - pair_event[left, right, 2, 2] * denom_other

                pair_num = torch.zeros(beliefs.shape[0], active_count, dtype=dtype, device=device)
                for subset in range(1 << opp_count):
                    modes = []
                    ties = 0
                    for opp_idx in range(opp_count):
                        mode = 1 if (subset >> opp_idx) & 1 else 0
                        modes.append(mode)
                        ties += mode
                    other_factor = torch.ones(
                        beliefs.shape[0],
                        active_count,
                        dtype=dtype,
                        device=device,
                    )
                    for opp_idx in other:
                        other_factor = other_factor * scalar[opp_idx, modes[opp_idx]]
                    pair_num = pair_num + (
                        pair_event[left, right, modes[left], modes[right]]
                        * other_factor
                        / float(ties + 1)
                    )
            numerator = numerator - pair_num

        if wedge_p4 is not None and opp_count == 3:
            wedge_num_all, wedge_den_all = wedge_p4
            numerator = numerator + wedge_num_all[:, hero]
            denominator = denominator + wedge_den_all[:, hero]
            equities.append(_aggregate_from_num_denom(beliefs[:, hero], numerator, denominator))
            numerator_active[:, hero] = numerator
            denominator_active[:, hero] = denominator
            equity_active[:, hero] = safe_divide_by_hand(numerator, denominator)
            continue

        for edge_idx, first in enumerate(edges):
            for second in edges[edge_idx + 1 :]:
                nodes = set(first) | set(second)
                if len(nodes) == 4:
                    left_a, left_b = first
                    right_a, right_b = second
                    denominator = denominator + (
                        pair_event[left_a, left_b, 2, 2]
                        * pair_event[right_a, right_b, 2, 2]
                    )

                    pair_pair_num = torch.zeros(
                        beliefs.shape[0],
                        active_count,
                        dtype=dtype,
                        device=device,
                    )
                    for subset in range(1 << opp_count):
                        modes = []
                        ties = 0
                        for opp_idx in range(opp_count):
                            mode = 1 if (subset >> opp_idx) & 1 else 0
                            modes.append(mode)
                            ties += mode
                        pair_pair_num = pair_pair_num + (
                            pair_event[left_a, left_b, modes[left_a], modes[left_b]]
                            * pair_event[
                                right_a,
                                right_b,
                                modes[right_a],
                                modes[right_b],
                            ]
                            / float(ties + 1)
                        )
                    numerator = numerator + pair_pair_num
                    continue

                center = next(node for node in nodes if node in first and node in second)
                sides = [node for node in nodes if node != center]
                isolated = [node for node in range(opp_count) if node not in nodes]
                side_a, side_b = sides

                wedge = (
                    weighted_rel[center, 2]
                    * conflict_response[side_a, 2]
                    * conflict_response[side_b, 2]
                ).sum(dim=-1)
                denom_other = torch.ones(beliefs.shape[0], active_count, dtype=dtype, device=device)
                for opp_idx in isolated:
                    denom_other = denom_other * scalar[opp_idx, 2]
                denominator = denominator + wedge * denom_other

                if opp_count == 3:
                    wedge_modes = torch.einsum(
                        "xbhk,ybhk,zbhk->xyzbh",
                        weighted_rel[center, :2],
                        conflict_response[side_a, :2],
                        conflict_response[side_b, :2],
                    )
                    numerator = numerator + (
                        wedge_modes * tie_weight_3[:, :, :, None, None]
                    ).sum(dim=(0, 1, 2))
                    continue

                wedge_num = torch.zeros(beliefs.shape[0], active_count, dtype=dtype, device=device)
                for subset in range(1 << opp_count):
                    modes = []
                    ties = 0
                    for opp_idx in range(opp_count):
                        mode = 1 if (subset >> opp_idx) & 1 else 0
                        modes.append(mode)
                        ties += mode
                    factor = (
                        weighted_rel[center, modes[center]]
                        * conflict_response[side_a, modes[side_a]]
                        * conflict_response[side_b, modes[side_b]]
                    ).sum(dim=-1)
                    for opp_idx in isolated:
                        factor = factor * scalar[opp_idx, modes[opp_idx]]
                    wedge_num = wedge_num + factor / float(ties + 1)
                numerator = numerator + wedge_num

        equities.append(_aggregate_from_num_denom(beliefs[:, hero], numerator, denominator))
        numerator_active[:, hero] = numerator
        denominator_active[:, hero] = denominator
        equity_active[:, hero] = safe_divide_by_hand(numerator, denominator)

    numerator_by_hand, denominator_by_hand, equity_by_hand = _scatter_active_outputs(
        ctx,
        numerator_active,
        denominator_active,
        equity_active,
    )
    result = torch.stack(equities, dim=1).to(torch.float32)
    if prepared.beliefs.device.type == "cuda":
        torch.cuda.synchronize(prepared.beliefs.device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand.to(torch.float32),
        aggregate_equity=result,
        denominator_by_hand=denominator_by_hand,
        numerator_by_hand=numerator_by_hand,
        seconds=time.perf_counter() - start,
    )


def tier3_second_order_opp_collision(prepared: PreparedShowdown) -> TierResult:
    return _tier_result_from_by_hand(tier3_second_order_opp_collision_by_hand(prepared))


def _factor_sum_all_distinct_batched(
    factors: list[tuple[tuple[int, ...], torch.Tensor]],
    variable_count: int,
    multiplicity: int,
    *,
    hand_count: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    letters = "cdef"
    batch_size = factors[0][1].shape[0]
    total = torch.zeros(batch_size, hand_count, dtype=dtype, device=device)
    for partition, coefficient in _set_partitions_with_mobius(variable_count):
        variable_to_block = [0] * variable_count
        for block_idx, block in enumerate(partition):
            for variable in block:
                variable_to_block[variable] = block_idx

        scalar = torch.ones(batch_size, hand_count, dtype=dtype, device=device)
        operands: list[torch.Tensor] = []
        subscripts: list[str] = []
        for scope, tensor in factors:
            if not scope:
                scalar = scalar * tensor
                continue
            collapsed = tuple(variable_to_block[variable] for variable in scope)
            if len(scope) == 1:
                operands.append(tensor)
                subscripts.append("bh" + letters[collapsed[0]])
            elif collapsed[0] == collapsed[1]:
                operands.append(tensor.diagonal(dim1=2, dim2=3))
                subscripts.append("bh" + letters[collapsed[0]])
            else:
                operands.append(tensor)
                subscripts.append("bh" + letters[collapsed[0]] + letters[collapsed[1]])

        if operands:
            equation = ",".join(["bh", *subscripts]) + "->bh"
            value = torch.einsum(equation, scalar, *operands)
        else:
            value = scalar
        total += float(coefficient) * value

    return total / float(multiplicity)


def _pattern_factor_sum_batched(
    forced_by_opp: list[list[int]],
    modes: list[int],
    scalar: torch.Tensor,
    card: torch.Tensor,
    pair: torch.Tensor,
    *,
    hand_count: int,
    variable_count: int,
    multiplicity: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    factors: list[tuple[tuple[int, ...], torch.Tensor]] = []
    for opp_idx, mode in enumerate(modes):
        forced_vars = forced_by_opp[opp_idx]
        if not forced_vars:
            factors.append(((), scalar[opp_idx, mode]))
        elif len(forced_vars) == 1:
            factors.append(((forced_vars[0],), card[opp_idx, mode]))
        else:
            factors.append(((forced_vars[0], forced_vars[1]), pair[opp_idx, mode]))
    return _factor_sum_all_distinct_batched(
        factors,
        variable_count,
        multiplicity,
        hand_count=hand_count,
        dtype=dtype,
        device=device,
    )


def tier4_third_degree_card_collision_by_hand(
    prepared: PreparedShowdown,
) -> PerHandEquityResult:
    """Exact IE truncated after three shared-card collision variables."""
    start = time.perf_counter()
    dtype = torch.float32
    ctx = _active_context(prepared, dtype=dtype)
    beliefs = ctx.beliefs
    ranks = ctx.ranks
    players = beliefs.shape[1]
    device = beliefs.device
    active_count = ctx.active_ids.shape[1]

    active_contains = _active_contains_matrix(ctx, dtype=dtype)
    pair_ids = ctx.local_pair_ids
    pair_valid = pair_ids >= 0
    safe_pair_ids = pair_ids.clamp_min(0)

    relations = list(_hand_relations(ctx, dtype=dtype))
    pair_masks = prepared.hand_masks.reshape(NUM_HANDS)[safe_pair_ids]
    pair_disjoint = ((ctx.masks[:, :, None, None] & pair_masks[:, None, :, :]) == 0) & pair_valid[
        :,
        None,
        :,
        :,
    ]
    pair_ranks = prepared.hand_ranks.reshape(prepared.beliefs.shape[0], NUM_HANDS).gather(
        1,
        safe_pair_ids.reshape(prepared.beliefs.shape[0], -1),
    ).reshape(prepared.beliefs.shape[0], 47, 47)

    scalar_all = torch.empty(players, 3, beliefs.shape[0], active_count, dtype=dtype, device=device)
    card_all = torch.empty(players, 3, beliefs.shape[0], active_count, 47, dtype=dtype, device=device)
    pair_all = torch.empty(
        players,
        3,
        beliefs.shape[0],
        active_count,
        47,
        47,
        dtype=dtype,
        device=device,
    )
    for player in range(players):
        belief = beliefs[:, player]
        weighted_contains = belief[:, :, None] * active_contains
        full_belief = prepared.beliefs[:, player].to(dtype)
        pair_mass = full_belief.gather(
            1,
            safe_pair_ids.reshape(prepared.beliefs.shape[0], -1),
        ).reshape(prepared.beliefs.shape[0], 47, 47)
        pair_mass = pair_mass * pair_valid.to(dtype)
        for mode, relation in enumerate(relations):
            scalar_all[player, mode] = relation.matmul(belief.unsqueeze(-1)).squeeze(-1)
            card_all[player, mode] = relation.matmul(weighted_contains)
            if mode == 0:
                pair_rel = pair_disjoint & (pair_ranks[:, None, :, :] < ranks[:, :, None, None])
            elif mode == 1:
                pair_rel = pair_disjoint & (pair_ranks[:, None, :, :] == ranks[:, :, None, None])
            else:
                pair_rel = pair_disjoint
            pair_all[player, mode] = pair_rel.to(dtype) * pair_mass[:, None, :, :]

    equities: list[torch.Tensor] = []
    numerator_active = torch.zeros(
        beliefs.shape[0],
        players,
        active_count,
        dtype=dtype,
        device=device,
    )
    denominator_active = torch.zeros_like(numerator_active)
    equity_active = torch.zeros_like(numerator_active)
    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        opp_count = len(opponents)
        scalar = scalar_all[opponents]
        card = card_all[opponents]
        pair = pair_all[opponents]

        numerator = torch.zeros(beliefs.shape[0], active_count, dtype=dtype, device=device)
        denominator = torch.ones(beliefs.shape[0], active_count, dtype=dtype, device=device)
        for opp_idx in range(opp_count):
            denominator = denominator * scalar[opp_idx, 2]

        for subset in range(1 << opp_count):
            prod = torch.ones(beliefs.shape[0], active_count, dtype=dtype, device=device)
            ties = 0
            for opp_idx in range(opp_count):
                mode = 1 if (subset >> opp_idx) & 1 else 0
                ties += mode
                prod = prod * scalar[opp_idx, mode]
            numerator = numerator + prod / float(ties + 1)

        for pattern_spec in _exact_pattern_specs(opp_count):
            if pattern_spec.variable_count > 3:
                continue
            forced_by_opp = [list(forced) for forced in pattern_spec.forced_by_opp]
            variable_count = pattern_spec.variable_count
            multiplicity = pattern_spec.multiplicity
            sign = pattern_spec.sign
            denominator = denominator + float(sign) * _pattern_factor_sum_batched(
                forced_by_opp,
                [2] * opp_count,
                scalar,
                card,
                pair,
                hand_count=active_count,
                variable_count=variable_count,
                multiplicity=multiplicity,
                dtype=dtype,
                device=device,
            )

            pattern_num = torch.zeros(beliefs.shape[0], active_count, dtype=dtype, device=device)
            for subset in range(1 << opp_count):
                modes = []
                ties = 0
                for opp_idx in range(opp_count):
                    mode = 1 if (subset >> opp_idx) & 1 else 0
                    modes.append(mode)
                    ties += mode
                pattern_num = pattern_num + _pattern_factor_sum_batched(
                    forced_by_opp,
                    modes,
                    scalar,
                    card,
                    pair,
                    hand_count=active_count,
                    variable_count=variable_count,
                    multiplicity=multiplicity,
                    dtype=dtype,
                    device=device,
                ) / float(ties + 1)
            numerator = numerator + float(sign) * pattern_num

        equities.append(_aggregate_from_num_denom(beliefs[:, hero], numerator, denominator))
        numerator_active[:, hero] = numerator
        denominator_active[:, hero] = denominator
        equity_active[:, hero] = safe_divide_by_hand(numerator, denominator)

    numerator_by_hand, denominator_by_hand, equity_by_hand = _scatter_active_outputs(
        ctx,
        numerator_active,
        denominator_active,
        equity_active,
    )
    result = torch.stack(equities, dim=1).to(torch.float32)
    if prepared.beliefs.device.type == "cuda":
        torch.cuda.synchronize(prepared.beliefs.device)
    return PerHandEquityResult(
        equity_by_hand=equity_by_hand.to(torch.float32),
        aggregate_equity=result,
        denominator_by_hand=denominator_by_hand,
        numerator_by_hand=numerator_by_hand,
        seconds=time.perf_counter() - start,
    )


def tier4_third_degree_card_collision(prepared: PreparedShowdown) -> TierResult:
    return _tier_result_from_by_hand(tier4_third_degree_card_collision_by_hand(prepared))


def _fmt_vector(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in values.reshape(-1).tolist()) + "]"


def _summary(errors: torch.Tensor) -> str:
    return (
        f"mean_abs={errors.abs().mean().item():.6g} "
        f"max_abs={errors.abs().max().item():.6g} "
        f"rmse={errors.square().mean().sqrt().item():.6g}"
    )


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    tier1_rows = []
    tier2_rows = []
    tier3_rows = []
    tier4_rows = []
    exact_rows = []
    tier1_seconds = 0.0
    tier2_seconds = 0.0
    tier3_seconds = 0.0
    tier4_seconds = 0.0
    exact_seconds = 0.0
    tier2_negative = 0
    tier3_negative = 0
    tier4_negative = 0
    tier2_min_denom = float("inf")
    tier3_min_denom = float("inf")
    tier4_min_denom = float("inf")

    for board_idx in range(args.boards):
        prepared = prepare_random_showdown(
            players=args.players,
            device=device,
            generator=generator,
            concentration=args.concentration,
        )
        exact = exact_nway_ie(prepared, pattern_chunk=args.pattern_chunk)
        tier1 = tier1_hero_removal(prepared)
        tier2 = tier2_first_order_opp_collision(prepared)
        tier3 = tier3_second_order_opp_collision(prepared)
        tier4 = tier4_third_degree_card_collision(prepared)
        exact_rows.append(exact.equity.detach().cpu())
        tier1_rows.append(tier1.equity.detach().cpu())
        tier2_rows.append(tier2.equity.detach().cpu())
        tier3_rows.append(tier3.equity.detach().cpu())
        tier4_rows.append(tier4.equity.detach().cpu())
        exact_seconds += exact.seconds
        tier1_seconds += tier1.seconds
        tier2_seconds += tier2.seconds
        tier3_seconds += tier3.seconds
        tier4_seconds += tier4.seconds
        tier2_negative += tier2.negative_denom_count
        tier3_negative += tier3.negative_denom_count
        tier4_negative += tier4.negative_denom_count
        tier2_min_denom = min(tier2_min_denom, tier2.min_denom)
        tier3_min_denom = min(tier3_min_denom, tier3.min_denom)
        tier4_min_denom = min(tier4_min_denom, tier4.min_denom)

        print(
            f"board={board_idx} cards={prepared.board.reshape(-1).detach().cpu().tolist()} "
            f"exact={_fmt_vector(exact.equity.cpu())} "
            f"tier1={_fmt_vector(tier1.equity.cpu())} "
            f"tier2={_fmt_vector(tier2.equity.cpu())} "
            f"tier3={_fmt_vector(tier3.equity.cpu())} "
            f"tier4={_fmt_vector(tier4.equity.cpu())}",
            flush=True,
        )
        print(
            f"  tier1_err={_fmt_vector((tier1.equity - exact.equity).cpu())} "
            f"tier2_err={_fmt_vector((tier2.equity - exact.equity).cpu())} "
            f"tier3_err={_fmt_vector((tier3.equity - exact.equity).cpu())} "
            f"tier4_err={_fmt_vector((tier4.equity - exact.equity).cpu())}",
            flush=True,
        )

    exact_all = torch.cat(exact_rows, dim=0)
    tier1_all = torch.cat(tier1_rows, dim=0)
    tier2_all = torch.cat(tier2_rows, dim=0)
    tier3_all = torch.cat(tier3_rows, dim=0)
    tier4_all = torch.cat(tier4_rows, dim=0)
    tier1_err = tier1_all - exact_all
    tier2_err = tier2_all - exact_all
    tier3_err = tier3_all - exact_all
    tier4_err = tier4_all - exact_all
    tier1_shape = tier1_all / tier1_all.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
    tier2_shape = tier2_all / tier2_all.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
    tier3_shape = tier3_all / tier3_all.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
    tier4_shape = tier4_all / tier4_all.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
    tier1_shape_err = tier1_shape - exact_all
    tier2_shape_err = tier2_shape - exact_all
    tier3_shape_err = tier3_shape - exact_all
    tier4_shape_err = tier4_shape - exact_all
    print("\nsummary")
    print(f"players={args.players} boards={args.boards} concentration={args.concentration}")
    print(f"exact_sum_mean={exact_all.sum(dim=1).mean().item():.6f}")
    print(f"tier1_sum_mean={tier1_all.sum(dim=1).mean().item():.6f}")
    print(f"tier2_sum_mean={tier2_all.sum(dim=1).mean().item():.6f}")
    print(f"tier3_sum_mean={tier3_all.sum(dim=1).mean().item():.6f}")
    print(f"tier4_sum_mean={tier4_all.sum(dim=1).mean().item():.6f}")
    print(f"tier1 {_summary(tier1_err)}")
    print(f"tier2 {_summary(tier2_err)}")
    print(f"tier3 {_summary(tier3_err)}")
    print(f"tier4 {_summary(tier4_err)}")
    print(f"tier1_sum_normalized {_summary(tier1_shape_err)}")
    print(f"tier2_sum_normalized {_summary(tier2_shape_err)}")
    print(f"tier3_sum_normalized {_summary(tier3_shape_err)}")
    print(f"tier4_sum_normalized {_summary(tier4_shape_err)}")
    print(
        "seconds "
        f"exact={exact_seconds:.3f} tier1={tier1_seconds:.3f} "
        f"tier2={tier2_seconds:.3f} tier3={tier3_seconds:.3f} "
        f"tier4={tier4_seconds:.3f}"
    )
    print(f"tier2_min_denom={tier2_min_denom:.6e} tier2_negative_denoms={tier2_negative}")
    print(f"tier3_min_denom={tier3_min_denom:.6e} tier3_negative_denoms={tier3_negative}")
    print(f"tier4_min_denom={tier4_min_denom:.6e} tier4_negative_denoms={tier4_negative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=5)
    parser.add_argument("--boards", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pattern-chunk", type=int, default=64)
    parser.add_argument("--concentration", type=float, default=1.0)
    args = parser.parse_args()
    if args.players != 5:
        raise ValueError("This comparison is intended for exact 5-way runs.")
    run(args)


if __name__ == "__main__":
    main()
