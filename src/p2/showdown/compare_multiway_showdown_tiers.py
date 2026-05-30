from __future__ import annotations

import argparse
import os
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
class _TierBoardContext:
    ranks: torch.Tensor
    masks: torch.Tensor
    combos: torch.Tensor
    active_ids: torch.Tensor
    active_cards: torch.Tensor
    local_c0: torch.Tensor
    local_c1: torch.Tensor
    local_pair_ids: torch.Tensor
    pair_p_ids: torch.Tensor
    pair_q_ids: torch.Tensor
    pair_p_rank_flags: torch.Tensor
    pair_q_rank_flags: torch.Tensor
    pair_rank_flags: torch.Tensor
    allowed: torch.Tensor
    order: torch.Tensor
    sorted_ranks: torch.Tensor
    sorted_group_id: torch.Tensor
    sorted_c0: torch.Tensor
    sorted_c1: torch.Tensor
    sorted_card_positions: torch.Tensor
    slot_lower_by_card: torch.Tensor
    slot_tie_by_card: torch.Tensor
    slot_lower_tie_by_card: torch.Tensor
    sorted_contains: torch.Tensor
    lower_end: torch.Tensor
    tie_end: torch.Tensor
    lower_group_end: torch.Tensor
    tie_group_end: torch.Tensor
    rank_group_count: torch.Tensor
    max_rank_groups: int


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
    pair_p_ids: torch.Tensor
    pair_q_ids: torch.Tensor
    pair_p_rank_flags: torch.Tensor
    pair_q_rank_flags: torch.Tensor
    pair_rank_flags: torch.Tensor
    allowed: torch.Tensor
    order: torch.Tensor
    sorted_ranks: torch.Tensor
    sorted_group_id: torch.Tensor
    sorted_c0: torch.Tensor
    sorted_c1: torch.Tensor
    sorted_card_positions: torch.Tensor
    slot_lower_by_card: torch.Tensor
    slot_tie_by_card: torch.Tensor
    slot_lower_tie_by_card: torch.Tensor
    sorted_contains: torch.Tensor
    lower_end: torch.Tensor
    tie_end: torch.Tensor
    lower_group_end: torch.Tensor
    tie_group_end: torch.Tensor
    rank_group_count: torch.Tensor
    max_rank_groups: int


_P4_PLAYER_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_P4_PLAYER_PAIR_INDEX = {pair: idx for idx, pair in enumerate(_P4_PLAYER_PAIRS)}
_TIER_LOCAL_CARDS = 47
_TIER_CARD_SLOT_CAP = 64
_TIER_HANDS_PER_LOCAL_CARD = 46


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


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


def _aggregate_all_active_from_num_denom(
    beliefs: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return (
        (beliefs * numerator).sum(dim=-1)
        / (beliefs * denominator).sum(dim=-1).clamp_min(1.0e-30)
    ).to(torch.float32)


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


def _build_tier_board_context(prepared: PreparedShowdown) -> _TierBoardContext:
    allowed = board_allowed_hands(prepared.board)
    batch_size = prepared.beliefs.shape[0]
    active_count = 1081
    active_ids = torch.topk(allowed.to(torch.int8), active_count, dim=1).indices.to(torch.int32)
    combos = prepared.combos.to(torch.int32)
    active_combos = combos[active_ids]
    ranks = prepared.hand_ranks.reshape(batch_size, NUM_HANDS).gather(1, active_ids)
    hand_masks = prepared.hand_masks.reshape(NUM_HANDS)
    masks = hand_masks[active_ids]

    board_mask = torch.ones(batch_size, 52, dtype=torch.bool, device=prepared.beliefs.device)
    board_mask.scatter_(1, prepared.board.long(), False)
    active_cards = torch.topk(board_mask.to(torch.int8), 47, dim=1).indices.to(torch.int32)
    card_to_local = torch.full(
        (batch_size, 52),
        -1,
        dtype=torch.int32,
        device=prepared.beliefs.device,
    )
    card_to_local.scatter_(
        1,
        active_cards,
        torch.arange(
            47,
            device=prepared.beliefs.device,
            dtype=torch.int32,
        )[None, :].expand(batch_size, -1),
    )
    local_c0 = card_to_local.gather(1, active_combos[..., 0])
    local_c1 = card_to_local.gather(1, active_combos[..., 1])
    pair_lookup = _pair_lookup(prepared.beliefs.device)
    local_pair_ids = pair_lookup[active_cards[:, :, None], active_cards[:, None, :]].to(torch.int32)
    pair_p_ids = local_pair_ids.gather(
        2,
        local_c0[:, None, :].expand(-1, 47, -1),
    ).permute(0, 2, 1).contiguous()
    pair_q_ids = local_pair_ids.gather(
        2,
        local_c1[:, None, :].expand(-1, 47, -1),
    ).permute(0, 2, 1).contiguous()
    full_ranks = prepared.hand_ranks.reshape(batch_size, NUM_HANDS)
    pair_p_rank = full_ranks.gather(
        1,
        pair_p_ids.clamp_min(0).long().reshape(batch_size, -1),
    ).reshape(batch_size, active_count, 47)
    pair_q_rank = full_ranks.gather(
        1,
        pair_q_ids.clamp_min(0).long().reshape(batch_size, -1),
    ).reshape(batch_size, active_count, 47)
    valid_p = pair_p_ids >= 0
    valid_q = pair_q_ids >= 0
    rank_h = ranks[:, :, None]
    pair_p_rank_flags = (
        ((valid_p & (pair_p_rank < rank_h)).to(torch.uint8))
        | ((valid_p & (pair_p_rank == rank_h)).to(torch.uint8) << 1)
    ).contiguous()
    pair_q_rank_flags = (
        ((valid_q & (pair_q_rank < rank_h)).to(torch.uint8))
        | ((valid_q & (pair_q_rank == rank_h)).to(torch.uint8) << 1)
    ).contiguous()
    pair_rank_flags = (
        pair_p_rank_flags.to(torch.int16) | (pair_q_rank_flags.to(torch.int16) << 2)
    ).to(torch.uint8).contiguous()
    order = torch.argsort(ranks, dim=1).to(torch.int32)
    sorted_ranks = ranks.gather(1, order)
    is_group_start = torch.ones(
        batch_size,
        active_count,
        dtype=torch.bool,
        device=prepared.beliefs.device,
    )
    is_group_start[:, 1:] = sorted_ranks[:, 1:] != sorted_ranks[:, :-1]
    sorted_group_id = is_group_start.cumsum(dim=1, dtype=torch.int32) - 1
    rank_group_id = torch.empty_like(sorted_group_id)
    rank_group_id.scatter_(1, order.long(), sorted_group_id)
    rank_group_count = (sorted_group_id[:, -1] + 1).to(torch.int32).contiguous()
    max_rank_groups = int(rank_group_count.max().item())
    sorted_c0 = local_c0.gather(1, order)
    sorted_c1 = local_c1.gather(1, order)
    sorted_contains = torch.zeros(
        batch_size,
        active_count,
        47,
        dtype=torch.float32,
        device=prepared.beliefs.device,
    )
    board_ids = torch.arange(
        batch_size,
        device=prepared.beliefs.device,
        dtype=torch.int32,
    )[:, None]
    hand_ids = torch.arange(
        active_count,
        device=prepared.beliefs.device,
        dtype=torch.int32,
    )[None, :]
    sorted_contains[board_ids, hand_ids, sorted_c0] = 1.0
    sorted_contains[board_ids, hand_ids, sorted_c1] = 1.0
    lower_end = torch.searchsorted(sorted_ranks, ranks, right=False).to(torch.int32)
    tie_end = torch.searchsorted(sorted_ranks, ranks, right=True).to(torch.int32)
    lower_group_end = rank_group_id.to(torch.int32).contiguous()
    tie_group_end = (rank_group_id + 1).to(torch.int32).contiguous()
    local_card = torch.arange(
        _TIER_LOCAL_CARDS,
        device=prepared.beliefs.device,
        dtype=torch.int32,
    )
    sorted_has_card = (sorted_c0[:, :, None] == local_card) | (
        sorted_c1[:, :, None] == local_card
    )
    sorted_position = torch.arange(
        active_count,
        device=prepared.beliefs.device,
        dtype=torch.int32,
    )[None, :, None]
    sorted_card_positions = torch.where(
        sorted_has_card,
        sorted_position,
        torch.full((), active_count, device=prepared.beliefs.device, dtype=torch.int32),
    ).transpose(1, 2)
    sorted_card_positions = (
        sorted_card_positions.sort(dim=2).values[:, :, :_TIER_CARD_SLOT_CAP].contiguous()
    )
    pack_positions = os.environ.setdefault("P2_SHOWDOWN_TIER2_PACK_POSITIONS", "1")
    pack_min_batch = _env_int("P2_SHOWDOWN_TIER2_PACK_POSITIONS_MIN_BATCH", 1024)
    should_pack_positions = (
        pack_positions == "1"
        or (pack_positions == "auto" and batch_size >= pack_min_batch)
    )
    if should_pack_positions:
        sorted_card_positions = sorted_card_positions.to(torch.int16).contiguous()
    slot_lower_by_card, slot_tie_by_card = _tier2_card_slots_from_positions(
        sorted_card_positions,
        lower_end,
        tie_end,
    )
    slot_lower_tie_by_card = (
        slot_lower_by_card.to(torch.int16) | (slot_tie_by_card.to(torch.int16) << 6)
    ).contiguous()
    return _TierBoardContext(
        ranks=ranks,
        masks=masks,
        combos=active_combos,
        active_ids=active_ids,
        active_cards=active_cards,
        local_c0=local_c0,
        local_c1=local_c1,
        local_pair_ids=local_pair_ids,
        pair_p_ids=pair_p_ids,
        pair_q_ids=pair_q_ids,
        pair_p_rank_flags=pair_p_rank_flags,
        pair_q_rank_flags=pair_q_rank_flags,
        pair_rank_flags=pair_rank_flags,
        allowed=allowed,
        order=order,
        sorted_ranks=sorted_ranks,
        sorted_group_id=sorted_group_id.contiguous(),
        sorted_c0=sorted_c0,
        sorted_c1=sorted_c1,
        sorted_card_positions=sorted_card_positions,
        slot_lower_by_card=slot_lower_by_card,
        slot_tie_by_card=slot_tie_by_card,
        slot_lower_tie_by_card=slot_lower_tie_by_card,
        sorted_contains=sorted_contains,
        lower_end=lower_end,
        tie_end=tie_end,
        lower_group_end=lower_group_end,
        tie_group_end=tie_group_end,
        rank_group_count=rank_group_count,
        max_rank_groups=max_rank_groups,
    )


def _tier_board_context(prepared: PreparedShowdown) -> _TierBoardContext:
    cached = getattr(prepared, "_p2_tier_board_context", None)
    if cached is None:
        cached = _build_tier_board_context(prepared)
        setattr(prepared, "_p2_tier_board_context", cached)
    return cached


def _active_context(
    prepared: PreparedShowdown,
    *,
    dtype: torch.dtype,
) -> _ActiveTierContext:
    board_ctx = _tier_board_context(prepared)
    beliefs = prepared.beliefs.to(dtype).gather(
        2,
        board_ctx.active_ids[:, None, :].expand(-1, prepared.beliefs.shape[1], -1),
    )
    return _ActiveTierContext(
        beliefs=beliefs,
        ranks=board_ctx.ranks,
        masks=board_ctx.masks,
        combos=board_ctx.combos,
        active_ids=board_ctx.active_ids,
        active_cards=board_ctx.active_cards,
        local_c0=board_ctx.local_c0,
        local_c1=board_ctx.local_c1,
        local_pair_ids=board_ctx.local_pair_ids,
        pair_p_ids=board_ctx.pair_p_ids,
        pair_q_ids=board_ctx.pair_q_ids,
        pair_p_rank_flags=board_ctx.pair_p_rank_flags,
        pair_q_rank_flags=board_ctx.pair_q_rank_flags,
        pair_rank_flags=board_ctx.pair_rank_flags,
        allowed=board_ctx.allowed,
        order=board_ctx.order,
        sorted_ranks=board_ctx.sorted_ranks,
        sorted_group_id=board_ctx.sorted_group_id,
        sorted_c0=board_ctx.sorted_c0,
        sorted_c1=board_ctx.sorted_c1,
        sorted_card_positions=board_ctx.sorted_card_positions,
        slot_lower_by_card=board_ctx.slot_lower_by_card,
        slot_tie_by_card=board_ctx.slot_tie_by_card,
        slot_lower_tie_by_card=board_ctx.slot_lower_tie_by_card,
        sorted_contains=board_ctx.sorted_contains,
        lower_end=board_ctx.lower_end,
        tie_end=board_ctx.tie_end,
        lower_group_end=board_ctx.lower_group_end,
        tie_group_end=board_ctx.tie_group_end,
        rank_group_count=board_ctx.rank_group_count,
        max_rank_groups=board_ctx.max_rank_groups,
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


def _local_belief_matrix(
    beliefs: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
) -> torch.Tensor:
    batch_size, players, _ = beliefs.shape
    card_count = 47
    out = torch.zeros(
        batch_size,
        players,
        card_count,
        card_count,
        dtype=beliefs.dtype,
        device=beliefs.device,
    )
    board_offsets = (
        torch.arange(batch_size, device=beliefs.device, dtype=torch.int64)[:, None, None]
        * players
        * card_count
        * card_count
    )
    player_offsets = (
        torch.arange(players, device=beliefs.device, dtype=torch.int64)[None, :, None]
        * card_count
        * card_count
    )
    c0 = local_c0[:, None, :].long()
    c1 = local_c1[:, None, :].long()
    values = beliefs.reshape(-1)
    flat = out.reshape(-1)
    flat.scatter_(0, (board_offsets + player_offsets + c0 * card_count + c1).reshape(-1), values)
    flat.scatter_(0, (board_offsets + player_offsets + c1 * card_count + c0).reshape(-1), values)
    return out


def _subtract_same_pair_events(
    pair_event_all: torch.Tensor,
    same_all: torch.Tensor,
) -> None:
    players = pair_event_all.shape[0]
    if players == 4:
        for left, right in _P4_PLAYER_PAIRS:
            for mode in range(3):
                pair_event_all[left, right, mode, mode] -= same_all[left, right, mode]
        return
    for mode in range(3):
        pair_event_all[:, :, mode, mode] -= same_all[:, :, mode]


def _pair_event_lookup(
    pair_event_all: torch.Tensor,
    opponents: list[int],
    left: int,
    right: int,
) -> torch.Tensor:
    if pair_event_all.dim() == 5:
        return pair_event_all[_P4_PLAYER_PAIR_INDEX[(opponents[left], opponents[right])]]
    return pair_event_all[opponents[left], opponents[right]]


def _pair_event_all_from_card(
    card_all: torch.Tensor,
    same_all: torch.Tensor,
    *,
    finish_only: bool = False,
) -> torch.Tensor:
    compact = _p4_pair_event_triton(card_all, same_all, finish_only=finish_only)
    if compact is not None:
        return compact
    pair_event_all = torch.einsum("pmbhc,qnbhc->pqmnbh", card_all, card_all)
    _subtract_same_pair_events(pair_event_all, same_all)
    return pair_event_all


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


def _tier2_card_slots_from_positions(
    sorted_card_positions: torch.Tensor,
    lower_end: torch.Tensor,
    tie_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, card_count, slot_cap = sorted_card_positions.shape
    active_count = lower_end.shape[1]
    slot_lower = torch.empty(
        batch_size,
        active_count,
        card_count,
        dtype=torch.uint8,
        device=sorted_card_positions.device,
    )
    slot_tie = torch.empty_like(slot_lower)
    if triton is not None and sorted_card_positions.device.type == "cuda":
        block_h = 16
        grid = (batch_size, triton.cdiv(active_count, block_h), card_count)
        _tier2_card_slot_kernel[grid](
            sorted_card_positions.contiguous(),
            lower_end.contiguous(),
            tie_end.contiguous(),
            slot_lower,
            slot_tie,
            H=active_count,
            CARD_COUNT=card_count,
            SLOT_CAP=slot_cap,
            BLOCK_H=block_h,
            num_warps=2,
        )
        return slot_lower.contiguous(), slot_tie.contiguous()

    for card in range(card_count):
        positions = sorted_card_positions[:, card]
        slot_lower[:, :, card] = (
            positions[:, None, :] < lower_end[:, :, None]
        ).sum(dim=2).to(torch.uint8)
        slot_tie[:, :, card] = (
            positions[:, None, :] < tie_end[:, :, None]
        ).sum(dim=2).to(torch.uint8)
    return slot_lower.contiguous(), slot_tie.contiguous()


if triton is not None:

    @triton.jit
    def _tier2_card_slot_kernel(
        card_positions,
        lower_end,
        tie_end,
        slot_lower_out,
        slot_tie_out,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        card = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        slot = tl.arange(0, SLOT_CAP)
        h_mask = h < H
        positions = tl.load(card_positions + (b * CARD_COUNT + card) * SLOT_CAP + slot).to(tl.int32)
        lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
        tie = tl.load(tie_end + b * H + h, mask=h_mask, other=0)
        lower_count = tl.sum(tl.where(positions[None, :] < lower[:, None], 1, 0), axis=1)
        tie_count = tl.sum(tl.where(positions[None, :] < tie[:, None], 1, 0), axis=1)
        out_base = (b * H + h) * CARD_COUNT + card
        tl.store(slot_lower_out + out_base, lower_count, mask=h_mask)
        tl.store(slot_tie_out + out_base, tie_count, mask=h_mask)

    @triton.jit
    def _tier2_p4_sparse_scalar_pair_prefix_kernel(
        sorted_beliefs,
        scalar_prefix,
        pair_prefix,
        H: tl.constexpr,
        H1: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)
        h_mask = h < H
        b0 = tl.load(sorted_beliefs + (b * 4 + 0) * H + h, mask=h_mask, other=0.0)
        b1 = tl.load(sorted_beliefs + (b * 4 + 1) * H + h, mask=h_mask, other=0.0)
        b2 = tl.load(sorted_beliefs + (b * 4 + 2) * H + h, mask=h_mask, other=0.0)
        b3 = tl.load(sorted_beliefs + (b * 4 + 3) * H + h, mask=h_mask, other=0.0)
        c0 = tl.cumsum(b0, axis=0)
        c1 = tl.cumsum(b1, axis=0)
        c2 = tl.cumsum(b2, axis=0)
        c3 = tl.cumsum(b3, axis=0)
        scalar_base = b * 4 * H1
        tl.store(scalar_prefix + scalar_base + 0 * H1, 0.0)
        tl.store(scalar_prefix + scalar_base + 1 * H1, 0.0)
        tl.store(scalar_prefix + scalar_base + 2 * H1, 0.0)
        tl.store(scalar_prefix + scalar_base + 3 * H1, 0.0)
        tl.store(scalar_prefix + scalar_base + 0 * H1 + h + 1, c0, mask=h_mask)
        tl.store(scalar_prefix + scalar_base + 1 * H1 + h + 1, c1, mask=h_mask)
        tl.store(scalar_prefix + scalar_base + 2 * H1 + h + 1, c2, mask=h_mask)
        tl.store(scalar_prefix + scalar_base + 3 * H1 + h + 1, c3, mask=h_mask)

        v01 = b0 * b1
        v02 = b0 * b2
        v03 = b0 * b3
        v12 = b1 * b2
        v13 = b1 * b3
        v23 = b2 * b3
        p01 = tl.cumsum(v01, axis=0)
        p02 = tl.cumsum(v02, axis=0)
        p03 = tl.cumsum(v03, axis=0)
        p12 = tl.cumsum(v12, axis=0)
        p13 = tl.cumsum(v13, axis=0)
        p23 = tl.cumsum(v23, axis=0)
        pair_base = b * 6 * H1
        tl.store(pair_prefix + pair_base + 0 * H1, 0.0)
        tl.store(pair_prefix + pair_base + 1 * H1, 0.0)
        tl.store(pair_prefix + pair_base + 2 * H1, 0.0)
        tl.store(pair_prefix + pair_base + 3 * H1, 0.0)
        tl.store(pair_prefix + pair_base + 4 * H1, 0.0)
        tl.store(pair_prefix + pair_base + 5 * H1, 0.0)
        tl.store(pair_prefix + pair_base + 0 * H1 + h + 1, p01, mask=h_mask)
        tl.store(pair_prefix + pair_base + 1 * H1 + h + 1, p02, mask=h_mask)
        tl.store(pair_prefix + pair_base + 2 * H1 + h + 1, p03, mask=h_mask)
        tl.store(pair_prefix + pair_base + 3 * H1 + h + 1, p12, mask=h_mask)
        tl.store(pair_prefix + pair_base + 4 * H1 + h + 1, p13, mask=h_mask)
        tl.store(pair_prefix + pair_base + 5 * H1 + h + 1, p23, mask=h_mask)

    @triton.jit
    def _tier2_p4_sparse_card_cumsum_kernel(
        sorted_beliefs,
        card_positions,
        player_card_cumsum,
        pair_card_cumsum,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
    ):
        b = tl.program_id(0)
        card = tl.program_id(1)
        slot = tl.arange(0, SLOT_CAP)
        positions = tl.load(card_positions + (b * CARD_COUNT + card) * SLOT_CAP + slot).to(tl.int32)
        in_range = positions < H
        safe_pos = tl.where(in_range, positions, 0)
        b0 = tl.load(sorted_beliefs + (b * 4 + 0) * H + safe_pos)
        b1 = tl.load(sorted_beliefs + (b * 4 + 1) * H + safe_pos)
        b2 = tl.load(sorted_beliefs + (b * 4 + 2) * H + safe_pos)
        b3 = tl.load(sorted_beliefs + (b * 4 + 3) * H + safe_pos)
        b0 = tl.where(in_range, b0, 0.0)
        b1 = tl.where(in_range, b1, 0.0)
        b2 = tl.where(in_range, b2, 0.0)
        b3 = tl.where(in_range, b3, 0.0)
        c0 = tl.cumsum(b0, axis=0)
        c1 = tl.cumsum(b1, axis=0)
        c2 = tl.cumsum(b2, axis=0)
        c3 = tl.cumsum(b3, axis=0)
        player_base = (b * 4 * CARD_COUNT + card) * SLOT_CAP
        tl.store(player_card_cumsum + player_base + 0 * CARD_COUNT * SLOT_CAP + slot, c0)
        tl.store(player_card_cumsum + player_base + 1 * CARD_COUNT * SLOT_CAP + slot, c1)
        tl.store(player_card_cumsum + player_base + 2 * CARD_COUNT * SLOT_CAP + slot, c2)
        tl.store(player_card_cumsum + player_base + 3 * CARD_COUNT * SLOT_CAP + slot, c3)

        v01 = b0 * b1
        v02 = b0 * b2
        v03 = b0 * b3
        v12 = b1 * b2
        v13 = b1 * b3
        v23 = b2 * b3
        p01 = tl.cumsum(v01, axis=0)
        p02 = tl.cumsum(v02, axis=0)
        p03 = tl.cumsum(v03, axis=0)
        p12 = tl.cumsum(v12, axis=0)
        p13 = tl.cumsum(v13, axis=0)
        p23 = tl.cumsum(v23, axis=0)
        pair_base = (b * 6 * CARD_COUNT + card) * SLOT_CAP
        tl.store(pair_card_cumsum + pair_base + 0 * CARD_COUNT * SLOT_CAP + slot, p01)
        tl.store(pair_card_cumsum + pair_base + 1 * CARD_COUNT * SLOT_CAP + slot, p02)
        tl.store(pair_card_cumsum + pair_base + 2 * CARD_COUNT * SLOT_CAP + slot, p03)
        tl.store(pair_card_cumsum + pair_base + 3 * CARD_COUNT * SLOT_CAP + slot, p12)
        tl.store(pair_card_cumsum + pair_base + 4 * CARD_COUNT * SLOT_CAP + slot, p13)
        tl.store(pair_card_cumsum + pair_base + 5 * CARD_COUNT * SLOT_CAP + slot, p23)

    @triton.jit
    def _tier2_p4_sparse_scalar_same_kernel(
        beliefs,
        scalar_prefix,
        pair_prefix,
        player_card_cumsum,
        pair_card_cumsum,
        local_c0,
        local_c1,
        lower_end,
        tie_end,
        slot_lower_by_card,
        slot_tie_by_card,
        scalar_all,
        same_all,
        local_belief_matrix,
        B: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        TOTAL_SLOT: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H
        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
        tie = tl.load(tie_end + b * H + h, mask=h_mask, other=0)
        slot_base = (b * H + h) * CARD_COUNT
        sl0 = tl.load(slot_lower_by_card + slot_base + c0, mask=h_mask, other=0).to(tl.int32)
        sl1 = tl.load(slot_lower_by_card + slot_base + c1, mask=h_mask, other=0).to(tl.int32)
        st0 = tl.load(slot_tie_by_card + slot_base + c0, mask=h_mask, other=0).to(tl.int32)
        st1 = tl.load(slot_tie_by_card + slot_base + c1, mask=h_mask, other=0).to(tl.int32)

        for player in tl.static_range(0, 4):
            belief = tl.load(beliefs + (b * 4 + player) * H + h, mask=h_mask, other=0.0)
            matrix_base = (b * 4 + player) * CARD_COUNT * CARD_COUNT
            tl.store(local_belief_matrix + matrix_base + c0 * CARD_COUNT + c1, belief, mask=h_mask)
            tl.store(local_belief_matrix + matrix_base + c1 * CARD_COUNT + c0, belief, mask=h_mask)
            prefix_base = (b * 4 + player) * H1
            card_base0 = ((b * 4 + player) * CARD_COUNT + c0) * SLOT_CAP
            card_base1 = ((b * 4 + player) * CARD_COUNT + c1) * SLOT_CAP
            for mode in tl.static_range(0, 3):
                start = tl.where(mode == 1, lower, 0)
                end = tl.where(mode == 0, lower, tl.where(mode == 1, tie, H))
                scalar = tl.load(scalar_prefix + prefix_base + end, mask=h_mask, other=0.0) - tl.load(
                    scalar_prefix + prefix_base + start,
                    mask=h_mask,
                    other=0.0,
                )
                slot_start0 = tl.where(mode == 1, sl0, 0)
                slot_end0 = tl.where(mode == 0, sl0, tl.where(mode == 1, st0, TOTAL_SLOT))
                slot_start1 = tl.where(mode == 1, sl1, 0)
                slot_end1 = tl.where(mode == 0, sl1, tl.where(mode == 1, st1, TOTAL_SLOT))
                idx_s0 = tl.maximum(slot_start0 - 1, 0)
                idx_e0 = tl.maximum(slot_end0 - 1, 0)
                idx_s1 = tl.maximum(slot_start1 - 1, 0)
                idx_e1 = tl.maximum(slot_end1 - 1, 0)
                card0_start = tl.load(
                    player_card_cumsum + card_base0 + idx_s0,
                    mask=h_mask & (slot_start0 > 0),
                    other=0.0,
                )
                card0_end = tl.load(
                    player_card_cumsum + card_base0 + idx_e0,
                    mask=h_mask & (slot_end0 > 0),
                    other=0.0,
                )
                card1_start = tl.load(
                    player_card_cumsum + card_base1 + idx_s1,
                    mask=h_mask & (slot_start1 > 0),
                    other=0.0,
                )
                card1_end = tl.load(
                    player_card_cumsum + card_base1 + idx_e1,
                    mask=h_mask & (slot_end1 > 0),
                    other=0.0,
                )
                edge = tl.where(mode == 0, 0.0, belief)
                value = scalar - (card0_end - card0_start) - (card1_end - card1_start) + edge
                tl.store(
                    scalar_all + ((player * 3 + mode) * B + b) * H + h,
                    value,
                    mask=h_mask,
                )

        b0 = tl.load(beliefs + (b * 4 + 0) * H + h, mask=h_mask, other=0.0)
        b1 = tl.load(beliefs + (b * 4 + 1) * H + h, mask=h_mask, other=0.0)
        b2 = tl.load(beliefs + (b * 4 + 2) * H + h, mask=h_mask, other=0.0)
        b3 = tl.load(beliefs + (b * 4 + 3) * H + h, mask=h_mask, other=0.0)
        for pair in tl.static_range(0, 6):
            left_b = tl.where(pair == 0, b0, tl.where(pair == 1, b0, tl.where(pair == 2, b0, tl.where(pair == 3, b1, tl.where(pair == 4, b1, b2)))))
            right_b = tl.where(pair == 0, b1, tl.where(pair == 1, b2, tl.where(pair == 2, b3, tl.where(pair == 3, b2, tl.where(pair == 4, b3, b3)))))
            edge = left_b * right_b
            pair_prefix_base = (b * 6 + pair) * H1
            pair_card_base0 = ((b * 6 + pair) * CARD_COUNT + c0) * SLOT_CAP
            pair_card_base1 = ((b * 6 + pair) * CARD_COUNT + c1) * SLOT_CAP
            for mode in tl.static_range(0, 3):
                start = tl.where(mode == 1, lower, 0)
                end = tl.where(mode == 0, lower, tl.where(mode == 1, tie, H))
                scalar = tl.load(pair_prefix + pair_prefix_base + end, mask=h_mask, other=0.0) - tl.load(
                    pair_prefix + pair_prefix_base + start,
                    mask=h_mask,
                    other=0.0,
                )
                slot_start0 = tl.where(mode == 1, sl0, 0)
                slot_end0 = tl.where(mode == 0, sl0, tl.where(mode == 1, st0, TOTAL_SLOT))
                slot_start1 = tl.where(mode == 1, sl1, 0)
                slot_end1 = tl.where(mode == 0, sl1, tl.where(mode == 1, st1, TOTAL_SLOT))
                idx_s0 = tl.maximum(slot_start0 - 1, 0)
                idx_e0 = tl.maximum(slot_end0 - 1, 0)
                idx_s1 = tl.maximum(slot_start1 - 1, 0)
                idx_e1 = tl.maximum(slot_end1 - 1, 0)
                card0_start = tl.load(
                    pair_card_cumsum + pair_card_base0 + idx_s0,
                    mask=h_mask & (slot_start0 > 0),
                    other=0.0,
                )
                card0_end = tl.load(
                    pair_card_cumsum + pair_card_base0 + idx_e0,
                    mask=h_mask & (slot_end0 > 0),
                    other=0.0,
                )
                card1_start = tl.load(
                    pair_card_cumsum + pair_card_base1 + idx_s1,
                    mask=h_mask & (slot_start1 > 0),
                    other=0.0,
                )
                card1_end = tl.load(
                    pair_card_cumsum + pair_card_base1 + idx_e1,
                    mask=h_mask & (slot_end1 > 0),
                    other=0.0,
                )
                value = scalar - (card0_end - card0_start) - (card1_end - card1_start)
                value += tl.where(mode == 0, 0.0, edge)
                tl.store(same_all + ((pair * 3 + mode) * B + b) * H + h, value, mask=h_mask)

    @triton.jit
    def _tier2_prefix_scalar_card_kernel(
        scalar_prefix,
        card_prefix,
        beliefs,
        local_belief_matrix,
        local_c0,
        local_c1,
        pair_p_rank_flags,
        pair_q_rank_flags,
        lower_end,
        tie_end,
        group_count,
        scalar_out,
        card_out,
        B: tl.constexpr,
        P: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        MODE: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        player = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H

        if MODE == 0:
            lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
            start = 0
            end = lower
        elif MODE == 1:
            lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
            tie = tl.load(tie_end + b * H + h, mask=h_mask, other=0)
            start = lower
            end = tie
        else:
            start = 0
            end = tl.load(group_count + b)

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)

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
        if MODE == 0:
            edge = 0.0
        else:
            edge = tl.load(beliefs + (b * P + player) * H + h, mask=h_mask, other=0.0)
        scalar = scalar - hero0 - hero1 + edge
        tl.store(
            scalar_out + ((player * 3 + MODE) * B + b) * H + h,
            scalar,
            mask=h_mask,
        )

        card = tl.arange(0, BLOCK_C)
        valid_card = card < CARD_COUNT
        interval = tl.load(
            card_prefix + card_base + end[:, None] * CARD_COUNT + card[None, :],
            mask=h_mask[:, None] & valid_card[None, :],
            other=0.0,
        ) - tl.load(
            card_prefix + card_base + start[:, None] * CARD_COUNT + card[None, :],
            mask=h_mask[:, None] & valid_card[None, :],
            other=0.0,
        )

        pair_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        hc_mask = h_mask[:, None] & valid_card[None, :]
        valid_p = hc_mask & (card[None, :] != c0[:, None])
        valid_q = hc_mask & (card[None, :] != c1[:, None])
        if MODE == 0:
            flags_p = tl.load(pair_p_rank_flags + pair_base, mask=hc_mask, other=0)
            flags_q = tl.load(pair_q_rank_flags + pair_base, mask=hc_mask, other=0)
            row_p = valid_p & ((flags_p & 1) != 0)
            row_q = valid_q & ((flags_q & 1) != 0)
        elif MODE == 1:
            flags_p = tl.load(pair_p_rank_flags + pair_base, mask=hc_mask, other=0)
            flags_q = tl.load(pair_q_rank_flags + pair_base, mask=hc_mask, other=0)
            row_p = valid_p & ((flags_p & 2) != 0)
            row_q = valid_q & ((flags_q & 2) != 0)
        else:
            row_p = valid_p
            row_q = valid_q
        matrix_base = ((b * P + player) * CARD_COUNT + card[None, :]) * CARD_COUNT
        corr = tl.load(local_belief_matrix + matrix_base + c0[:, None], mask=row_p, other=0.0)
        corr += tl.load(local_belief_matrix + matrix_base + c1[:, None], mask=row_q, other=0.0)
        value = interval - corr
        value = tl.where((card[None, :] == c0[:, None]) | (card[None, :] == c1[:, None]), 0.0, value)
        card_out_base = (((player * 3 + MODE) * B + b) * H + h[:, None]) * CARD_COUNT
        tl.store(card_out + card_out_base + card[None, :], value, mask=hc_mask)

    @triton.jit
    def _tier2_prefix_scalar_only_kernel(
        scalar_prefix,
        card_prefix,
        beliefs,
        local_c0,
        local_c1,
        lower_end,
        tie_end,
        group_count,
        scalar_out,
        B: tl.constexpr,
        P: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        MODE: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        player = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H

        if MODE == 0:
            lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
            start = 0
            end = lower
        elif MODE == 1:
            lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
            tie = tl.load(tie_end + b * H + h, mask=h_mask, other=0)
            start = lower
            end = tie
        else:
            start = 0
            end = tl.load(group_count + b)

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        scalar_base = (b * P + player) * H1
        scalar = tl.load(scalar_prefix + scalar_base + end) - tl.load(
            scalar_prefix + scalar_base + start
        )
        card_base = ((b * P + player) * H1) * CARD_COUNT
        hero0 = tl.load(card_prefix + card_base + end * CARD_COUNT + c0) - tl.load(
            card_prefix + card_base + start * CARD_COUNT + c0
        )
        hero1 = tl.load(card_prefix + card_base + end * CARD_COUNT + c1) - tl.load(
            card_prefix + card_base + start * CARD_COUNT + c1
        )
        if MODE == 0:
            edge = 0.0
        else:
            edge = tl.load(beliefs + (b * P + player) * H + h, mask=h_mask, other=0.0)
        value = scalar - hero0 - hero1 + edge
        tl.store(
            scalar_out + ((player * 3 + MODE) * B + b) * H + h,
            value,
            mask=h_mask,
        )

    @triton.jit
    def _tier2_prefix_same_kernel(
        pair_prefix,
        pair_card_prefix,
        beliefs,
        local_c0,
        local_c1,
        lower_end,
        tie_end,
        group_count,
        same_out,
        B: tl.constexpr,
        P: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        USE_P4_UNORDERED: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        pair_mode = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H
        mode = pair_mode % 3
        pair = pair_mode // 3
        if USE_P4_UNORDERED:
            left = tl.where(pair < 3, 0, tl.where(pair < 5, 1, 2))
            right = tl.where(pair < 3, pair + 1, tl.where(pair < 5, pair - 1, 3))
            pair_prefix_base = (b * 6 + pair) * H1
            pair_card_base = ((b * 6 + pair) * H1) * CARD_COUNT
        else:
            right = pair % P
            left = pair // P
            pair_prefix_base = ((b * P + left) * P + right) * H1
            pair_card_base = (((b * P + left) * P + right) * H1) * CARD_COUNT

        lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
        tie = tl.load(tie_end + b * H + h, mask=h_mask, other=0)
        total = tl.load(group_count + b)
        start = tl.where(mode == 1, lower, 0)
        end = tl.where(mode == 0, lower, tl.where(mode == 1, tie, total))

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        scalar = tl.load(pair_prefix + pair_prefix_base + end) - tl.load(
            pair_prefix + pair_prefix_base + start
        )
        card0 = tl.load(pair_card_prefix + pair_card_base + end * CARD_COUNT + c0) - tl.load(
            pair_card_prefix + pair_card_base + start * CARD_COUNT + c0
        )
        card1 = tl.load(pair_card_prefix + pair_card_base + end * CARD_COUNT + c1) - tl.load(
            pair_card_prefix + pair_card_base + start * CARD_COUNT + c1
        )
        edge = tl.load(beliefs + (b * P + left) * H + h, mask=h_mask, other=0.0) * tl.load(
            beliefs + (b * P + right) * H + h,
            mask=h_mask,
            other=0.0,
        )
        edge = tl.where(mode == 0, 0.0, edge)
        value = scalar - card0 - card1 + edge
        same_pair = tl.where(USE_P4_UNORDERED, pair, left * P + right)
        tl.store(
            same_out + ((same_pair * 3 + mode) * B + b) * H + h,
            value,
            mask=h_mask,
        )

    @triton.jit
    def _tier2_p4_group_accum_kernel(
        sorted_beliefs,
        sorted_group_id,
        sorted_c0,
        sorted_c1,
        scalar_group,
        card_group,
        pair_group,
        pair_card_group,
        local_belief_matrix,
        B: tl.constexpr,
        H: tl.constexpr,
        G: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H
        group = tl.load(sorted_group_id + b * H + h, mask=h_mask, other=0)
        c0 = tl.load(sorted_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(sorted_c1 + b * H + h, mask=h_mask, other=0)

        b0 = tl.load(sorted_beliefs + (b * 4 + 0) * H + h, mask=h_mask, other=0.0)
        b1 = tl.load(sorted_beliefs + (b * 4 + 1) * H + h, mask=h_mask, other=0.0)
        b2 = tl.load(sorted_beliefs + (b * 4 + 2) * H + h, mask=h_mask, other=0.0)
        b3 = tl.load(sorted_beliefs + (b * 4 + 3) * H + h, mask=h_mask, other=0.0)

        scalar_base = b * 4 * G + group
        card_base = b * 4 * G * CARD_COUNT + group * CARD_COUNT
        tl.atomic_add(scalar_group + scalar_base + 0 * G, b0, sem="relaxed", mask=h_mask)
        tl.atomic_add(scalar_group + scalar_base + 1 * G, b1, sem="relaxed", mask=h_mask)
        tl.atomic_add(scalar_group + scalar_base + 2 * G, b2, sem="relaxed", mask=h_mask)
        tl.atomic_add(scalar_group + scalar_base + 3 * G, b3, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 0 * G * CARD_COUNT + c0, b0, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 0 * G * CARD_COUNT + c1, b0, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 1 * G * CARD_COUNT + c0, b1, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 1 * G * CARD_COUNT + c1, b1, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 2 * G * CARD_COUNT + c0, b2, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 2 * G * CARD_COUNT + c1, b2, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 3 * G * CARD_COUNT + c0, b3, sem="relaxed", mask=h_mask)
        tl.atomic_add(card_group + card_base + 3 * G * CARD_COUNT + c1, b3, sem="relaxed", mask=h_mask)

        matrix_base = (b * 4) * CARD_COUNT * CARD_COUNT
        matrix01 = c0 * CARD_COUNT + c1
        matrix10 = c1 * CARD_COUNT + c0
        tl.store(local_belief_matrix + matrix_base + matrix01, b0, mask=h_mask)
        tl.store(local_belief_matrix + matrix_base + matrix10, b0, mask=h_mask)
        tl.store(local_belief_matrix + matrix_base + CARD_COUNT * CARD_COUNT + matrix01, b1, mask=h_mask)
        tl.store(local_belief_matrix + matrix_base + CARD_COUNT * CARD_COUNT + matrix10, b1, mask=h_mask)
        tl.store(local_belief_matrix + matrix_base + 2 * CARD_COUNT * CARD_COUNT + matrix01, b2, mask=h_mask)
        tl.store(local_belief_matrix + matrix_base + 2 * CARD_COUNT * CARD_COUNT + matrix10, b2, mask=h_mask)
        tl.store(local_belief_matrix + matrix_base + 3 * CARD_COUNT * CARD_COUNT + matrix01, b3, mask=h_mask)
        tl.store(local_belief_matrix + matrix_base + 3 * CARD_COUNT * CARD_COUNT + matrix10, b3, mask=h_mask)

        v01 = b0 * b1
        v02 = b0 * b2
        v03 = b0 * b3
        v12 = b1 * b2
        v13 = b1 * b3
        v23 = b2 * b3
        pair_base = b * 6 * G + group
        pair_card_base = b * 6 * G * CARD_COUNT + group * CARD_COUNT
        tl.atomic_add(pair_group + pair_base + 0 * G, v01, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_group + pair_base + 1 * G, v02, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_group + pair_base + 2 * G, v03, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_group + pair_base + 3 * G, v12, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_group + pair_base + 4 * G, v13, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_group + pair_base + 5 * G, v23, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 0 * G * CARD_COUNT + c0, v01, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 0 * G * CARD_COUNT + c1, v01, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 1 * G * CARD_COUNT + c0, v02, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 1 * G * CARD_COUNT + c1, v02, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 2 * G * CARD_COUNT + c0, v03, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 2 * G * CARD_COUNT + c1, v03, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 3 * G * CARD_COUNT + c0, v12, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 3 * G * CARD_COUNT + c1, v12, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 4 * G * CARD_COUNT + c0, v13, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 4 * G * CARD_COUNT + c1, v13, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 5 * G * CARD_COUNT + c0, v23, sem="relaxed", mask=h_mask)
        tl.atomic_add(pair_card_group + pair_card_base + 5 * G * CARD_COUNT + c1, v23, sem="relaxed", mask=h_mask)

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
        h_block = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H
        hero_c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=-1)
        hero_c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=-1)
        rank_h = tl.load(ranks + b * H + h, mask=h_mask, other=0)

        opp0 = tl.where(0 < hero, 0, 1)
        opp1 = tl.where(1 < hero, 1, 2)
        opp2 = tl.where(2 < hero, 2, 3)

        o0_base0 = (((opp0 * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
        o0_base1 = o0_base0 + B * H * CARD_COUNT
        o0_base2 = o0_base0 + 2 * B * H * CARD_COUNT
        o1_base0 = (((opp1 * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
        o1_base1 = o1_base0 + B * H * CARD_COUNT
        o1_base2 = o1_base0 + 2 * B * H * CARD_COUNT
        o2_base0 = (((opp2 * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
        o2_base1 = o2_base0 + B * H * CARD_COUNT
        o2_base2 = o2_base0 + 2 * B * H * CARD_COUNT

        den_total = tl.zeros((BLOCK_H,), dtype=tl.float32)
        num_total = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for k_block in tl.range(0, K_BLOCKS):
            k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            hk_mask = (h[:, None] < H) & (k[None, :] < H)
            k_c0 = tl.load(local_c0 + b * H + k, mask=k < H, other=-2)
            k_c1 = tl.load(local_c1 + b * H + k, mask=k < H, other=-3)
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

            b0 = tl.load(beliefs + (b * 4 + opp0) * H + k, mask=k < H, other=0.0)
            b1 = tl.load(beliefs + (b * 4 + opp1) * H + k, mask=k < H, other=0.0)
            b2 = tl.load(beliefs + (b * 4 + opp2) * H + k, mask=k < H, other=0.0)
            w00 = tl.where(lower, b0[None, :], 0.0)
            w01 = tl.where(tied, b0[None, :], 0.0)
            w02 = tl.where(disjoint, b0[None, :], 0.0)
            w10 = tl.where(lower, b1[None, :], 0.0)
            w11 = tl.where(tied, b1[None, :], 0.0)
            w12 = tl.where(disjoint, b1[None, :], 0.0)
            w20 = tl.where(lower, b2[None, :], 0.0)
            w21 = tl.where(tied, b2[None, :], 0.0)
            w22 = tl.where(disjoint, b2[None, :], 0.0)

            r00 = tl.load(card_all + o0_base0 + k_c0[None, :], mask=hk_mask, other=0.0)
            r00 += tl.load(card_all + o0_base0 + k_c1[None, :], mask=hk_mask, other=0.0)
            r00 -= w00
            r01 = tl.load(card_all + o0_base1 + k_c0[None, :], mask=hk_mask, other=0.0)
            r01 += tl.load(card_all + o0_base1 + k_c1[None, :], mask=hk_mask, other=0.0)
            r01 -= w01
            r02 = tl.load(card_all + o0_base2 + k_c0[None, :], mask=hk_mask, other=0.0)
            r02 += tl.load(card_all + o0_base2 + k_c1[None, :], mask=hk_mask, other=0.0)
            r02 -= w02

            r10 = tl.load(card_all + o1_base0 + k_c0[None, :], mask=hk_mask, other=0.0)
            r10 += tl.load(card_all + o1_base0 + k_c1[None, :], mask=hk_mask, other=0.0)
            r10 -= w10
            r11 = tl.load(card_all + o1_base1 + k_c0[None, :], mask=hk_mask, other=0.0)
            r11 += tl.load(card_all + o1_base1 + k_c1[None, :], mask=hk_mask, other=0.0)
            r11 -= w11
            r12 = tl.load(card_all + o1_base2 + k_c0[None, :], mask=hk_mask, other=0.0)
            r12 += tl.load(card_all + o1_base2 + k_c1[None, :], mask=hk_mask, other=0.0)
            r12 -= w12

            r20 = tl.load(card_all + o2_base0 + k_c0[None, :], mask=hk_mask, other=0.0)
            r20 += tl.load(card_all + o2_base0 + k_c1[None, :], mask=hk_mask, other=0.0)
            r20 -= w20
            r21 = tl.load(card_all + o2_base1 + k_c0[None, :], mask=hk_mask, other=0.0)
            r21 += tl.load(card_all + o2_base1 + k_c1[None, :], mask=hk_mask, other=0.0)
            r21 -= w21
            r22 = tl.load(card_all + o2_base2 + k_c0[None, :], mask=hk_mask, other=0.0)
            r22 += tl.load(card_all + o2_base2 + k_c1[None, :], mask=hk_mask, other=0.0)
            r22 -= w22

            combo0 = r10 * r20 + 0.5 * (r11 * r20 + r10 * r21) + (1.0 / 3.0) * r11 * r21
            combo1 = 0.5 * r10 * r20 + (1.0 / 3.0) * (r11 * r20 + r10 * r21)
            combo1 += 0.25 * r11 * r21
            den_total += tl.sum(w02 * r12 * r22, axis=1)
            num_total += tl.sum(w00 * combo0 + w01 * combo1, axis=1)

            combo0 = r00 * r20 + 0.5 * (r01 * r20 + r00 * r21) + (1.0 / 3.0) * r01 * r21
            combo1 = 0.5 * r00 * r20 + (1.0 / 3.0) * (r01 * r20 + r00 * r21)
            combo1 += 0.25 * r01 * r21
            den_total += tl.sum(w12 * r02 * r22, axis=1)
            num_total += tl.sum(w10 * combo0 + w11 * combo1, axis=1)

            combo0 = r00 * r10 + 0.5 * (r01 * r10 + r00 * r11) + (1.0 / 3.0) * r01 * r11
            combo1 = 0.5 * r00 * r10 + (1.0 / 3.0) * (r01 * r10 + r00 * r11)
            combo1 += 0.25 * r01 * r11
            den_total += tl.sum(w22 * r02 * r12, axis=1)
            num_total += tl.sum(w20 * combo0 + w21 * combo1, axis=1)

        out_base = (b * 4 + hero) * H + h
        tl.store(wedge_den_out + out_base, den_total, mask=h_mask)
        tl.store(wedge_num_out + out_base, num_total, mask=h_mask)

    @triton.jit
    def _tier3_wedge_p4_all_heroes_kernel(
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
        PART_K_BLOCKS: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
        COMPUTE_DEN: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        k_part = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H
        hero_c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=-1)
        hero_c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=-1)
        rank_h = tl.load(ranks + b * H + h, mask=h_mask, other=0)

        p0_base0 = (((0 * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
        p0_base1 = p0_base0 + B * H * CARD_COUNT
        p0_base2 = p0_base0 + 2 * B * H * CARD_COUNT
        p1_base0 = (((1 * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
        p1_base1 = p1_base0 + B * H * CARD_COUNT
        p1_base2 = p1_base0 + 2 * B * H * CARD_COUNT
        p2_base0 = (((2 * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
        p2_base1 = p2_base0 + B * H * CARD_COUNT
        p2_base2 = p2_base0 + 2 * B * H * CARD_COUNT
        p3_base0 = (((3 * 3 + 0) * B + b) * H + h[:, None]) * CARD_COUNT
        p3_base1 = p3_base0 + B * H * CARD_COUNT
        p3_base2 = p3_base0 + 2 * B * H * CARD_COUNT

        den0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        den1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        den2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        den3 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        num0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        num1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        num2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        num3 = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for k_local in tl.range(0, PART_K_BLOCKS):
            k_block = k_part * PART_K_BLOCKS + k_local
            k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
            hk_mask = (h[:, None] < H) & (k[None, :] < H)
            k_mask = k < H
            k_c0 = tl.load(local_c0 + b * H + k, mask=k_mask, other=-2)
            k_c1 = tl.load(local_c1 + b * H + k, mask=k_mask, other=-3)
            rank_k = tl.load(ranks + b * H + k, mask=k_mask, other=-1)
            disjoint = (
                (k_c0[None, :] != hero_c0[:, None])
                & (k_c0[None, :] != hero_c1[:, None])
                & (k_c1[None, :] != hero_c0[:, None])
                & (k_c1[None, :] != hero_c1[:, None])
                & hk_mask
            )
            lower = disjoint & (rank_k[None, :] < rank_h[:, None])
            tied = disjoint & (rank_k[None, :] == rank_h[:, None])

            b0 = tl.load(beliefs + (b * 4 + 0) * H + k, mask=k_mask, other=0.0)
            b1 = tl.load(beliefs + (b * 4 + 1) * H + k, mask=k_mask, other=0.0)
            b2 = tl.load(beliefs + (b * 4 + 2) * H + k, mask=k_mask, other=0.0)
            b3 = tl.load(beliefs + (b * 4 + 3) * H + k, mask=k_mask, other=0.0)
            w00 = tl.where(lower, b0[None, :], 0.0)
            w01 = tl.where(tied, b0[None, :], 0.0)
            w02 = tl.where(disjoint, b0[None, :], 0.0)
            w10 = tl.where(lower, b1[None, :], 0.0)
            w11 = tl.where(tied, b1[None, :], 0.0)
            w20 = tl.where(lower, b2[None, :], 0.0)
            w21 = tl.where(tied, b2[None, :], 0.0)
            w30 = tl.where(lower, b3[None, :], 0.0)
            w31 = tl.where(tied, b3[None, :], 0.0)
            if COMPUTE_DEN:
                w02 = tl.where(disjoint, b0[None, :], 0.0)
                w12 = tl.where(disjoint, b1[None, :], 0.0)
                w22 = tl.where(disjoint, b2[None, :], 0.0)
                w32 = tl.where(disjoint, b3[None, :], 0.0)

            r00 = tl.load(card_all + p0_base0 + k_c0[None, :], mask=hk_mask, other=0.0)
            r00 += tl.load(card_all + p0_base0 + k_c1[None, :], mask=hk_mask, other=0.0)
            r00 -= w00
            r01 = tl.load(card_all + p0_base1 + k_c0[None, :], mask=hk_mask, other=0.0)
            r01 += tl.load(card_all + p0_base1 + k_c1[None, :], mask=hk_mask, other=0.0)
            r01 -= w01
            if COMPUTE_DEN:
                r02 = tl.load(card_all + p0_base2 + k_c0[None, :], mask=hk_mask, other=0.0)
                r02 += tl.load(card_all + p0_base2 + k_c1[None, :], mask=hk_mask, other=0.0)
                r02 -= w02

            r10 = tl.load(card_all + p1_base0 + k_c0[None, :], mask=hk_mask, other=0.0)
            r10 += tl.load(card_all + p1_base0 + k_c1[None, :], mask=hk_mask, other=0.0)
            r10 -= w10
            r11 = tl.load(card_all + p1_base1 + k_c0[None, :], mask=hk_mask, other=0.0)
            r11 += tl.load(card_all + p1_base1 + k_c1[None, :], mask=hk_mask, other=0.0)
            r11 -= w11
            if COMPUTE_DEN:
                r12 = tl.load(card_all + p1_base2 + k_c0[None, :], mask=hk_mask, other=0.0)
                r12 += tl.load(card_all + p1_base2 + k_c1[None, :], mask=hk_mask, other=0.0)
                r12 -= w12

            r20 = tl.load(card_all + p2_base0 + k_c0[None, :], mask=hk_mask, other=0.0)
            r20 += tl.load(card_all + p2_base0 + k_c1[None, :], mask=hk_mask, other=0.0)
            r20 -= w20
            r21 = tl.load(card_all + p2_base1 + k_c0[None, :], mask=hk_mask, other=0.0)
            r21 += tl.load(card_all + p2_base1 + k_c1[None, :], mask=hk_mask, other=0.0)
            r21 -= w21
            if COMPUTE_DEN:
                r22 = tl.load(card_all + p2_base2 + k_c0[None, :], mask=hk_mask, other=0.0)
                r22 += tl.load(card_all + p2_base2 + k_c1[None, :], mask=hk_mask, other=0.0)
                r22 -= w22

            r30 = tl.load(card_all + p3_base0 + k_c0[None, :], mask=hk_mask, other=0.0)
            r30 += tl.load(card_all + p3_base0 + k_c1[None, :], mask=hk_mask, other=0.0)
            r30 -= w30
            r31 = tl.load(card_all + p3_base1 + k_c0[None, :], mask=hk_mask, other=0.0)
            r31 += tl.load(card_all + p3_base1 + k_c1[None, :], mask=hk_mask, other=0.0)
            r31 -= w31
            if COMPUTE_DEN:
                r32 = tl.load(card_all + p3_base2 + k_c0[None, :], mask=hk_mask, other=0.0)
                r32 += tl.load(card_all + p3_base2 + k_c1[None, :], mask=hk_mask, other=0.0)
                r32 -= w32

            # Exact two-point quadrature for 1 / (tie_count + 1) over three opponents.
            q0 = 0.21132486540518713
            q1 = 0.7886751345948129
            r0q0 = r00 + q0 * r01
            r0q1 = r00 + q1 * r01
            r1q0 = r10 + q0 * r11
            r1q1 = r10 + q1 * r11
            r2q0 = r20 + q0 * r21
            r2q1 = r20 + q1 * r21
            r3q0 = r30 + q0 * r31
            r3q1 = r30 + q1 * r31
            w0q0 = w00 + q0 * w01
            w0q1 = w00 + q1 * w01
            w1q0 = w10 + q0 * w11
            w1q1 = w10 + q1 * w11
            w2q0 = w20 + q0 * w21
            w2q1 = w20 + q1 * w21
            w3q0 = w30 + q0 * w31
            w3q1 = w30 + q1 * w31
            mix0_12 = r1q0 * r2q0
            mix1_12 = r1q1 * r2q1
            mix0_13 = r1q0 * r3q0
            mix1_13 = r1q1 * r3q1
            mix0_23 = r2q0 * r3q0
            mix1_23 = r2q1 * r3q1

            num0 += tl.sum(
                0.5
                * (
                    w1q0 * mix0_23
                    + w1q1 * mix1_23
                    + w2q0 * mix0_13
                    + w2q1 * mix1_13
                    + w3q0 * mix0_12
                    + w3q1 * mix1_12
                ),
                axis=1,
            )
            if COMPUTE_DEN:
                den_12 = r12 * r22
                den_13 = r12 * r32
                den_23 = r22 * r32
                den0 += tl.sum(w12 * den_23 + w22 * den_13 + w32 * den_12, axis=1)

            mix0_02 = r0q0 * r2q0
            mix1_02 = r0q1 * r2q1
            mix0_03 = r0q0 * r3q0
            mix1_03 = r0q1 * r3q1

            num1 += tl.sum(
                0.5
                * (
                    w0q0 * mix0_23
                    + w0q1 * mix1_23
                    + w2q0 * mix0_03
                    + w2q1 * mix1_03
                    + w3q0 * mix0_02
                    + w3q1 * mix1_02
                ),
                axis=1,
            )
            if COMPUTE_DEN:
                den_02 = r02 * r22
                den_03 = r02 * r32
                den1 += tl.sum(w02 * den_23 + w22 * den_03 + w32 * den_02, axis=1)

            mix0_01 = r0q0 * r1q0
            mix1_01 = r0q1 * r1q1

            num2 += tl.sum(
                0.5
                * (
                    w0q0 * mix0_13
                    + w0q1 * mix1_13
                    + w1q0 * mix0_03
                    + w1q1 * mix1_03
                    + w3q0 * mix0_01
                    + w3q1 * mix1_01
                ),
                axis=1,
            )
            if COMPUTE_DEN:
                den_01 = r02 * r12
                den2 += tl.sum(w02 * den_13 + w12 * den_03 + w32 * den_01, axis=1)

            num3 += tl.sum(
                0.5
                * (
                    w0q0 * mix0_12
                    + w0q1 * mix1_12
                    + w1q0 * mix0_02
                    + w1q1 * mix1_02
                    + w2q0 * mix0_01
                    + w2q1 * mix1_01
                ),
                axis=1,
            )
            if COMPUTE_DEN:
                den3 += tl.sum(w02 * den_12 + w12 * den_02 + w22 * den_01, axis=1)

        out_base = (b * 4) * H + h
        if SPLIT_K == 1:
            tl.store(wedge_num_out + out_base, num0, mask=h_mask)
            tl.store(wedge_num_out + out_base + H, num1, mask=h_mask)
            tl.store(wedge_num_out + out_base + 2 * H, num2, mask=h_mask)
            tl.store(wedge_num_out + out_base + 3 * H, num3, mask=h_mask)
            if COMPUTE_DEN:
                tl.store(wedge_den_out + out_base, den0, mask=h_mask)
                tl.store(wedge_den_out + out_base + H, den1, mask=h_mask)
                tl.store(wedge_den_out + out_base + 2 * H, den2, mask=h_mask)
                tl.store(wedge_den_out + out_base + 3 * H, den3, mask=h_mask)
        else:
            tl.atomic_add(wedge_num_out + out_base, num0, sem="relaxed", mask=h_mask)
            tl.atomic_add(wedge_num_out + out_base + H, num1, sem="relaxed", mask=h_mask)
            tl.atomic_add(wedge_num_out + out_base + 2 * H, num2, sem="relaxed", mask=h_mask)
            tl.atomic_add(wedge_num_out + out_base + 3 * H, num3, sem="relaxed", mask=h_mask)
            if COMPUTE_DEN:
                tl.atomic_add(wedge_den_out + out_base, den0, sem="relaxed", mask=h_mask)
                tl.atomic_add(wedge_den_out + out_base + H, den1, sem="relaxed", mask=h_mask)
                tl.atomic_add(wedge_den_out + out_base + 2 * H, den2, sem="relaxed", mask=h_mask)
                tl.atomic_add(wedge_den_out + out_base + 3 * H, den3, sem="relaxed", mask=h_mask)

    @triton.jit
    def _p4_pair_event_kernel(
        card_all,
        same_all,
        pair_event_out,
        B: tl.constexpr,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        pair = tl.program_id(1)
        h_block = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        hc_mask = h_mask[:, None] & (card[None, :] < CARD_COUNT)
        left = tl.where(pair < 3, 0, tl.where(pair < 5, 1, 2))
        right = tl.where(pair < 3, pair + 1, tl.where(pair < 5, pair - 1, 3))

        for left_mode in tl.static_range(0, 3):
            left_base = (((left * 3 + left_mode) * B + b) * H + h[:, None]) * CARD_COUNT
            left_cards = tl.load(card_all + left_base + card[None, :], mask=hc_mask, other=0.0)
            for right_mode in tl.static_range(0, 3):
                right_base = (((right * 3 + right_mode) * B + b) * H + h[:, None]) * CARD_COUNT
                right_cards = tl.load(
                    card_all + right_base + card[None, :],
                    mask=hc_mask,
                    other=0.0,
                )
                value = tl.sum(left_cards * right_cards, axis=1)
                if left_mode == right_mode:
                    value -= tl.load(
                        same_all + ((pair * 3 + left_mode) * B + b) * H + h,
                        mask=h_mask,
                        other=0.0,
                    )
                out_base = (((pair * 3 + left_mode) * 3 + right_mode) * B + b) * H
                tl.store(pair_event_out + out_base + h, value, mask=h_mask)

    @triton.jit
    def _p4_pair_event_finish_kernel(
        card_all,
        same_all,
        pair_event_out,
        B: tl.constexpr,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        pair = tl.program_id(1)
        h_block = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        hc_mask = h_mask[:, None] & (card[None, :] < CARD_COUNT)
        left = tl.where(pair < 3, 0, tl.where(pair < 5, 1, 2))
        right = tl.where(pair < 3, pair + 1, tl.where(pair < 5, pair - 1, 3))

        for event in tl.static_range(0, 5):
            left_mode = tl.where(
                event == 2,
                1,
                tl.where(event == 3, 1, tl.where(event == 4, 2, 0)),
            )
            right_mode = tl.where(event == 1, 1, tl.where(event == 3, 1, tl.where(event == 4, 2, 0)))
            left_base = (((left * 3 + left_mode) * B + b) * H + h[:, None]) * CARD_COUNT
            right_base = (((right * 3 + right_mode) * B + b) * H + h[:, None]) * CARD_COUNT
            left_cards = tl.load(card_all + left_base + card[None, :], mask=hc_mask, other=0.0)
            right_cards = tl.load(card_all + right_base + card[None, :], mask=hc_mask, other=0.0)
            value = tl.sum(left_cards * right_cards, axis=1)
            if event == 0 or event == 3 or event == 4:
                value -= tl.load(
                    same_all + ((pair * 3 + left_mode) * B + b) * H + h,
                    mask=h_mask,
                    other=0.0,
                )
            out_base = ((pair * 5 + event) * B + b) * H
            tl.store(pair_event_out + out_base + h, value, mask=h_mask)

    @triton.jit
    def _p4_pair_event_from_prefix_finish_kernel(
        card_prefix,
        local_belief_matrix,
        local_c0,
        local_c1,
        pair_p_rank_flags,
        pair_q_rank_flags,
        lower_end,
        tie_end,
        group_count,
        same_all,
        pair_event_out,
        B: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        pair = tl.program_id(1)
        h_block = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        valid_card = card < CARD_COUNT
        hc_mask = h_mask[:, None] & valid_card[None, :]
        left = tl.where(pair < 3, 0, tl.where(pair < 5, 1, 2))
        right = tl.where(pair < 3, pair + 1, tl.where(pair < 5, pair - 1, 3))

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
        tie = tl.load(tie_end + b * H + h, mask=h_mask, other=0)
        total = tl.load(group_count + b)

        pair_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        valid_p = hc_mask & (card[None, :] != c0[:, None])
        valid_q = hc_mask & (card[None, :] != c1[:, None])
        flags_p = tl.load(pair_p_rank_flags + pair_base, mask=hc_mask, other=0)
        flags_q = tl.load(pair_q_rank_flags + pair_base, mask=hc_mask, other=0)
        matrix_left_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * CARD_COUNT
        matrix_right_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * CARD_COUNT

        left_prefix_base = ((b * 4 + left) * H1) * CARD_COUNT
        right_prefix_base = ((b * 4 + right) * H1) * CARD_COUNT
        left_lower_raw = tl.load(
            card_prefix + left_prefix_base + lower[:, None] * CARD_COUNT + card[None, :],
            mask=hc_mask,
            other=0.0,
        )
        right_lower_raw = tl.load(
            card_prefix + right_prefix_base + lower[:, None] * CARD_COUNT + card[None, :],
            mask=hc_mask,
            other=0.0,
        )
        left_tie_end = tl.load(
            card_prefix + left_prefix_base + tie[:, None] * CARD_COUNT + card[None, :],
            mask=hc_mask,
            other=0.0,
        )
        right_tie_end = tl.load(
            card_prefix + right_prefix_base + tie[:, None] * CARD_COUNT + card[None, :],
            mask=hc_mask,
            other=0.0,
        )
        left_total_raw = tl.load(
            card_prefix + left_prefix_base + total * CARD_COUNT + card[None, :],
            mask=hc_mask,
            other=0.0,
        )
        right_total_raw = tl.load(
            card_prefix + right_prefix_base + total * CARD_COUNT + card[None, :],
            mask=hc_mask,
            other=0.0,
        )
        left_tie_raw = left_tie_end - left_lower_raw
        right_tie_raw = right_tie_end - right_lower_raw

        row_lower_p = valid_p & ((flags_p & 1) != 0)
        row_lower_q = valid_q & ((flags_q & 1) != 0)
        row_tie_p = valid_p & ((flags_p & 2) != 0)
        row_tie_q = valid_q & ((flags_q & 2) != 0)

        left_lower_corr = tl.load(
            local_belief_matrix + matrix_left_base + c0[:, None],
            mask=row_lower_p,
            other=0.0,
        )
        left_lower_corr += tl.load(
            local_belief_matrix + matrix_left_base + c1[:, None],
            mask=row_lower_q,
            other=0.0,
        )
        right_lower_corr = tl.load(
            local_belief_matrix + matrix_right_base + c0[:, None],
            mask=row_lower_p,
            other=0.0,
        )
        right_lower_corr += tl.load(
            local_belief_matrix + matrix_right_base + c1[:, None],
            mask=row_lower_q,
            other=0.0,
        )
        left_tie_corr = tl.load(
            local_belief_matrix + matrix_left_base + c0[:, None],
            mask=row_tie_p,
            other=0.0,
        )
        left_tie_corr += tl.load(
            local_belief_matrix + matrix_left_base + c1[:, None],
            mask=row_tie_q,
            other=0.0,
        )
        right_tie_corr = tl.load(
            local_belief_matrix + matrix_right_base + c0[:, None],
            mask=row_tie_p,
            other=0.0,
        )
        right_tie_corr += tl.load(
            local_belief_matrix + matrix_right_base + c1[:, None],
            mask=row_tie_q,
            other=0.0,
        )
        left_total_corr = tl.load(
            local_belief_matrix + matrix_left_base + c0[:, None],
            mask=valid_p,
            other=0.0,
        )
        left_total_corr += tl.load(
            local_belief_matrix + matrix_left_base + c1[:, None],
            mask=valid_q,
            other=0.0,
        )
        right_total_corr = tl.load(
            local_belief_matrix + matrix_right_base + c0[:, None],
            mask=valid_p,
            other=0.0,
        )
        right_total_corr += tl.load(
            local_belief_matrix + matrix_right_base + c1[:, None],
            mask=valid_q,
            other=0.0,
        )
        blocked = (card[None, :] == c0[:, None]) | (card[None, :] == c1[:, None])
        left_lower = tl.where(blocked, 0.0, left_lower_raw - left_lower_corr)
        right_lower = tl.where(blocked, 0.0, right_lower_raw - right_lower_corr)
        left_tie = tl.where(blocked, 0.0, left_tie_raw - left_tie_corr)
        right_tie = tl.where(blocked, 0.0, right_tie_raw - right_tie_corr)
        left_total = tl.where(blocked, 0.0, left_total_raw - left_total_corr)
        right_total = tl.where(blocked, 0.0, right_total_raw - right_total_corr)

        same0 = tl.load(same_all + ((pair * 3 + 0) * B + b) * H + h, mask=h_mask, other=0.0)
        same1 = tl.load(same_all + ((pair * 3 + 1) * B + b) * H + h, mask=h_mask, other=0.0)
        same2 = tl.load(same_all + ((pair * 3 + 2) * B + b) * H + h, mask=h_mask, other=0.0)
        value0 = tl.sum(left_lower * right_lower, axis=1) - same0
        value1 = tl.sum(left_lower * right_tie, axis=1)
        value2 = tl.sum(left_tie * right_lower, axis=1)
        value3 = tl.sum(left_tie * right_tie, axis=1) - same1
        value4 = tl.sum(left_total * right_total, axis=1) - same2
        out_base = (pair * 5 * B + b) * H
        tl.store(pair_event_out + out_base + h, value0, mask=h_mask)
        tl.store(pair_event_out + out_base + B * H + h, value1, mask=h_mask)
        tl.store(pair_event_out + out_base + 2 * B * H + h, value2, mask=h_mask)
        tl.store(pair_event_out + out_base + 3 * B * H + h, value3, mask=h_mask)
        tl.store(pair_event_out + out_base + 4 * B * H + h, value4, mask=h_mask)

    @triton.jit
    def _p4_pair_event_from_sparse_finish_kernel(
        player_card_cumsum,
        local_belief_matrix,
        local_c0,
        local_c1,
        pair_p_rank_flags,
        pair_q_rank_flags,
        slot_lower_by_card,
        slot_tie_by_card,
        same_all,
        pair_event_out,
        B: tl.constexpr,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        TOTAL_SLOT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        COMPUTE_TOTAL: tl.constexpr,
    ):
        b = tl.program_id(0)
        pair = tl.program_id(1)
        h_block = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        valid_card = card < CARD_COUNT
        hc_mask = h_mask[:, None] & valid_card[None, :]
        left = tl.where(pair < 3, 0, tl.where(pair < 5, 1, 2))
        right = tl.where(pair < 3, pair + 1, tl.where(pair < 5, pair - 1, 3))

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        slot_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        slot_lower = tl.load(slot_lower_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
        slot_tie = tl.load(slot_tie_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)

        pair_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        valid_p = hc_mask & (card[None, :] != c0[:, None])
        valid_q = hc_mask & (card[None, :] != c1[:, None])
        flags_p = tl.load(pair_p_rank_flags + pair_base, mask=hc_mask, other=0)
        flags_q = tl.load(pair_q_rank_flags + pair_base, mask=hc_mask, other=0)
        matrix_left_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * CARD_COUNT
        matrix_right_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * CARD_COUNT
        left_card_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * SLOT_CAP
        right_card_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * SLOT_CAP

        idx_lower = tl.maximum(slot_lower - 1, 0)
        idx_tie = tl.maximum(slot_tie - 1, 0)
        idx_total = TOTAL_SLOT - 1
        left_lower_raw = tl.load(
            player_card_cumsum + left_card_base + idx_lower,
            mask=hc_mask & (slot_lower > 0),
            other=0.0,
        )
        right_lower_raw = tl.load(
            player_card_cumsum + right_card_base + idx_lower,
            mask=hc_mask & (slot_lower > 0),
            other=0.0,
        )
        left_tie_end = tl.load(
            player_card_cumsum + left_card_base + idx_tie,
            mask=hc_mask & (slot_tie > 0),
            other=0.0,
        )
        right_tie_end = tl.load(
            player_card_cumsum + right_card_base + idx_tie,
            mask=hc_mask & (slot_tie > 0),
            other=0.0,
        )
        left_tie_raw = left_tie_end - left_lower_raw
        right_tie_raw = right_tie_end - right_lower_raw

        row_lower_p = valid_p & ((flags_p & 1) != 0)
        row_lower_q = valid_q & ((flags_q & 1) != 0)
        row_tie_p = valid_p & ((flags_p & 2) != 0)
        row_tie_q = valid_q & ((flags_q & 2) != 0)

        left_lower_corr = tl.load(
            local_belief_matrix + matrix_left_base + c0[:, None],
            mask=row_lower_p,
            other=0.0,
        )
        left_lower_corr += tl.load(
            local_belief_matrix + matrix_left_base + c1[:, None],
            mask=row_lower_q,
            other=0.0,
        )
        right_lower_corr = tl.load(
            local_belief_matrix + matrix_right_base + c0[:, None],
            mask=row_lower_p,
            other=0.0,
        )
        right_lower_corr += tl.load(
            local_belief_matrix + matrix_right_base + c1[:, None],
            mask=row_lower_q,
            other=0.0,
        )
        left_tie_corr = tl.load(
            local_belief_matrix + matrix_left_base + c0[:, None],
            mask=row_tie_p,
            other=0.0,
        )
        left_tie_corr += tl.load(
            local_belief_matrix + matrix_left_base + c1[:, None],
            mask=row_tie_q,
            other=0.0,
        )
        right_tie_corr = tl.load(
            local_belief_matrix + matrix_right_base + c0[:, None],
            mask=row_tie_p,
            other=0.0,
        )
        right_tie_corr += tl.load(
            local_belief_matrix + matrix_right_base + c1[:, None],
            mask=row_tie_q,
            other=0.0,
        )
        blocked = (card[None, :] == c0[:, None]) | (card[None, :] == c1[:, None])
        left_lower = tl.where(blocked, 0.0, left_lower_raw - left_lower_corr)
        right_lower = tl.where(blocked, 0.0, right_lower_raw - right_lower_corr)
        left_tie = tl.where(blocked, 0.0, left_tie_raw - left_tie_corr)
        right_tie = tl.where(blocked, 0.0, right_tie_raw - right_tie_corr)

        same0 = tl.load(same_all + ((pair * 3 + 0) * B + b) * H + h, mask=h_mask, other=0.0)
        same1 = tl.load(same_all + ((pair * 3 + 1) * B + b) * H + h, mask=h_mask, other=0.0)
        value0 = tl.sum(left_lower * right_lower, axis=1) - same0
        value1 = tl.sum(left_lower * right_tie, axis=1)
        value2 = tl.sum(left_tie * right_lower, axis=1)
        value3 = tl.sum(left_tie * right_tie, axis=1) - same1
        out_base = (pair * 5 * B + b) * H
        tl.store(pair_event_out + out_base + h, value0, mask=h_mask)
        tl.store(pair_event_out + out_base + B * H + h, value1, mask=h_mask)
        tl.store(pair_event_out + out_base + 2 * B * H + h, value2, mask=h_mask)
        tl.store(pair_event_out + out_base + 3 * B * H + h, value3, mask=h_mask)
        if COMPUTE_TOTAL:
            left_total_raw = tl.load(
                player_card_cumsum + left_card_base + idx_total,
                mask=hc_mask,
                other=0.0,
            )
            right_total_raw = tl.load(
                player_card_cumsum + right_card_base + idx_total,
                mask=hc_mask,
                other=0.0,
            )
            left_total_corr = tl.load(
                local_belief_matrix + matrix_left_base + c0[:, None],
                mask=valid_p,
                other=0.0,
            )
            left_total_corr += tl.load(
                local_belief_matrix + matrix_left_base + c1[:, None],
                mask=valid_q,
                other=0.0,
            )
            right_total_corr = tl.load(
                local_belief_matrix + matrix_right_base + c0[:, None],
                mask=valid_p,
                other=0.0,
            )
            right_total_corr += tl.load(
                local_belief_matrix + matrix_right_base + c1[:, None],
                mask=valid_q,
                other=0.0,
            )
            left_total = tl.where(blocked, 0.0, left_total_raw - left_total_corr)
            right_total = tl.where(blocked, 0.0, right_total_raw - right_total_corr)
            same2 = tl.load(
                same_all + ((pair * 3 + 2) * B + b) * H + h,
                mask=h_mask,
                other=0.0,
            )
            value4 = tl.sum(left_total * right_total, axis=1) - same2
            tl.store(pair_event_out + out_base + 4 * B * H + h, value4, mask=h_mask)

    @triton.jit
    def _tier2_direct_load_player_modes(
        player_card_cumsum,
        local_belief_matrix,
        b,
        card,
        idx_lower,
        idx_tie,
        idx_total,
        hc_mask,
        slot_lower,
        slot_tie,
        c0,
        c1,
        row_lower_p,
        row_lower_q,
        row_tie_p,
        row_tie_q,
        valid_p,
        valid_q,
        blocked,
        player: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
    ):
        card_base = ((b * 4 + player) * CARD_COUNT + card[None, :]) * SLOT_CAP
        lower_raw = tl.load(
            player_card_cumsum + card_base + idx_lower,
            mask=hc_mask & (slot_lower > 0),
            other=0.0,
        )
        tie_end = tl.load(
            player_card_cumsum + card_base + idx_tie,
            mask=hc_mask & (slot_tie > 0),
            other=0.0,
        )
        total_raw = tl.load(
            player_card_cumsum + card_base + idx_total,
            mask=hc_mask,
            other=0.0,
        )
        tie_raw = tie_end - lower_raw

        matrix_base = ((b * 4 + player) * CARD_COUNT + card[None, :]) * CARD_COUNT
        lower_corr = tl.load(
            local_belief_matrix + matrix_base + c0[:, None],
            mask=row_lower_p,
            other=0.0,
        )
        lower_corr += tl.load(
            local_belief_matrix + matrix_base + c1[:, None],
            mask=row_lower_q,
            other=0.0,
        )
        tie_corr = tl.load(
            local_belief_matrix + matrix_base + c0[:, None],
            mask=row_tie_p,
            other=0.0,
        )
        tie_corr += tl.load(
            local_belief_matrix + matrix_base + c1[:, None],
            mask=row_tie_q,
            other=0.0,
        )
        total_corr = tl.load(
            local_belief_matrix + matrix_base + c0[:, None],
            mask=valid_p,
            other=0.0,
        )
        total_corr += tl.load(
            local_belief_matrix + matrix_base + c1[:, None],
            mask=valid_q,
            other=0.0,
        )
        lower = tl.where(blocked, 0.0, lower_raw - lower_corr)
        tie = tl.where(blocked, 0.0, tie_raw - tie_corr)
        total = tl.where(blocked, 0.0, total_raw - total_corr)
        return lower, tie, total

    @triton.jit
    def _tier2_direct_pair_terms(
        left_lower,
        left_tie,
        left_total,
        right_lower,
        right_tie,
        right_total,
        same_all,
        b,
        h,
        h_mask,
        pair: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
    ):
        same0 = tl.load(
            same_all + ((pair * 3 + 0) * B + b) * H + h,
            mask=h_mask,
            other=0.0,
        )
        same1 = tl.load(
            same_all + ((pair * 3 + 1) * B + b) * H + h,
            mask=h_mask,
            other=0.0,
        )
        same2 = tl.load(
            same_all + ((pair * 3 + 2) * B + b) * H + h,
            mask=h_mask,
            other=0.0,
        )
        pair00 = tl.sum(left_lower * right_lower, axis=1) - same0
        pair01 = tl.sum(left_lower * right_tie, axis=1)
        pair10 = tl.sum(left_tie * right_lower, axis=1)
        pair11 = tl.sum(left_tie * right_tie, axis=1) - same1
        pair_total = tl.sum(left_total * right_total, axis=1) - same2
        pair10_plus_01 = pair10 + pair01
        other0 = pair00 + 0.5 * pair10_plus_01 + (1.0 / 3.0) * pair11
        other1 = 0.5 * pair00 + (1.0 / 3.0) * pair10_plus_01 + 0.25 * pair11
        return other0, other1, pair_total

    @triton.jit(
        do_not_specialize_on_alignment=[
            "scalar_all",
            "player_card_cumsum",
            "local_belief_matrix",
            "local_c0",
            "local_c1",
            "pair_p_rank_flags",
            "pair_q_rank_flags",
            "pair_rank_flags",
            "slot_lower_by_card",
            "slot_tie_by_card",
            "slot_lower_tie_by_card",
            "same_all",
            "numerator_out",
            "denominator_out",
            "equity_out",
        ],
    )
    def _tier2_p4_sparse_direct_finish_plain_kernel(
        scalar_all,
        player_card_cumsum,
        local_belief_matrix,
        local_c0,
        local_c1,
        pair_p_rank_flags,
        pair_q_rank_flags,
        pair_rank_flags,
        slot_lower_by_card,
        slot_tie_by_card,
        slot_lower_tie_by_card,
        same_all,
        numerator_out,
        denominator_out,
        equity_out,
        B: tl.constexpr,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        TOTAL_SLOT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        USE_COMPACT_LUT: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        valid_card = card < CARD_COUNT
        hc_mask = h_mask[:, None] & valid_card[None, :]

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        slot_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        if USE_COMPACT_LUT:
            slot_lut = tl.load(slot_lower_tie_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
            slot_lower = slot_lut & 63
            slot_tie = slot_lut >> 6
        else:
            slot_lower = tl.load(slot_lower_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
            slot_tie = tl.load(slot_tie_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
        idx_lower = tl.maximum(slot_lower - 1, 0)
        idx_tie = tl.maximum(slot_tie - 1, 0)
        idx_total = TOTAL_SLOT - 1

        valid_p = hc_mask & (card[None, :] != c0[:, None])
        valid_q = hc_mask & (card[None, :] != c1[:, None])
        if USE_COMPACT_LUT:
            flags = tl.load(pair_rank_flags + slot_base, mask=hc_mask, other=0)
            row_lower_p = valid_p & ((flags & 1) != 0)
            row_tie_p = valid_p & ((flags & 2) != 0)
            row_lower_q = valid_q & ((flags & 4) != 0)
            row_tie_q = valid_q & ((flags & 8) != 0)
        else:
            flags_p = tl.load(pair_p_rank_flags + slot_base, mask=hc_mask, other=0)
            flags_q = tl.load(pair_q_rank_flags + slot_base, mask=hc_mask, other=0)
            row_lower_p = valid_p & ((flags_p & 1) != 0)
            row_lower_q = valid_q & ((flags_q & 1) != 0)
            row_tie_p = valid_p & ((flags_p & 2) != 0)
            row_tie_q = valid_q & ((flags_q & 2) != 0)
        blocked = (card[None, :] == c0[:, None]) | (card[None, :] == c1[:, None])

        base0 = ((0 * 3) * B + b) * H + h
        base1 = ((1 * 3) * B + b) * H + h
        base2 = ((2 * 3) * B + b) * H + h
        base3 = ((3 * 3) * B + b) * H + h
        l0 = tl.load(scalar_all + base0, mask=h_mask, other=0.0)
        t0 = tl.load(scalar_all + base0 + B * H, mask=h_mask, other=0.0)
        d0 = tl.load(scalar_all + base0 + 2 * B * H, mask=h_mask, other=0.0)
        l1 = tl.load(scalar_all + base1, mask=h_mask, other=0.0)
        t1 = tl.load(scalar_all + base1 + B * H, mask=h_mask, other=0.0)
        d1 = tl.load(scalar_all + base1 + 2 * B * H, mask=h_mask, other=0.0)
        l2 = tl.load(scalar_all + base2, mask=h_mask, other=0.0)
        t2 = tl.load(scalar_all + base2 + B * H, mask=h_mask, other=0.0)
        d2 = tl.load(scalar_all + base2 + 2 * B * H, mask=h_mask, other=0.0)
        l3 = tl.load(scalar_all + base3, mask=h_mask, other=0.0)
        t3 = tl.load(scalar_all + base3 + B * H, mask=h_mask, other=0.0)
        d3 = tl.load(scalar_all + base3 + 2 * B * H, mask=h_mask, other=0.0)

        num0 = (
            l1 * l2 * l3
            + 0.5 * (t1 * l2 * l3 + l1 * t2 * l3 + l1 * l2 * t3)
            + (1.0 / 3.0) * (t1 * t2 * l3 + t1 * l2 * t3 + l1 * t2 * t3)
            + 0.25 * t1 * t2 * t3
        )
        den0 = d1 * d2 * d3
        num1 = (
            l0 * l2 * l3
            + 0.5 * (t0 * l2 * l3 + l0 * t2 * l3 + l0 * l2 * t3)
            + (1.0 / 3.0) * (t0 * t2 * l3 + t0 * l2 * t3 + l0 * t2 * t3)
            + 0.25 * t0 * t2 * t3
        )
        den1 = d0 * d2 * d3
        num2 = (
            l0 * l1 * l3
            + 0.5 * (t0 * l1 * l3 + l0 * t1 * l3 + l0 * l1 * t3)
            + (1.0 / 3.0) * (t0 * t1 * l3 + t0 * l1 * t3 + l0 * t1 * t3)
            + 0.25 * t0 * t1 * t3
        )
        den2 = d0 * d1 * d3
        num3 = (
            l0 * l1 * l2
            + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
            + (1.0 / 3.0) * (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2)
            + 0.25 * t0 * t1 * t2
        )
        den3 = d0 * d1 * d2

        for pair in tl.static_range(0, 6):
            if pair == 0:
                left = 0
                right = 1
            elif pair == 1:
                left = 0
                right = 2
            elif pair == 2:
                left = 0
                right = 3
            elif pair == 3:
                left = 1
                right = 2
            elif pair == 4:
                left = 1
                right = 3
            else:
                left = 2
                right = 3

            left_card_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * SLOT_CAP
            right_card_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * SLOT_CAP
            left_lower_raw = tl.load(
                player_card_cumsum + left_card_base + idx_lower,
                mask=hc_mask & (slot_lower > 0),
                other=0.0,
            )
            right_lower_raw = tl.load(
                player_card_cumsum + right_card_base + idx_lower,
                mask=hc_mask & (slot_lower > 0),
                other=0.0,
            )
            left_tie_end = tl.load(
                player_card_cumsum + left_card_base + idx_tie,
                mask=hc_mask & (slot_tie > 0),
                other=0.0,
            )
            right_tie_end = tl.load(
                player_card_cumsum + right_card_base + idx_tie,
                mask=hc_mask & (slot_tie > 0),
                other=0.0,
            )
            left_total_raw = tl.load(
                player_card_cumsum + left_card_base + idx_total,
                mask=hc_mask,
                other=0.0,
            )
            right_total_raw = tl.load(
                player_card_cumsum + right_card_base + idx_total,
                mask=hc_mask,
                other=0.0,
            )
            left_tie_raw = left_tie_end - left_lower_raw
            right_tie_raw = right_tie_end - right_lower_raw

            matrix_left_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * CARD_COUNT
            matrix_right_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * CARD_COUNT
            left_lower_corr = tl.load(
                local_belief_matrix + matrix_left_base + c0[:, None],
                mask=row_lower_p,
                other=0.0,
            )
            left_lower_corr += tl.load(
                local_belief_matrix + matrix_left_base + c1[:, None],
                mask=row_lower_q,
                other=0.0,
            )
            right_lower_corr = tl.load(
                local_belief_matrix + matrix_right_base + c0[:, None],
                mask=row_lower_p,
                other=0.0,
            )
            right_lower_corr += tl.load(
                local_belief_matrix + matrix_right_base + c1[:, None],
                mask=row_lower_q,
                other=0.0,
            )
            left_tie_corr = tl.load(
                local_belief_matrix + matrix_left_base + c0[:, None],
                mask=row_tie_p,
                other=0.0,
            )
            left_tie_corr += tl.load(
                local_belief_matrix + matrix_left_base + c1[:, None],
                mask=row_tie_q,
                other=0.0,
            )
            right_tie_corr = tl.load(
                local_belief_matrix + matrix_right_base + c0[:, None],
                mask=row_tie_p,
                other=0.0,
            )
            right_tie_corr += tl.load(
                local_belief_matrix + matrix_right_base + c1[:, None],
                mask=row_tie_q,
                other=0.0,
            )
            left_total_corr = tl.load(
                local_belief_matrix + matrix_left_base + c0[:, None],
                mask=valid_p,
                other=0.0,
            )
            left_total_corr += tl.load(
                local_belief_matrix + matrix_left_base + c1[:, None],
                mask=valid_q,
                other=0.0,
            )
            right_total_corr = tl.load(
                local_belief_matrix + matrix_right_base + c0[:, None],
                mask=valid_p,
                other=0.0,
            )
            right_total_corr += tl.load(
                local_belief_matrix + matrix_right_base + c1[:, None],
                mask=valid_q,
                other=0.0,
            )

            left_lower = tl.where(blocked, 0.0, left_lower_raw - left_lower_corr)
            right_lower = tl.where(blocked, 0.0, right_lower_raw - right_lower_corr)
            left_tie = tl.where(blocked, 0.0, left_tie_raw - left_tie_corr)
            right_tie = tl.where(blocked, 0.0, right_tie_raw - right_tie_corr)
            left_total = tl.where(blocked, 0.0, left_total_raw - left_total_corr)
            right_total = tl.where(blocked, 0.0, right_total_raw - right_total_corr)

            same0 = tl.load(
                same_all + ((pair * 3 + 0) * B + b) * H + h,
                mask=h_mask,
                other=0.0,
            )
            same1 = tl.load(
                same_all + ((pair * 3 + 1) * B + b) * H + h,
                mask=h_mask,
                other=0.0,
            )
            same2 = tl.load(
                same_all + ((pair * 3 + 2) * B + b) * H + h,
                mask=h_mask,
                other=0.0,
            )
            pair00 = tl.sum(left_lower * right_lower, axis=1) - same0
            pair01 = tl.sum(left_lower * right_tie, axis=1)
            pair10 = tl.sum(left_tie * right_lower, axis=1)
            pair11 = tl.sum(left_tie * right_tie, axis=1) - same1
            pair_total = tl.sum(left_total * right_total, axis=1) - same2
            pair10_plus_01 = pair10 + pair01
            other0 = pair00 + 0.5 * pair10_plus_01 + (1.0 / 3.0) * pair11
            other1 = 0.5 * pair00 + (1.0 / 3.0) * pair10_plus_01 + 0.25 * pair11

            if pair == 0:
                num2 -= other0 * l3 + other1 * t3
                den2 -= pair_total * d3
                num3 -= other0 * l2 + other1 * t2
                den3 -= pair_total * d2
            elif pair == 1:
                num1 -= other0 * l3 + other1 * t3
                den1 -= pair_total * d3
                num3 -= other0 * l1 + other1 * t1
                den3 -= pair_total * d1
            elif pair == 2:
                num1 -= other0 * l2 + other1 * t2
                den1 -= pair_total * d2
                num2 -= other0 * l1 + other1 * t1
                den2 -= pair_total * d1
            elif pair == 3:
                num0 -= other0 * l3 + other1 * t3
                den0 -= pair_total * d3
                num3 -= other0 * l0 + other1 * t0
                den3 -= pair_total * d0
            elif pair == 4:
                num0 -= other0 * l2 + other1 * t2
                den0 -= pair_total * d2
                num2 -= other0 * l0 + other1 * t0
                den2 -= pair_total * d0
            else:
                num0 -= other0 * l1 + other1 * t1
                den0 -= pair_total * d1
                num1 -= other0 * l0 + other1 * t0
                den1 -= pair_total * d0

        out_base = (b * 4) * H + h
        tl.store(numerator_out + out_base, num0, mask=h_mask)
        tl.store(denominator_out + out_base, den0, mask=h_mask)
        tl.store(equity_out + out_base, tl.where(den0 > 0.0, num0 / tl.maximum(den0, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + H, num1, mask=h_mask)
        tl.store(denominator_out + out_base + H, den1, mask=h_mask)
        tl.store(equity_out + out_base + H, tl.where(den1 > 0.0, num1 / tl.maximum(den1, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 2 * H, num2, mask=h_mask)
        tl.store(denominator_out + out_base + 2 * H, den2, mask=h_mask)
        tl.store(equity_out + out_base + 2 * H, tl.where(den2 > 0.0, num2 / tl.maximum(den2, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 3 * H, num3, mask=h_mask)
        tl.store(denominator_out + out_base + 3 * H, den3, mask=h_mask)
        tl.store(equity_out + out_base + 3 * H, tl.where(den3 > 0.0, num3 / tl.maximum(den3, 1.0e-30), 0.0), mask=h_mask)

    @triton.jit(
        do_not_specialize_on_alignment=[
            "scalar_all",
            "player_card_cumsum",
            "local_belief_matrix",
            "local_c0",
            "local_c1",
            "pair_p_rank_flags",
            "pair_q_rank_flags",
            "pair_rank_flags",
            "slot_lower_by_card",
            "slot_tie_by_card",
            "slot_lower_tie_by_card",
            "same_all",
            "numerator_out",
            "denominator_out",
            "equity_out",
        ],
    )
    def _tier2_p4_sparse_direct_finish_reuse_kernel(
        scalar_all,
        player_card_cumsum,
        local_belief_matrix,
        local_c0,
        local_c1,
        pair_p_rank_flags,
        pair_q_rank_flags,
        pair_rank_flags,
        slot_lower_by_card,
        slot_tie_by_card,
        slot_lower_tie_by_card,
        same_all,
        numerator_out,
        denominator_out,
        equity_out,
        B: tl.constexpr,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        TOTAL_SLOT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        USE_COMPACT_LUT: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        valid_card = card < CARD_COUNT
        hc_mask = h_mask[:, None] & valid_card[None, :]

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        slot_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        if USE_COMPACT_LUT:
            slot_lut = tl.load(slot_lower_tie_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
            slot_lower = slot_lut & 63
            slot_tie = slot_lut >> 6
        else:
            slot_lower = tl.load(slot_lower_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
            slot_tie = tl.load(slot_tie_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
        idx_lower = tl.maximum(slot_lower - 1, 0)
        idx_tie = tl.maximum(slot_tie - 1, 0)
        idx_total = TOTAL_SLOT - 1

        valid_p = hc_mask & (card[None, :] != c0[:, None])
        valid_q = hc_mask & (card[None, :] != c1[:, None])
        if USE_COMPACT_LUT:
            flags = tl.load(pair_rank_flags + slot_base, mask=hc_mask, other=0)
            row_lower_p = valid_p & ((flags & 1) != 0)
            row_tie_p = valid_p & ((flags & 2) != 0)
            row_lower_q = valid_q & ((flags & 4) != 0)
            row_tie_q = valid_q & ((flags & 8) != 0)
        else:
            flags_p = tl.load(pair_p_rank_flags + slot_base, mask=hc_mask, other=0)
            flags_q = tl.load(pair_q_rank_flags + slot_base, mask=hc_mask, other=0)
            row_lower_p = valid_p & ((flags_p & 1) != 0)
            row_lower_q = valid_q & ((flags_q & 1) != 0)
            row_tie_p = valid_p & ((flags_p & 2) != 0)
            row_tie_q = valid_q & ((flags_q & 2) != 0)
        blocked = (card[None, :] == c0[:, None]) | (card[None, :] == c1[:, None])

        base0 = ((0 * 3) * B + b) * H + h
        base1 = ((1 * 3) * B + b) * H + h
        base2 = ((2 * 3) * B + b) * H + h
        base3 = ((3 * 3) * B + b) * H + h
        l0 = tl.load(scalar_all + base0, mask=h_mask, other=0.0)
        t0 = tl.load(scalar_all + base0 + B * H, mask=h_mask, other=0.0)
        d0 = tl.load(scalar_all + base0 + 2 * B * H, mask=h_mask, other=0.0)
        l1 = tl.load(scalar_all + base1, mask=h_mask, other=0.0)
        t1 = tl.load(scalar_all + base1 + B * H, mask=h_mask, other=0.0)
        d1 = tl.load(scalar_all + base1 + 2 * B * H, mask=h_mask, other=0.0)
        l2 = tl.load(scalar_all + base2, mask=h_mask, other=0.0)
        t2 = tl.load(scalar_all + base2 + B * H, mask=h_mask, other=0.0)
        d2 = tl.load(scalar_all + base2 + 2 * B * H, mask=h_mask, other=0.0)
        l3 = tl.load(scalar_all + base3, mask=h_mask, other=0.0)
        t3 = tl.load(scalar_all + base3 + B * H, mask=h_mask, other=0.0)
        d3 = tl.load(scalar_all + base3 + 2 * B * H, mask=h_mask, other=0.0)

        num0 = (
            l1 * l2 * l3
            + 0.5 * (t1 * l2 * l3 + l1 * t2 * l3 + l1 * l2 * t3)
            + (1.0 / 3.0) * (t1 * t2 * l3 + t1 * l2 * t3 + l1 * t2 * t3)
            + 0.25 * t1 * t2 * t3
        )
        den0 = d1 * d2 * d3
        num1 = (
            l0 * l2 * l3
            + 0.5 * (t0 * l2 * l3 + l0 * t2 * l3 + l0 * l2 * t3)
            + (1.0 / 3.0) * (t0 * t2 * l3 + t0 * l2 * t3 + l0 * t2 * t3)
            + 0.25 * t0 * t2 * t3
        )
        den1 = d0 * d2 * d3
        num2 = (
            l0 * l1 * l3
            + 0.5 * (t0 * l1 * l3 + l0 * t1 * l3 + l0 * l1 * t3)
            + (1.0 / 3.0) * (t0 * t1 * l3 + t0 * l1 * t3 + l0 * t1 * t3)
            + 0.25 * t0 * t1 * t3
        )
        den2 = d0 * d1 * d3
        num3 = (
            l0 * l1 * l2
            + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
            + (1.0 / 3.0) * (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2)
            + 0.25 * t0 * t1 * t2
        )
        den3 = d0 * d1 * d2

        p0_lower, p0_tie, p0_total = _tier2_direct_load_player_modes(
            player_card_cumsum,
            local_belief_matrix,
            b,
            card,
            idx_lower,
            idx_tie,
            idx_total,
            hc_mask,
            slot_lower,
            slot_tie,
            c0,
            c1,
            row_lower_p,
            row_lower_q,
            row_tie_p,
            row_tie_q,
            valid_p,
            valid_q,
            blocked,
            player=0,
            CARD_COUNT=CARD_COUNT,
            SLOT_CAP=SLOT_CAP,
        )
        p1_lower, p1_tie, p1_total = _tier2_direct_load_player_modes(
            player_card_cumsum,
            local_belief_matrix,
            b,
            card,
            idx_lower,
            idx_tie,
            idx_total,
            hc_mask,
            slot_lower,
            slot_tie,
            c0,
            c1,
            row_lower_p,
            row_lower_q,
            row_tie_p,
            row_tie_q,
            valid_p,
            valid_q,
            blocked,
            player=1,
            CARD_COUNT=CARD_COUNT,
            SLOT_CAP=SLOT_CAP,
        )
        other0, other1, pair_total = _tier2_direct_pair_terms(
            p0_lower, p0_tie, p0_total, p1_lower, p1_tie, p1_total, same_all, b, h, h_mask,
            pair=0, B=B, H=H,
        )
        num2 -= other0 * l3 + other1 * t3
        den2 -= pair_total * d3
        num3 -= other0 * l2 + other1 * t2
        den3 -= pair_total * d2

        p2_lower, p2_tie, p2_total = _tier2_direct_load_player_modes(
            player_card_cumsum,
            local_belief_matrix,
            b,
            card,
            idx_lower,
            idx_tie,
            idx_total,
            hc_mask,
            slot_lower,
            slot_tie,
            c0,
            c1,
            row_lower_p,
            row_lower_q,
            row_tie_p,
            row_tie_q,
            valid_p,
            valid_q,
            blocked,
            player=2,
            CARD_COUNT=CARD_COUNT,
            SLOT_CAP=SLOT_CAP,
        )
        other0, other1, pair_total = _tier2_direct_pair_terms(
            p0_lower, p0_tie, p0_total, p2_lower, p2_tie, p2_total, same_all, b, h, h_mask,
            pair=1, B=B, H=H,
        )
        num1 -= other0 * l3 + other1 * t3
        den1 -= pair_total * d3
        num3 -= other0 * l1 + other1 * t1
        den3 -= pair_total * d1

        p3_lower, p3_tie, p3_total = _tier2_direct_load_player_modes(
            player_card_cumsum,
            local_belief_matrix,
            b,
            card,
            idx_lower,
            idx_tie,
            idx_total,
            hc_mask,
            slot_lower,
            slot_tie,
            c0,
            c1,
            row_lower_p,
            row_lower_q,
            row_tie_p,
            row_tie_q,
            valid_p,
            valid_q,
            blocked,
            player=3,
            CARD_COUNT=CARD_COUNT,
            SLOT_CAP=SLOT_CAP,
        )
        other0, other1, pair_total = _tier2_direct_pair_terms(
            p0_lower, p0_tie, p0_total, p3_lower, p3_tie, p3_total, same_all, b, h, h_mask,
            pair=2, B=B, H=H,
        )
        num1 -= other0 * l2 + other1 * t2
        den1 -= pair_total * d2
        num2 -= other0 * l1 + other1 * t1
        den2 -= pair_total * d1

        other0, other1, pair_total = _tier2_direct_pair_terms(
            p1_lower, p1_tie, p1_total, p2_lower, p2_tie, p2_total, same_all, b, h, h_mask,
            pair=3, B=B, H=H,
        )
        num0 -= other0 * l3 + other1 * t3
        den0 -= pair_total * d3
        num3 -= other0 * l0 + other1 * t0
        den3 -= pair_total * d0

        other0, other1, pair_total = _tier2_direct_pair_terms(
            p1_lower, p1_tie, p1_total, p3_lower, p3_tie, p3_total, same_all, b, h, h_mask,
            pair=4, B=B, H=H,
        )
        num0 -= other0 * l2 + other1 * t2
        den0 -= pair_total * d2
        num2 -= other0 * l0 + other1 * t0
        den2 -= pair_total * d0

        other0, other1, pair_total = _tier2_direct_pair_terms(
            p2_lower, p2_tie, p2_total, p3_lower, p3_tie, p3_total, same_all, b, h, h_mask,
            pair=5, B=B, H=H,
        )
        num0 -= other0 * l1 + other1 * t1
        den0 -= pair_total * d1
        num1 -= other0 * l0 + other1 * t0
        den1 -= pair_total * d0

        out_base = (b * 4) * H + h
        tl.store(numerator_out + out_base, num0, mask=h_mask)
        tl.store(denominator_out + out_base, den0, mask=h_mask)
        tl.store(equity_out + out_base, tl.where(den0 > 0.0, num0 / tl.maximum(den0, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + H, num1, mask=h_mask)
        tl.store(denominator_out + out_base + H, den1, mask=h_mask)
        tl.store(equity_out + out_base + H, tl.where(den1 > 0.0, num1 / tl.maximum(den1, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 2 * H, num2, mask=h_mask)
        tl.store(denominator_out + out_base + 2 * H, den2, mask=h_mask)
        tl.store(equity_out + out_base + 2 * H, tl.where(den2 > 0.0, num2 / tl.maximum(den2, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 3 * H, num3, mask=h_mask)
        tl.store(denominator_out + out_base + 3 * H, den3, mask=h_mask)
        tl.store(equity_out + out_base + 3 * H, tl.where(den3 > 0.0, num3 / tl.maximum(den3, 1.0e-30), 0.0), mask=h_mask)

    @triton.jit
    def _tier2_p4_sparse_direct_finish_kernel(
        scalar_all,
        player_card_cumsum,
        local_belief_matrix,
        total_desc_rowdot,
        total_desc_ltr,
        total_desc_rtl,
        total_desc_cross,
        local_c0,
        local_c1,
        pair_p_rank_flags,
        pair_q_rank_flags,
        slot_lower_by_card,
        slot_tie_by_card,
        same_all,
        numerator_out,
        denominator_out,
        equity_out,
        B: tl.constexpr,
        H: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        TOTAL_SLOT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
        USE_TOTAL_DESC: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        valid_card = card < CARD_COUNT
        hc_mask = h_mask[:, None] & valid_card[None, :]

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        slot_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        slot_lower = tl.load(slot_lower_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
        slot_tie = tl.load(slot_tie_by_card + slot_base, mask=hc_mask, other=0).to(tl.int32)
        idx_lower = tl.maximum(slot_lower - 1, 0)
        idx_tie = tl.maximum(slot_tie - 1, 0)
        idx_total = TOTAL_SLOT - 1

        valid_p = hc_mask & (card[None, :] != c0[:, None])
        valid_q = hc_mask & (card[None, :] != c1[:, None])
        flags_p = tl.load(pair_p_rank_flags + slot_base, mask=hc_mask, other=0)
        flags_q = tl.load(pair_q_rank_flags + slot_base, mask=hc_mask, other=0)
        row_lower_p = valid_p & ((flags_p & 1) != 0)
        row_lower_q = valid_q & ((flags_q & 1) != 0)
        row_tie_p = valid_p & ((flags_p & 2) != 0)
        row_tie_q = valid_q & ((flags_q & 2) != 0)
        blocked = (card[None, :] == c0[:, None]) | (card[None, :] == c1[:, None])

        base0 = ((0 * 3) * B + b) * H + h
        base1 = ((1 * 3) * B + b) * H + h
        base2 = ((2 * 3) * B + b) * H + h
        base3 = ((3 * 3) * B + b) * H + h
        l0 = tl.load(scalar_all + base0, mask=h_mask, other=0.0)
        t0 = tl.load(scalar_all + base0 + B * H, mask=h_mask, other=0.0)
        d0 = tl.load(scalar_all + base0 + 2 * B * H, mask=h_mask, other=0.0)
        l1 = tl.load(scalar_all + base1, mask=h_mask, other=0.0)
        t1 = tl.load(scalar_all + base1 + B * H, mask=h_mask, other=0.0)
        d1 = tl.load(scalar_all + base1 + 2 * B * H, mask=h_mask, other=0.0)
        l2 = tl.load(scalar_all + base2, mask=h_mask, other=0.0)
        t2 = tl.load(scalar_all + base2 + B * H, mask=h_mask, other=0.0)
        d2 = tl.load(scalar_all + base2 + 2 * B * H, mask=h_mask, other=0.0)
        l3 = tl.load(scalar_all + base3, mask=h_mask, other=0.0)
        t3 = tl.load(scalar_all + base3 + B * H, mask=h_mask, other=0.0)
        d3 = tl.load(scalar_all + base3 + 2 * B * H, mask=h_mask, other=0.0)

        num0 = (
            l1 * l2 * l3
            + 0.5 * (t1 * l2 * l3 + l1 * t2 * l3 + l1 * l2 * t3)
            + (1.0 / 3.0) * (t1 * t2 * l3 + t1 * l2 * t3 + l1 * t2 * t3)
            + 0.25 * t1 * t2 * t3
        )
        den0 = d1 * d2 * d3
        num1 = (
            l0 * l2 * l3
            + 0.5 * (t0 * l2 * l3 + l0 * t2 * l3 + l0 * l2 * t3)
            + (1.0 / 3.0) * (t0 * t2 * l3 + t0 * l2 * t3 + l0 * t2 * t3)
            + 0.25 * t0 * t2 * t3
        )
        den1 = d0 * d2 * d3
        num2 = (
            l0 * l1 * l3
            + 0.5 * (t0 * l1 * l3 + l0 * t1 * l3 + l0 * l1 * t3)
            + (1.0 / 3.0) * (t0 * t1 * l3 + t0 * l1 * t3 + l0 * t1 * t3)
            + 0.25 * t0 * t1 * t3
        )
        den2 = d0 * d1 * d3
        num3 = (
            l0 * l1 * l2
            + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
            + (1.0 / 3.0) * (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2)
            + 0.25 * t0 * t1 * t2
        )
        den3 = d0 * d1 * d2

        for pair in tl.static_range(0, 6):
            if pair == 0:
                left = 0
                right = 1
            elif pair == 1:
                left = 0
                right = 2
            elif pair == 2:
                left = 0
                right = 3
            elif pair == 3:
                left = 1
                right = 2
            elif pair == 4:
                left = 1
                right = 3
            else:
                left = 2
                right = 3

            left_card_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * SLOT_CAP
            right_card_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * SLOT_CAP
            left_lower_raw = tl.load(
                player_card_cumsum + left_card_base + idx_lower,
                mask=hc_mask & (slot_lower > 0),
                other=0.0,
            )
            right_lower_raw = tl.load(
                player_card_cumsum + right_card_base + idx_lower,
                mask=hc_mask & (slot_lower > 0),
                other=0.0,
            )
            left_tie_end = tl.load(
                player_card_cumsum + left_card_base + idx_tie,
                mask=hc_mask & (slot_tie > 0),
                other=0.0,
            )
            right_tie_end = tl.load(
                player_card_cumsum + right_card_base + idx_tie,
                mask=hc_mask & (slot_tie > 0),
                other=0.0,
            )
            left_tie_raw = left_tie_end - left_lower_raw
            right_tie_raw = right_tie_end - right_lower_raw

            matrix_left_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * CARD_COUNT
            matrix_right_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * CARD_COUNT
            left_lower_corr = tl.load(
                local_belief_matrix + matrix_left_base + c0[:, None],
                mask=row_lower_p,
                other=0.0,
            )
            left_lower_corr += tl.load(
                local_belief_matrix + matrix_left_base + c1[:, None],
                mask=row_lower_q,
                other=0.0,
            )
            right_lower_corr = tl.load(
                local_belief_matrix + matrix_right_base + c0[:, None],
                mask=row_lower_p,
                other=0.0,
            )
            right_lower_corr += tl.load(
                local_belief_matrix + matrix_right_base + c1[:, None],
                mask=row_lower_q,
                other=0.0,
            )
            left_tie_corr = tl.load(
                local_belief_matrix + matrix_left_base + c0[:, None],
                mask=row_tie_p,
                other=0.0,
            )
            left_tie_corr += tl.load(
                local_belief_matrix + matrix_left_base + c1[:, None],
                mask=row_tie_q,
                other=0.0,
            )
            right_tie_corr = tl.load(
                local_belief_matrix + matrix_right_base + c0[:, None],
                mask=row_tie_p,
                other=0.0,
            )
            right_tie_corr += tl.load(
                local_belief_matrix + matrix_right_base + c1[:, None],
                mask=row_tie_q,
                other=0.0,
            )
            left_lower = tl.where(blocked, 0.0, left_lower_raw - left_lower_corr)
            right_lower = tl.where(blocked, 0.0, right_lower_raw - right_lower_corr)
            left_tie = tl.where(blocked, 0.0, left_tie_raw - left_tie_corr)
            right_tie = tl.where(blocked, 0.0, right_tie_raw - right_tie_corr)

            same0 = tl.load(same_all + ((pair * 3 + 0) * B + b) * H + h, mask=h_mask, other=0.0)
            same1 = tl.load(same_all + ((pair * 3 + 1) * B + b) * H + h, mask=h_mask, other=0.0)
            same2 = tl.load(same_all + ((pair * 3 + 2) * B + b) * H + h, mask=h_mask, other=0.0)
            pair00 = tl.sum(left_lower * right_lower, axis=1) - same0
            pair01 = tl.sum(left_lower * right_tie, axis=1)
            pair10 = tl.sum(left_tie * right_lower, axis=1)
            pair11 = tl.sum(left_tie * right_tie, axis=1) - same1

            left_total_base0 = ((b * 4 + left) * CARD_COUNT + c0) * SLOT_CAP
            left_total_base1 = ((b * 4 + left) * CARD_COUNT + c1) * SLOT_CAP
            right_total_base0 = ((b * 4 + right) * CARD_COUNT + c0) * SLOT_CAP
            right_total_base1 = ((b * 4 + right) * CARD_COUNT + c1) * SLOT_CAP
            lr_a = tl.load(player_card_cumsum + left_total_base0 + idx_total, mask=h_mask, other=0.0)
            lr_b = tl.load(player_card_cumsum + left_total_base1 + idx_total, mask=h_mask, other=0.0)
            rr_a = tl.load(player_card_cumsum + right_total_base0 + idx_total, mask=h_mask, other=0.0)
            rr_b = tl.load(player_card_cumsum + right_total_base1 + idx_total, mask=h_mask, other=0.0)
            pair_vec_base = (b * 6 + pair) * CARD_COUNT
            ltr_a = tl.load(total_desc_ltr + pair_vec_base + c0, mask=h_mask, other=0.0)
            ltr_b = tl.load(total_desc_ltr + pair_vec_base + c1, mask=h_mask, other=0.0)
            rtl_a = tl.load(total_desc_rtl + pair_vec_base + c0, mask=h_mask, other=0.0)
            rtl_b = tl.load(total_desc_rtl + pair_vec_base + c1, mask=h_mask, other=0.0)
            pair_cross_base = (b * 6 + pair) * CARD_COUNT * CARD_COUNT
            left_matrix_base = (b * 4 + left) * CARD_COUNT * CARD_COUNT
            right_matrix_base = (b * 4 + right) * CARD_COUNT * CARD_COUNT
            left_ab = tl.load(
                local_belief_matrix + left_matrix_base + c0 * CARD_COUNT + c1,
                mask=h_mask,
                other=0.0,
            )
            right_ab = tl.load(
                local_belief_matrix + right_matrix_base + c0 * CARD_COUNT + c1,
                mask=h_mask,
                other=0.0,
            )
            cross_aa = tl.load(
                total_desc_cross + pair_cross_base + c0 * CARD_COUNT + c0,
                mask=h_mask,
                other=0.0,
            )
            cross_ab = tl.load(
                total_desc_cross + pair_cross_base + c0 * CARD_COUNT + c1,
                mask=h_mask,
                other=0.0,
            )
            cross_ba = tl.load(
                total_desc_cross + pair_cross_base + c1 * CARD_COUNT + c0,
                mask=h_mask,
                other=0.0,
            )
            cross_bb = tl.load(
                total_desc_cross + pair_cross_base + c1 * CARD_COUNT + c1,
                mask=h_mask,
                other=0.0,
            )
            rowdot = tl.load(total_desc_rowdot + b * 6 + pair)
            row_excl = rowdot - lr_a * rr_a - lr_b * rr_b
            row_excl -= ltr_a - lr_b * right_ab + ltr_b - lr_a * right_ab
            row_excl -= rtl_a - rr_b * left_ab + rtl_b - rr_a * left_ab
            row_excl += cross_aa + cross_ab + cross_ba + cross_bb - 2.0 * left_ab * right_ab
            pair_total = row_excl - same2
            pair10_plus_01 = pair10 + pair01
            other0 = pair00 + 0.5 * pair10_plus_01 + (1.0 / 3.0) * pair11
            other1 = 0.5 * pair00 + (1.0 / 3.0) * pair10_plus_01 + 0.25 * pair11

            if pair == 0:
                num2 -= other0 * l3 + other1 * t3
                den2 -= pair_total * d3
                num3 -= other0 * l2 + other1 * t2
                den3 -= pair_total * d2
            elif pair == 1:
                num1 -= other0 * l3 + other1 * t3
                den1 -= pair_total * d3
                num3 -= other0 * l1 + other1 * t1
                den3 -= pair_total * d1
            elif pair == 2:
                num1 -= other0 * l2 + other1 * t2
                den1 -= pair_total * d2
                num2 -= other0 * l1 + other1 * t1
                den2 -= pair_total * d1
            elif pair == 3:
                num0 -= other0 * l3 + other1 * t3
                den0 -= pair_total * d3
                num3 -= other0 * l0 + other1 * t0
                den3 -= pair_total * d0
            elif pair == 4:
                num0 -= other0 * l2 + other1 * t2
                den0 -= pair_total * d2
                num2 -= other0 * l0 + other1 * t0
                den2 -= pair_total * d0
            else:
                num0 -= other0 * l1 + other1 * t1
                den0 -= pair_total * d1
                num1 -= other0 * l0 + other1 * t0
                den1 -= pair_total * d0

        out_base = (b * 4) * H + h
        tl.store(numerator_out + out_base, num0, mask=h_mask)
        tl.store(denominator_out + out_base, den0, mask=h_mask)
        tl.store(equity_out + out_base, tl.where(den0 > 0.0, num0 / tl.maximum(den0, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + H, num1, mask=h_mask)
        tl.store(denominator_out + out_base + H, den1, mask=h_mask)
        tl.store(equity_out + out_base + H, tl.where(den1 > 0.0, num1 / tl.maximum(den1, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 2 * H, num2, mask=h_mask)
        tl.store(denominator_out + out_base + 2 * H, den2, mask=h_mask)
        tl.store(equity_out + out_base + 2 * H, tl.where(den2 > 0.0, num2 / tl.maximum(den2, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 3 * H, num3, mask=h_mask)
        tl.store(denominator_out + out_base + 3 * H, den3, mask=h_mask)
        tl.store(equity_out + out_base + 3 * H, tl.where(den3 > 0.0, num3 / tl.maximum(den3, 1.0e-30), 0.0), mask=h_mask)

    @triton.jit
    def _tier2_total_pair_vec_desc_kernel(
        local_belief_matrix,
        player_card_cumsum,
        rowdot_out,
        ltr_out,
        rtl_out,
        B: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        TOTAL_SLOT: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        pair = tl.program_id(1)
        card = tl.arange(0, BLOCK_C)
        k = tl.arange(0, BLOCK_C)
        valid_card = card < CARD_COUNT
        valid_k = k < CARD_COUNT
        left = tl.where(pair < 3, 0, tl.where(pair < 5, 1, 2))
        right = tl.where(pair < 3, pair + 1, tl.where(pair < 5, pair - 1, 3))

        left_rows = tl.load(
            player_card_cumsum + ((b * 4 + left) * CARD_COUNT + k) * SLOT_CAP + (TOTAL_SLOT - 1),
            mask=valid_k,
            other=0.0,
        )
        right_rows = tl.load(
            player_card_cumsum + ((b * 4 + right) * CARD_COUNT + k) * SLOT_CAP + (TOTAL_SLOT - 1),
            mask=valid_k,
            other=0.0,
        )
        rowdot = tl.sum(left_rows * right_rows, axis=0)
        tl.store(rowdot_out + b * 6 + pair, rowdot)

        right_matrix_base = (b * 4 + right) * CARD_COUNT * CARD_COUNT
        left_matrix_base = (b * 4 + left) * CARD_COUNT * CARD_COUNT
        right_vals = tl.load(
            local_belief_matrix + right_matrix_base + card[:, None] * CARD_COUNT + k[None, :],
            mask=valid_card[:, None] & valid_k[None, :] & (card[:, None] != k[None, :]),
            other=0.0,
        )
        left_vals = tl.load(
            local_belief_matrix + left_matrix_base + card[:, None] * CARD_COUNT + k[None, :],
            mask=valid_card[:, None] & valid_k[None, :] & (card[:, None] != k[None, :]),
            other=0.0,
        )
        ltr = tl.sum(right_vals * left_rows[None, :], axis=1)
        rtl = tl.sum(left_vals * right_rows[None, :], axis=1)
        out_base = (b * 6 + pair) * CARD_COUNT + card
        tl.store(ltr_out + out_base, ltr, mask=valid_card)
        tl.store(rtl_out + out_base, rtl, mask=valid_card)

    @triton.jit
    def _tier2_total_pair_cross_desc_kernel(
        local_belief_matrix,
        cross_out,
        B: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        TILES_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        pair = tl.program_id(1)
        tile = tl.program_id(2)
        m_block = tile // TILES_N
        n_block = tile - m_block * TILES_N
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        left = tl.where(pair < 3, 0, tl.where(pair < 5, 1, 2))
        right = tl.where(pair < 3, pair + 1, tl.where(pair < 5, pair - 1, 3))
        left_matrix_base = (b * 4 + left) * CARD_COUNT * CARD_COUNT
        right_matrix_base = (b * 4 + right) * CARD_COUNT * CARD_COUNT
        w = tl.load(
            local_belief_matrix + left_matrix_base + offs_m[:, None] * CARD_COUNT + offs_k[None, :],
            mask=(offs_m[:, None] < CARD_COUNT)
            & (offs_k[None, :] < CARD_COUNT)
            & (offs_m[:, None] != offs_k[None, :]),
            other=0.0,
        )
        v = tl.load(
            local_belief_matrix + right_matrix_base + offs_k[:, None] * CARD_COUNT + offs_n[None, :],
            mask=(offs_k[:, None] < CARD_COUNT)
            & (offs_n[None, :] < CARD_COUNT)
            & (offs_k[:, None] != offs_n[None, :]),
            other=0.0,
        )
        acc = tl.dot(w, v, input_precision="tf32")
        out_base = (b * 6 + pair) * CARD_COUNT * CARD_COUNT
        tl.store(
            cross_out + out_base + offs_m[:, None] * CARD_COUNT + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < CARD_COUNT) & (offs_n[None, :] < CARD_COUNT),
        )

    @triton.jit
    def _tier2_p4_sparse_prefix_direct_finish_kernel(
        beliefs,
        full_beliefs,
        scalar_prefix,
        pair_prefix,
        player_card_cumsum,
        pair_card_cumsum,
        local_c0,
        local_c1,
        pair_p_ids,
        pair_q_ids,
        pair_p_rank_flags,
        pair_q_rank_flags,
        lower_end,
        tie_end,
        slot_lower_by_card,
        slot_tie_by_card,
        numerator_out,
        denominator_out,
        equity_out,
        B: tl.constexpr,
        H: tl.constexpr,
        H1: tl.constexpr,
        FULL_HANDS: tl.constexpr,
        CARD_COUNT: tl.constexpr,
        SLOT_CAP: tl.constexpr,
        TOTAL_SLOT: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.program_id(0)
        h_block = tl.program_id(1)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        card = tl.arange(0, BLOCK_C)
        h_mask = h < H
        valid_card = card < CARD_COUNT
        hc_mask = h_mask[:, None] & valid_card[None, :]

        c0 = tl.load(local_c0 + b * H + h, mask=h_mask, other=0)
        c1 = tl.load(local_c1 + b * H + h, mask=h_mask, other=0)
        lower = tl.load(lower_end + b * H + h, mask=h_mask, other=0)
        tie = tl.load(tie_end + b * H + h, mask=h_mask, other=0)

        slot_h_base = (b * H + h) * CARD_COUNT
        sl0 = tl.load(slot_lower_by_card + slot_h_base + c0, mask=h_mask, other=0).to(tl.int32)
        sl1 = tl.load(slot_lower_by_card + slot_h_base + c1, mask=h_mask, other=0).to(tl.int32)
        st0 = tl.load(slot_tie_by_card + slot_h_base + c0, mask=h_mask, other=0).to(tl.int32)
        st1 = tl.load(slot_tie_by_card + slot_h_base + c1, mask=h_mask, other=0).to(tl.int32)

        for player in tl.static_range(0, 4):
            prefix_base = (b * 4 + player) * H1
            card_base0 = ((b * 4 + player) * CARD_COUNT + c0) * SLOT_CAP
            card_base1 = ((b * 4 + player) * CARD_COUNT + c1) * SLOT_CAP
            belief = tl.load(beliefs + (b * 4 + player) * H + h, mask=h_mask, other=0.0)

            idx_l0 = tl.maximum(sl0 - 1, 0)
            idx_l1 = tl.maximum(sl1 - 1, 0)
            lower_scalar = tl.load(scalar_prefix + prefix_base + lower, mask=h_mask, other=0.0) - tl.load(
                scalar_prefix + prefix_base
            )
            lower_c0 = tl.load(player_card_cumsum + card_base0 + idx_l0, mask=h_mask & (sl0 > 0), other=0.0)
            lower_c1 = tl.load(player_card_cumsum + card_base1 + idx_l1, mask=h_mask & (sl1 > 0), other=0.0)
            lower_value = lower_scalar - lower_c0 - lower_c1

            idx_t0 = tl.maximum(st0 - 1, 0)
            idx_t1 = tl.maximum(st1 - 1, 0)
            tie_scalar = tl.load(scalar_prefix + prefix_base + tie, mask=h_mask, other=0.0) - tl.load(
                scalar_prefix + prefix_base + lower,
                mask=h_mask,
                other=0.0,
            )
            tie_c0_end = tl.load(player_card_cumsum + card_base0 + idx_t0, mask=h_mask & (st0 > 0), other=0.0)
            tie_c1_end = tl.load(player_card_cumsum + card_base1 + idx_t1, mask=h_mask & (st1 > 0), other=0.0)
            tie_value = tie_scalar - (tie_c0_end - lower_c0) - (tie_c1_end - lower_c1) + belief

            total_scalar = tl.load(scalar_prefix + prefix_base + H) - tl.load(scalar_prefix + prefix_base)
            total_c0 = tl.load(
                player_card_cumsum + card_base0 + (TOTAL_SLOT - 1),
                mask=h_mask,
                other=0.0,
            )
            total_c1 = tl.load(
                player_card_cumsum + card_base1 + (TOTAL_SLOT - 1),
                mask=h_mask,
                other=0.0,
            )
            total_value = total_scalar - total_c0 - total_c1 + belief

            if player == 0:
                l0 = lower_value
                t0 = tie_value
                d0 = total_value
                b0 = belief
            elif player == 1:
                l1 = lower_value
                t1 = tie_value
                d1 = total_value
                b1 = belief
            elif player == 2:
                l2 = lower_value
                t2 = tie_value
                d2 = total_value
                b2 = belief
            else:
                l3 = lower_value
                t3 = tie_value
                d3 = total_value
                b3 = belief

        num0 = (
            l1 * l2 * l3
            + 0.5 * (t1 * l2 * l3 + l1 * t2 * l3 + l1 * l2 * t3)
            + (1.0 / 3.0) * (t1 * t2 * l3 + t1 * l2 * t3 + l1 * t2 * t3)
            + 0.25 * t1 * t2 * t3
        )
        den0 = d1 * d2 * d3
        num1 = (
            l0 * l2 * l3
            + 0.5 * (t0 * l2 * l3 + l0 * t2 * l3 + l0 * l2 * t3)
            + (1.0 / 3.0) * (t0 * t2 * l3 + t0 * l2 * t3 + l0 * t2 * t3)
            + 0.25 * t0 * t2 * t3
        )
        den1 = d0 * d2 * d3
        num2 = (
            l0 * l1 * l3
            + 0.5 * (t0 * l1 * l3 + l0 * t1 * l3 + l0 * l1 * t3)
            + (1.0 / 3.0) * (t0 * t1 * l3 + t0 * l1 * t3 + l0 * t1 * t3)
            + 0.25 * t0 * t1 * t3
        )
        den2 = d0 * d1 * d3
        num3 = (
            l0 * l1 * l2
            + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
            + (1.0 / 3.0) * (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2)
            + 0.25 * t0 * t1 * t2
        )
        den3 = d0 * d1 * d2

        card_slot_base = (b * H + h[:, None]) * CARD_COUNT + card[None, :]
        slot_lower = tl.load(slot_lower_by_card + card_slot_base, mask=hc_mask, other=0).to(tl.int32)
        slot_tie = tl.load(slot_tie_by_card + card_slot_base, mask=hc_mask, other=0).to(tl.int32)
        idx_lower = tl.maximum(slot_lower - 1, 0)
        idx_tie = tl.maximum(slot_tie - 1, 0)
        idx_total = TOTAL_SLOT - 1

        pair_p_id = tl.load(pair_p_ids + card_slot_base, mask=hc_mask, other=0).to(tl.int32)
        pair_q_id = tl.load(pair_q_ids + card_slot_base, mask=hc_mask, other=0).to(tl.int32)
        valid_p = hc_mask & (card[None, :] != c0[:, None])
        valid_q = hc_mask & (card[None, :] != c1[:, None])
        flags_p = tl.load(pair_p_rank_flags + card_slot_base, mask=hc_mask, other=0)
        flags_q = tl.load(pair_q_rank_flags + card_slot_base, mask=hc_mask, other=0)
        row_lower_p = valid_p & ((flags_p & 1) != 0)
        row_lower_q = valid_q & ((flags_q & 1) != 0)
        row_tie_p = valid_p & ((flags_p & 2) != 0)
        row_tie_q = valid_q & ((flags_q & 2) != 0)
        blocked = (card[None, :] == c0[:, None]) | (card[None, :] == c1[:, None])

        for pair in tl.static_range(0, 6):
            if pair == 0:
                left = 0
                right = 1
                left_b = b0
                right_b = b1
            elif pair == 1:
                left = 0
                right = 2
                left_b = b0
                right_b = b2
            elif pair == 2:
                left = 0
                right = 3
                left_b = b0
                right_b = b3
            elif pair == 3:
                left = 1
                right = 2
                left_b = b1
                right_b = b2
            elif pair == 4:
                left = 1
                right = 3
                left_b = b1
                right_b = b3
            else:
                left = 2
                right = 3
                left_b = b2
                right_b = b3

            pair_prefix_base = (b * 6 + pair) * H1
            pair_card_base0 = ((b * 6 + pair) * CARD_COUNT + c0) * SLOT_CAP
            pair_card_base1 = ((b * 6 + pair) * CARD_COUNT + c1) * SLOT_CAP
            pair_edge = left_b * right_b
            pair_l0 = tl.load(pair_card_cumsum + pair_card_base0 + idx_l0, mask=h_mask & (sl0 > 0), other=0.0)
            pair_l1 = tl.load(pair_card_cumsum + pair_card_base1 + idx_l1, mask=h_mask & (sl1 > 0), other=0.0)
            same0 = tl.load(pair_prefix + pair_prefix_base + lower, mask=h_mask, other=0.0) - tl.load(
                pair_prefix + pair_prefix_base
            )
            same0 = same0 - pair_l0 - pair_l1
            pair_t0 = tl.load(pair_card_cumsum + pair_card_base0 + idx_t0, mask=h_mask & (st0 > 0), other=0.0)
            pair_t1 = tl.load(pair_card_cumsum + pair_card_base1 + idx_t1, mask=h_mask & (st1 > 0), other=0.0)
            same1 = tl.load(pair_prefix + pair_prefix_base + tie, mask=h_mask, other=0.0) - tl.load(
                pair_prefix + pair_prefix_base + lower,
                mask=h_mask,
                other=0.0,
            )
            same1 = same1 - (pair_t0 - pair_l0) - (pair_t1 - pair_l1) + pair_edge
            pair_total0 = tl.load(
                pair_card_cumsum + pair_card_base0 + (TOTAL_SLOT - 1),
                mask=h_mask,
                other=0.0,
            )
            pair_total1 = tl.load(
                pair_card_cumsum + pair_card_base1 + (TOTAL_SLOT - 1),
                mask=h_mask,
                other=0.0,
            )
            same2 = tl.load(pair_prefix + pair_prefix_base + H) - tl.load(pair_prefix + pair_prefix_base)
            same2 = same2 - pair_total0 - pair_total1 + pair_edge

            left_card_base = ((b * 4 + left) * CARD_COUNT + card[None, :]) * SLOT_CAP
            right_card_base = ((b * 4 + right) * CARD_COUNT + card[None, :]) * SLOT_CAP
            left_lower_raw = tl.load(
                player_card_cumsum + left_card_base + idx_lower,
                mask=hc_mask & (slot_lower > 0),
                other=0.0,
            )
            right_lower_raw = tl.load(
                player_card_cumsum + right_card_base + idx_lower,
                mask=hc_mask & (slot_lower > 0),
                other=0.0,
            )
            left_tie_end = tl.load(
                player_card_cumsum + left_card_base + idx_tie,
                mask=hc_mask & (slot_tie > 0),
                other=0.0,
            )
            right_tie_end = tl.load(
                player_card_cumsum + right_card_base + idx_tie,
                mask=hc_mask & (slot_tie > 0),
                other=0.0,
            )
            left_total_raw = tl.load(
                player_card_cumsum + left_card_base + idx_total,
                mask=hc_mask,
                other=0.0,
            )
            right_total_raw = tl.load(
                player_card_cumsum + right_card_base + idx_total,
                mask=hc_mask,
                other=0.0,
            )
            left_tie_raw = left_tie_end - left_lower_raw
            right_tie_raw = right_tie_end - right_lower_raw

            full_left_base = (b * 4 + left) * FULL_HANDS
            full_right_base = (b * 4 + right) * FULL_HANDS
            left_p = tl.load(full_beliefs + full_left_base + pair_p_id, mask=valid_p, other=0.0)
            left_q = tl.load(full_beliefs + full_left_base + pair_q_id, mask=valid_q, other=0.0)
            right_p = tl.load(full_beliefs + full_right_base + pair_p_id, mask=valid_p, other=0.0)
            right_q = tl.load(full_beliefs + full_right_base + pair_q_id, mask=valid_q, other=0.0)

            left_lower_corr = tl.where(row_lower_p, left_p, 0.0) + tl.where(row_lower_q, left_q, 0.0)
            right_lower_corr = tl.where(row_lower_p, right_p, 0.0) + tl.where(row_lower_q, right_q, 0.0)
            left_tie_corr = tl.where(row_tie_p, left_p, 0.0) + tl.where(row_tie_q, left_q, 0.0)
            right_tie_corr = tl.where(row_tie_p, right_p, 0.0) + tl.where(row_tie_q, right_q, 0.0)
            left_total_corr = tl.where(valid_p, left_p, 0.0) + tl.where(valid_q, left_q, 0.0)
            right_total_corr = tl.where(valid_p, right_p, 0.0) + tl.where(valid_q, right_q, 0.0)

            left_lower = tl.where(blocked, 0.0, left_lower_raw - left_lower_corr)
            right_lower = tl.where(blocked, 0.0, right_lower_raw - right_lower_corr)
            left_tie = tl.where(blocked, 0.0, left_tie_raw - left_tie_corr)
            right_tie = tl.where(blocked, 0.0, right_tie_raw - right_tie_corr)
            left_total = tl.where(blocked, 0.0, left_total_raw - left_total_corr)
            right_total = tl.where(blocked, 0.0, right_total_raw - right_total_corr)

            pair00 = tl.sum(left_lower * right_lower, axis=1) - same0
            pair01 = tl.sum(left_lower * right_tie, axis=1)
            pair10 = tl.sum(left_tie * right_lower, axis=1)
            pair11 = tl.sum(left_tie * right_tie, axis=1) - same1
            pair_total = tl.sum(left_total * right_total, axis=1) - same2
            pair10_plus_01 = pair10 + pair01
            other0 = pair00 + 0.5 * pair10_plus_01 + (1.0 / 3.0) * pair11
            other1 = 0.5 * pair00 + (1.0 / 3.0) * pair10_plus_01 + 0.25 * pair11

            if pair == 0:
                num2 -= other0 * l3 + other1 * t3
                den2 -= pair_total * d3
                num3 -= other0 * l2 + other1 * t2
                den3 -= pair_total * d2
            elif pair == 1:
                num1 -= other0 * l3 + other1 * t3
                den1 -= pair_total * d3
                num3 -= other0 * l1 + other1 * t1
                den3 -= pair_total * d1
            elif pair == 2:
                num1 -= other0 * l2 + other1 * t2
                den1 -= pair_total * d2
                num2 -= other0 * l1 + other1 * t1
                den2 -= pair_total * d1
            elif pair == 3:
                num0 -= other0 * l3 + other1 * t3
                den0 -= pair_total * d3
                num3 -= other0 * l0 + other1 * t0
                den3 -= pair_total * d0
            elif pair == 4:
                num0 -= other0 * l2 + other1 * t2
                den0 -= pair_total * d2
                num2 -= other0 * l0 + other1 * t0
                den2 -= pair_total * d0
            else:
                num0 -= other0 * l1 + other1 * t1
                den0 -= pair_total * d1
                num1 -= other0 * l0 + other1 * t0
                den1 -= pair_total * d0

        out_base = (b * 4) * H + h
        tl.store(numerator_out + out_base, num0, mask=h_mask)
        tl.store(denominator_out + out_base, den0, mask=h_mask)
        tl.store(equity_out + out_base, tl.where(den0 > 0.0, num0 / tl.maximum(den0, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + H, num1, mask=h_mask)
        tl.store(denominator_out + out_base + H, den1, mask=h_mask)
        tl.store(equity_out + out_base + H, tl.where(den1 > 0.0, num1 / tl.maximum(den1, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 2 * H, num2, mask=h_mask)
        tl.store(denominator_out + out_base + 2 * H, den2, mask=h_mask)
        tl.store(equity_out + out_base + 2 * H, tl.where(den2 > 0.0, num2 / tl.maximum(den2, 1.0e-30), 0.0), mask=h_mask)
        tl.store(numerator_out + out_base + 3 * H, num3, mask=h_mask)
        tl.store(denominator_out + out_base + 3 * H, den3, mask=h_mask)
        tl.store(equity_out + out_base + 3 * H, tl.where(den3 > 0.0, num3 / tl.maximum(den3, 1.0e-30), 0.0), mask=h_mask)

    @triton.jit
    def _tier2_p4_finish_kernel(
        scalar_all,
        pair_event_all,
        numerator_out,
        denominator_out,
        equity_out,
        B: tl.constexpr,
        H: tl.constexpr,
        BLOCK_H: tl.constexpr,
        FINISH_ONLY_EVENTS: tl.constexpr,
    ):
        b = tl.program_id(0)
        hero = tl.program_id(1)
        h = tl.program_id(2) * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = h < H

        opp0 = tl.where(0 < hero, 0, 1)
        opp1 = tl.where(1 < hero, 1, 2)
        opp2 = tl.where(2 < hero, 2, 3)

        base0 = ((opp0 * 3) * B + b) * H + h
        base1 = ((opp1 * 3) * B + b) * H + h
        base2 = ((opp2 * 3) * B + b) * H + h
        l0 = tl.load(scalar_all + base0, mask=h_mask, other=0.0)
        t0 = tl.load(scalar_all + base0 + B * H, mask=h_mask, other=0.0)
        d0 = tl.load(scalar_all + base0 + 2 * B * H, mask=h_mask, other=0.0)
        l1 = tl.load(scalar_all + base1, mask=h_mask, other=0.0)
        t1 = tl.load(scalar_all + base1 + B * H, mask=h_mask, other=0.0)
        d1 = tl.load(scalar_all + base1 + 2 * B * H, mask=h_mask, other=0.0)
        l2 = tl.load(scalar_all + base2, mask=h_mask, other=0.0)
        t2 = tl.load(scalar_all + base2 + B * H, mask=h_mask, other=0.0)
        d2 = tl.load(scalar_all + base2 + 2 * B * H, mask=h_mask, other=0.0)

        numerator = (
            l0 * l1 * l2
            + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
            + (1.0 / 3.0) * (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2)
            + 0.25 * t0 * t1 * t2
        )
        denominator = d0 * d1 * d2

        for edge in tl.static_range(0, 3):
            left = tl.where(edge == 0, opp0, tl.where(edge == 1, opp0, opp1))
            right = tl.where(edge == 0, opp1, tl.where(edge == 1, opp2, opp2))
            other_l = tl.where(edge == 0, l2, tl.where(edge == 1, l1, l0))
            other_t = tl.where(edge == 0, t2, tl.where(edge == 1, t1, t0))
            other_d = tl.where(edge == 0, d2, tl.where(edge == 1, d1, d0))
            pair = tl.where(
                left == 0,
                right - 1,
                tl.where(left == 1, right + 1, 5),
            )
            pair_stride = tl.where(FINISH_ONLY_EVENTS, 5, 9)
            pair_base = (pair * pair_stride * B + b) * H + h
            pair00 = tl.load(pair_event_all + pair_base, mask=h_mask, other=0.0)
            pair01 = tl.load(pair_event_all + pair_base + B * H, mask=h_mask, other=0.0)
            pair10_offset = tl.where(FINISH_ONLY_EVENTS, 2, 3) * B * H
            pair11_offset = tl.where(FINISH_ONLY_EVENTS, 3, 4) * B * H
            pair_total_offset = tl.where(FINISH_ONLY_EVENTS, 4, 8) * B * H
            pair10 = tl.load(pair_event_all + pair_base + pair10_offset, mask=h_mask, other=0.0)
            pair11 = tl.load(pair_event_all + pair_base + pair11_offset, mask=h_mask, other=0.0)
            pair_total = tl.load(
                pair_event_all + pair_base + pair_total_offset,
                mask=h_mask,
                other=0.0,
            )
            pair10_plus_01 = pair10 + pair01
            other0 = pair00 + 0.5 * pair10_plus_01 + (1.0 / 3.0) * pair11
            other1 = 0.5 * pair00 + (1.0 / 3.0) * pair10_plus_01 + 0.25 * pair11
            numerator -= other0 * other_l + other1 * other_t
            denominator -= pair_total * other_d

        out_base = (b * 4 + hero) * H + h
        tl.store(numerator_out + out_base, numerator, mask=h_mask)
        tl.store(denominator_out + out_base, denominator, mask=h_mask)
        equity = tl.where(denominator > 0.0, numerator / tl.maximum(denominator, 1.0e-30), 0.0)
        tl.store(equity_out + out_base, equity, mask=h_mask)


def _tier2_prefix_factors_triton(
    *,
    scalar_prefix: torch.Tensor,
    card_prefix: torch.Tensor,
    pair_prefix: torch.Tensor,
    pair_card_prefix: torch.Tensor,
    beliefs: torch.Tensor,
    local_belief_matrix: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    pair_p_rank_flags: torch.Tensor,
    pair_q_rank_flags: torch.Tensor,
    lower_end: torch.Tensor,
    tie_end: torch.Tensor,
    group_count: torch.Tensor,
    dtype: torch.dtype,
    same_num_warps: int,
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
    same_all = (
        torch.empty(
            6,
            3,
            batch_size,
            active_count,
            dtype=dtype,
            device=device,
        )
        if players == 4
        else torch.empty(
            players,
            players,
            3,
            batch_size,
            active_count,
            dtype=dtype,
            device=device,
        )
    )
    scalar_block_h = 4
    grid_scalar = (batch_size, triton.cdiv(active_count, scalar_block_h), players)
    for mode in range(3):
        _tier2_prefix_scalar_card_kernel[grid_scalar](
            scalar_prefix,
            card_prefix,
            beliefs,
            local_belief_matrix,
            local_c0,
            local_c1,
            pair_p_rank_flags,
            pair_q_rank_flags,
            lower_end,
            tie_end,
            group_count,
            scalar_all,
            card_all,
            B=batch_size,
            P=players,
            H=active_count,
            H1=scalar_prefix.shape[2],
            CARD_COUNT=47,
            BLOCK_H=scalar_block_h,
            BLOCK_C=64,
            MODE=mode,
            num_warps=2,
        )
    same_pair_count = 6 if players == 4 else players * players
    same_block_h = 16
    grid_same = (batch_size, triton.cdiv(active_count, same_block_h), same_pair_count * 3)
    _tier2_prefix_same_kernel[grid_same](
        pair_prefix,
        pair_card_prefix,
        beliefs,
        local_c0,
        local_c1,
        lower_end,
        tie_end,
        group_count,
        same_all,
        B=batch_size,
        P=players,
        H=active_count,
        H1=pair_prefix.shape[-1],
        CARD_COUNT=47,
        BLOCK_H=same_block_h,
        USE_P4_UNORDERED=players == 4,
        num_warps=same_num_warps,
    )
    return scalar_all, card_all, same_all


def _tier2_scalar_same_from_prefix_triton(
    *,
    scalar_prefix: torch.Tensor,
    card_prefix: torch.Tensor,
    pair_prefix: torch.Tensor,
    pair_card_prefix: torch.Tensor,
    beliefs: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    lower_end: torch.Tensor,
    tie_end: torch.Tensor,
    group_count: torch.Tensor,
    dtype: torch.dtype,
    same_num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or beliefs.device.type != "cuda" or beliefs.shape[1] != 4:
        return None
    batch_size, players, active_count = beliefs.shape
    device = beliefs.device
    scalar_all = torch.empty(players, 3, batch_size, active_count, dtype=dtype, device=device)
    same_all = torch.empty(6, 3, batch_size, active_count, dtype=dtype, device=device)
    scalar_block_h = 16
    grid_scalar = (batch_size, triton.cdiv(active_count, scalar_block_h), players)
    for mode in range(3):
        _tier2_prefix_scalar_only_kernel[grid_scalar](
            scalar_prefix,
            card_prefix,
            beliefs,
            local_c0,
            local_c1,
            lower_end,
            tie_end,
            group_count,
            scalar_all,
            B=batch_size,
            P=players,
            H=active_count,
            H1=scalar_prefix.shape[2],
            CARD_COUNT=47,
            BLOCK_H=scalar_block_h,
            MODE=mode,
            num_warps=1,
        )
    same_block_h = 16
    grid_same = (batch_size, triton.cdiv(active_count, same_block_h), 18)
    _tier2_prefix_same_kernel[grid_same](
        pair_prefix,
        pair_card_prefix,
        beliefs,
        local_c0,
        local_c1,
        lower_end,
        tie_end,
        group_count,
        same_all,
        B=batch_size,
        P=players,
        H=active_count,
        H1=pair_prefix.shape[-1],
        CARD_COUNT=47,
        BLOCK_H=same_block_h,
        USE_P4_UNORDERED=True,
        num_warps=same_num_warps,
    )
    return scalar_all, same_all


def _tier2_p4_pair_event_from_prefix_triton(
    *,
    card_prefix: torch.Tensor,
    local_belief_matrix: torch.Tensor,
    local_c0: torch.Tensor,
    local_c1: torch.Tensor,
    pair_p_rank_flags: torch.Tensor,
    pair_q_rank_flags: torch.Tensor,
    lower_end: torch.Tensor,
    tie_end: torch.Tensor,
    group_count: torch.Tensor,
    same_all: torch.Tensor,
) -> torch.Tensor | None:
    if triton is None or card_prefix.device.type != "cuda" or card_prefix.shape[1] != 4:
        return None
    batch_size = card_prefix.shape[0]
    active_count = local_c0.shape[1]
    pair_event = torch.empty(
        6,
        5,
        batch_size,
        active_count,
        dtype=card_prefix.dtype,
        device=card_prefix.device,
    )
    block_h = _env_int("P2_SHOWDOWN_TIER2_DIRECT_PAIR_BLOCK_H", 8)
    block_c = 64
    grid = (batch_size, 6, triton.cdiv(active_count, block_h))
    _p4_pair_event_from_prefix_finish_kernel[grid](
        card_prefix.contiguous(),
        local_belief_matrix.contiguous(),
        local_c0.contiguous(),
        local_c1.contiguous(),
        pair_p_rank_flags.contiguous(),
        pair_q_rank_flags.contiguous(),
        lower_end.contiguous(),
        tie_end.contiguous(),
        group_count.contiguous(),
        same_all.contiguous(),
        pair_event,
        B=batch_size,
        H=active_count,
        H1=card_prefix.shape[2],
        CARD_COUNT=47,
        BLOCK_H=block_h,
        BLOCK_C=block_c,
        num_warps=2,
    )
    return pair_event


def _tier2_p4_sparse_prefixes_triton(
    *,
    sorted_beliefs: torch.Tensor,
    sorted_card_positions: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or sorted_beliefs.device.type != "cuda" or sorted_beliefs.shape[1] != 4:
        return None
    batch_size, _, active_count = sorted_beliefs.shape
    device = sorted_beliefs.device
    scalar_prefix = torch.empty(batch_size, 4, active_count + 1, dtype=dtype, device=device)
    pair_prefix = torch.empty(batch_size, 6, active_count + 1, dtype=dtype, device=device)
    block_h = 1
    while block_h < active_count:
        block_h *= 2
    _tier2_p4_sparse_scalar_pair_prefix_kernel[(batch_size,)](
        sorted_beliefs.contiguous(),
        scalar_prefix,
        pair_prefix,
        H=active_count,
        H1=active_count + 1,
        BLOCK_H=block_h,
        num_warps=8,
    )
    player_card_cumsum = torch.empty(
        batch_size,
        4,
        _TIER_LOCAL_CARDS,
        _TIER_CARD_SLOT_CAP,
        dtype=dtype,
        device=device,
    )
    pair_card_cumsum = torch.empty(
        batch_size,
        6,
        _TIER_LOCAL_CARDS,
        _TIER_CARD_SLOT_CAP,
        dtype=dtype,
        device=device,
    )
    _tier2_p4_sparse_card_cumsum_kernel[(batch_size, _TIER_LOCAL_CARDS)](
        sorted_beliefs.contiguous(),
        sorted_card_positions.contiguous(),
        player_card_cumsum,
        pair_card_cumsum,
        H=active_count,
        CARD_COUNT=_TIER_LOCAL_CARDS,
        SLOT_CAP=_TIER_CARD_SLOT_CAP,
        num_warps=2,
    )
    return scalar_prefix, pair_prefix, player_card_cumsum, pair_card_cumsum


def _tier2_p4_sparse_scalar_same_triton(
    *,
    beliefs: torch.Tensor,
    scalar_prefix: torch.Tensor,
    pair_prefix: torch.Tensor,
    player_card_cumsum: torch.Tensor,
    pair_card_cumsum: torch.Tensor,
    ctx: _ActiveTierContext,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or beliefs.device.type != "cuda" or beliefs.shape[1] != 4:
        return None
    batch_size, _, active_count = beliefs.shape
    scalar_all = torch.empty(4, 3, batch_size, active_count, dtype=dtype, device=beliefs.device)
    same_all = torch.empty(6, 3, batch_size, active_count, dtype=dtype, device=beliefs.device)
    local_belief_matrix = torch.empty(
        batch_size,
        4,
        _TIER_LOCAL_CARDS,
        _TIER_LOCAL_CARDS,
        dtype=dtype,
        device=beliefs.device,
    )
    block_h = _env_int("P2_SHOWDOWN_TIER2_SPARSE_SCALAR_BLOCK_H", 16)
    grid = (batch_size, triton.cdiv(active_count, block_h))
    _tier2_p4_sparse_scalar_same_kernel[grid](
        beliefs.contiguous(),
        scalar_prefix.contiguous(),
        pair_prefix.contiguous(),
        player_card_cumsum.contiguous(),
        pair_card_cumsum.contiguous(),
        ctx.local_c0.contiguous(),
        ctx.local_c1.contiguous(),
        ctx.lower_end.contiguous(),
        ctx.tie_end.contiguous(),
        ctx.slot_lower_by_card.contiguous(),
        ctx.slot_tie_by_card.contiguous(),
        scalar_all,
        same_all,
        local_belief_matrix,
        B=batch_size,
        H=active_count,
        H1=active_count + 1,
        CARD_COUNT=_TIER_LOCAL_CARDS,
        SLOT_CAP=_TIER_CARD_SLOT_CAP,
        TOTAL_SLOT=_TIER_HANDS_PER_LOCAL_CARD,
        BLOCK_H=block_h,
        num_warps=1,
    )
    return scalar_all, same_all, local_belief_matrix


def _tier2_p4_pair_event_from_sparse_triton(
    *,
    player_card_cumsum: torch.Tensor,
    local_belief_matrix: torch.Tensor,
    ctx: _ActiveTierContext,
    same_all: torch.Tensor,
) -> torch.Tensor | None:
    if triton is None or player_card_cumsum.device.type != "cuda":
        return None
    batch_size = player_card_cumsum.shape[0]
    active_count = ctx.local_c0.shape[1]
    pair_event = torch.empty(
        6,
        5,
        batch_size,
        active_count,
        dtype=player_card_cumsum.dtype,
        device=player_card_cumsum.device,
    )
    block_h = _env_int("P2_SHOWDOWN_TIER2_SPARSE_PAIR_BLOCK_H", 16)
    direct_total = os.environ.get("P2_SHOWDOWN_TIER2_TOTAL_PIE") == "1"
    grid = (batch_size, 6, triton.cdiv(active_count, block_h))
    _p4_pair_event_from_sparse_finish_kernel[grid](
        player_card_cumsum.contiguous(),
        local_belief_matrix.contiguous(),
        ctx.local_c0.contiguous(),
        ctx.local_c1.contiguous(),
        ctx.pair_p_rank_flags.contiguous(),
        ctx.pair_q_rank_flags.contiguous(),
        ctx.slot_lower_by_card.contiguous(),
        ctx.slot_tie_by_card.contiguous(),
        same_all.contiguous(),
        pair_event,
        B=batch_size,
        H=active_count,
        CARD_COUNT=_TIER_LOCAL_CARDS,
        SLOT_CAP=_TIER_CARD_SLOT_CAP,
        TOTAL_SLOT=_TIER_HANDS_PER_LOCAL_CARD,
        BLOCK_H=block_h,
        BLOCK_C=64,
        COMPUTE_TOTAL=not direct_total,
        num_warps=2,
    )
    if direct_total:
        _tier2_p4_fill_total_pair_event_pie(pair_event, local_belief_matrix, ctx)
    return pair_event


def _tier2_p4_total_pair_descriptors_triton(
    *,
    local_belief_matrix: torch.Tensor,
    player_card_cumsum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or local_belief_matrix.device.type != "cuda":
        return None
    batch_size = local_belief_matrix.shape[0]
    dtype = local_belief_matrix.dtype
    device = local_belief_matrix.device
    rowdot = torch.empty(batch_size, 6, dtype=dtype, device=device)
    ltr = torch.empty(batch_size, 6, _TIER_LOCAL_CARDS, dtype=dtype, device=device)
    rtl = torch.empty_like(ltr)
    cross = torch.empty(
        batch_size,
        6,
        _TIER_LOCAL_CARDS,
        _TIER_LOCAL_CARDS,
        dtype=dtype,
        device=device,
    )
    vec_block = 64
    _tier2_total_pair_vec_desc_kernel[(batch_size, 6)](
        local_belief_matrix.contiguous(),
        player_card_cumsum.contiguous(),
        rowdot,
        ltr,
        rtl,
        B=batch_size,
        CARD_COUNT=_TIER_LOCAL_CARDS,
        SLOT_CAP=_TIER_CARD_SLOT_CAP,
        TOTAL_SLOT=_TIER_HANDS_PER_LOCAL_CARD,
        BLOCK_C=vec_block,
        num_warps=2,
    )
    block_m = _env_int("P2_SHOWDOWN_TIER2_TOTAL_DESC_BLOCK_M", 16)
    block_n = _env_int("P2_SHOWDOWN_TIER2_TOTAL_DESC_BLOCK_N", 16)
    tiles_m = triton.cdiv(_TIER_LOCAL_CARDS, block_m)
    tiles_n = triton.cdiv(_TIER_LOCAL_CARDS, block_n)
    grid_cross = (
        batch_size,
        6,
        tiles_m * tiles_n,
    )
    _tier2_total_pair_cross_desc_kernel[grid_cross](
        local_belief_matrix.contiguous(),
        cross,
        B=batch_size,
        CARD_COUNT=_TIER_LOCAL_CARDS,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=64,
        TILES_N=tiles_n,
        num_warps=_env_int("P2_SHOWDOWN_TIER2_TOTAL_DESC_WARPS", 4),
    )
    return rowdot, ltr, rtl, cross


def _tier2_p4_sparse_direct_finish_triton(
    *,
    scalar_all: torch.Tensor,
    player_card_cumsum: torch.Tensor,
    local_belief_matrix: torch.Tensor,
    ctx: _ActiveTierContext,
    same_all: torch.Tensor,
    reuse_vectors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        triton is None
        or scalar_all.device.type != "cuda"
        or scalar_all.shape[0] != 4
        or player_card_cumsum.device.type != "cuda"
    ):
        return None
    batch_size = scalar_all.shape[2]
    active_count = scalar_all.shape[3]
    numerator = torch.empty(
        batch_size,
        4,
        active_count,
        dtype=scalar_all.dtype,
        device=scalar_all.device,
    )
    denominator = torch.empty_like(numerator)
    equity = torch.empty_like(numerator)
    use_total_desc = os.environ.get("P2_SHOWDOWN_TIER2_TOTAL_DESC") == "1"
    use_compact_lut = os.environ.setdefault("P2_SHOWDOWN_TIER2_COMPACT_LUT", "1") == "1"
    if reuse_vectors:
        block_h = _env_int(
            "P2_SHOWDOWN_TIER2_REUSE_FINISH_BLOCK_H",
            16,
        )
        num_warps = _env_int(
            "P2_SHOWDOWN_TIER2_REUSE_FINISH_WARPS",
            4,
        )
    else:
        block_h = _env_int("P2_SHOWDOWN_TIER2_DIRECT_FINISH_BLOCK_H", 8)
        num_warps = _env_int("P2_SHOWDOWN_TIER2_DIRECT_FINISH_WARPS", 2)
    grid = (batch_size, triton.cdiv(active_count, block_h))
    if not use_total_desc:
        finish_kernel = (
            _tier2_p4_sparse_direct_finish_reuse_kernel
            if reuse_vectors
            else _tier2_p4_sparse_direct_finish_plain_kernel
        )
        finish_kernel[grid](
            scalar_all.contiguous(),
            player_card_cumsum.contiguous(),
            local_belief_matrix.contiguous(),
            ctx.local_c0.contiguous(),
            ctx.local_c1.contiguous(),
            ctx.pair_p_rank_flags.contiguous(),
            ctx.pair_q_rank_flags.contiguous(),
            ctx.pair_rank_flags.contiguous(),
            ctx.slot_lower_by_card.contiguous(),
            ctx.slot_tie_by_card.contiguous(),
            ctx.slot_lower_tie_by_card.contiguous(),
            same_all.contiguous(),
            numerator,
            denominator,
            equity,
            B=batch_size,
            H=active_count,
            CARD_COUNT=_TIER_LOCAL_CARDS,
            SLOT_CAP=_TIER_CARD_SLOT_CAP,
            TOTAL_SLOT=_TIER_HANDS_PER_LOCAL_CARD,
            BLOCK_H=block_h,
            BLOCK_C=64,
            USE_COMPACT_LUT=use_compact_lut,
            num_warps=num_warps,
        )
        return numerator, denominator, equity
    total_desc = _tier2_p4_total_pair_descriptors_triton(
        local_belief_matrix=local_belief_matrix,
        player_card_cumsum=player_card_cumsum,
    )
    if total_desc is None:
        return None
    total_desc_rowdot, total_desc_ltr, total_desc_rtl, total_desc_cross = total_desc
    _tier2_p4_sparse_direct_finish_kernel[grid](
        scalar_all.contiguous(),
        player_card_cumsum.contiguous(),
        local_belief_matrix.contiguous(),
        total_desc_rowdot.contiguous(),
        total_desc_ltr.contiguous(),
        total_desc_rtl.contiguous(),
        total_desc_cross.contiguous(),
        ctx.local_c0.contiguous(),
        ctx.local_c1.contiguous(),
        ctx.pair_p_rank_flags.contiguous(),
        ctx.pair_q_rank_flags.contiguous(),
        ctx.slot_lower_by_card.contiguous(),
        ctx.slot_tie_by_card.contiguous(),
        same_all.contiguous(),
        numerator,
        denominator,
        equity,
        B=batch_size,
        H=active_count,
        CARD_COUNT=_TIER_LOCAL_CARDS,
        SLOT_CAP=_TIER_CARD_SLOT_CAP,
        TOTAL_SLOT=_TIER_HANDS_PER_LOCAL_CARD,
        BLOCK_H=block_h,
        BLOCK_C=64,
        USE_TOTAL_DESC=True,
        num_warps=num_warps,
    )
    return numerator, denominator, equity


def _tier2_p4_sparse_prefix_direct_finish_triton(
    *,
    beliefs: torch.Tensor,
    full_beliefs: torch.Tensor,
    scalar_prefix: torch.Tensor,
    pair_prefix: torch.Tensor,
    player_card_cumsum: torch.Tensor,
    pair_card_cumsum: torch.Tensor,
    ctx: _ActiveTierContext,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        triton is None
        or beliefs.device.type != "cuda"
        or beliefs.shape[1] != 4
        or full_beliefs.device.type != "cuda"
    ):
        return None
    batch_size = beliefs.shape[0]
    active_count = beliefs.shape[2]
    numerator = torch.empty(
        batch_size,
        4,
        active_count,
        dtype=beliefs.dtype,
        device=beliefs.device,
    )
    denominator = torch.empty_like(numerator)
    equity = torch.empty_like(numerator)
    block_h = _env_int("P2_SHOWDOWN_TIER2_PREFIX_DIRECT_BLOCK_H", 4)
    num_warps = _env_int("P2_SHOWDOWN_TIER2_PREFIX_DIRECT_WARPS", 2)
    grid = (batch_size, triton.cdiv(active_count, block_h))
    _tier2_p4_sparse_prefix_direct_finish_kernel[grid](
        beliefs.contiguous(),
        full_beliefs.contiguous(),
        scalar_prefix.contiguous(),
        pair_prefix.contiguous(),
        player_card_cumsum.contiguous(),
        pair_card_cumsum.contiguous(),
        ctx.local_c0.contiguous(),
        ctx.local_c1.contiguous(),
        ctx.pair_p_ids.contiguous(),
        ctx.pair_q_ids.contiguous(),
        ctx.pair_p_rank_flags.contiguous(),
        ctx.pair_q_rank_flags.contiguous(),
        ctx.lower_end.contiguous(),
        ctx.tie_end.contiguous(),
        ctx.slot_lower_by_card.contiguous(),
        ctx.slot_tie_by_card.contiguous(),
        numerator,
        denominator,
        equity,
        B=batch_size,
        H=active_count,
        H1=active_count + 1,
        FULL_HANDS=NUM_HANDS,
        CARD_COUNT=_TIER_LOCAL_CARDS,
        SLOT_CAP=_TIER_CARD_SLOT_CAP,
        TOTAL_SLOT=_TIER_HANDS_PER_LOCAL_CARD,
        BLOCK_H=block_h,
        BLOCK_C=64,
        num_warps=num_warps,
    )
    return numerator, denominator, equity


def _tier2_p4_fill_total_pair_event_pie(
    pair_event: torch.Tensor,
    local_belief_matrix: torch.Tensor,
    ctx: _ActiveTierContext,
) -> None:
    pair_left = torch.tensor(
        [pair[0] for pair in _P4_PLAYER_PAIRS],
        dtype=torch.long,
        device=local_belief_matrix.device,
    )
    pair_right = torch.tensor(
        [pair[1] for pair in _P4_PLAYER_PAIRS],
        dtype=torch.long,
        device=local_belief_matrix.device,
    )
    mats = local_belief_matrix.clone()
    diag = torch.arange(_TIER_LOCAL_CARDS, device=mats.device)
    mats[:, :, diag, diag] = 0.0
    left_mats = mats[:, pair_left]
    right_mats = mats[:, pair_right]
    left_rows = left_mats.sum(dim=3)
    right_rows = right_mats.sum(dim=3)
    rowdot = (left_rows * right_rows).sum(dim=2)
    same = 0.5 * (left_mats * right_mats).sum(dim=(2, 3))
    left_to_right = torch.matmul(right_mats, left_rows.unsqueeze(-1)).squeeze(-1)
    right_to_left = torch.matmul(left_mats, right_rows.unsqueeze(-1)).squeeze(-1)
    cross = torch.matmul(left_mats, right_mats)

    a = ctx.local_c0.long()
    b = ctx.local_c1.long()
    gather_a = a[:, None, :].expand(-1, 6, -1)
    gather_b = b[:, None, :].expand(-1, 6, -1)
    lr_a = left_rows.gather(2, gather_a)
    lr_b = left_rows.gather(2, gather_b)
    rr_a = right_rows.gather(2, gather_a)
    rr_b = right_rows.gather(2, gather_b)
    ltr_a = left_to_right.gather(2, gather_a)
    ltr_b = left_to_right.gather(2, gather_b)
    rtl_a = right_to_left.gather(2, gather_a)
    rtl_b = right_to_left.gather(2, gather_b)

    batch_idx = torch.arange(local_belief_matrix.shape[0], device=mats.device)[:, None, None]
    pair_idx = torch.arange(6, device=mats.device)[None, :, None]
    a_idx = a[:, None, :]
    b_idx = b[:, None, :]
    left_ab = left_mats[batch_idx, pair_idx, a_idx, b_idx]
    right_ab = right_mats[batch_idx, pair_idx, a_idx, b_idx]
    cross_aa = cross[batch_idx, pair_idx, a_idx, a_idx]
    cross_ab = cross[batch_idx, pair_idx, a_idx, b_idx]
    cross_ba = cross[batch_idx, pair_idx, b_idx, a_idx]
    cross_bb = cross[batch_idx, pair_idx, b_idx, b_idx]

    row_excl = rowdot[:, :, None] - lr_a * rr_a - lr_b * rr_b
    row_excl = row_excl - (ltr_a - lr_b * right_ab + ltr_b - lr_a * right_ab)
    row_excl = row_excl - (rtl_a - rr_b * left_ab + rtl_b - rr_a * left_ab)
    row_excl = row_excl + (
        cross_aa + cross_ab + cross_ba + cross_bb - 2.0 * left_ab * right_ab
    )
    same_excl = same[:, :, None] - cross_aa - cross_bb + left_ab * right_ab
    pair_event[:, 4] = (row_excl - same_excl).permute(1, 0, 2)


def _tier2_p4_group_prefixes_triton(
    *,
    sorted_beliefs: torch.Tensor,
    sorted_group_id: torch.Tensor,
    sorted_c0: torch.Tensor,
    sorted_c1: torch.Tensor,
    group_count: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or sorted_beliefs.device.type != "cuda" or sorted_beliefs.shape[1] != 4:
        return None
    batch_size, _, active_count = sorted_beliefs.shape
    device = sorted_beliefs.device
    scalar_group = torch.zeros(batch_size, 4, group_count, dtype=dtype, device=device)
    card_group = torch.zeros(batch_size, 4, group_count, 47, dtype=dtype, device=device)
    pair_group = torch.zeros(batch_size, 6, group_count, dtype=dtype, device=device)
    pair_card_group = torch.zeros(batch_size, 6, group_count, 47, dtype=dtype, device=device)
    local_belief_matrix = torch.empty(batch_size, 4, 47, 47, dtype=dtype, device=device)
    block_h = 128
    grid = (batch_size, triton.cdiv(active_count, block_h))
    _tier2_p4_group_accum_kernel[grid](
        sorted_beliefs.contiguous(),
        sorted_group_id.contiguous(),
        sorted_c0.contiguous(),
        sorted_c1.contiguous(),
        scalar_group,
        card_group,
        pair_group,
        pair_card_group,
        local_belief_matrix,
        B=batch_size,
        H=active_count,
        G=group_count,
        CARD_COUNT=47,
        BLOCK_H=block_h,
        num_warps=4,
    )
    scalar_prefix = torch.empty(
        batch_size,
        4,
        group_count + 1,
        dtype=dtype,
        device=device,
    )
    scalar_prefix[:, :, 0] = 0.0
    torch.cumsum(scalar_group, dim=2, out=scalar_prefix[:, :, 1:])
    card_prefix = torch.empty(
        batch_size,
        4,
        group_count + 1,
        47,
        dtype=dtype,
        device=device,
    )
    card_prefix[:, :, 0] = 0.0
    torch.cumsum(card_group, dim=2, out=card_prefix[:, :, 1:])
    pair_prefix = torch.empty(
        batch_size,
        6,
        group_count + 1,
        dtype=dtype,
        device=device,
    )
    pair_prefix[:, :, 0] = 0.0
    torch.cumsum(pair_group, dim=2, out=pair_prefix[:, :, 1:])
    pair_card_prefix = torch.empty(
        batch_size,
        6,
        group_count + 1,
        47,
        dtype=dtype,
        device=device,
    )
    pair_card_prefix[:, :, 0] = 0.0
    torch.cumsum(pair_card_group, dim=2, out=pair_card_prefix[:, :, 1:])
    return scalar_prefix, card_prefix, pair_prefix, pair_card_prefix, local_belief_matrix


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
    block_h = _env_int("P2_SHOWDOWN_WEDGE_BLOCK_H", 4)
    block_k = _env_int("P2_SHOWDOWN_WEDGE_BLOCK_K", 16)
    k_blocks = triton.cdiv(active_count, block_k)
    all_heroes = _env_int("P2_SHOWDOWN_WEDGE_ALL_HEROES", 1)
    split_k = _env_int("P2_SHOWDOWN_WEDGE_SPLIT_K", 2 if all_heroes else 1)
    part_k_blocks = triton.cdiv(k_blocks, split_k)
    num_warps = _env_int("P2_SHOWDOWN_WEDGE_NUM_WARPS", 1)
    num_stages = _env_int("P2_SHOWDOWN_WEDGE_NUM_STAGES", 2 if all_heroes else 3)
    maxnreg = _env_int("P2_SHOWDOWN_WEDGE_MAXNREG", 0)
    num_out = torch.empty(
        batch_size,
        4,
        active_count,
        dtype=beliefs.dtype,
        device=beliefs.device,
    )
    den_out = torch.empty_like(num_out)
    h_blocks = triton.cdiv(active_count, block_h)
    if all_heroes:
        if split_k > 1:
            num_out.zero_()
            den_out.zero_()
        grid_all = (batch_size, h_blocks, split_k)
        if maxnreg > 0:
            _tier3_wedge_p4_all_heroes_kernel[grid_all](
                beliefs.contiguous(),
                ranks.contiguous(),
                local_c0.contiguous(),
                local_c1.contiguous(),
                card_all.contiguous(),
                num_out,
                den_out,
                B=batch_size,
                H=active_count,
                CARD_COUNT=47,
                K_BLOCKS=k_blocks,
                PART_K_BLOCKS=part_k_blocks,
                SPLIT_K=split_k,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                COMPUTE_DEN=True,
                num_warps=num_warps,
                num_stages=num_stages,
                maxnreg=maxnreg,
            )
        else:
            _tier3_wedge_p4_all_heroes_kernel[grid_all](
                beliefs.contiguous(),
                ranks.contiguous(),
                local_c0.contiguous(),
                local_c1.contiguous(),
                card_all.contiguous(),
                num_out,
                den_out,
                B=batch_size,
                H=active_count,
                CARD_COUNT=47,
                K_BLOCKS=k_blocks,
                PART_K_BLOCKS=part_k_blocks,
                SPLIT_K=split_k,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                COMPUTE_DEN=True,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    else:
        grid = (batch_size, 4, h_blocks)
        if maxnreg > 0:
            _tier3_wedge_p4_kernel[grid](
                beliefs.contiguous(),
                ranks.contiguous(),
                local_c0.contiguous(),
                local_c1.contiguous(),
                card_all.contiguous(),
                num_out,
                den_out,
                B=batch_size,
                H=active_count,
                CARD_COUNT=47,
                K_BLOCKS=k_blocks,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
                maxnreg=maxnreg,
            )
        else:
            _tier3_wedge_p4_kernel[grid](
                beliefs.contiguous(),
                ranks.contiguous(),
                local_c0.contiguous(),
                local_c1.contiguous(),
                card_all.contiguous(),
                num_out,
                den_out,
                B=batch_size,
                H=active_count,
                CARD_COUNT=47,
                K_BLOCKS=k_blocks,
                BLOCK_H=block_h,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return num_out, den_out


def _p4_pair_event_triton(
    card_all: torch.Tensor,
    same_all: torch.Tensor,
    *,
    finish_only: bool = False,
) -> torch.Tensor | None:
    if triton is None or card_all.device.type != "cuda" or card_all.shape[0] != 4:
        return None
    batch_size = card_all.shape[2]
    active_count = card_all.shape[3]
    block_h = 4
    if finish_only:
        out = torch.empty(
            6,
            5,
            batch_size,
            active_count,
            dtype=card_all.dtype,
            device=card_all.device,
        )
        grid = (batch_size, 6, triton.cdiv(active_count, block_h))
        _p4_pair_event_finish_kernel[grid](
            card_all.contiguous(),
            same_all.contiguous(),
            out,
            B=batch_size,
            H=active_count,
            CARD_COUNT=47,
            BLOCK_H=block_h,
            BLOCK_C=64,
            num_warps=1,
        )
        return out
    out = torch.empty(
        6,
        3,
        3,
        batch_size,
        active_count,
        dtype=card_all.dtype,
        device=card_all.device,
    )
    grid = (batch_size, 6, triton.cdiv(active_count, block_h))
    _p4_pair_event_kernel[grid](
        card_all.contiguous(),
        same_all.contiguous(),
        out,
        B=batch_size,
        H=active_count,
        CARD_COUNT=47,
        BLOCK_H=block_h,
        BLOCK_C=64,
        num_warps=1,
    )
    return out


def _tier2_p4_finish_triton(
    scalar_all: torch.Tensor,
    pair_event_all: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        triton is None
        or scalar_all.device.type != "cuda"
        or scalar_all.shape[0] != 4
        or pair_event_all.dim() not in (4, 5)
    ):
        return None
    batch_size = scalar_all.shape[2]
    active_count = scalar_all.shape[3]
    numerator = torch.empty(
        batch_size,
        4,
        active_count,
        dtype=scalar_all.dtype,
        device=scalar_all.device,
    )
    denominator = torch.empty_like(numerator)
    equity = torch.empty_like(numerator)
    block_h = 64
    grid = (batch_size, 4, triton.cdiv(active_count, block_h))
    _tier2_p4_finish_kernel[grid](
        scalar_all.contiguous(),
        pair_event_all.contiguous(),
        numerator,
        denominator,
        equity,
        B=batch_size,
        H=active_count,
        BLOCK_H=block_h,
        FINISH_ONLY_EVENTS=pair_event_all.dim() == 4,
        num_warps=1,
    )
    return numerator, denominator, equity


def _tier2_prefix_factors(
    prepared: PreparedShowdown,
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
    same_num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beliefs = ctx.beliefs
    batch_size, players, active_count = beliefs.shape
    device = beliefs.device
    local_c0, local_c1 = _active_local_combo_cards(ctx)
    sorted_beliefs = beliefs.gather(2, ctx.order[:, None, :].expand(-1, players, -1))
    sorted_contains = (
        ctx.sorted_contains
        if ctx.sorted_contains.dtype == dtype
        else ctx.sorted_contains.to(dtype)
    )
    group_count = ctx.max_rank_groups
    use_compact_p4_pairs = triton is not None and device.type == "cuda" and players == 4
    group_prefixes = (
        _tier2_p4_group_prefixes_triton(
            sorted_beliefs=sorted_beliefs,
            sorted_group_id=ctx.sorted_group_id,
            sorted_c0=ctx.sorted_c0,
            sorted_c1=ctx.sorted_c1,
            group_count=group_count,
            dtype=dtype,
        )
        if use_compact_p4_pairs
        else None
    )
    if group_prefixes is not None:
        (
            scalar_prefix,
            card_prefix,
            pair_prefix,
            pair_card_prefix,
            local_belief_matrix,
        ) = group_prefixes
    else:
        local_belief_matrix = None
        group_index = ctx.sorted_group_id.long()
        group_index_players = group_index[:, None, :].expand(-1, players, -1)

        scalar_prefix = torch.empty(
            batch_size,
            players,
            group_count + 1,
            dtype=dtype,
            device=device,
        )
        scalar_prefix[:, :, 0] = 0.0
        scalar_group = torch.zeros(
            batch_size,
            players,
            group_count,
            dtype=dtype,
            device=device,
        )
        scalar_group.scatter_add_(2, group_index_players, sorted_beliefs)
        torch.cumsum(scalar_group, dim=2, out=scalar_prefix[:, :, 1:])
        card_values = sorted_beliefs[:, :, :, None] * sorted_contains[:, None, :, :]
        card_prefix = torch.empty(
            batch_size,
            players,
            group_count + 1,
            47,
            dtype=dtype,
            device=device,
        )
        card_prefix[:, :, 0] = 0.0
        card_group = torch.zeros(
            batch_size,
            players,
            group_count,
            47,
            dtype=dtype,
            device=device,
        )
        card_group.scatter_add_(
            2,
            group_index_players[..., None].expand(-1, -1, -1, 47),
            card_values,
        )
        torch.cumsum(card_group, dim=2, out=card_prefix[:, :, 1:])

        if use_compact_p4_pairs:
            pair_values = torch.stack(
                (
                    sorted_beliefs[:, 0] * sorted_beliefs[:, 1],
                    sorted_beliefs[:, 0] * sorted_beliefs[:, 2],
                    sorted_beliefs[:, 0] * sorted_beliefs[:, 3],
                    sorted_beliefs[:, 1] * sorted_beliefs[:, 2],
                    sorted_beliefs[:, 1] * sorted_beliefs[:, 3],
                    sorted_beliefs[:, 2] * sorted_beliefs[:, 3],
                ),
                dim=1,
            )
            pair_prefix = torch.empty(batch_size, 6, group_count + 1, dtype=dtype, device=device)
            pair_prefix[:, :, 0] = 0.0
            pair_group = torch.zeros(
                batch_size,
                6,
                group_count,
                dtype=dtype,
                device=device,
            )
            pair_group.scatter_add_(
                2,
                group_index[:, None, :].expand(-1, 6, -1),
                pair_values,
            )
            torch.cumsum(pair_group, dim=2, out=pair_prefix[:, :, 1:])
            pair_card_values = pair_values[:, :, :, None] * sorted_contains[:, None, :, :]
            pair_card_prefix = torch.empty(
                batch_size,
                6,
                group_count + 1,
                47,
                dtype=dtype,
                device=device,
            )
            pair_card_prefix[:, :, 0] = 0.0
            pair_card_group = torch.zeros(
                batch_size,
                6,
                group_count,
                47,
                dtype=dtype,
                device=device,
            )
            pair_card_group.scatter_add_(
                2,
                group_index[:, None, :, None].expand(-1, 6, -1, 47),
                pair_card_values,
            )
            torch.cumsum(pair_card_group, dim=2, out=pair_card_prefix[:, :, 1:])
        else:
            pair_values = sorted_beliefs[:, :, None, :] * sorted_beliefs[:, None, :, :]
            pair_prefix = torch.empty(
                batch_size,
                players,
                players,
                group_count + 1,
                dtype=dtype,
                device=device,
            )
            pair_prefix[:, :, :, 0] = 0.0
            pair_group = torch.zeros(
                batch_size,
                players,
                players,
                group_count,
                dtype=dtype,
                device=device,
            )
            pair_group.scatter_add_(
                3,
                group_index[:, None, None, :].expand(-1, players, players, -1),
                pair_values,
            )
            torch.cumsum(pair_group, dim=3, out=pair_prefix[:, :, :, 1:])
            pair_card_values = pair_values[:, :, :, :, None] * sorted_contains[:, None, None, :, :]
            pair_card_prefix = torch.empty(
                batch_size,
                players,
                players,
                group_count + 1,
                47,
                dtype=dtype,
                device=device,
            )
            pair_card_prefix[:, :, :, 0] = 0.0
            pair_card_group = torch.zeros(
                batch_size,
                players,
                players,
                group_count,
                47,
                dtype=dtype,
                device=device,
            )
            pair_card_group.scatter_add_(
                3,
                group_index[:, None, None, :, None].expand(-1, players, players, -1, 47),
                pair_card_values,
            )
            torch.cumsum(pair_card_group, dim=3, out=pair_card_prefix[:, :, :, 1:])

    lower_end = ctx.lower_group_end
    tie_end = ctx.tie_group_end
    total_end = ctx.rank_group_count
    full_beliefs = prepared.beliefs.to(dtype).contiguous()
    pair_p_ids = ctx.pair_p_ids
    pair_q_ids = ctx.pair_q_ids
    if triton is not None and device.type == "cuda":
        if local_belief_matrix is None:
            local_belief_matrix = _local_belief_matrix(beliefs, local_c0, local_c1)
        return _tier2_prefix_factors_triton(
            scalar_prefix=scalar_prefix.contiguous(),
            card_prefix=card_prefix.contiguous(),
            pair_prefix=pair_prefix.contiguous(),
            pair_card_prefix=pair_card_prefix.contiguous(),
            beliefs=beliefs.contiguous(),
            local_belief_matrix=local_belief_matrix.contiguous(),
            local_c0=local_c0.contiguous(),
            local_c1=local_c1.contiguous(),
            pair_p_rank_flags=ctx.pair_p_rank_flags.contiguous(),
            pair_q_rank_flags=ctx.pair_q_rank_flags.contiguous(),
            lower_end=lower_end.contiguous(),
            tie_end=tie_end.contiguous(),
            group_count=total_end.contiguous(),
            dtype=dtype,
            same_num_warps=same_num_warps,
        )

    zero_index = torch.zeros_like(lower_end)
    total_end_expanded = total_end[:, None].expand(-1, active_count)
    starts = [zero_index, lower_end, zero_index]
    ends = [lower_end, tie_end, total_end_expanded]

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


def _tier2_p4_direct_pair_from_group_prefixes(
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    beliefs = ctx.beliefs
    if triton is None or beliefs.device.type != "cuda" or beliefs.shape[1] != 4:
        return None
    sorted_beliefs = beliefs.gather(2, ctx.order[:, None, :].expand(-1, 4, -1))
    group_prefixes = _tier2_p4_group_prefixes_triton(
        sorted_beliefs=sorted_beliefs,
        sorted_group_id=ctx.sorted_group_id,
        sorted_c0=ctx.sorted_c0,
        sorted_c1=ctx.sorted_c1,
        group_count=ctx.max_rank_groups,
        dtype=dtype,
    )
    if group_prefixes is None:
        return None
    scalar_prefix, card_prefix, pair_prefix, pair_card_prefix, local_belief_matrix = (
        group_prefixes
    )
    scalar_same = _tier2_scalar_same_from_prefix_triton(
        scalar_prefix=scalar_prefix,
        card_prefix=card_prefix,
        pair_prefix=pair_prefix,
        pair_card_prefix=pair_card_prefix,
        beliefs=beliefs.contiguous(),
        local_c0=ctx.local_c0,
        local_c1=ctx.local_c1,
        lower_end=ctx.lower_group_end,
        tie_end=ctx.tie_group_end,
        group_count=ctx.rank_group_count,
        dtype=dtype,
    )
    if scalar_same is None:
        return None
    scalar_all, same_all = scalar_same
    pair_event_all = _tier2_p4_pair_event_from_prefix_triton(
        card_prefix=card_prefix,
        local_belief_matrix=local_belief_matrix,
        local_c0=ctx.local_c0,
        local_c1=ctx.local_c1,
        pair_p_rank_flags=ctx.pair_p_rank_flags,
        pair_q_rank_flags=ctx.pair_q_rank_flags,
        lower_end=ctx.lower_group_end,
        tie_end=ctx.tie_group_end,
        group_count=ctx.rank_group_count,
        same_all=same_all,
    )
    if pair_event_all is None:
        return None
    return scalar_all, pair_event_all


def _tier2_p4_sparse_by_card(
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    beliefs = ctx.beliefs
    if triton is None or beliefs.device.type != "cuda" or beliefs.shape[1] != 4:
        return None
    sorted_beliefs = beliefs.gather(2, ctx.order[:, None, :].expand(-1, 4, -1))
    prefixes = _tier2_p4_sparse_prefixes_triton(
        sorted_beliefs=sorted_beliefs,
        sorted_card_positions=ctx.sorted_card_positions,
        dtype=dtype,
    )
    if prefixes is None:
        return None
    scalar_prefix, pair_prefix, player_card_cumsum, pair_card_cumsum = prefixes
    scalar_same = _tier2_p4_sparse_scalar_same_triton(
        beliefs=beliefs,
        scalar_prefix=scalar_prefix,
        pair_prefix=pair_prefix,
        player_card_cumsum=player_card_cumsum,
        pair_card_cumsum=pair_card_cumsum,
        ctx=ctx,
        dtype=dtype,
    )
    if scalar_same is None:
        return None
    scalar_all, same_all, local_belief_matrix = scalar_same
    pair_event_all = _tier2_p4_pair_event_from_sparse_triton(
        player_card_cumsum=player_card_cumsum,
        local_belief_matrix=local_belief_matrix,
        ctx=ctx,
        same_all=same_all,
    )
    if pair_event_all is None:
        return None
    return scalar_all, pair_event_all


def _tier2_p4_sparse_direct_finish_by_card(
    prepared: PreparedShowdown,
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
    reuse_vectors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    beliefs = ctx.beliefs
    if triton is None or beliefs.device.type != "cuda" or beliefs.shape[1] != 4:
        return None
    sorted_beliefs = beliefs.gather(2, ctx.order[:, None, :].expand(-1, 4, -1))
    prefixes = _tier2_p4_sparse_prefixes_triton(
        sorted_beliefs=sorted_beliefs,
        sorted_card_positions=ctx.sorted_card_positions,
        dtype=dtype,
    )
    if prefixes is None:
        return None
    scalar_prefix, pair_prefix, player_card_cumsum, pair_card_cumsum = prefixes
    scalar_same = _tier2_p4_sparse_scalar_same_triton(
        beliefs=beliefs,
        scalar_prefix=scalar_prefix,
        pair_prefix=pair_prefix,
        player_card_cumsum=player_card_cumsum,
        pair_card_cumsum=pair_card_cumsum,
        ctx=ctx,
        dtype=dtype,
    )
    if scalar_same is None:
        return None
    scalar_all, same_all, local_belief_matrix = scalar_same
    return _tier2_p4_sparse_direct_finish_triton(
        scalar_all=scalar_all,
        player_card_cumsum=player_card_cumsum,
        local_belief_matrix=local_belief_matrix,
        ctx=ctx,
        same_all=same_all,
        reuse_vectors=reuse_vectors,
    )


def _tier2_p4_sparse_prefix_direct_finish_by_card(
    prepared: PreparedShowdown,
    ctx: _ActiveTierContext,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    beliefs = ctx.beliefs
    if triton is None or beliefs.device.type != "cuda" or beliefs.shape[1] != 4:
        return None
    sorted_beliefs = beliefs.gather(2, ctx.order[:, None, :].expand(-1, 4, -1))
    prefixes = _tier2_p4_sparse_prefixes_triton(
        sorted_beliefs=sorted_beliefs,
        sorted_card_positions=ctx.sorted_card_positions,
        dtype=dtype,
    )
    if prefixes is None:
        return None
    scalar_prefix, pair_prefix, player_card_cumsum, pair_card_cumsum = prefixes
    return _tier2_p4_sparse_prefix_direct_finish_triton(
        beliefs=beliefs,
        full_beliefs=prepared.beliefs.to(dtype=dtype),
        scalar_prefix=scalar_prefix,
        pair_prefix=pair_prefix,
        player_card_cumsum=player_card_cumsum,
        pair_card_cumsum=pair_card_cumsum,
        ctx=ctx,
    )


def tier1_hero_removal_by_hand(prepared: PreparedShowdown) -> PerHandEquityResult:
    start = time.perf_counter()
    dtype = torch.float32
    ctx = _active_context(prepared, dtype=dtype)
    beliefs = ctx.beliefs
    players = beliefs.shape[1]
    lower_rel, tie_rel, total_rel = _hand_relations(ctx, dtype=dtype)
    rel_f = [lower_rel, tie_rel, total_rel]

    equities: list[torch.Tensor] = []
    numerator_active = torch.empty(
        beliefs.shape[0],
        players,
        ctx.active_ids.shape[1],
        dtype=dtype,
        device=beliefs.device,
    )
    denominator_active = torch.empty_like(numerator_active)
    equity_active = torch.empty_like(numerator_active)
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
    by_card_mode = os.environ.get("P2_SHOWDOWN_TIER2_BY_CARD", "3")
    direct_finish = None
    if by_card_mode == "4":
        direct_finish = _tier2_p4_sparse_prefix_direct_finish_by_card(
            prepared,
            ctx,
            dtype=dtype,
        )
    elif by_card_mode == "5":
        direct_finish = _tier2_p4_sparse_direct_finish_by_card(
            prepared,
            ctx,
            dtype=dtype,
            reuse_vectors=True,
        )
    elif by_card_mode == "3":
        reuse_vectors = os.environ.get("P2_SHOWDOWN_TIER2_REUSE_VECTORS", "1") == "1"
        direct_finish = _tier2_p4_sparse_direct_finish_by_card(
            prepared,
            ctx,
            dtype=dtype,
            reuse_vectors=reuse_vectors,
        )
    if direct_finish is not None:
        numerator_active, denominator_active, equity_active = direct_finish
        numerator_by_hand, denominator_by_hand, equity_by_hand = _scatter_active_outputs(
            ctx,
            numerator_active,
            denominator_active,
            equity_active,
        )
        result = _aggregate_all_active_from_num_denom(
            beliefs,
            numerator_active,
            denominator_active,
        )
        if prepared.beliefs.device.type == "cuda":
            torch.cuda.synchronize(prepared.beliefs.device)
        return PerHandEquityResult(
            equity_by_hand=equity_by_hand.to(torch.float32),
            aggregate_equity=result,
            denominator_by_hand=denominator_by_hand,
            numerator_by_hand=numerator_by_hand,
            seconds=time.perf_counter() - start,
        )
    if by_card_mode == "2":
        direct_pair = _tier2_p4_sparse_by_card(ctx, dtype=dtype)
    elif by_card_mode == "1":
        direct_pair = _tier2_p4_direct_pair_from_group_prefixes(ctx, dtype=dtype)
    else:
        direct_pair = None
    if direct_pair is not None:
        scalar_all, pair_event_all = direct_pair
    else:
        scalar_all, card_all, same_all = _tier2_prefix_factors(prepared, ctx, dtype=dtype)
        pair_event_all = _pair_event_all_from_card(card_all, same_all, finish_only=True)
    p4_finish = _tier2_p4_finish_triton(scalar_all, pair_event_all)
    if p4_finish is not None:
        numerator_active, denominator_active, equity_active = p4_finish
        numerator_by_hand, denominator_by_hand, equity_by_hand = _scatter_active_outputs(
            ctx,
            numerator_active,
            denominator_active,
            equity_active,
        )
        result = _aggregate_all_active_from_num_denom(
            beliefs,
            numerator_active,
            denominator_active,
        )
        if prepared.beliefs.device.type == "cuda":
            torch.cuda.synchronize(prepared.beliefs.device)
        return PerHandEquityResult(
            equity_by_hand=equity_by_hand.to(torch.float32),
            aggregate_equity=result,
            denominator_by_hand=denominator_by_hand,
            numerator_by_hand=numerator_by_hand,
            seconds=time.perf_counter() - start,
        )

    equities: list[torch.Tensor] = []
    numerator_active = torch.empty(
        beliefs.shape[0],
        players,
        active_count,
        dtype=dtype,
        device=device,
    )
    denominator_active = torch.empty_like(numerator_active)
    equity_active = torch.empty_like(numerator_active)
    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        opp_count = len(opponents)

        if opp_count == 3:
            scalar = [scalar_all[player] for player in opponents]
            l0, l1, l2 = scalar[0][0], scalar[1][0], scalar[2][0]
            t0, t1, t2 = scalar[0][1], scalar[1][1], scalar[2][1]
            numerator = (
                l0 * l1 * l2
                + 0.5 * (t0 * l1 * l2 + l0 * t1 * l2 + l0 * l1 * t2)
                + (t0 * t1 * l2 + t0 * l1 * t2 + l0 * t1 * t2) / 3.0
                + 0.25 * t0 * t1 * t2
            )
            denominator = scalar[0][2] * scalar[1][2] * scalar[2][2]
        else:
            scalar = scalar_all[opponents]
            lower = scalar[:, 0].permute(1, 2, 0).contiguous()
            tied = scalar[:, 1].permute(1, 2, 0).contiguous()
            denom_terms = scalar[:, 2].permute(1, 2, 0).contiguous()
            numerator = _independent_share_numerators(lower, tied)
            denominator = denom_terms.prod(dim=-1)

        for left in range(opp_count):
            for right in range(left + 1, opp_count):
                other = [idx for idx in range(opp_count) if idx not in (left, right)]
                pair_lr = _pair_event_lookup(pair_event_all, opponents, left, right)
                if opp_count == 3:
                    other_idx = other[0]
                    denominator = (
                        denominator - pair_lr[2, 2] * scalar[other_idx][2]
                    )
                    pair_num = _tier2_pair_num_three_opponents(
                        pair_lr,
                        scalar[other_idx],
                    )
                else:
                    pair_total = pair_lr[2, 2]
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
                        pair_factor = pair_lr[left_mode, right_mode]
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

    scalar_all, card_all, same_all = _tier2_prefix_factors(
        prepared,
        ctx,
        dtype=dtype,
        same_num_warps=2,
    )
    local_c0, local_c1 = _active_local_combo_cards(ctx)
    use_streaming_wedge = (
        triton is not None
        and players == 4
        and device.type == "cuda"
        and beliefs.shape[0] > 128
    )
    pair_event_all = _pair_event_all_from_card(
        card_all,
        same_all,
        finish_only=use_streaming_wedge,
    )
    wedge_p4 = (
        _tier3_wedge_p4_triton(beliefs, ctx.ranks, local_c0, local_c1, card_all)
        if use_streaming_wedge
        else None
    )
    p4_finish = _tier2_p4_finish_triton(scalar_all, pair_event_all) if wedge_p4 is not None else None
    if wedge_p4 is not None and p4_finish is not None:
        numerator_active, denominator_active, equity_active = p4_finish
        wedge_num_all, wedge_den_all = wedge_p4
        numerator_active.add_(wedge_num_all)
        denominator_active.add_(wedge_den_all)
        torch.div(numerator_active, denominator_active.clamp_min(1.0e-30), out=equity_active)
        equity_active.masked_fill_(denominator_active <= 0.0, 0.0)
        numerator_by_hand, denominator_by_hand, equity_by_hand = _scatter_active_outputs(
            ctx,
            numerator_active,
            denominator_active,
            equity_active,
        )
        result = _aggregate_all_active_from_num_denom(
            beliefs,
            numerator_active,
            denominator_active,
        )
        if prepared.beliefs.device.type == "cuda":
            torch.cuda.synchronize(prepared.beliefs.device)
        return PerHandEquityResult(
            equity_by_hand=equity_by_hand.to(torch.float32),
            aggregate_equity=result,
            denominator_by_hand=denominator_by_hand,
            numerator_by_hand=numerator_by_hand,
            seconds=time.perf_counter() - start,
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
    tie_weight_3 = None
    if wedge_p4 is None:
        tie_weight_3 = torch.empty(2, 2, 2, dtype=dtype, device=device)
        for mode0 in range(2):
            for mode1 in range(2):
                for mode2 in range(2):
                    tie_weight_3[mode0, mode1, mode2] = 1.0 / float(
                        mode0 + mode1 + mode2 + 1
                    )

    equities: list[torch.Tensor] = []
    numerator_active = torch.empty(
        beliefs.shape[0],
        players,
        active_count,
        dtype=dtype,
        device=device,
    )
    denominator_active = torch.empty_like(numerator_active)
    equity_active = torch.empty_like(numerator_active)
    for hero in range(players):
        opponents = [player for player in range(players) if player != hero]
        opp_count = len(opponents)
        edges = [
            (left, right)
            for left in range(opp_count)
            for right in range(left + 1, opp_count)
        ]
        scalar = scalar_all[opponents]
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
            pair_lr = _pair_event_lookup(pair_event_all, opponents, left, right)
            if opp_count == 3:
                other_idx = other[0]
                denominator = denominator - pair_lr[2, 2] * scalar[other_idx, 2]
                pair_num = _tier2_pair_num_three_opponents(
                    pair_lr,
                    scalar[other_idx],
                )
            else:
                denom_other = torch.ones(beliefs.shape[0], active_count, dtype=dtype, device=device)
                for opp_idx in other:
                    denom_other = denom_other * scalar[opp_idx, 2]
                denominator = denominator - pair_lr[2, 2] * denom_other

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
                        pair_lr[modes[left], modes[right]]
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
                    pair_first = _pair_event_lookup(pair_event_all, opponents, left_a, left_b)
                    pair_second = _pair_event_lookup(pair_event_all, opponents, right_a, right_b)
                    denominator = denominator + (
                        pair_first[2, 2]
                        * pair_second[2, 2]
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
                            pair_first[modes[left_a], modes[left_b]]
                            * pair_second[modes[right_a], modes[right_b]]
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
                    if tie_weight_3 is None:
                        raise RuntimeError("tie_weight_3 is required for dense tier3")
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
    numerator_active = torch.empty(
        beliefs.shape[0],
        players,
        active_count,
        dtype=dtype,
        device=device,
    )
    denominator_active = torch.empty_like(numerator_active)
    equity_active = torch.empty_like(numerator_active)
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
