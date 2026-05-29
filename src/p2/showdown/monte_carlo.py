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
from .results import PerHandEquityResult, safe_divide_by_hand


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
    combos = prepared.combos.long()
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
]
