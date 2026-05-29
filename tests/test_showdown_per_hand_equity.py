from __future__ import annotations

import torch
import pytest

from p2.env.card_utils import NUM_HANDS, board_allowed_hands
from p2.env.rules import rank_hands
from p2.showdown.approximate import (
    tier1_hero_removal,
    tier1_hero_removal_by_hand,
    tier2_first_order_opp_collision,
    tier2_first_order_opp_collision_by_hand,
    tier3_second_order_opp_collision,
    tier3_second_order_opp_collision_by_hand,
    tier4_third_degree_card_collision,
    tier4_third_degree_card_collision_by_hand,
)
from p2.showdown.compare_multiway_showdown_tiers import prepare_random_showdown, random_beliefs
from p2.showdown.exact import exact_nway_ie_axb_by_hand, exact_nway_ie_fast_by_hand
from p2.showdown.multiway_showdown_estimators import (
    PreparedShowdown,
    exact_nway_ie,
    exact_nway_ie_by_hand,
)


def _prepared(players: int, seed: int = 123):
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    return prepare_random_showdown(
        players=players,
        device=device,
        generator=generator,
        concentration=1.0,
    )


def _prepared_batch(players: int, batch_size: int = 2, seed: int = 123) -> PreparedShowdown:
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(seed)
    boards = torch.topk(
        torch.rand(batch_size, 52, device=device, generator=generator),
        5,
        dim=1,
    ).indices
    beliefs = random_beliefs(
        boards,
        players,
        generator=generator,
        concentration=1.0,
    )
    hand_ranks, _ = rank_hands(boards)
    combos = _prepared(players=players, seed=seed).combos
    hand_masks = ((1 << combos[:, 0]) | (1 << combos[:, 1])).long().contiguous()
    return PreparedShowdown(
        board=boards,
        beliefs=beliefs,
        hand_ranks=hand_ranks,
        combos=combos,
        hand_masks=hand_masks,
        setup_seconds=0.0,
    )


def _slice_prepared(prepared: PreparedShowdown, board_idx: int) -> PreparedShowdown:
    return PreparedShowdown(
        board=prepared.board[board_idx : board_idx + 1],
        beliefs=prepared.beliefs[board_idx : board_idx + 1],
        hand_ranks=prepared.hand_ranks[board_idx : board_idx + 1],
        combos=prepared.combos,
        hand_masks=prepared.hand_masks,
        setup_seconds=prepared.setup_seconds,
    )


def test_exact_two_player_by_hand_matches_direct_pairwise_formula() -> None:
    prepared = _prepared(players=2)
    result = exact_nway_ie_by_hand(prepared, pattern_chunk=64)
    beliefs = prepared.beliefs[0].to(torch.float64)
    ranks = prepared.hand_ranks.reshape(NUM_HANDS)
    masks = prepared.hand_masks.reshape(NUM_HANDS)
    disjoint = (masks[:, None] & masks[None, :]) == 0
    allowed = board_allowed_hands(prepared.board)[0]

    for hero in range(2):
        opp = 1 - hero
        worse = disjoint & (ranks[None, :] < ranks[:, None])
        tied = disjoint & (ranks[None, :] == ranks[:, None])
        denom = disjoint.to(torch.float64).matmul(beliefs[opp])
        numer = worse.to(torch.float64).matmul(beliefs[opp])
        numer = numer + 0.5 * tied.to(torch.float64).matmul(beliefs[opp])
        expected = torch.where(
            allowed & (denom > 0),
            (numer / denom.clamp_min(1.0e-30)).to(torch.float32),
            torch.zeros(NUM_HANDS, dtype=torch.float32),
        )
        assert torch.allclose(result.equity_by_hand[hero], expected)


@pytest.mark.slow
def test_exact_by_hand_aggregate_matches_aggregate_oracle() -> None:
    prepared = _prepared(players=3)
    by_hand = exact_nway_ie_by_hand(prepared, pattern_chunk=64)
    aggregate = exact_nway_ie(prepared, pattern_chunk=64)

    assert torch.equal(by_hand.aggregate_equity, aggregate.equity)


@pytest.mark.slow
def test_axb_by_hand_matches_oracle_by_hand_for_four_players() -> None:
    prepared = _prepared(players=4)
    oracle = exact_nway_ie_by_hand(prepared, pattern_chunk=64)
    axb = exact_nway_ie_axb_by_hand(prepared)
    fast = exact_nway_ie_fast_by_hand(prepared, pattern_chunk=64)

    assert torch.equal(axb.aggregate_equity, oracle.aggregate_equity)
    assert torch.equal(fast.aggregate_equity, oracle.aggregate_equity)
    assert axb.equity_by_hand.shape == (1, 4, NUM_HANDS)
    assert fast.equity_by_hand.shape == (1, 4, NUM_HANDS)
    assert torch.allclose(axb.equity_by_hand[0], oracle.equity_by_hand, atol=1.0e-6)
    assert torch.allclose(fast.equity_by_hand[0], oracle.equity_by_hand, atol=1.0e-6)


@pytest.mark.slow
def test_axb_by_hand_batched_matches_sliced_calls() -> None:
    prepared = _prepared_batch(players=4, batch_size=2)
    batched = exact_nway_ie_axb_by_hand(prepared)
    assert batched.equity_by_hand.shape == (2, 4, NUM_HANDS)
    assert batched.denominator_by_hand.shape == (2, 4, NUM_HANDS)
    assert batched.aggregate_equity.shape == (2, 4)

    for board_idx in range(2):
        sliced = exact_nway_ie_axb_by_hand(_slice_prepared(prepared, board_idx))
        assert torch.equal(batched.aggregate_equity[board_idx], sliced.aggregate_equity[0])
        assert torch.equal(batched.equity_by_hand[board_idx], sliced.equity_by_hand[0])
        assert torch.equal(
            batched.denominator_by_hand[board_idx],
            sliced.denominator_by_hand[0],
        )


@pytest.mark.slow
def test_tier_by_hand_aggregates_match_aggregate_wrappers() -> None:
    prepared = _prepared(players=3)
    cases = [
        (tier1_hero_removal_by_hand, tier1_hero_removal),
        (tier2_first_order_opp_collision_by_hand, tier2_first_order_opp_collision),
        (tier3_second_order_opp_collision_by_hand, tier3_second_order_opp_collision),
        (tier4_third_degree_card_collision_by_hand, tier4_third_degree_card_collision),
    ]

    for by_hand_fn, aggregate_fn in cases:
        by_hand = by_hand_fn(prepared)
        aggregate = aggregate_fn(prepared)
        assert by_hand.equity_by_hand.shape == (1, prepared.beliefs.shape[1], NUM_HANDS)
        assert by_hand.denominator_by_hand.shape == (1, prepared.beliefs.shape[1], NUM_HANDS)
        assert by_hand.aggregate_equity.shape == (1, prepared.beliefs.shape[1])
        assert torch.equal(by_hand.aggregate_equity, aggregate.equity)


@pytest.mark.slow
def test_tier_by_hand_batched_matches_sliced_calls() -> None:
    prepared = _prepared_batch(players=3, batch_size=2)
    cases = [
        tier1_hero_removal_by_hand,
        tier2_first_order_opp_collision_by_hand,
        tier3_second_order_opp_collision_by_hand,
        tier4_third_degree_card_collision_by_hand,
    ]

    for by_hand_fn in cases:
        batched = by_hand_fn(prepared)
        assert batched.equity_by_hand.shape == (2, 3, NUM_HANDS)
        assert batched.denominator_by_hand.shape == (2, 3, NUM_HANDS)
        assert batched.aggregate_equity.shape == (2, 3)
        for board_idx in range(2):
            sliced = by_hand_fn(_slice_prepared(prepared, board_idx))
            assert torch.allclose(
                batched.equity_by_hand[board_idx],
                sliced.equity_by_hand[0],
                atol=1.0e-8,
            )
            assert torch.allclose(
                batched.denominator_by_hand[board_idx],
                sliced.denominator_by_hand[0],
                atol=1.0e-10,
            )
            assert torch.allclose(
                batched.aggregate_equity[board_idx],
                sliced.aggregate_equity[0],
                atol=1.0e-8,
            )


@pytest.mark.slow
def test_blocked_hands_are_zero_in_by_hand_outputs() -> None:
    prepared = _prepared(players=4)
    allowed = board_allowed_hands(prepared.board)[0]
    blocked = ~allowed

    exact = exact_nway_ie_axb_by_hand(prepared)
    tier1 = tier1_hero_removal_by_hand(prepared)

    assert torch.count_nonzero(exact.equity_by_hand[:, :, blocked]) == 0
    assert torch.count_nonzero(exact.denominator_by_hand[:, :, blocked]) == 0
    assert torch.count_nonzero(tier1.equity_by_hand[:, :, blocked]) == 0
    assert torch.count_nonzero(tier1.denominator_by_hand[:, :, blocked]) == 0
