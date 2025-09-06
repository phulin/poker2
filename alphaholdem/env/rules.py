from __future__ import annotations

from typing import Iterable, List, Tuple, Sequence
import random
import torch

# Card encoding: 0..51, rank = c % 13 (2..A), suit = c // 13 (0..3)

RANKS = list(range(13))  # 0..12 means 2..A
SUITS = list(range(4))


def new_shuffled_deck(rng: random.Random) -> List[int]:
    deck = list(range(52))
    rng.shuffle(deck)
    return deck


def rank(card: int) -> int:
    return card % 13


def suit(card: int) -> int:
    return card // 13


# Tensor helpers


def cards_to_onehot_indices(cards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert numeric card tensor to (suit_idx, rank_idx) tensors.

    cards: tensor of any shape with integral dtype; returns tensors of same shape
    containing suit indices [0..3] and rank indices [0..12].
    """
    cards = cards.to(torch.long)
    suits = cards // 13
    ranks = cards % 13
    return suits, ranks


# 7-card evaluator (simple, deterministic; optimized enough for HU env)
# Hand strength as a tuple: (category, tiebreakers...) where higher is better
# Categories (high to low): 8=StraightFlush,7=FourKind,6=FullHouse,5=Flush,4=Straight,3=ThreeKind,2=TwoPair,1=OnePair,0=HighCard


def evaluate_7(cards: Iterable[int]) -> Tuple[int, Tuple[int, ...]]:
    cs = list(cards)
    assert len(cs) == 7
    ranks = [rank(c) for c in cs]
    suits = [suit(c) for c in cs]

    # Count ranks
    rc = [0] * 13
    for r in ranks:
        rc[r] += 1

    # Flush detection
    sc = [0] * 4
    for s in suits:
        sc[s] += 1
    flush_suit = next((s for s, n in enumerate(sc) if n >= 5), None)

    # Straight detection helper (with wheel A-2-3-4-5)
    def best_straight(mask: List[bool]) -> int | None:
        present = {i for i, v in enumerate(mask) if v}
        # High-to-low search; return highest straight high-card index
        for high in range(12, 3, -1):
            needed = {high, high - 1, high - 2, high - 3, high - 4}
            if needed.issubset(present):
                return high
        # Wheel
        if {12, 0, 1, 2, 3}.issubset(present):
            return 3
        return None

    rank_mask = [rc[r] > 0 for r in range(13)]
    straight_high = best_straight(rank_mask)

    # Straight flush
    if flush_suit is not None:
        flush_rs = [rank(c) for c in cs if suit(c) == flush_suit]
        mask = [False] * 13
        for r in flush_rs:
            mask[r] = True
        sf_high = best_straight(mask)
        if sf_high is not None:
            return (8, (sf_high,))

    # Four of a kind
    if 4 in rc:
        four = max(r for r in range(13) if rc[r] == 4)
        kickers = sorted(
            [r for r in range(13) if r != four and rc[r] > 0], reverse=True
        )
        return (7, (four, kickers[0]))

    # Full house
    trips = sorted([r for r in range(13) if rc[r] == 3], reverse=True)
    pairs = sorted(
        [r for r in range(13) if rc[r] >= 2 and r not in trips], reverse=True
    )
    if trips and (len(trips) >= 2 or pairs):
        three = trips[0]
        two = trips[1] if len(trips) >= 2 else pairs[0]
        return (6, (three, two))

    # Flush
    if flush_suit is not None:
        flush_rs = sorted([rank(c) for c in cs if suit(c) == flush_suit], reverse=True)
        top5 = tuple(flush_rs[:5])
        return (5, top5)

    # Straight
    if straight_high is not None:
        return (4, (straight_high,))

    # Three of a kind
    if trips:
        three = trips[0]
        kickers = sorted(
            [r for r in range(13) if r != three and rc[r] > 0], reverse=True
        )[:2]
        return (3, (three, *kickers))

    # Two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        kicker = max(r for r in range(13) if rc[r] > 0 and r not in (p1, p2))
        return (2, (p1, p2, kicker))

    # One pair
    if len(pairs) == 1:
        p = pairs[0]
        kickers = sorted([r for r in range(13) if r != p and rc[r] > 0], reverse=True)[
            :3
        ]
        return (1, (p, *kickers))

    # High card
    top5 = sorted([r for r in range(13) if rc[r] > 0], reverse=True)[:5]
    return (0, tuple(top5))


def compare_7(a: Iterable[int], b: Iterable[int]) -> int:
    va = evaluate_7(a)
    vb = evaluate_7(b)
    if va > vb:
        return 1
    if va < vb:
        return -1
    return 0


def compare_7_batch(
    a_batch: torch.Tensor,
    b_batch: torch.Tensor,
) -> torch.Tensor:
    """Vectorized comparison for batches of 7-card hands using one-hot planes.

    Inputs must be one-hot planes [N,4,13]; returns [N] in {1, -1, 0}.
    The score layout per hand is [category, t1, t2, t3, t4, t5] where
    category is in {8:SF, 7:4K, 6:FH, 5:F, 4:S, 3:3K, 2:2P, 1:1P, 0:HC} and
    t1..t5 are tiebreaker ranks in descending priority for that category.
    """
    assert isinstance(a_batch, torch.Tensor) and isinstance(b_batch, torch.Tensor)
    assert (
        a_batch.shape == b_batch.shape
        and a_batch.dim() == 3
        and a_batch.shape[1:] == (4, 13)
    )
    device = a_batch.device

    def eval_from_counts(
        rank_counts: torch.Tensor, suit_counts: torch.Tensor
    ) -> torch.Tensor:
        """Compute category/tiebreakers from rank/suit counts.

        rank_counts: [N,13] (# of each rank present)
        suit_counts: [N,4]  (# of cards per suit)
        """
        N = rank_counts.size(0)
        rank_present = rank_counts > 0  # [N,13]
        num_suit_cards, flush_suit_index = suit_counts.max(dim=1)
        has_flush = num_suit_cards >= 5

        # Straight detection (highest straight high rank index)
        x = rank_present.to(torch.float32).unsqueeze(1)  # [N,1,13]
        window = torch.ones(1, 1, 5, device=device)
        conv = torch.nn.functional.conv1d(x, window).squeeze(1)  # [N,9]
        five_in_window = conv == 5
        highs_range = torch.arange(4, 13, device=device)  # 4..12
        straight_candidates = torch.where(
            five_in_window,
            highs_range.view(1, -1).expand_as(five_in_window),
            torch.full_like(five_in_window, -1, dtype=torch.long),
        )
        straight_high, _ = straight_candidates.max(dim=1)
        # Wheel straight (A-2-3-4-5)
        has_wheel = (
            rank_present[:, 12]
            & rank_present[:, 0]
            & rank_present[:, 1]
            & rank_present[:, 2]
            & rank_present[:, 3]
        )
        straight_high = torch.where(
            (straight_high < 0) & has_wheel,
            torch.full_like(straight_high, 3),
            straight_high,
        )
        has_straight = straight_high >= 0

        # Score tensor [N,6]
        score = torch.full((N, 6), -1, dtype=torch.long, device=device)

        # Four of a kind
        is_four_mask = rank_counts == 4
        four_rank = (
            torch.where(
                is_four_mask,
                torch.arange(13, device=device).view(1, -1).expand_as(rank_counts),
                torch.full_like(rank_counts, -1),
            )
            .max(dim=1)
            .values
        )
        has_four = four_rank >= 0
        is_four = has_four
        score[is_four, 0] = 7
        score[is_four, 1] = four_rank[is_four]
        # best kicker excluding quads rank
        ranks_all = torch.arange(13, device=device).view(1, -1).expand_as(rank_present)
        exclude_quads = torch.zeros_like(rank_present)
        exclude_quads[torch.arange(N, device=device), four_rank.clamp(min=0)] = True
        kicker_mask = rank_present & (~exclude_quads)
        kicker_vals = torch.where(
            kicker_mask, ranks_all, torch.full_like(ranks_all, -1)
        )
        kicker_sorted, _ = torch.sort(kicker_vals, dim=1, descending=True)
        score[is_four, 2] = kicker_sorted[is_four, 0]

        # Trips and pairs
        is_trip_mask = rank_counts == 3
        is_pair_or_better = rank_counts >= 2
        rank_idx = torch.arange(13, device=device).view(1, -1).expand_as(rank_counts)
        trips_vals = torch.where(is_trip_mask, rank_idx, torch.full_like(rank_idx, -1))
        trips_sorted, _ = torch.sort(trips_vals, dim=1, descending=True)
        pairs_vals = torch.where(
            is_pair_or_better, rank_idx, torch.full_like(rank_idx, -1)
        )
        pairs_sorted, _ = torch.sort(pairs_vals, dim=1, descending=True)

        # Full house
        has_trip_any = is_trip_mask.any(dim=1)
        top_trip = trips_sorted[:, 0]
        pairs_excl_top_trip = is_pair_or_better.clone()
        pairs_excl_top_trip[torch.arange(N, device=device), top_trip.clamp(min=0)] = (
            False
        )
        pairs_excl_vals = torch.where(
            pairs_excl_top_trip, rank_idx, torch.full_like(rank_idx, -1)
        )
        pairs_excl_sorted, _ = torch.sort(pairs_excl_vals, dim=1, descending=True)
        has_pair_excl = pairs_excl_sorted[:, 0] >= 0
        second_trip = trips_sorted[:, 1]
        has_second_trip = second_trip >= 0
        is_full = has_trip_any & (has_second_trip | has_pair_excl)
        need_full = (score[:, 0] < 0) & is_full
        score[need_full, 0] = 6
        score[need_full, 1] = top_trip[need_full]
        best_pair_or_trip = torch.where(
            has_second_trip, second_trip, pairs_excl_sorted[:, 0]
        )
        score[need_full, 2] = best_pair_or_trip[need_full]

        # Flush: category filled later in one-hot path with exact tiebreakers
        # Straight
        is_straight = (score[:, 0] < 0) & has_straight
        score[is_straight, 0] = 4
        score[is_straight, 1] = straight_high[is_straight]

        # Three of a kind
        is_trips = (score[:, 0] < 0) & has_trip_any
        score[is_trips, 0] = 3
        score[is_trips, 1] = top_trip[is_trips]
        exclude_trip = rank_present.clone()
        exclude_trip[torch.arange(N, device=device), top_trip.clamp(min=0)] = False
        kick_vals = torch.where(exclude_trip, ranks_all, torch.full_like(ranks_all, -1))
        kick_sorted, _ = torch.sort(kick_vals, dim=1, descending=True)
        score[is_trips, 2] = kick_sorted[is_trips, 0]
        score[is_trips, 3] = kick_sorted[is_trips, 1]

        # Two pair
        has_two_pairs = (pairs_sorted[:, 0] >= 0) & (pairs_sorted[:, 1] >= 0)
        is_two_pair = (score[:, 0] < 0) & has_two_pairs
        score[is_two_pair, 0] = 2
        score[is_two_pair, 1] = pairs_sorted[is_two_pair, 0]
        score[is_two_pair, 2] = pairs_sorted[is_two_pair, 1]
        exclude_pairs = rank_present.clone()
        exclude_pairs[
            torch.arange(N, device=device), pairs_sorted[:, 0].clamp(min=0)
        ] = False
        exclude_pairs[
            torch.arange(N, device=device), pairs_sorted[:, 1].clamp(min=0)
        ] = False
        kp_vals = torch.where(exclude_pairs, ranks_all, torch.full_like(ranks_all, -1))
        kp_sorted, _ = torch.sort(kp_vals, dim=1, descending=True)
        score[is_two_pair, 3] = kp_sorted[is_two_pair, 0]

        # One pair
        has_one_pair = pairs_sorted[:, 0] >= 0
        is_one_pair = (score[:, 0] < 0) & has_one_pair
        score[is_one_pair, 0] = 1
        score[is_one_pair, 1] = pairs_sorted[is_one_pair, 0]
        excl_pair = rank_present.clone()
        excl_pair[torch.arange(N, device=device), pairs_sorted[:, 0].clamp(min=0)] = (
            False
        )
        kp3_vals = torch.where(excl_pair, ranks_all, torch.full_like(ranks_all, -1))
        kp3_sorted, _ = torch.sort(kp3_vals, dim=1, descending=True)
        score[is_one_pair, 2:5] = kp3_sorted[is_one_pair, :3]

        # High card
        is_high = score[:, 0] < 0
        score[is_high, 0] = 0
        high_sorted, _ = torch.sort(
            torch.where(rank_present, ranks_all, torch.full_like(ranks_all, -1)),
            dim=1,
            descending=True,
        )
        score[is_high, 1:6] = high_sorted[is_high, :5]
        return score

    # One-hot path: compute counts and exact flush/straight-flush information
    def eval_onehot_full(onehot_planes: torch.Tensor) -> torch.Tensor:
        """Evaluate scores from one-hot planes with exact flush and straight-flush handling."""
        N = onehot_planes.size(0)
        rank_counts = onehot_planes.sum(dim=1).to(torch.long)  # [N,13]
        suit_counts = onehot_planes.sum(dim=2).to(torch.long)  # [N,4]
        rank_present = rank_counts > 0
        score = eval_from_counts(rank_counts, suit_counts)

        # Exact flush tiebreakers and straight-flush override
        num_suit_cards, _ = suit_counts.max(dim=1)
        has_flush = num_suit_cards >= 5
        if has_flush.any():
            ranks_all = (
                torch.arange(13, device=device).view(1, -1).expand_as(rank_present)
            )
            flush_suit_index = suit_counts.argmax(dim=1)
            suit_mask = torch.nn.functional.one_hot(flush_suit_index, num_classes=4).to(
                onehot_planes.dtype
            )
            suit_mask = suit_mask.view(N, 4, 1)
            present_flush = (onehot_planes > 0.5) & (suit_mask > 0.5)
            present_flush_ranks = present_flush.any(dim=1)  # [N,13]
            # Top-5 flush ranks for tiebreakers
            flush_rank_vals = torch.where(
                present_flush_ranks, ranks_all, torch.full_like(ranks_all, -1)
            )
            flush_sorted, _ = torch.sort(flush_rank_vals, dim=1, descending=True)
            top5_flush = flush_sorted[:, :5]
            # Upgrade any category below flush to flush
            need_flush = has_flush & (score[:, 0] < 5)
            score[need_flush, 0] = 5
            # Fill tiebreakers for flush rows
            is_flush_now = has_flush & (score[:, 0] == 5)
            score[is_flush_now, 1:6] = top5_flush[is_flush_now, :]
            # Straight-flush high
            xf = present_flush_ranks.to(torch.float32).unsqueeze(1)
            convf = (
                torch.nn.functional.conv1d(
                    xf, torch.ones(1, 1, 5, device=device)
                ).squeeze(1)
                == 5
            )
            highs = torch.arange(4, 13, device=device)
            candf = torch.where(
                convf,
                highs.view(1, -1).expand_as(convf),
                torch.full_like(convf, -1, dtype=torch.long),
            )
            sf_high, _ = candf.max(dim=1)
            wheelf = (
                present_flush_ranks[:, 12]
                & present_flush_ranks[:, 0]
                & present_flush_ranks[:, 1]
                & present_flush_ranks[:, 2]
                & present_flush_ranks[:, 3]
            )
            sf_high = torch.where(
                (sf_high < 0) & wheelf, torch.full_like(sf_high, 3), sf_high
            )
            has_sf = has_flush & (sf_high >= 0)
            upgrade_to_sf = has_sf & (score[:, 0] <= 5)
            score[upgrade_to_sf, 0] = 8
            score[upgrade_to_sf, 1] = sf_high[upgrade_to_sf]
        return score

    sa = eval_onehot_full(a_batch.to(torch.float32))
    sb = eval_onehot_full(b_batch.to(torch.float32))

    # Lexicographic compare
    res = torch.zeros(sa.size(0), dtype=torch.int64, device=device)
    eq = torch.ones(sa.size(0), dtype=torch.bool, device=device)
    for k in range(sa.size(1)):
        gt = (sa[:, k] > sb[:, k]) & eq
        lt = (sa[:, k] < sb[:, k]) & eq
        res[gt] = 1
        res[lt] = -1
        eq = eq & (sa[:, k] == sb[:, k])
    return res


def compare_7_batch_onehot(
    a_onehot: torch.Tensor, b_onehot: torch.Tensor
) -> torch.Tensor:
    """Compare batched 7-card hands given as one-hot [N, 4, 13] planes.

    Returns [N] in {1, -1, 0}.
    """
    assert a_onehot.dim() == 3 and a_onehot.shape[1:] == (4, 13)
    assert b_onehot.shape == a_onehot.shape
    return compare_7_batch(a_onehot, b_onehot)
