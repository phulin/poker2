from __future__ import annotations

from typing import Iterable, List, Tuple
import random

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
        # mask[r] indicates presence of rank r
        # consider A as high and low (rank 12 as Ace)
        run = 0
        best_high = None
        for r in list(range(13)) + [12]:  # wrap Ace low
            present = mask[r if r < 13 else 0]
            if present:
                run += 1
                if run >= 5:
                    best_high = r if r < 13 else 3  # high rank index of straight
            else:
                run = 0
        return best_high

    rank_mask = [rc[r] > 0 for r in range(13)]
    straight_high = best_straight(rank_mask)

    # Straight flush
    if flush_suit is not None:
        flush_ranks = [rank(c) for c in cs if suit(c) == flush_suit]
        mask = [False] * 13
        for r in flush_ranks:
            mask[r] = True
        sf_high = best_straight(mask)
        if sf_high is not None:
            return (8, (sf_high,))

    # Four of a kind
    if 4 in rc:
        four = max(r for r in range(13) if rc[r] == 4)
        kickers = sorted([r for r in range(13) if r != four and rc[r] > 0], reverse=True)
        return (7, (four, kickers[0]))

    # Full house
    trips = sorted([r for r in range(13) if rc[r] == 3], reverse=True)
    pairs = sorted([r for r in range(13) if rc[r] >= 2 and r not in trips], reverse=True)
    if trips and (len(trips) >= 2 or pairs):
        three = trips[0]
        two = trips[1] if len(trips) >= 2 else pairs[0]
        return (6, (three, two))

    # Flush
    if flush_suit is not None:
        flush_cards = sorted([r for r in ranks if suit(cs[ranks.index(r)]) == flush_suit], reverse=True)
        top5 = sorted(flush_ranks, reverse=True)[:5]  # type: ignore[name-defined]
        return (5, tuple(top5))

    # Straight
    if straight_high is not None:
        return (4, (straight_high,))

    # Three of a kind
    if trips:
        three = trips[0]
        kickers = sorted([r for r in range(13) if r != three and rc[r] > 0], reverse=True)[:2]
        return (3, (three, *kickers))

    # Two pair
    if len(pairs) >= 2:
        p1, p2 = pairs[:2]
        kicker = max(r for r in range(13) if rc[r] > 0 and r not in (p1, p2))
        return (2, (p1, p2, kicker))

    # One pair
    if len(pairs) == 1:
        p = pairs[0]
        kickers = sorted([r for r in range(13) if r != p and rc[r] > 0], reverse=True)[:3]
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
