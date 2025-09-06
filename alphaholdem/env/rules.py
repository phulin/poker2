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
    a_batch: Sequence[Sequence[int]] | torch.Tensor,
    b_batch: Sequence[Sequence[int]] | torch.Tensor,
) -> torch.Tensor:
    """Vectorized comparison for batches of 7-card hands using torch ops.

    Inputs may be tensors [N,7] or sequences; outputs tensor [N] with 1/-1/0.
    """
    if not isinstance(a_batch, torch.Tensor):
        a_batch = torch.tensor(a_batch, dtype=torch.long)
    if not isinstance(b_batch, torch.Tensor):
        b_batch = torch.tensor(b_batch, dtype=torch.long)
    assert (
        a_batch.shape == b_batch.shape and a_batch.dim() == 2 and a_batch.size(1) == 7
    )
    device = a_batch.device

    def eval_batch(cards: torch.Tensor) -> torch.Tensor:
        # cards: [N,7]
        N = cards.size(0)
        ranks = cards % 13  # [N,7]
        suits = cards // 13  # [N,7]
        ar = torch.arange(13, device=device).view(1, 1, 13)
        # Rank counts [N,13]
        rc = (
            torch.nn.functional.one_hot(ranks, num_classes=13).sum(dim=1).to(torch.long)
        )
        present = rc > 0  # [N,13]
        # Suit counts [N,4]
        sc = torch.nn.functional.one_hot(suits, num_classes=4).sum(dim=1).to(torch.long)
        flush_suit_counts, flush_suit_idx = sc.max(dim=1)
        has_flush = flush_suit_counts >= 5

        # Straight high via conv over present mask
        x = present.to(torch.float32).unsqueeze(1)  # [N,1,13]
        w = torch.ones(1, 1, 5, device=device)
        conv = torch.nn.functional.conv1d(x, w)  # [N,1,9]
        five = conv.squeeze(1) == 5  # [N,9], window ending at index i+4
        # Highest high = i+4 for True positions
        highs = torch.arange(4, 13, device=device)  # 4..12
        # Mask multiply to get candidate highs, -1 where false
        cand = torch.where(
            five,
            highs.view(1, -1).expand_as(five),
            torch.full_like(five, -1, dtype=torch.long),
        )
        straight_high, _ = cand.max(dim=1)  # [-1 or 4..12]
        # Wheel check: A-2-3-4-5
        wheel = (
            present[:, 12]
            & present[:, 0]
            & present[:, 1]
            & present[:, 2]
            & present[:, 3]
        )
        straight_high = torch.where(
            (straight_high < 0) & wheel,
            torch.full_like(straight_high, 3),
            straight_high,
        )
        has_straight = straight_high >= 0

        # Straight flush
        # Mask cards of flush suit
        suits_eq = suits == flush_suit_idx.view(-1, 1)
        suits_eq = suits_eq & has_flush.view(-1, 1)
        # Present ranks in flush suit: any over cards
        present_flush = (
            torch.nn.functional.one_hot(ranks, 13).bool() & suits_eq.unsqueeze(-1)
        ).any(
            dim=1
        )  # [N,13]
        xf = present_flush.to(torch.float32).unsqueeze(1)
        convf = torch.nn.functional.conv1d(xf, w).squeeze(1) == 5
        candf = torch.where(
            convf,
            highs.view(1, -1).expand_as(convf),
            torch.full_like(convf, -1, dtype=torch.long),
        )
        sf_high, _ = candf.max(dim=1)
        wheelf = (
            present_flush[:, 12]
            & present_flush[:, 0]
            & present_flush[:, 1]
            & present_flush[:, 2]
            & present_flush[:, 3]
        )
        sf_high = torch.where(
            (sf_high < 0) & wheelf, torch.full_like(sf_high, 3), sf_high
        )
        has_sf = has_flush & (sf_high >= 0)

        # Categories and tiebreakers packed into [N,6]
        score = torch.full((N, 6), -1, dtype=torch.long, device=device)

        # Four of a kind
        four_mask = rc == 4
        four_rank = (
            torch.where(
                four_mask,
                torch.arange(13, device=device).view(1, -1).expand_as(rc),
                torch.full_like(rc, -1),
            )
            .max(dim=1)
            .values
        )
        has_four = four_rank >= 0

        # Trips and pairs
        trips_mask = rc == 3
        pair_mask = rc >= 2
        rank_idx = torch.arange(13, device=device).view(1, -1).expand_as(rc)
        trips_vals = torch.where(trips_mask, rank_idx, torch.full_like(rank_idx, -1))
        trips_sorted, _ = torch.sort(trips_vals, dim=1, descending=True)
        pairs_vals = torch.where(pair_mask, rank_idx, torch.full_like(rank_idx, -1))
        pairs_sorted, _ = torch.sort(pairs_vals, dim=1, descending=True)

        # Flush top5 ranks
        # Build ranks present for flush suit already computed (present_flush)
        top5_flush = torch.full((N, 5), -1, dtype=torch.long, device=device)
        if has_flush.any():
            # Convert present_flush to rank list by masking ranks
            ranks_all = (
                torch.arange(13, device=device).view(1, -1).expand_as(present_flush)
            )
            flush_rank_vals = torch.where(
                present_flush, ranks_all, torch.full_like(ranks_all, -1)
            )
            flush_sorted, _ = torch.sort(flush_rank_vals, dim=1, descending=True)
            top5_flush = flush_sorted[:, :5]

        # High card top5
        ranks_all = torch.arange(13, device=device).view(1, -1).expand_as(present)
        present_vals = torch.where(present, ranks_all, torch.full_like(ranks_all, -1))
        high_sorted, _ = torch.sort(present_vals, dim=1, descending=True)
        top5 = high_sorted[:, :5]

        # Category selection priority
        # 8 Straight Flush
        sel = has_sf
        score[sel, 0] = 8
        score[sel, 1] = sf_high[sel]

        # 7 Four of a Kind
        sel = (~sel) & has_four
        score[sel, 0] = 7
        score[sel, 1] = four_rank[sel]
        # best kicker = highest rank present excluding four_rank
        excl = torch.zeros_like(present)
        excl[torch.arange(N, device=device), four_rank.clamp(min=0)] = True
        kicker_mask = present & (~excl)
        kicker_vals = torch.where(
            kicker_mask, ranks_all, torch.full_like(ranks_all, -1)
        )
        kicker_sorted, _ = torch.sort(kicker_vals, dim=1, descending=True)
        score[sel, 2] = kicker_sorted[sel, 0]

        # 6 Full House: trips and then pair
        has_trip = trips_mask.any(dim=1)
        # choose top trip
        top_trip = trips_sorted[:, 0]
        # For pair part: either second trip or any pair excluding top_trip
        # Build pairs excluding top_trip
        exclude_top_trip = pair_mask.clone()
        exclude_top_trip[torch.arange(N, device=device), top_trip.clamp(min=0)] = False
        pairs_excl_vals = torch.where(
            exclude_top_trip, rank_idx, torch.full_like(rank_idx, -1)
        )
        pairs_excl_sorted, _ = torch.sort(pairs_excl_vals, dim=1, descending=True)
        has_pair_excl = pairs_excl_sorted[:, 0] >= 0
        second_trip = trips_sorted[:, 1]
        has_second_trip = second_trip >= 0
        full_mask = has_trip & (has_second_trip | has_pair_excl)
        sel2 = (score[:, 0] < 0) & full_mask
        score[sel2, 0] = 6
        score[sel2, 1] = top_trip[sel2]
        # choose best of second_trip vs pair_excl
        pair_choice = torch.where(has_second_trip, second_trip, pairs_excl_sorted[:, 0])
        score[sel2, 2] = pair_choice[sel2]

        # 5 Flush
        sel3 = (score[:, 0] < 0) & has_flush
        score[sel3, 0] = 5
        score[sel3, 1:6] = top5_flush[sel3, :]

        # 4 Straight
        sel4 = (score[:, 0] < 0) & has_straight
        score[sel4, 0] = 4
        score[sel4, 1] = straight_high[sel4]

        # 3 Three of a kind
        sel5 = (score[:, 0] < 0) & has_trip
        score[sel5, 0] = 3
        score[sel5, 1] = top_trip[sel5]
        # kickers: top two excluding trip rank
        excl_trip = present.clone()
        excl_trip[torch.arange(N, device=device), top_trip.clamp(min=0)] = False
        kick_vals = torch.where(excl_trip, ranks_all, torch.full_like(ranks_all, -1))
        kick_sorted, _ = torch.sort(kick_vals, dim=1, descending=True)
        score[sel5, 2] = kick_sorted[sel5, 0]
        score[sel5, 3] = kick_sorted[sel5, 1]

        # 2 Two Pair
        has_two_pairs = (pairs_sorted[:, 0] >= 0) & (pairs_sorted[:, 1] >= 0)
        sel6 = (score[:, 0] < 0) & has_two_pairs
        score[sel6, 0] = 2
        score[sel6, 1] = pairs_sorted[sel6, 0]
        score[sel6, 2] = pairs_sorted[sel6, 1]
        # kicker excluding those pairs
        excl_pairs = present.clone()
        excl_pairs[torch.arange(N, device=device), pairs_sorted[:, 0].clamp(min=0)] = (
            False
        )
        excl_pairs[torch.arange(N, device=device), pairs_sorted[:, 1].clamp(min=0)] = (
            False
        )
        kp_vals = torch.where(excl_pairs, ranks_all, torch.full_like(ranks_all, -1))
        kp_sorted, _ = torch.sort(kp_vals, dim=1, descending=True)
        score[sel6, 3] = kp_sorted[sel6, 0]

        # 1 One Pair
        has_one_pair = pairs_sorted[:, 0] >= 0
        sel7 = (score[:, 0] < 0) & has_one_pair
        score[sel7, 0] = 1
        score[sel7, 1] = pairs_sorted[sel7, 0]
        # three kickers excluding pair rank
        excl_pair = present.clone()
        excl_pair[torch.arange(N, device=device), pairs_sorted[:, 0].clamp(min=0)] = (
            False
        )
        kp3_vals = torch.where(excl_pair, ranks_all, torch.full_like(ranks_all, -1))
        kp3_sorted, _ = torch.sort(kp3_vals, dim=1, descending=True)
        score[sel7, 2:5] = kp3_sorted[sel7, :3]

        # 0 High Card
        sel8 = score[:, 0] < 0
        score[sel8, 0] = 0
        score[sel8, 1:6] = top5[sel8, :]

        return score

    sa = eval_batch(a_batch)
    sb = eval_batch(b_batch)

    # Lexicographic compare across columns
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

    Expects exactly 7 cards set per sample; returns [N] in {1, -1, 0}.
    """
    assert a_onehot.dim() == 3 and a_onehot.shape[1:] == (4, 13)
    assert b_onehot.shape == a_onehot.shape
    device = a_onehot.device
    N = a_onehot.size(0)

    # Convert one-hot to list of card ints [N,7] for reuse of vector evaluator
    def planes_to_cards(x: torch.Tensor) -> torch.Tensor:
        # x: [N,4,13] -> gather suit/rank indices
        # Find positions where x==1
        idxs = (x > 0.5).nonzero(as_tuple=False)  # [K, 3]: (n, s, r)
        # Bucket by n
        cards = [[] for _ in range(N)]
        for n, s, r in idxs.tolist():
            cards[n].append(s * 13 + r)
        # Pad/truncate to 7
        out = torch.full((N, 7), -1, dtype=torch.long, device=device)
        for n in range(N):
            vals = cards[n][:7]
            assert len(vals) == 7, "Each sample must have 7 cards set"
            out[n, :] = torch.tensor(vals, dtype=torch.long, device=device)
        return out

    a_cards = planes_to_cards(a_onehot)
    b_cards = planes_to_cards(b_onehot)
    return compare_7_batch(a_cards, b_cards)
