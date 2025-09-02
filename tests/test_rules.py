from __future__ import annotations

from alphaholdem.env import rules

# Helper to make card by rank/suit (r: 0..12 for 2..A, s: 0..3)

def C(r: int, s: int) -> int:
    return s * 13 + r


def test_straight_flush_vs_four_kind():
    # Straight flush: 9-T-J-Q-K of suit 0
    sf = [C(7, 0), C(8, 0), C(9, 0), C(10, 0), C(11, 0), C(0, 1), C(1, 2)]
    # Four of a kind: four 9s + kickers
    fk = [C(7, 0), C(7, 1), C(7, 2), C(7, 3), C(2, 0), C(3, 1), C(4, 2)]
    assert rules.compare_7(sf, fk) == 1


def test_full_house_vs_flush():
    # Full house: 7-7-7-5-5
    fh = [C(5, 0), C(5, 1), C(7, 0), C(7, 1), C(7, 2), C(2, 1), C(3, 1)]
    # Flush (weaker than FH): five hearts suit 1
    fl = [C(2, 1), C(6, 1), C(8, 1), C(9, 1), C(11, 1), C(0, 0), C(1, 2)]
    assert rules.compare_7(fh, fl) == 1


def test_straight_wheel():
    # Wheel straight A-2-3-4-5: ranks (12,0,1,2,3)
    st = [C(12, 0), C(0, 1), C(1, 2), C(2, 3), C(3, 0), C(5, 1), C(8, 2)]
    cat, tie = rules.evaluate_7(st)
    assert cat == 4


def test_one_pair_tie_breakers():
    # Pair of 9s with high kickers
    a = [C(7, 0), C(7, 1), C(12, 0), C(11, 0), C(10, 0), C(2, 1), C(3, 2)]
    # Pair of 9s with lower kicker
    b = [C(7, 2), C(7, 3), C(9, 0), C(8, 0), C(5, 0), C(2, 2), C(3, 3)]
    assert rules.compare_7(a, b) == 1


def test_two_pair_tiebreaker_second_pair():
    # A: K-K and T-T with A kicker
    a = [C(11, 0), C(11, 1), C(8, 0), C(8, 2), C(12, 0), C(3, 1), C(4, 2)]
    # B: K-K and 9-9 with A kicker
    b = [C(11, 2), C(11, 3), C(7, 0), C(7, 2), C(12, 1), C(2, 1), C(3, 2)]
    assert rules.compare_7(a, b) == 1


def test_two_pair_tiebreaker_kicker():
    # Same pairs K-K and T-T, A kicker vs Q kicker
    a = [C(11, 0), C(11, 1), C(8, 0), C(8, 2), C(12, 0), C(3, 1), C(4, 2)]
    b = [C(11, 2), C(11, 3), C(8, 1), C(8, 3), C(10, 1), C(2, 1), C(3, 2)]
    assert rules.compare_7(a, b) == 1


def test_flush_tiebreakers():
    # A: Flush in suit 1 with A,K,9,5,2
    a = [C(12, 1), C(11, 1), C(7, 1), C(3, 1), C(0, 1), C(2, 0), C(4, 2)]
    # B: Flush in suit 1 with A,Q,9,5,2
    b = [C(12, 1), C(10, 1), C(7, 1), C(3, 1), C(0, 1), C(5, 0), C(6, 2)]
    assert rules.compare_7(a, b) == 1
