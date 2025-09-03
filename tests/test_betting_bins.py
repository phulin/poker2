from __future__ import annotations

from alphaholdem.encoding.betting_bins import choose_bin, bin_to_amount


def test_choose_bin_and_amount_mapping():
    pot = 100
    to_call = 20
    stack = 200
    nb = 8

    # Desired around pot-sized should pick bin with mult=1.0 (index 4)
    idx = choose_bin(pot, stack, nb, desired=1.0)
    assert idx == 4

    # Map bins back to concrete amounts
    assert bin_to_amount(pot, to_call, stack, 0) == 0  # fold
    assert bin_to_amount(pot, to_call, stack, 1) == to_call  # check/call

    # Half-pot bet min constrained by to_call
    amt_half = bin_to_amount(pot, to_call, stack, 2)
    assert amt_half >= to_call and amt_half <= stack + to_call

    # All-in maps to stack+to_call
    assert bin_to_amount(pot, to_call, stack, 7) == stack + to_call
