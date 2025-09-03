from __future__ import annotations

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.types import Action
from alphaholdem.encoding.action_mapping import (
    _action_to_bin_idx,
    _bin_to_target_action,
)


def test_env_action_bins_align_total_committed_reference():
    env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
    s = env.reset()

    # Preflop SB to act: to_call=10, total_committed = 30 + 10 + 20 = 60
    # A raise mapped from bin 4 (1.0x total_committed) should target 10 + 60 = 70
    target = _bin_to_target_action(4, s, 9)
    assert target.kind == "raise" and target.amount == 70

    # Ensure env offers a raise amount close to 70
    legal = env.legal_actions()
    raises = [a.amount for a in legal if a.kind == "raise"]
    assert 70 in raises

    # If we select that raise, mapping back should give bin 4
    bin_idx = _action_to_bin_idx(Action("raise", amount=70), s, 9)
    assert bin_idx == 4


def test_postflop_bet_alignment():
    env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
    s = env.reset()
    # Complete preflop to go to flop
    s, *_ = env.step(Action("call", amount=10))
    s, *_ = env.step(Action("check", amount=0))

    # On flop: total_committed = pot + committeds = 40 + 0 + 0 = 40
    # Bin 2 (0.5x) bet should be 20
    target = _bin_to_target_action(2, s, 9)
    assert target.kind == "bet" and target.amount == 20
    legal = env.legal_actions()
    bets = [a.amount for a in legal if a.kind == "bet"]
    assert 20 in bets
    assert _action_to_bin_idx(Action("bet", amount=20), s, 9) == 2


def test_raise_alignment_after_bet():
    env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=123)
    s = env.reset()
    # Go to flop
    s, *_ = env.step(Action("call", amount=10))
    s, *_ = env.step(Action("check", amount=0))
    # SB bets pot (40)
    s, *_ = env.step(Action("bet", amount=40))
    # Now BB faces to_call=40, total_committed = pot + committeds = 80 + 0 + 40 = 120
    # Bin 4 (1.0x) raise target: 40 + 120 = 160
    target = _bin_to_target_action(4, s, 9)
    assert target.kind == "raise" and target.amount == 160
    legal = env.legal_actions()
    raises = [a.amount for a in legal if a.kind == "raise"]
    assert 160 in raises
    assert _action_to_bin_idx(Action("raise", amount=160), s, 9) == 4
