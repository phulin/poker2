from __future__ import annotations

import pytest

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.types import Action


def make_env(
    starting_stack: int = 1000, sb: int = 5, bb: int = 10, seed: int = 123
) -> HUNLEnv:
    return HUNLEnv(starting_stack=starting_stack, sb=sb, bb=bb, seed=seed)


def test_sb_folds_preflop_reward_is_zero_relative_to_after_blinds_baseline():
    env = make_env()
    s = env.reset()
    # SB acts first preflop; allowed to fold facing the BB
    assert s.to_act in (0, 1)
    sb_seat = s.to_act
    next_state, reward, done, _ = env.step(Action("fold"))
    assert done is True
    # Reward is from acting player's perspective (SB). With baseline after blinds, no extra chips lost → 0
    assert reward == 0


def test_sb_calls_then_folds_to_bb_allin_loses_only_call_amount():
    env = make_env()
    s = env.reset()
    sb_seat = s.to_act
    opp = 1 - sb_seat
    to_call = s.players[opp].committed - s.players[sb_seat].committed
    assert to_call > 0

    # SB calls
    s, r, d, _ = env.step(Action("call", amount=to_call))
    assert d is False
    # BB shoves
    bb_stack = s.players[opp].stack
    s, r, d, _ = env.step(Action("allin", amount=bb_stack))
    assert d is False
    # SB folds now
    s, reward, done, _ = env.step(Action("fold"))
    assert done is True
    # Reward is from SB perspective; with after-blinds baseline, they only lose the call amount beyond the blind
    assert reward == -to_call


def test_bb_folds_to_sb_bet_reward_is_zero_relative_to_after_blinds_baseline():
    env = make_env()
    s = env.reset()
    sb_seat = s.to_act
    opp = 1 - sb_seat
    # SB makes a small bet (any positive amount within stack)
    bet_amt = max(1, s.big_blind)  # a simple valid bet size
    s, r, d, _ = env.step(Action("bet", amount=bet_amt))
    assert d is False
    # BB folds facing the bet
    s, reward, done, _ = env.step(Action("fold"))
    assert done is True
    # Reward is from BB perspective; with baseline after blinds, no extra chips lost beyond blind → 0
    assert reward == 0
