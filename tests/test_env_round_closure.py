from __future__ import annotations

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.types import Action


def test_check_check_closes_round_and_advances_street():
    env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=42)
    s = env.reset()

    # SB calls, BB checks -> should go to flop
    s, _, _, _ = env.step(Action("call", amount=10))
    s, _, _, _ = env.step(Action("check", amount=0))
    assert s.street == "flop"

    # On flop: check-check should advance to turn
    s, _, _, _ = env.step(Action("check", amount=0))
    s, _, _, _ = env.step(Action("check", amount=0))
    assert s.street == "turn"

    # On turn: check-check should advance to river
    s, _, _, _ = env.step(Action("check", amount=0))
    s, _, _, _ = env.step(Action("check", amount=0))
    assert s.street == "river"

    # On river: check-check should go to showdown/terminal
    s, r1, d1, _ = env.step(Action("check", amount=0))
    assert not d1
    s, r2, d2, _ = env.step(Action("check", amount=0))
    assert d2
    assert s.street == "showdown"
