from __future__ import annotations

import torch

from alphaholdem.env.types import GameState, PlayerState
from alphaholdem.models.cnn import ActionsHUEncoderV1, CardsPlanesV1


def make_state(street: str, board: list[int]) -> GameState:
    p0 = PlayerState(stack=1000, hole_cards=[0, 1])  # ranks 0,1 suits 0
    p1 = PlayerState(stack=1000, hole_cards=[2, 3])
    return GameState(
        button=0,
        street=street,
        deck=list(range(4, 52)),
        board=board,
        pot=0,
        to_act=0,
        small_blind=50,
        big_blind=100,
        min_raise=100,
        last_aggressive_amount=100,
        players=(p0, p1),
    )


def test_cards_planes_v1_values_and_shapes():
    enc = CardsPlanesV1(config=None)
    # Board: ranks 4,5,6,7 suit 0
    s = make_state("turn", [4, 5, 6, 7])
    t = enc.encode_cards(s, seat=0)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (6, 4, 13)
    assert t.dtype == torch.float32

    hole, flop, turn, river, public, all_cards = t

    # Hole cards [0,1] → suit 0, ranks 0 and 1
    assert hole.sum().item() == 2
    assert hole[0, 0].item() == 1.0
    assert hole[0, 1].item() == 1.0
    assert hole[0].sum().item() == 2

    # Flop from board [4,5,6]
    assert flop.sum().item() == 3
    assert flop[0, 4].item() == 1.0
    assert flop[0, 5].item() == 1.0
    assert flop[0, 6].item() == 1.0

    # Turn from board [7]
    assert turn.sum().item() == 1
    assert turn[0, 7].item() == 1.0

    # River empty at turn
    assert river.sum().item() == 0

    # Public shows all 4 board cards
    assert public.sum().item() == 4
    for r in [4, 5, 6, 7]:
        assert public[0, r].item() == 1.0

    # All cards = hole + board (6 ones total)
    assert all_cards.sum().item() == 6
    for r in [0, 1, 4, 5, 6, 7]:
        assert all_cards[0, r].item() == 1.0


def test_actions_hu_v1_values_and_shapes():
    enc = ActionsHUEncoderV1(history_actions_per_round=6)
    s = make_state("flop", [4, 5, 6])
    nb = 8
    a = enc.encode_actions(s, seat=0)
    assert isinstance(a, torch.Tensor)
    assert a.shape == (24, 4, nb)
    assert a.dtype == torch.float32

    # Inject simple action history and verify population
    s.action_history = [
        ("preflop", 0, "raise", 40, 20, 60),
        ("preflop", 1, "call", 40, 20, 60),
        ("flop", 0, "bet", 20, 0, 40),
    ]
    a = enc.encode_actions(s, seat=0)
    # Preflop first slot should reflect player0 raise in one of the latest slots
    preflop_slot_last = 0 * 6 + 5
    assert a[preflop_slot_last, 2].sum().item() >= 1.0  # sum plane marks events
    # Flop last slot has legal plane filled
    flop_slot_last = 1 * 6 + 5
    assert a[flop_slot_last, 3].sum().item() > 0
