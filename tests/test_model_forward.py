from __future__ import annotations

import torch

from alphaholdem.models.cnn import CardsPlanesV1, ActionsHUEncoderV1
from alphaholdem.encoding.action_mapping import bin_to_action, get_legal_mask
from alphaholdem.models.cnn import SiameseConvNetV1
from alphaholdem.models.heads import CategoricalPolicyV1
from alphaholdem.env.types import GameState, PlayerState
from alphaholdem.env.hunl_env import HUNLEnv


def make_state(street: str, board: list[int]) -> GameState:
    p0 = PlayerState(stack=1000, hole_cards=[0, 1])
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


def test_siamese_convnet_forward_and_policy_action():
    nb = 8
    cards_enc = CardsPlanesV1()
    actions_enc = ActionsHUEncoderV1(history_actions_per_round=6)
    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=128,
        actions_hidden=128,
        fusion_hidden=128,
        num_actions=nb,
    )
    policy = CategoricalPolicyV1()

    s = make_state("turn", [4, 5, 6, 7])
    cards = cards_enc.encode_cards(s, seat=0).unsqueeze(0)  # (1, 6, 4, 13)
    actions = actions_enc.encode_actions(s, seat=0).unsqueeze(0)  # (1, 24, 4, 8)

    logits, value = model(cards, actions)
    assert logits.shape == (1, nb)
    assert value.shape == (1,)

    a, logp = policy.action(logits.squeeze(0))
    assert 0 <= a < nb
    assert torch.isfinite(torch.tensor(logp))


def test_action_mapping_with_env():
    env = HUNLEnv(starting_stack=1000, sb=50, bb=100)
    state = env.reset()
    nb = 8

    # Test legal mask
    legal_mask = get_legal_mask(state, nb, torch.float32)
    assert legal_mask.shape == (nb,)
    assert legal_mask.sum() > 0  # should have some legal actions

    # Test bin to action mapping
    for bin_idx in range(nb):
        action = bin_to_action(bin_idx, state, nb)
        assert action.kind in ["fold", "check", "call", "bet", "raise", "allin"]
        if action.kind in ["bet", "raise", "call", "allin"]:
            assert action.amount >= 0
