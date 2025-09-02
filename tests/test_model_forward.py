from __future__ import annotations

import torch

from alphaholdem.encoding.cards_encoder import CardsPlanesV1
from alphaholdem.encoding.actions_encoder import ActionsHUEncoderV1
from alphaholdem.models.siamese_convnet import SiameseConvNetV1
from alphaholdem.models.heads import CategoricalPolicyV1
from alphaholdem.env.types import GameState, PlayerState


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
    nb = 9
    cards_enc = CardsPlanesV1()
    actions_enc = ActionsHUEncoderV1(history_actions_per_round=6)
    model = SiameseConvNetV1(cards_channels=6, actions_channels=24, fusion_hidden=128, num_actions=nb)
    policy = CategoricalPolicyV1()

    s = make_state("turn", [4, 5, 6, 7])
    cards = cards_enc.encode_cards(s, seat=0).unsqueeze(0)  # (1, 6, 4, 13)
    actions = actions_enc.encode_actions(s, seat=0, num_bet_bins=nb).unsqueeze(0)  # (1, 24, 4, 9)

    logits, value = model(cards, actions)
    assert logits.shape == (1, nb)
    assert value.shape == (1,)

    a, logp = policy.action(logits.squeeze(0))
    assert 0 <= a < nb
    assert torch.isfinite(torch.tensor(logp))
