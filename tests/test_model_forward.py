from __future__ import annotations

import torch
import torch.nn as nn

from p2.encoding.action_mapping import bin_to_action, get_legal_mask
from p2.env.card_utils import NUM_HANDS
from p2.env.hunl_env import HUNLEnv
from p2.env.types import GameState, PlayerState
from p2.models.cnn import ActionsHUEncoderV1, CardsPlanesV1, SiameseConvNetV1
from p2.models.cnn.cnn_embedding_data import CNNEmbeddingData
from p2.models.mlp.better_features import context_length
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.policy import CategoricalPolicyV1


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

    # Create CNNEmbeddingData
    embedding_data = CNNEmbeddingData(cards=cards, actions=actions)

    # Get model outputs
    outputs = model(embedding_data)
    logits = outputs.policy_logits
    value = outputs.value
    assert logits.shape == (1, nb)
    assert value.shape == (1,)

    a, logp = policy.action(logits.squeeze(0))
    assert 0 <= a < nb
    assert torch.isfinite(torch.tensor(logp))


def test_better_ffn_uses_rmsnorm_and_forward_shapes():
    batch_size = 2
    num_actions = 4
    num_players = 2
    model = BetterFFN(
        num_actions=num_actions,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(0))

    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())
    assert any(isinstance(module, nn.RMSNorm) for module in model.modules())
    assert not any(name.endswith(".norm.bias") for name in model.state_dict())

    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    features = MLPFeatures(
        context=torch.zeros(batch_size, context_length(num_players)),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )

    output = model(features)
    static_output = model(
        features,
        static_base_features=model.static_feature_base(features),
    )

    assert output.policy_logits.shape == (batch_size, NUM_HANDS, num_actions)
    assert output.hand_values.shape == (batch_size, num_players, NUM_HANDS)
    assert output.value.shape == (batch_size, num_players)
    torch.testing.assert_close(static_output.policy_logits, output.policy_logits)
    torch.testing.assert_close(static_output.hand_values, output.hand_values)
    torch.testing.assert_close(static_output.value, output.value)
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.hand_values).all()


def test_better_ffn_range_hidden_dim_zero_uses_hand_embedding_with_ffn_width():
    batch_size = 2
    num_actions = 4
    num_players = 2
    hidden_dim = 16
    ffn_dim = 32
    model = BetterFFN(
        num_actions=num_actions,
        hidden_dim=hidden_dim,
        range_hidden_dim=0,
        ffn_dim=ffn_dim,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
    )

    assert model.belief_proj.linear_in.weight.shape == (
        ffn_dim,
        num_players * hidden_dim,
    )
    assert model.belief_proj.linear_out.weight.shape == (
        hidden_dim,
        ffn_dim,
    )
    assert "belief_proj.linear.weight" not in model.state_dict()

    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    features = MLPFeatures(
        context=torch.zeros(batch_size, context_length(num_players)),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )

    output = model(features)

    assert output.policy_logits.shape == (batch_size, NUM_HANDS, num_actions)
    assert output.hand_values.shape == (batch_size, num_players, NUM_HANDS)
    assert output.value.shape == (batch_size, num_players)


def test_better_ffn_hand_embedding_preserves_rank_suit_assignment():
    model = BetterFFN(
        num_actions=4,
        hidden_dim=4,
        range_hidden_dim=4,
        ffn_dim=8,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=2,
    )
    with torch.no_grad():
        cards = torch.arange(52, dtype=model.card_embedding.weight.dtype)
        model.card_embedding.weight.zero_()
        model.card_embedding.weight[:, 0] = cards
        model.card_embedding.weight[:, 1] = cards.square()

    combos = model.hand_combos
    ah_ks = torch.tensor([12, 24])
    as_kh = torch.tensor([11, 25])
    ah_ks_idx = torch.nonzero((combos == ah_ks).all(dim=1), as_tuple=False).item()
    as_kh_idx = torch.nonzero((combos == as_kh).all(dim=1), as_tuple=False).item()

    hand_emb = model._hand_embedding()

    assert not torch.allclose(hand_emb[ah_ks_idx], hand_emb[as_kh_idx])


def test_better_ffn_rank_and_suit_board_bilinear_interactions():
    batch_size = 2
    num_actions = 4
    num_players = 2
    interaction_dim = 64
    model = BetterFFN(
        num_actions=num_actions,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        board_interaction_dim=interaction_dim,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(0))

    assert model.rank_pair_low_embedding.weight.shape == (91, interaction_dim)
    assert model.suit_pair_low_embedding.weight.shape == (10, interaction_dim)
    assert model.board_rank_low.weight.shape == (interaction_dim, 13)
    assert model.board_suit_low.weight.shape == (interaction_dim, 4)

    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    features = MLPFeatures(
        context=torch.zeros(batch_size, context_length(num_players)),
        street=torch.tensor([1, 3], dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.tensor(
            [
                [12, 25, 38, -1, -1],
                [0, 14, 28, 42, 51],
            ],
            dtype=torch.long,
        ),
        beliefs=beliefs.view(batch_size, -1),
    )

    output = model(features)
    static_output = model(
        features,
        static_base_features=model.static_feature_base(features),
    )

    assert output.policy_logits.shape == (batch_size, NUM_HANDS, num_actions)
    assert output.hand_values.shape == (batch_size, num_players, NUM_HANDS)
    assert output.value.shape == (batch_size, num_players)
    torch.testing.assert_close(static_output.policy_logits, output.policy_logits)
    torch.testing.assert_close(static_output.hand_values, output.hand_values)
    torch.testing.assert_close(static_output.value, output.value)
    assert torch.isfinite(output.policy_logits).all()
    assert torch.isfinite(output.hand_values).all()


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
