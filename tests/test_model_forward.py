from __future__ import annotations

import math

import torch
import torch.nn as nn

from p2.encoding.action_mapping import bin_to_action, get_legal_mask
from p2.env.card_utils import NUM_HANDS, calculate_unblocked_mass
from p2.env.hunl_env import HUNLEnv
from p2.env.types import GameState, PlayerState
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.cnn import ActionsHUEncoderV1, CardsPlanesV1, SiameseConvNetV1
from p2.models.cnn.cnn_embedding_data import CNNEmbeddingData
from p2.core.structured_config import StreetValueHeads
import p2.models.mlp.better_ffn as better_ffn_module
from p2.models.mlp.better_features import (
    ChancePhase,
    ValueScalarContext,
    context_length,
    value_context_length,
)
from p2.models.mlp.better_ffn import (
    HAND_DYNAMIC_FEATURE_DIM,
    BetterFFN,
    BetterPolicyFFN,
    BetterSplitFFN,
    BetterStreetValueFFN,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
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
        policy_rank=8,
        policy_hand_bias_rank=4,
        enforce_zero_sum=False,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(0))

    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())
    assert any(isinstance(module, nn.RMSNorm) for module in model.modules())
    assert not any(name.endswith(".norm.bias") for name in model.state_dict())
    assert "policy_head.1.linear_out.weight" not in model.state_dict()
    assert model.policy_hand_proj.linear_out.weight.shape == (
        model.policy_rank,
        model.hidden_dim,
    )
    assert model.policy_action_head.linear_out.weight.shape == (
        num_actions * model.policy_rank,
        model.hidden_dim,
    )
    assert model.policy_hand_bias.linear_out.weight.shape == (
        model.policy_hand_bias_rank,
        model.hidden_dim,
    )
    assert model.policy_hand_bias_action.linear_out.weight.shape == (
        num_actions * model.policy_hand_bias_rank,
        model.hidden_dim,
    )
    assert model.policy_dynamic_coeff.linear_out.weight.shape == (
        num_actions * HAND_DYNAMIC_FEATURE_DIM,
        model.hidden_dim,
    )
    assert model.hand_value_head[-1].linear_out.weight.shape == (
        num_players * NUM_HANDS,
        model.hidden_dim,
    )
    assert not hasattr(model, "policy_factor_scale")

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


def test_better_ffn_initial_policy_is_near_uniform():
    batch_size = 4
    num_actions = 5
    num_players = 2
    model = BetterFFN(
        num_actions=num_actions,
        hidden_dim=64,
        range_hidden_dim=16,
        ffn_dim=64,
        num_hidden_layers=2,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        board_interaction_dim=16,
        policy_rank=16,
        policy_hand_bias_rank=8,
        value_output_init_scale=0.1,
        nonlinearity="leaky_relu",
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(0))
    model.eval()

    context = torch.zeros(batch_size, context_length(num_players))
    context[:, 3] = 1.5
    context[:, 4] = 1.0
    context[:, 5] = 4.6
    context[:, 6] = 0.7
    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    features = MLPFeatures(
        context=context,
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.arange(batch_size, dtype=torch.long) % num_players,
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )

    with torch.no_grad():
        output = model(features, include_policy=True, include_value=True)
        logits = output.policy_logits
        probs = torch.softmax(logits, dim=-1)
        entropy = (-(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)).mean()

    assert logits.std() < 0.25
    assert entropy > math.log(num_actions) - 0.03
    assert output.hand_values is not None
    assert 0.01 < output.hand_values.std() < 0.2


def test_better_ffn_policy_logit_branches_initialize_with_small_gains():
    num_actions = 5
    num_players = 2
    model = BetterFFN(
        num_actions=num_actions,
        hidden_dim=32,
        range_hidden_dim=8,
        ffn_dim=64,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
        nonlinearity="leaky_relu",
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(1))

    expected_gains = (
        (model.policy_action_head, 0.1),
        (model.policy_hand_bias_action, 0.01),
        (model.policy_dynamic_coeff, 0.01),
        (model.policy_action_bias, 0.01),
    )
    for projection, expected_gain in expected_gains:
        spectral_norm = torch.linalg.matrix_norm(
            projection.linear_out.weight.detach(), ord=2
        )
        torch.testing.assert_close(
            spectral_norm,
            torch.tensor(expected_gain),
            atol=1e-6,
            rtol=1e-6,
        )


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


def test_better_split_ffn_policy_and_value_shapes_and_parameters():
    batch_size = 2
    num_actions = 4
    num_players = 2
    policy_model = BetterPolicyFFN(
        num_actions=num_actions,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
        enforce_zero_sum=False,
    )
    value_model = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
    )
    policy_model.init_weights(torch.Generator(device="cpu").manual_seed(0))
    value_model.init_weights(torch.Generator(device="cpu").manual_seed(1))

    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    policy_features = MLPFeatures(
        context=torch.zeros(batch_size, context_length(num_players)),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    value_context = torch.zeros(batch_size, value_context_length(num_players))
    value_context[:, ValueScalarContext.CHANCE_PHASE.value] = torch.tensor(
        [ChancePhase.PRE_CHANCE.value, ChancePhase.POST_CHANCE.value],
        dtype=torch.float32,
    )
    value_features = MLPFeatures(
        context=value_context,
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )

    assert policy_model.forward_policy(policy_features).shape == (
        batch_size,
        NUM_HANDS,
        num_actions,
    )
    assert value_model.forward_pre(value_features).shape == (
        batch_size,
        num_players,
        NUM_HANDS,
    )
    assert value_model.forward_post(value_features).shape == (
        batch_size,
        num_players,
        NUM_HANDS,
    )
    assert not any("value" in name for name, _ in policy_model.named_parameters())
    assert not any("policy" in name for name, _ in value_model.named_parameters())


def test_better_policy_ffn_multiway_forward_shape():
    batch_size = 2
    num_actions = 4
    num_players = 3
    policy_model = BetterPolicyFFN(
        num_actions=num_actions,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=48,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
        enforce_zero_sum=False,
    )
    policy_model.init_weights(torch.Generator(device="cpu").manual_seed(0))
    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    features = MLPFeatures(
        context=torch.zeros(batch_size, context_length(num_players)),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.tensor([0, 2], dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.reshape(batch_size, -1),
    )

    logits = policy_model.forward_policy(features)

    assert logits.shape == (batch_size, NUM_HANDS, num_actions)


def test_better_split_ffn_fixed_value_head_uses_compiled_value_forward():
    batch_size = 3
    num_actions = 4
    num_players = 2
    policy_model = BetterPolicyFFN(
        num_actions=num_actions,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
    )
    value_model = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
    )
    model = BetterSplitFFN(policy_model=policy_model, value_model=value_model)

    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    value_context = torch.zeros(batch_size, value_context_length(num_players))
    value_context[:, ValueScalarContext.CHANCE_PHASE.value] = torch.tensor(
        [
            ChancePhase.PRE_CHANCE.value,
            ChancePhase.POST_CHANCE.value,
            ChancePhase.PRE_CHANCE.value,
        ],
        dtype=torch.float32,
    )
    features = MLPFeatures(
        context=value_context,
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    static_base = torch.randn(batch_size, value_model.hidden_dim)
    calls = []

    def fake_forward_value(
        sub_features,
        latent=None,
        apply_zero_sum=True,
        static_base_features=None,
        value_head="auto",
    ):
        del latent, apply_zero_sum
        calls.append(
            (
                value_head,
                len(sub_features),
                None if static_base_features is None else tuple(static_base_features.shape),
            )
        )
        fill = 1.0 if value_head == "pre" else 2.0
        hand_values = sub_features.beliefs.new_full(
            (len(sub_features), num_players, NUM_HANDS), fill
        )
        return ModelOutput(value=hand_values.mean(dim=-1), hand_values=hand_values)

    value_model._compiled_forward_value = fake_forward_value

    output = model(
        features,
        include_policy=False,
        include_value=True,
        static_base_features=static_base,
        value_head="pre",
    )

    assert calls == [
        ("pre", batch_size, (batch_size, value_model.hidden_dim)),
    ]
    assert output.hand_values is not None
    torch.testing.assert_close(output.hand_values[0], torch.ones_like(output.hand_values[0]))
    torch.testing.assert_close(output.hand_values[1], torch.ones_like(output.hand_values[1]))
    torch.testing.assert_close(output.hand_values[2], torch.ones_like(output.hand_values[2]))


def test_compile_forward_modes_can_leave_policy_eager_with_static_values(monkeypatch):
    compiled_calls = []

    def fake_compile(fn, **kwargs):
        compiled_calls.append((fn.__name__, dict(kwargs)))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    policy_model = BetterPolicyFFN(
        num_actions=4,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=2,
        policy_rank=8,
        policy_hand_bias_rank=4,
    )
    value_model = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=2,
        policy_rank=8,
        policy_hand_bias_rank=4,
    )
    model = BetterSplitFFN(policy_model=policy_model, value_model=value_model)

    model.compile_forward_modes(
        dynamic=False,
        policy_dynamic=True,
        policy_compile=False,
    )

    assert model.policy_model._compiled_forward_policy is None
    assert model.policy_model._compiled_forward_policy_dynamic_batch is False
    assert model.value_model._compiled_forward_value_dynamic_batch is False
    assert model.value_model._compiled_forward_value_static_base_dynamic_batch is False
    assert not any(name == "policy_forward_features_only" for name, _ in compiled_calls)
    assert any(
        name == "value_forward" and kwargs["dynamic"] is False
        for name, kwargs in compiled_calls
    )

    model.compile_forward_modes(
        dynamic=True,
        policy_dynamic=True,
        policy_compile=False,
    )

    assert model.value_model._compiled_forward_value_dynamic_batch is True
    assert model.value_model._compiled_forward_value_static_base_dynamic_batch is False


def test_dynamic_batch_marking_uses_hint_maybe_mark_dynamic(monkeypatch):
    # The batch dim is a hint, not a hard constraint: a rare shape must be
    # allowed to recompile instead of raising ConstraintViolationError.
    calls = []

    def fake_maybe_mark_dynamic(tensor, dim):
        calls.append((tensor.shape, dim))

    monkeypatch.setattr(torch._dynamo, "maybe_mark_dynamic", fake_maybe_mark_dynamic)

    BaseMLPModel._mark_dynamic_batch(torch.zeros(3, 2))

    assert calls == [(torch.Size([3, 2]), 0)]


def test_better_street_value_auto_matches_phase_heads():
    batch_size = 2
    num_players = 2
    model = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(3))
    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    context = torch.zeros(batch_size, value_context_length(num_players))
    context[:, ValueScalarContext.CHANCE_PHASE.value] = torch.tensor(
        [ChancePhase.PRE_CHANCE.value, ChancePhase.POST_CHANCE.value],
        dtype=torch.float32,
    )
    features = MLPFeatures(
        context=context,
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )

    auto = model.forward_value(features).hand_values
    pre = model.forward_value(features[:1], value_head="pre").hand_values
    post = model.forward_value(features[1:], value_head="post").hand_values

    assert auto is not None
    torch.testing.assert_close(auto[:1], pre)
    torch.testing.assert_close(auto[1:], post)


def test_better_street_value_accepts_legacy_scalar_context_width():
    batch_size = 2
    num_players = 2
    model = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(11))
    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    legacy_context = torch.zeros(
        batch_size,
        ValueScalarContext.NUM_SCALAR_CONTEXT.value,
    )
    legacy_context[:, ValueScalarContext.CHANCE_PHASE.value] = torch.tensor(
        [ChancePhase.PRE_CHANCE.value, ChancePhase.POST_CHANCE.value],
        dtype=torch.float32,
    )
    features = MLPFeatures(
        context=legacy_context,
        street=torch.full((batch_size,), 3, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )

    output = model.forward_value(features)

    assert output.hand_values is not None
    assert output.hand_values.shape == (batch_size, num_players, NUM_HANDS)


def test_better_street_value_can_omit_unused_phase_head():
    pre_only = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=2,
        value_heads=StreetValueHeads.pre,
    )
    post_only = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=2,
        value_heads=StreetValueHeads.post,
    )

    pre_keys = set(pre_only.state_dict())
    post_keys = set(post_only.state_dict())

    assert any(key.startswith("pre_value_head.") for key in pre_keys)
    assert not any(key.startswith("post_value_head.") for key in pre_keys)
    assert any(key.startswith("post_value_head.") for key in post_keys)
    assert not any(key.startswith("pre_value_head.") for key in post_keys)

    beliefs = torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32)
    features = MLPFeatures(
        context=torch.zeros(1, value_context_length(2)),
        street=torch.zeros(1, dtype=torch.long),
        to_act=torch.zeros(1, dtype=torch.long),
        board=torch.full((1, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(1, -1),
    )

    assert pre_only(features, include_policy=False).hand_values.shape == (
        1,
        2,
        NUM_HANDS,
    )
    assert post_only(features, include_policy=False).hand_values.shape == (
        1,
        2,
        NUM_HANDS,
    )


def test_better_street_value_auto_uses_graph_safe_dispatch(monkeypatch):
    batch_size = 2
    num_players = 2
    model = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(4))
    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    context = torch.zeros(batch_size, value_context_length(num_players))
    context[:, ValueScalarContext.CHANCE_PHASE.value] = torch.tensor(
        [ChancePhase.PRE_CHANCE.value, ChancePhase.POST_CHANCE.value],
        dtype=torch.float32,
    )
    features = MLPFeatures(
        context=context,
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )

    monkeypatch.setattr(
        better_ffn_module, "_is_cuda_graph_capturing", lambda tensor: True
    )

    auto = model.forward_value(features).hand_values
    pre = model.forward_value(features[:1], value_head="pre").hand_values
    post = model.forward_value(features[1:], value_head="post").hand_values

    assert auto is not None
    torch.testing.assert_close(auto[:1], pre)
    torch.testing.assert_close(auto[1:], post)


def test_better_street_value_phase_conditioning_changes_output():
    batch_size = 1
    num_players = 2
    model = BetterStreetValueFFN(
        num_actions=1,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        value_output_init_scale=0.1,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(2))
    assert torch.count_nonzero(model.belief_phase_shift.weight) == 0
    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    context = torch.zeros(batch_size, value_context_length(num_players))
    features = MLPFeatures(
        context=context.clone(),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.view(batch_size, -1),
    )
    pre_features = features.clone()
    pre_features.context[:, ValueScalarContext.CHANCE_PHASE.value] = float(
        ChancePhase.PRE_CHANCE.value
    )

    post = model.forward_pre(features)
    pre = model.forward_pre(pre_features)

    assert not torch.allclose(post, pre)


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


def test_better_ffn_unblocked_mass_uses_registered_buffers():
    model = BetterFFN(
        num_actions=4,
        hidden_dim=16,
        range_hidden_dim=8,
        ffn_dim=32,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=2,
    )
    target = torch.rand(3, NUM_HANDS)

    actual = model._calculate_unblocked_mass(target)
    expected = calculate_unblocked_mass(target)

    torch.testing.assert_close(actual, expected)


def test_better_street_value_ffn_arch_proposal_forward_shapes():
    batch_size = 2
    num_players = 2
    proposal_kwargs = [
        {"value_per_hand_residual": True},
        {"board_conditioned_hand_embedding_dim": 4},
        {"cross_range_rank": 4},
        {"card_token_value_head_dim": 8},
        {"context_range_stats": True},
        {"postflop_multi_token_trunk": True},
        {"belief_second_moment": True},
        {"value_strength_bucket_count": 4},
        {"value_strength_bucket_count": 4, "value_strength_bucket_film": True},
        {"value_strength_bucket_count": 4, "value_strength_bucket_relative": True},
        {
            "value_strength_bucket_count": 4,
            "value_strength_bucket_film": True,
            "value_strength_bucket_relative": True,
        },
        {"value_head_rank": 4},
        {
            "value_head_rank": 4,
            "value_strength_bucket_count": 4,
            "value_strength_bucket_film": True,
        },
        {"value_hand_basis_rank": 4},
        {"belief_low_rank_dim": 4},
        {"belief_linear_encoder": True},
        {"belief_low_rank_dim": 4, "belief_linear_encoder": True},
        {"belief_low_rank_dim": 4, "belief_board_film": True},
        {"belief_low_rank_dim": 4, "belief_board_bilinear_rank": 4},
        {"belief_low_rank_dim": 4, "belief_board_mass_features": True},
        {"board_interaction_dim": 8, "board_interaction_gated": True},
        {"board_interaction_dim": 8, "board_interaction_skip_out": True},
        {
            "board_interaction_dim": 8,
            "board_interaction_skip_out": True,
            "board_interaction_gated": True,
        },
        {"belief_low_rank_dim": 4, "belief_low_rank_board_conditioned": True},
        {
            "belief_low_rank_dim": 4,
            "belief_second_moment": True,
            "belief_skip_matching_encoder": True,
        },
    ]
    beliefs = torch.full(
        (batch_size, num_players, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32
    )
    features = MLPFeatures(
        context=torch.zeros(batch_size, value_context_length(num_players)),
        street=torch.full((batch_size,), 3, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.tensor(
            [
                [12, 25, 38, 3, 17],
                [0, 14, 28, 42, 51],
            ],
            dtype=torch.long,
        ),
        beliefs=beliefs.view(batch_size, -1),
    )

    for kwargs in proposal_kwargs:
        model = BetterStreetValueFFN(
            hidden_dim=16,
            range_hidden_dim=8,
            ffn_dim=32,
            num_hidden_layers=1,
            num_policy_layers=1,
            num_value_layers=1,
            num_players=num_players,
            **kwargs,
        )
        model.init_weights(torch.Generator(device="cpu").manual_seed(0))

        output = model(features)

        assert output.value.shape == (batch_size, num_players)
        assert output.hand_values.shape == (batch_size, num_players, NUM_HANDS)


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
