from __future__ import annotations

import torch
import torch.nn.functional as F

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    calculate_unblocked_mass,
    collapse_1326_to_169,
    combo_suit_permutation_tensor,
    combo_to_preflop_class_tensor,
    expand_169_to_1326,
    preflop_class_compatibility_counts_tensor,
    preflop_class_multiplicity_tensor,
    preflop_class_unblocked_mass,
)
from p2.models.mlp.better_features import context_length
from p2.models.mlp.better_ffn import (
    BetterPreflopGatedTokenMixerPolicyFFN,
    BetterPreflopGatedTokenMixerValueFFN,
    BetterPreflopPolicyFFN,
    BetterPreflopTransformerPolicyFFN,
    BetterPreflopTransformerValueFFN,
    BetterPreflopValueFFN,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.rl.losses import RebelSupervisedLoss
from p2.rl.rebel_batch import RebelBatch


def _compact_features(batch_size: int = 3, num_players: int = 2) -> MLPFeatures:
    beliefs = torch.full(
        (batch_size, num_players, PREFLOP_HANDS),
        1.0 / PREFLOP_HANDS,
        dtype=torch.float32,
    )
    return MLPFeatures(
        context=torch.zeros(batch_size, context_length(num_players)),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=beliefs.reshape(batch_size, -1),
        hand_dim=PREFLOP_HANDS,
    )


def test_preflop_class_mapping_and_multiplicity() -> None:
    class_ids = combo_to_preflop_class_tensor()
    multiplicity = preflop_class_multiplicity_tensor()

    assert class_ids.shape == (NUM_HANDS,)
    assert int(class_ids.unique().numel()) == PREFLOP_HANDS
    assert multiplicity.shape == (PREFLOP_HANDS,)
    assert int(multiplicity.sum().item()) == NUM_HANDS
    assert set(multiplicity.tolist()) == {4.0, 6.0, 12.0}

    permuted = combo_suit_permutation_tensor()
    torch.testing.assert_close(class_ids[permuted], class_ids.expand_as(permuted))


def test_expand_collapse_round_trip_for_values_and_beliefs() -> None:
    values_169 = torch.randn(2, 2, PREFLOP_HANDS)
    values_1326 = expand_169_to_1326(values_169)

    torch.testing.assert_close(
        collapse_1326_to_169(values_1326, reduction="mean"),
        values_169,
    )

    beliefs_169 = torch.rand(2, 2, PREFLOP_HANDS)
    beliefs_169 = beliefs_169 / beliefs_169.sum(dim=-1, keepdim=True)
    beliefs_1326 = expand_169_to_1326(
        beliefs_169,
        divide_by_multiplicity=True,
    )
    torch.testing.assert_close(
        collapse_1326_to_169(beliefs_1326, reduction="sum"),
        beliefs_169,
    )


def test_preflop_class_unblocked_mass_matches_expanded_combo_projection() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    class_mass = torch.rand(3, 2, PREFLOP_HANDS, generator=generator)
    class_mass = class_mass / class_mass.sum(dim=-1, keepdim=True)

    compact_unblocked = preflop_class_unblocked_mass(class_mass)
    expanded_mass = expand_169_to_1326(
        class_mass,
        divide_by_multiplicity=True,
    )
    expanded_unblocked = calculate_unblocked_mass(expanded_mass)
    collapsed_unblocked = collapse_1326_to_169(expanded_unblocked, reduction="mean")

    torch.testing.assert_close(compact_unblocked, collapsed_unblocked)


def test_preflop_class_conditioned_policy_matches_expanded_combo_projection() -> None:
    generator = torch.Generator(device="cpu").manual_seed(124)
    actor_belief = torch.rand(4, PREFLOP_HANDS, generator=generator)
    actor_belief = actor_belief / actor_belief.sum(dim=-1, keepdim=True)
    actor_policy = torch.rand(4, PREFLOP_HANDS, generator=generator)

    compact_matchup = preflop_class_unblocked_mass(actor_belief)
    compact_policy_blocked = preflop_class_unblocked_mass(actor_belief * actor_policy)
    compact_conditioned = compact_policy_blocked / compact_matchup.clamp(min=1e-8)

    expanded_belief = expand_169_to_1326(actor_belief, divide_by_multiplicity=True)
    expanded_policy = expand_169_to_1326(actor_policy)
    expanded_matchup = calculate_unblocked_mass(expanded_belief)
    expanded_policy_blocked = calculate_unblocked_mass(
        expanded_belief * expanded_policy
    )
    expanded_conditioned = expanded_policy_blocked / expanded_matchup.clamp(min=1e-8)
    collapsed_conditioned = collapse_1326_to_169(
        expanded_conditioned,
        reduction="mean",
    )

    torch.testing.assert_close(compact_conditioned, collapsed_conditioned)


def test_preflop_class_compatibility_counts_are_class_symmetric() -> None:
    counts = preflop_class_compatibility_counts_tensor()
    multiplicity = preflop_class_multiplicity_tensor()

    assert counts.shape == (PREFLOP_HANDS, PREFLOP_HANDS)
    assert torch.all(counts >= 0)
    torch.testing.assert_close(
        counts * multiplicity[:, None],
        counts.T * multiplicity[None, :],
    )


def test_compact_169_value_loss_matches_expanded_weighted_loss() -> None:
    batch_size = 4
    num_players = 2
    generator = torch.Generator(device="cpu").manual_seed(127)
    beliefs_169 = torch.rand(
        batch_size, num_players, PREFLOP_HANDS, generator=generator
    )
    beliefs_169 = beliefs_169 / beliefs_169.sum(dim=-1, keepdim=True)
    beliefs_1326 = expand_169_to_1326(
        beliefs_169,
        divide_by_multiplicity=True,
    )
    pred_169 = torch.randn(batch_size, num_players, PREFLOP_HANDS)
    target_169 = torch.randn_like(pred_169)
    compact_batch = RebelBatch(
        features=MLPFeatures(
            context=torch.zeros(batch_size, context_length(num_players)),
            street=torch.zeros(batch_size, dtype=torch.long),
            to_act=torch.zeros(batch_size, dtype=torch.long),
            board=torch.full((batch_size, 5), -1, dtype=torch.long),
            beliefs=beliefs_169.reshape(batch_size, -1),
            hand_dim=PREFLOP_HANDS,
        ),
        legal_masks=torch.ones(batch_size, 4, dtype=torch.bool),
        value_targets=target_169,
    )
    expanded_batch = RebelBatch(
        features=MLPFeatures(
            context=torch.zeros(batch_size, context_length(num_players)),
            street=torch.zeros(batch_size, dtype=torch.long),
            to_act=torch.zeros(batch_size, dtype=torch.long),
            board=torch.full((batch_size, 5), -1, dtype=torch.long),
            beliefs=beliefs_1326.reshape(batch_size, -1),
        ),
        legal_masks=torch.ones(batch_size, 4, dtype=torch.bool),
        value_targets=expand_169_to_1326(target_169),
    )
    loss_fn = RebelSupervisedLoss(num_players=num_players)

    compact_loss = loss_fn.forward_value(
        ModelOutput(hand_values=pred_169),
        compact_batch,
    )["value_loss"]
    expanded_loss = loss_fn.forward_value(
        ModelOutput(hand_values=expand_169_to_1326(pred_169)),
        expanded_batch,
    )["value_loss"]

    torch.testing.assert_close(compact_loss, expanded_loss, rtol=1e-5, atol=1e-6)


def test_compact_policy_loss_uses_folded_opponent_belief_for_blockers() -> None:
    batch_size = 2
    num_players = 3
    num_actions = 4
    generator = torch.Generator(device="cpu").manual_seed(128)
    beliefs = torch.rand(batch_size, num_players, PREFLOP_HANDS, generator=generator)
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
    beliefs_changed = beliefs.clone()
    beliefs_changed[:, 2] = torch.flip(beliefs_changed[:, 2], dims=(-1,))
    logits = torch.randn(batch_size, PREFLOP_HANDS, num_actions, generator=generator)
    targets = F.softmax(
        torch.randn(batch_size, PREFLOP_HANDS, num_actions, generator=generator),
        dim=-1,
    )
    folded = torch.tensor(
        [[False, False, True], [False, False, True]],
        dtype=torch.bool,
    )

    def make_batch(beliefs_in: torch.Tensor) -> RebelBatch:
        return RebelBatch(
            features=MLPFeatures(
                context=torch.zeros(batch_size, context_length(num_players)),
                street=torch.zeros(batch_size, dtype=torch.long),
                to_act=torch.zeros(batch_size, dtype=torch.long),
                board=torch.full((batch_size, 5), -1, dtype=torch.long),
                beliefs=beliefs_in.reshape(batch_size, -1),
                hand_dim=PREFLOP_HANDS,
            ),
            legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
            policy_targets=targets,
            statistics={"has_folded": folded},
        )

    loss_fn = RebelSupervisedLoss(num_players=num_players)
    base = loss_fn.forward_policy(
        ModelOutput(policy_logits=logits), make_batch(beliefs)
    )
    changed = loss_fn.forward_policy(
        ModelOutput(policy_logits=logits),
        make_batch(beliefs_changed),
    )

    assert not torch.allclose(base["policy_weights"], changed["policy_weights"])
    assert not torch.allclose(base["policy_loss"], changed["policy_loss"])


def test_compact_value_loss_excludes_folded_focal_players() -> None:
    batch_size = 2
    num_players = 3
    generator = torch.Generator(device="cpu").manual_seed(129)
    beliefs = torch.rand(batch_size, num_players, PREFLOP_HANDS, generator=generator)
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
    pred = torch.randn(batch_size, num_players, PREFLOP_HANDS, generator=generator)
    target = torch.randn_like(pred)
    pred_changed = pred.clone()
    target_changed = target.clone()
    pred_changed[:, 2] = (
        torch.randn(
            batch_size,
            PREFLOP_HANDS,
            generator=generator,
        )
        * 100.0
    )
    target_changed[:, 2] = (
        torch.randn(
            batch_size,
            PREFLOP_HANDS,
            generator=generator,
        )
        * -100.0
    )
    folded = torch.tensor(
        [[False, False, True], [False, False, True]],
        dtype=torch.bool,
    )
    batch = RebelBatch(
        features=MLPFeatures(
            context=torch.zeros(batch_size, context_length(num_players)),
            street=torch.zeros(batch_size, dtype=torch.long),
            to_act=torch.zeros(batch_size, dtype=torch.long),
            board=torch.full((batch_size, 5), -1, dtype=torch.long),
            beliefs=beliefs.reshape(batch_size, -1),
            hand_dim=PREFLOP_HANDS,
        ),
        legal_masks=torch.ones(batch_size, 4, dtype=torch.bool),
        value_targets=target,
        statistics={"has_folded": folded},
    )
    changed_batch = RebelBatch(
        features=batch.features,
        legal_masks=batch.legal_masks,
        value_targets=target_changed,
        statistics=batch.statistics,
    )
    loss_fn = RebelSupervisedLoss(num_players=num_players)

    base = loss_fn.forward_value(ModelOutput(hand_values=pred), batch)
    changed_pred = loss_fn.forward_value(
        ModelOutput(hand_values=pred_changed),
        batch,
    )
    changed_target = loss_fn.forward_value(
        ModelOutput(hand_values=pred_changed),
        changed_batch,
    )

    torch.testing.assert_close(base["value_loss"], changed_pred["value_loss"])
    torch.testing.assert_close(base["value_loss"], changed_target["value_loss"])
    assert torch.all(base["value_weights"][:, 2] == 0)


def test_compact_policy_weights_match_expanded_combo_weights() -> None:
    batch_size = 3
    num_players = 2
    num_actions = 4
    generator = torch.Generator(device="cpu").manual_seed(125)
    beliefs_169 = torch.rand(
        batch_size, num_players, PREFLOP_HANDS, generator=generator
    )
    beliefs_169 = beliefs_169 / beliefs_169.sum(dim=-1, keepdim=True)
    beliefs_1326 = expand_169_to_1326(
        beliefs_169,
        divide_by_multiplicity=True,
    )
    context = torch.zeros(batch_size, context_length(num_players))
    street = torch.zeros(batch_size, dtype=torch.long)
    to_act = torch.tensor([0, 1, 0], dtype=torch.long)
    board = torch.full((batch_size, 5), -1, dtype=torch.long)
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

    logits_169 = torch.randn(
        batch_size, PREFLOP_HANDS, num_actions, generator=generator
    )
    class_ids = combo_to_preflop_class_tensor()
    logits_1326 = logits_169.index_select(1, class_ids)
    targets_169 = F.softmax(
        torch.randn(batch_size, PREFLOP_HANDS, num_actions, generator=generator),
        dim=-1,
    )
    targets_1326 = targets_169.index_select(1, class_ids)

    compact_batch = RebelBatch(
        features=MLPFeatures(
            context=context,
            street=street,
            to_act=to_act,
            board=board,
            beliefs=beliefs_169.reshape(batch_size, -1),
            hand_dim=PREFLOP_HANDS,
        ),
        legal_masks=legal_masks,
        policy_targets=targets_169,
    )
    expanded_batch = RebelBatch(
        features=MLPFeatures(
            context=context,
            street=street,
            to_act=to_act,
            board=board,
            beliefs=beliefs_1326.reshape(batch_size, -1),
        ),
        legal_masks=legal_masks,
        policy_targets=targets_1326,
    )
    loss_fn = RebelSupervisedLoss(num_players=num_players)

    compact_loss = loss_fn.forward_policy(
        ModelOutput(policy_logits=logits_169),
        compact_batch,
    )
    expanded_loss = loss_fn.forward_policy(
        ModelOutput(policy_logits=logits_1326),
        expanded_batch,
    )
    expanded_weights_169 = collapse_1326_to_169(
        expanded_loss["policy_weights"],
        reduction="sum",
    )

    torch.testing.assert_close(
        compact_loss["policy_weights"],
        expanded_weights_169,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        compact_loss["policy_loss"],
        expanded_loss["policy_loss"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_compact_preflop_model_shapes_and_policy_loss() -> None:
    batch_size = 2
    num_players = 2
    num_actions = 5
    features = _compact_features(batch_size, num_players)
    value_model = BetterPreflopValueFFN(
        num_actions=1,
        hidden_dim=32,
        range_hidden_dim=8,
        ffn_dim=64,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
    )
    policy_model = BetterPreflopPolicyFFN(
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
    )
    value_model.init_weights(torch.Generator(device="cpu").manual_seed(1))
    policy_model.init_weights(torch.Generator(device="cpu").manual_seed(2))

    value_output = value_model(features, include_policy=False)
    policy_output = policy_model(features, include_policy=True, include_value=False)

    assert value_output.hand_values.shape == (
        batch_size,
        num_players,
        PREFLOP_HANDS,
    )
    assert policy_output.policy_logits.shape == (
        batch_size,
        PREFLOP_HANDS,
        num_actions,
    )
    targets = torch.full(
        (batch_size, PREFLOP_HANDS, num_actions),
        1.0 / num_actions,
    )
    batch = RebelBatch(
        features=features,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
        policy_targets=targets,
    )
    loss = RebelSupervisedLoss(num_players=num_players).forward_policy(
        policy_output,
        batch,
    )
    assert torch.isfinite(loss["policy_loss"])


def test_compact_preflop_transformer_model_shapes_and_policy_loss() -> None:
    batch_size = 2
    num_players = 3
    num_actions = 5
    features = _compact_features(batch_size, num_players)
    features.to_act[:] = torch.tensor([0, 2], dtype=torch.long)
    value_model = BetterPreflopTransformerValueFFN(
        num_actions=1,
        hidden_dim=48,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=2,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=4,
        enforce_zero_sum=False,
    )
    policy_model = BetterPreflopTransformerPolicyFFN(
        num_actions=num_actions,
        hidden_dim=48,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=2,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=4,
        enforce_zero_sum=False,
    )
    value_model.init_weights(torch.Generator(device="cpu").manual_seed(4))
    policy_model.init_weights(torch.Generator(device="cpu").manual_seed(5))

    assert len(value_model.encoder) == 2
    assert len(value_model.value_encoder) == 1
    assert len(policy_model.encoder) == 2
    assert len(policy_model.policy_encoder) == 1

    value_output = value_model(features, include_policy=False)
    policy_output = policy_model(features, include_policy=True, include_value=False)

    assert value_output.hand_values.shape == (
        batch_size,
        num_players,
        PREFLOP_HANDS,
    )
    assert policy_output.policy_logits.shape == (
        batch_size,
        PREFLOP_HANDS,
        num_actions,
    )
    targets = torch.full(
        (batch_size, PREFLOP_HANDS, num_actions),
        1.0 / num_actions,
    )
    batch = RebelBatch(
        features=features,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
        policy_targets=targets,
    )
    loss = RebelSupervisedLoss(num_players=num_players).forward_policy(
        policy_output,
        batch,
    )
    assert torch.isfinite(loss["policy_loss"])


def test_compact_preflop_gated_token_mixer_model_shapes_and_policy_loss() -> None:
    batch_size = 2
    num_players = 3
    num_actions = 5
    features = _compact_features(batch_size, num_players)
    features.to_act[:] = torch.tensor([0, 2], dtype=torch.long)
    value_model = BetterPreflopGatedTokenMixerValueFFN(
        num_actions=1,
        hidden_dim=50,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=2,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=8,
        enforce_zero_sum=False,
    )
    policy_model = BetterPreflopGatedTokenMixerPolicyFFN(
        num_actions=num_actions,
        hidden_dim=50,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=2,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=num_players,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=8,
        enforce_zero_sum=False,
    )
    value_model.init_weights(torch.Generator(device="cpu").manual_seed(8))
    policy_model.init_weights(torch.Generator(device="cpu").manual_seed(9))

    assert len(value_model.encoder) == 2
    assert len(value_model.value_encoder) == 1
    assert len(policy_model.encoder) == 2
    assert len(policy_model.policy_encoder) == 1

    value_output = value_model(features, include_policy=False)
    policy_output = policy_model(features, include_policy=True, include_value=False)

    assert value_output.hand_values.shape == (
        batch_size,
        num_players,
        PREFLOP_HANDS,
    )
    assert policy_output.policy_logits.shape == (
        batch_size,
        PREFLOP_HANDS,
        num_actions,
    )
    targets = torch.full(
        (batch_size, PREFLOP_HANDS, num_actions),
        1.0 / num_actions,
    )
    batch = RebelBatch(
        features=features,
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
        policy_targets=targets,
    )
    loss = RebelSupervisedLoss(num_players=num_players).forward_policy(
        policy_output,
        batch,
    )
    assert torch.isfinite(loss["policy_loss"])


def test_compact_preflop_transformer_head_layers_work_without_shared_layers() -> None:
    features = _compact_features(batch_size=2, num_players=3)
    value_model = BetterPreflopTransformerValueFFN(
        num_actions=1,
        hidden_dim=48,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=0,
        num_policy_layers=2,
        num_value_layers=3,
        num_players=3,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=4,
        enforce_zero_sum=False,
    )
    policy_model = BetterPreflopTransformerPolicyFFN(
        num_actions=5,
        hidden_dim=48,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=0,
        num_policy_layers=2,
        num_value_layers=3,
        num_players=3,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=4,
        enforce_zero_sum=False,
    )

    assert len(value_model.encoder) == 0
    assert len(value_model.value_encoder) == 3
    assert len(policy_model.encoder) == 0
    assert len(policy_model.policy_encoder) == 2
    assert value_model(features, include_policy=False).hand_values.shape == (
        2,
        3,
        PREFLOP_HANDS,
    )
    assert policy_model(
        features, include_policy=True, include_value=False
    ).policy_logits.shape == (2, PREFLOP_HANDS, 5)


def test_compact_preflop_transformer_slot_moment_pool_shapes() -> None:
    features = _compact_features(batch_size=2, num_players=3)
    value_model = BetterPreflopTransformerValueFFN(
        num_actions=1,
        hidden_dim=48,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=3,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=4,
        range_slot_moment_slots=3,
        enforce_zero_sum=False,
    )
    policy_model = BetterPreflopTransformerPolicyFFN(
        num_actions=5,
        hidden_dim=48,
        range_hidden_dim=16,
        ffn_dim=96,
        num_hidden_layers=1,
        num_policy_layers=1,
        num_value_layers=1,
        num_players=3,
        policy_rank=8,
        policy_hand_bias_rank=4,
        transformer_heads=4,
        range_slot_moment_slots=3,
        enforce_zero_sum=False,
    )
    value_model.init_weights(torch.Generator(device="cpu").manual_seed(6))
    policy_model.init_weights(torch.Generator(device="cpu").manual_seed(7))

    assert value_model.range_slot_moment_pool is not None
    assert value_model.range_slot_moment_pool.slots == 3
    assert value_model.range_slot_moment_pool.slot_logits.shape == (PREFLOP_HANDS, 3)
    value_output = value_model(features, include_policy=False)
    policy_output = policy_model(features, include_policy=True, include_value=False)

    assert value_output.hand_values.shape == (2, 3, PREFLOP_HANDS)
    assert policy_output.policy_logits.shape == (2, PREFLOP_HANDS, 5)
    assert torch.isfinite(value_output.hand_values).all()
    assert torch.isfinite(policy_output.policy_logits).all()
