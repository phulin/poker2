import torch

import p2.models.mlp.mlp_features as mlp_features_module
from p2.env.card_utils import (
    HAND_EQUITY_ORDERING,
    IDX_TO_RANK,
    NUM_HANDS,
    PREFLOP_HANDS,
    combo_lookup_tensor,
    hand_combos_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.models.mlp.better_feature_encoder import (
    BetterFeatureEncoder,
    BetterPreflopValueFeatureEncoder,
    BetterPolicyFeatureEncoder,
    BetterStreetValueFeatureEncoder,
)
from p2.models.mlp.better_features import (
    ChancePhase,
    PlayerContext,
    ScalarContext,
    ValueScalarContext,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder


def make_env(num_envs: int = 4) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env.reset()
    return env


def test_bool_mask_indexing_keeps_feature_fields_aligned(monkeypatch):
    batch_size = 4
    features = MLPFeatures(
        context=torch.arange(batch_size * 3, dtype=torch.float32).reshape(
            batch_size, 3
        ),
        street=torch.arange(batch_size, dtype=torch.long),
        to_act=torch.arange(batch_size, dtype=torch.long) + 10,
        board=torch.arange(batch_size * 5, dtype=torch.long).reshape(batch_size, 5),
        beliefs=torch.arange(batch_size * 2 * NUM_HANDS, dtype=torch.float32).reshape(
            batch_size, 2 * NUM_HANDS
        ),
    )

    def fail_compiled_index_all(*args, **kwargs):
        raise AssertionError("boolean masks should bypass compiled _index_all")

    monkeypatch.setattr(mlp_features_module, "_index_all", fail_compiled_index_all)

    mask = torch.tensor([True, False, True, False])
    indexed = features[mask]

    torch.testing.assert_close(indexed.context, features.context[mask])
    torch.testing.assert_close(indexed.street, features.street[mask])
    torch.testing.assert_close(indexed.to_act, features.to_act[mask])
    torch.testing.assert_close(indexed.board, features.board[mask])
    torch.testing.assert_close(indexed.beliefs, features.beliefs[mask])


def test_mlp_features_accepts_multiway_beliefs():
    batch_size = 3
    num_players = 4
    features = MLPFeatures(
        context=torch.zeros(batch_size, 3),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.full((batch_size, 5), -1, dtype=torch.long),
        beliefs=torch.zeros(batch_size, num_players * NUM_HANDS),
    )

    assert features.num_players == num_players
    indexed = features[torch.tensor([0, 2])]
    assert indexed.beliefs.shape == (2, num_players * NUM_HANDS)
    assert indexed.num_players == num_players


def build_hand_name_to_combos(device: torch.device) -> dict[str, list[int]]:
    """Build a mapping from hand name to list of combo indices."""
    combos = hand_combos_tensor(device=device)
    hand_name_to_combos: dict[str, list[int]] = {}

    for idx in range(NUM_HANDS):
        c1, c2 = combos[idx]
        r1, r2 = c1 % 13, c2 % 13
        is_suited = c1 // 13 == c2 // 13

        # Create a canonical hand name from this combo
        if r1 == r2:
            hand_name = (
                f"{IDX_TO_RANK[int(r1)]}{IDX_TO_RANK[int(r1)]}"  # Pair like "AA"
            )
        else:
            rank_char_1 = IDX_TO_RANK[int(max(r1, r2))]
            rank_char_2 = IDX_TO_RANK[int(min(r1, r2))]
            suffix = "s" if is_suited else "o"
            hand_name = f"{rank_char_1}{rank_char_2}{suffix}"  # Like "AKs" or "AKo"

        if hand_name not in hand_name_to_combos:
            hand_name_to_combos[hand_name] = []
        hand_name_to_combos[hand_name].append(idx)

    return hand_name_to_combos


def test_permute_suits():
    """Test that permute_suits correctly permutes board cards and beliefs."""
    env = make_env(4)
    encoder = RebelFeatureEncoder(env, device=env.device, dtype=torch.float32)

    # Create random beliefs for each player
    generator = torch.Generator(device=env.device)
    generator.manual_seed(123)
    beliefs = torch.rand((4, 2, NUM_HANDS), generator=generator, device=env.device)
    # Normalize to sum to 1.0
    beliefs = beliefs / beliefs.sum(dim=2, keepdim=True)

    # Create features
    features = encoder.encode(beliefs)

    # Store original values
    original_board = features.board.clone()
    original_beliefs = features.beliefs.clone()

    # Verify beliefs sum to 1.0
    torch.testing.assert_close(
        original_beliefs[:, :NUM_HANDS].sum(dim=1),
        torch.ones(4, device=env.device),
    )

    # Apply suit permutation
    features.permute_suits(generator=generator)

    # Verify beliefs still sum to 1.0 (values are just reordered)
    torch.testing.assert_close(
        features.beliefs[:, :NUM_HANDS].sum(dim=1),
        torch.ones(4, device=env.device),
    )

    # Verify board cards were permuted
    # Board should be different unless permutation happened to be identity
    # Check that ranks are preserved but suits changed
    original_ranks = original_board % 13
    new_ranks = features.board % 13
    # Ranks should be the same (permutation only affects suits)
    torch.testing.assert_close(original_ranks, new_ranks)

    # Verify beliefs total sum is unchanged
    torch.testing.assert_close(original_beliefs.sum(), features.beliefs.sum())

    # For each hand class in HAND_EQUITY_ORDERING, verify suit permutation is correct
    hand_name_to_combos = build_hand_name_to_combos(env.device)

    # Check each player's beliefs
    for player_idx in range(2):
        player_beliefs_start = player_idx * NUM_HANDS
        player_beliefs_end = (player_idx + 1) * NUM_HANDS

        # Get original and permuted beliefs for this player
        orig_player_beliefs = original_beliefs[
            :, player_beliefs_start:player_beliefs_end
        ]
        permuted_player_beliefs = features.beliefs[
            :, player_beliefs_start:player_beliefs_end
        ]

        # For each hand class, check that the sum of beliefs across all suit combinations is preserved
        for hand_name in HAND_EQUITY_ORDERING:
            combo_indices = hand_name_to_combos[hand_name]
            orig_player_beliefs_combo = (
                orig_player_beliefs[:, combo_indices].sort(dim=1).values
            )
            permuted_player_beliefs_combo = (
                permuted_player_beliefs[:, combo_indices].sort(dim=1).values
            )

            # The sum should be the same (suit permutation just reorders within the hand class)
            torch.testing.assert_close(
                orig_player_beliefs_combo, permuted_player_beliefs_combo
            )


def test_better_feature_encoder_empty_indices():
    """Empty model batches should still produce well-shaped feature tensors."""
    env = make_env(4)
    encoder = BetterFeatureEncoder(env, device=env.device, dtype=torch.float32)
    beliefs = torch.full(
        (4, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
    )
    indices = torch.empty(0, dtype=torch.long, device=env.device)
    pre_chance_node = torch.zeros(4, dtype=torch.bool, device=env.device)

    features = encoder.encode(
        beliefs, pre_chance_node=pre_chance_node, indices=indices
    )

    assert features.context.shape == (
        0,
        ScalarContext.NUM_SCALAR_CONTEXT.value
        + PlayerContext.NUM_PLAYER_CONTEXT.value * 2,
    )
    assert features.street.shape == (0,)
    assert features.to_act.shape == (0,)
    assert features.board.shape == (0, 5)
    assert features.beliefs.shape == (0, 2 * NUM_HANDS)


def test_better_policy_and_value_feature_context_slots():
    env = make_env(2)
    env.actions_this_round[:] = torch.tensor([3, 5], device=env.device)
    policy_encoder = BetterPolicyFeatureEncoder(
        env, device=env.device, dtype=torch.float32
    )
    value_encoder = BetterStreetValueFeatureEncoder(
        env, device=env.device, dtype=torch.float32
    )
    beliefs = torch.full(
        (2, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
    )

    policy_features = policy_encoder.encode(beliefs, pre_chance_node=False)
    value_features = value_encoder.encode(beliefs, pre_chance_node=False)
    value_pre_features = value_encoder.encode(beliefs, pre_chance_node=True)

    torch.testing.assert_close(
        policy_features.context[:, ScalarContext.ACTIONS_ROUND.value],
        env.actions_this_round.to(torch.float32),
    )
    torch.testing.assert_close(
        value_features.context[:, ValueScalarContext.CHANCE_PHASE.value],
        torch.full((2,), float(ChancePhase.POST_CHANCE.value)),
    )
    torch.testing.assert_close(
        value_pre_features.context[:, ValueScalarContext.CHANCE_PHASE.value],
        torch.full((2,), float(ChancePhase.PRE_CHANCE.value)),
    )

    env.actions_this_round[:] = torch.tensor([0, 1], device=env.device)
    value_features_changed_actions = value_encoder.encode(
        beliefs, pre_chance_node=False
    )
    torch.testing.assert_close(value_features.context, value_features_changed_actions.context)


def test_better_feature_encoder_includes_multiway_betting_and_status_context():
    env = PBSEnv(
        num_envs=2,
        num_players=4,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
    )
    env.button[:] = torch.tensor([3, 1], device=env.device)
    env.to_act[:] = torch.tensor([1, 2], device=env.device)
    env.pot[:] = torch.tensor([220, 550], device=env.device)
    env.min_raise[:] = torch.tensor([40, 100], device=env.device)
    env.last_aggressive_amount[:] = torch.tensor([100, 200], device=env.device)
    env.actions_this_round[:] = torch.tensor([2, 4], device=env.device)
    env.scale[:] = torch.tensor([1000.0, 500.0], device=env.device)
    env.done.zero_()
    env.stacks[:] = torch.tensor(
        [[900, 850, 700, 1000], [500, 600, 150, 700]], device=env.device
    )
    env.committed[:] = torch.tensor(
        [[20, 100, 100, 0], [200, 200, 100, 50]], device=env.device
    )
    env.has_folded[:] = torch.tensor(
        [[False, False, False, True], [False, False, False, False]],
        device=env.device,
    )
    env.is_allin[:] = torch.tensor(
        [[False, False, True, False], [False, False, False, False]],
        device=env.device,
    )
    env.acted_this_round[:] = torch.tensor(
        [[True, False, True, False], [True, True, False, False]],
        device=env.device,
    )

    encoder = BetterPolicyFeatureEncoder(env, device=env.device, dtype=torch.float32)
    beliefs = torch.full(
        (2, 4, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
    )

    context = encoder.encode(beliefs).context
    value_context = BetterStreetValueFeatureEncoder(
        env, device=env.device, dtype=torch.float32
    ).encode(beliefs, pre_chance_node=False).context
    legal = env.legal_bins_mask()
    scale = env.scale.to(torch.float32)
    max_committed = env.committed.amax(dim=1).to(torch.float32)
    actor_committed = env.committed.gather(1, env.to_act[:, None]).squeeze(1).to(
        torch.float32
    )
    actor_to_call = (max_committed - actor_committed).clamp_min(0.0)

    torch.testing.assert_close(
        context[:, ScalarContext.MAX_COMMITTED.value], max_committed / scale
    )
    torch.testing.assert_close(
        context[:, ScalarContext.LAST_AGGRESSIVE_AMOUNT.value],
        env.last_aggressive_amount.to(torch.float32) / scale,
    )
    torch.testing.assert_close(
        context[:, ScalarContext.UNOPENED_OR_CHECKED_TO_ACTOR.value],
        (actor_to_call <= 0).to(torch.float32),
    )
    torch.testing.assert_close(
        context[:, ScalarContext.NUM_LEGAL_ACTIONS.value],
        legal.to(torch.float32).sum(dim=1) / legal.shape[1],
    )
    torch.testing.assert_close(
        context[:, ScalarContext.CAN_FOLD.value], legal[:, 0].to(torch.float32)
    )
    torch.testing.assert_close(
        context[:, ScalarContext.CAN_CALL.value], legal[:, 1].to(torch.float32)
    )
    torch.testing.assert_close(
        context[:, ScalarContext.CAN_RAISE.value],
        legal[:, 2:-1].any(dim=1).to(torch.float32),
    )
    torch.testing.assert_close(
        context[:, ScalarContext.CAN_ALLIN.value], legal[:, -1].to(torch.float32)
    )
    for policy_field, value_field in (
        (ScalarContext.MAX_COMMITTED, ValueScalarContext.MAX_COMMITTED),
        (
            ScalarContext.LAST_AGGRESSIVE_AMOUNT,
            ValueScalarContext.LAST_AGGRESSIVE_AMOUNT,
        ),
        (
            ScalarContext.UNOPENED_OR_CHECKED_TO_ACTOR,
            ValueScalarContext.UNOPENED_OR_CHECKED_TO_ACTOR,
        ),
        (ScalarContext.NUM_LEGAL_ACTIONS, ValueScalarContext.NUM_LEGAL_ACTIONS),
        (ScalarContext.CAN_FOLD, ValueScalarContext.CAN_FOLD),
        (ScalarContext.CAN_CALL, ValueScalarContext.CAN_CALL),
        (ScalarContext.CAN_RAISE, ValueScalarContext.CAN_RAISE),
        (ScalarContext.CAN_ALLIN, ValueScalarContext.CAN_ALLIN),
    ):
        torch.testing.assert_close(
            value_context[:, value_field.value],
            context[:, policy_field.value],
        )

    scalar_count = ScalarContext.NUM_SCALAR_CONTEXT.value
    num_players = env.num_players
    torch.testing.assert_close(
        value_context[:, ValueScalarContext.NUM_SCALAR_CONTEXT.value :],
        context[:, scalar_count:],
    )

    def player_ctx(field: PlayerContext) -> torch.Tensor:
        start = scalar_count + field.value * num_players
        return context[:, start : start + num_players]

    expected_rel_actor = torch.tensor(
        [[1.0, 0.0, 1.0 / 3.0, 2.0 / 3.0], [2.0 / 3.0, 1.0, 0.0, 1.0 / 3.0]],
        device=env.device,
    )
    expected_rel_button = torch.tensor(
        [[1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0], [1.0, 0.0, 1.0 / 3.0, 2.0 / 3.0]],
        device=env.device,
    )
    expected_to_call = torch.tensor(
        [[80.0, 0.0, 0.0, 100.0], [0.0, 0.0, 100.0, 150.0]],
        device=env.device,
    )

    torch.testing.assert_close(
        player_ctx(PlayerContext.FOLDED), env.has_folded.to(torch.float32)
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.ALL_IN), env.is_allin.to(torch.float32)
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.ACTED_THIS_ROUND),
        env.acted_this_round.to(torch.float32),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.IS_ACTOR),
        torch.tensor(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device=env.device
        ),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.REL_POS_TO_ACTOR), expected_rel_actor
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.REL_POS_TO_BUTTON), expected_rel_button
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.TO_CALL_SCALE), expected_to_call / scale[:, None]
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.TO_CALL_POT),
        expected_to_call / env.pot.to(torch.float32)[:, None],
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.STACK_AFTER_CALL_SCALE),
        (env.stacks.to(torch.float32) - expected_to_call).clamp_min(0.0)
        / scale[:, None],
    )


def test_better_preflop_value_feature_context_includes_actions_round():
    env = make_env(2)
    env.actions_this_round[:] = torch.tensor([4, 7], device=env.device)
    env.pot[:] = torch.tensor([150, 2500], device=env.device)
    encoder = BetterPreflopValueFeatureEncoder(
        env, device=env.device, dtype=torch.float32
    )
    beliefs = torch.full(
        (2, 2, PREFLOP_HANDS),
        1.0 / PREFLOP_HANDS,
        dtype=torch.float32,
        device=env.device,
    )

    features = encoder.encode(beliefs, pre_chance_node=False)

    assert features.hand_dim == PREFLOP_HANDS
    torch.testing.assert_close(
        features.context[:, ScalarContext.ACTIONS_ROUND.value],
        env.actions_this_round.to(torch.float32),
    )
    torch.testing.assert_close(
        features.context[:, ScalarContext.POT.value],
        env.pot.to(torch.float32) / env.scale.to(torch.float32),
    )

    env.actions_this_round[:] = torch.tensor([1, 2], device=env.device)
    changed = encoder.encode(beliefs, pre_chance_node=False)
    assert not torch.equal(features.context, changed.context)
    torch.testing.assert_close(
        changed.context[:, ScalarContext.ACTIONS_ROUND.value],
        env.actions_this_round.to(torch.float32),
    )


def test_better_feature_encoder_normalizes_chip_context_by_effective_stack():
    env = make_env(1)
    env.starting_stacks[0] = torch.tensor([500, 500], device=env.device)
    env.scale[0] = 500
    env.stacks[0] = torch.tensor([450, 400], device=env.device)
    env.committed[0] = torch.tensor([50, 100], device=env.device)
    env.pot[0] = 150
    env.min_raise[0] = 20

    encoder = BetterFeatureEncoder(env, device=env.device, dtype=torch.float32)
    beliefs = torch.full(
        (1, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
    )

    context = encoder.encode(beliefs).context
    scalar_count = ScalarContext.NUM_SCALAR_CONTEXT.value
    num_players = 2

    torch.testing.assert_close(
        context[:, ScalarContext.POT.value],
        torch.tensor([0.3], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        context[:, ScalarContext.MIN_RAISE.value],
        torch.tensor([0.04], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        context[:, ScalarContext.LOG_STACK_DEPTH_BB.value],
        torch.log(torch.tensor([50.0], dtype=torch.float32, device=env.device))
        / torch.log(torch.tensor([400.0], dtype=torch.float32, device=env.device)),
    )
    torch.testing.assert_close(
        context[:, ScalarContext.LOG_POT_BB.value],
        torch.log1p(torch.tensor([15.0], dtype=torch.float32, device=env.device)),
    )

    def player_ctx(field: PlayerContext, player: int) -> torch.Tensor:
        return context[:, scalar_count + field.value * num_players + player]

    torch.testing.assert_close(
        player_ctx(PlayerContext.STACK, 0),
        torch.tensor([0.9], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.STACK, 1),
        torch.tensor([0.8], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.COMMITTED, 0),
        torch.tensor([0.1], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.COMMITTED, 1),
        torch.tensor([0.2], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.SPR, 0),
        torch.tensor([3.0], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.SPR, 1),
        torch.tensor([400.0 / 150.0], dtype=torch.float32, device=env.device),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.LOG_COMMITTED_BB, 0),
        torch.log1p(torch.tensor([5.0], dtype=torch.float32, device=env.device)),
    )
    torch.testing.assert_close(
        player_ctx(PlayerContext.LOG_COMMITTED_BB, 1),
        torch.log1p(torch.tensor([10.0], dtype=torch.float32, device=env.device)),
    )


def _expected_remap(suit_permutation: torch.Tensor) -> torch.Tensor:
    device = suit_permutation.device
    hand_combos = hand_combos_tensor(device=device)
    combo_lookup = combo_lookup_tensor(device=device)

    all_ranks = (hand_combos % 13).unsqueeze(0)
    all_suits = (hand_combos // 13).unsqueeze(0)

    suit_perm_expanded = suit_permutation[:, None, :].expand(-1, NUM_HANDS, -1)
    new_suits = torch.gather(suit_perm_expanded, 2, all_suits)

    new_cards = all_ranks + new_suits * 13
    min_cards = torch.min(new_cards, dim=2).values
    max_cards = torch.max(new_cards, dim=2).values

    remap = combo_lookup[min_cards, max_cards].to(torch.long)
    return torch.argsort(remap, dim=1)


def test_permute_suits_permuted_board_and_beliefs():
    device = torch.device("cpu")
    batch_size = 1

    context = torch.zeros((batch_size, 1), device=device)
    street = torch.zeros(batch_size, device=device, dtype=torch.long)
    board = torch.tensor([[0, 13, 26, 39, -1]], device=device)
    board_clone = board.clone()

    # Beliefs must be (batch_size, 2 * NUM_HANDS)
    beliefs_raw = torch.rand(1, 2 * NUM_HANDS, dtype=torch.float32, device=device)
    beliefs = beliefs_raw / beliefs_raw.sum(dim=1, keepdim=True)  # Normalize
    beliefs_clone = beliefs.clone()

    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    state = generator.get_state()
    suit_permutation = torch.argsort(
        torch.rand((batch_size, 4), generator=generator), dim=-1
    )
    generator.set_state(state)

    features = MLPFeatures(
        context=context.clone(),
        street=street.clone(),
        to_act=torch.zeros(1, device=device, dtype=torch.long),
        board=board.clone(),
        beliefs=beliefs.clone(),
    )
    features.permute_suits(generator=generator)

    board_valid = board_clone >= 0
    ranks = (board_clone % 13).clamp(0, 12)
    suits = (board_clone // 13).clamp(0, 3)
    expected_suits = torch.gather(
        suit_permutation.unsqueeze(1).expand(-1, 5, -1),
        dim=2,
        index=suits.unsqueeze(2).to(torch.long),
    ).squeeze(2)
    expected_board = torch.where(board_valid, expected_suits * 13 + ranks, board_clone)
    assert torch.equal(features.board, expected_board)

    inverse_remap = _expected_remap(suit_permutation)
    # Split beliefs into two players, permute each, then concatenate
    p0_beliefs = beliefs_clone[:, :NUM_HANDS]
    p1_beliefs = beliefs_clone[:, NUM_HANDS:]
    expected_p0 = torch.gather(p0_beliefs, 1, inverse_remap)
    expected_p1 = torch.gather(p1_beliefs, 1, inverse_remap)
    expected_beliefs = torch.cat([expected_p0, expected_p1], dim=1)
    torch.testing.assert_close(features.beliefs, expected_beliefs)

    # Two-player belief tensor path.
    beliefs_two = torch.arange(
        2 * NUM_HANDS, dtype=torch.float32, device=device
    ).unsqueeze(0)
    generator_two = torch.Generator(device=device)
    generator_two.set_state(state)
    features_two = MLPFeatures(
        context=context.clone(),
        street=street.clone(),
        to_act=torch.zeros(1, device=device, dtype=torch.long),
        board=board.clone(),
        beliefs=beliefs_two.clone(),
    )
    features_two.permute_suits(generator=generator_two)
    assert torch.equal(features_two.board, expected_board)

    expected_two = torch.cat(
        [
            torch.gather(beliefs_two[:, :NUM_HANDS], 1, inverse_remap),
            torch.gather(beliefs_two[:, NUM_HANDS:], 1, inverse_remap),
        ],
        dim=1,
    )
    torch.testing.assert_close(features_two.beliefs, expected_two)
