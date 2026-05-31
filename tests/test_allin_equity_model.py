import json

import torch

from p2.allin import (
    PreflopAllInBatch,
    PreflopAllInEquityModel,
    estimate_preflop_allin_values,
    make_random_preflop_allin_batch,
)
from p2.allin.model import (
    OUTPUT_HEAD_INIT_BIAS,
    OUTPUT_HEAD_INIT_SCALE,
    _LeakyRMSBlock,
)
from p2.allin.train import (
    _pregenerated_player_permutations,
    _pregenerated_suit_permutation_idxs,
)
from p2.allin.training_data import (
    MANIFEST_NAME,
    TARGET_KEY,
    PregeneratedAllInDataset,
    batch_to_tensors,
    permute_allin_batch_by_players,
    permute_allin_batch_by_suit,
)
from p2.env.card_utils import (
    NUM_HANDS,
    combo_index,
    combo_suit_permutation_inverse_tensor,
)


def test_random_preflop_allin_batch_shapes_and_stack_distribution() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    batch = make_random_preflop_allin_batch(
        32,
        players=4,
        bb=100,
        device="cpu",
        generator=generator,
    )

    assert batch.beliefs.shape == (32, 4, NUM_HANDS)
    assert batch.starting_stacks.shape == (32, 4)
    assert batch.committed.shape == (32, 4)
    assert batch.allin_mask.shape == (32, 4)
    assert batch.folded_mask.shape == (32, 4)
    torch.testing.assert_close(
        batch.beliefs.sum(dim=-1),
        torch.ones(32, 4),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.all(batch.starting_stacks >= 10 * 100)
    assert torch.all(batch.starting_stacks <= 400 * 100)
    live_mask = ~batch.folded_mask
    assert torch.all(live_mask.sum(dim=1) >= 2)
    assert torch.all((live_mask & ~batch.allin_mask).sum(dim=1) <= 1)
    torch.testing.assert_close(batch.scale, batch.starting_stacks.mean(dim=1))


def test_allin_suit_permutation_remaps_beliefs_and_targets() -> None:
    players = 2
    batch_size = 3
    beliefs = torch.arange(
        batch_size * players * NUM_HANDS,
        dtype=torch.float32,
    ).view(batch_size, players, NUM_HANDS)
    targets = beliefs + 10_000.0
    batch = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=torch.ones(batch_size, players),
        committed=torch.zeros(batch_size, players),
        stacks_after=torch.ones(batch_size, players),
        allin_mask=torch.ones(batch_size, players, dtype=torch.bool),
        folded_mask=torch.zeros(batch_size, players, dtype=torch.bool),
        scale=torch.ones(batch_size),
    )
    perm_idxs = torch.tensor([0, 5, 17], dtype=torch.long)

    permuted_batch, permuted_targets = permute_allin_batch_by_suit(
        batch,
        targets,
        suit_permutation_idxs=perm_idxs,
    )

    inverse = combo_suit_permutation_inverse_tensor()[perm_idxs]
    expected_beliefs = torch.gather(
        beliefs,
        2,
        inverse[:, None, :].expand(-1, players, -1),
    )
    expected_targets = torch.gather(
        targets,
        2,
        inverse[:, None, :].expand(-1, players, -1),
    )
    torch.testing.assert_close(permuted_batch.beliefs, expected_beliefs)
    torch.testing.assert_close(permuted_targets, expected_targets)


def test_allin_player_permutation_remaps_features_and_targets() -> None:
    players = 4
    batch_size = 2
    beliefs = torch.arange(
        batch_size * players * NUM_HANDS,
        dtype=torch.float32,
    ).view(batch_size, players, NUM_HANDS)
    targets = beliefs + 10_000.0
    batch = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=torch.arange(batch_size * players, dtype=torch.float32).view(
            batch_size, players
        ),
        committed=torch.arange(
            100,
            100 + batch_size * players,
            dtype=torch.float32,
        ).view(batch_size, players),
        stacks_after=torch.arange(
            200,
            200 + batch_size * players,
            dtype=torch.float32,
        ).view(batch_size, players),
        allin_mask=torch.tensor(
            [[True, False, True, False], [False, True, False, True]]
        ),
        folded_mask=torch.tensor(
            [[False, False, True, False], [True, False, False, False]]
        ),
        scale=torch.arange(batch_size, dtype=torch.float32) + 1.0,
    )
    player_permutations = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])

    permuted_batch, permuted_targets = permute_allin_batch_by_players(
        batch,
        targets,
        player_permutations=player_permutations,
    )

    hand_index = player_permutations[:, :, None].expand(-1, -1, NUM_HANDS)
    torch.testing.assert_close(
        permuted_batch.beliefs,
        torch.gather(beliefs, 1, hand_index),
    )
    torch.testing.assert_close(
        permuted_targets,
        torch.gather(targets, 1, hand_index),
    )
    torch.testing.assert_close(
        permuted_batch.starting_stacks,
        torch.gather(batch.starting_stacks, 1, player_permutations),
    )
    torch.testing.assert_close(
        permuted_batch.committed,
        torch.gather(batch.committed, 1, player_permutations),
    )
    torch.testing.assert_close(
        permuted_batch.stacks_after,
        torch.gather(batch.stacks_after, 1, player_permutations),
    )
    assert torch.equal(
        permuted_batch.allin_mask,
        torch.gather(batch.allin_mask, 1, player_permutations),
    )
    assert torch.equal(
        permuted_batch.folded_mask,
        torch.gather(batch.folded_mask, 1, player_permutations),
    )
    torch.testing.assert_close(permuted_batch.scale, batch.scale)


def test_pregenerated_dataset_wraps_batches(tmp_path) -> None:
    players = 2
    examples = 3
    beliefs = torch.arange(
        examples * players * NUM_HANDS,
        dtype=torch.float32,
    ).view(examples, players, NUM_HANDS)
    batch = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=torch.arange(examples * players, dtype=torch.float32).view(
            examples, players
        ),
        committed=torch.zeros(examples, players),
        stacks_after=torch.ones(examples, players),
        allin_mask=torch.ones(examples, players, dtype=torch.bool),
        folded_mask=torch.zeros(examples, players, dtype=torch.bool),
        scale=torch.ones(examples),
    )
    targets = beliefs + 20_000.0
    torch.save(batch_to_tensors(batch, targets), tmp_path / "shard_000000.pt")
    manifest = {
        "format": "p2.allin.training_data.v1",
        "examples": examples,
        "players": players,
        "hands": NUM_HANDS,
        "feature_keys": [
            "beliefs",
            "starting_stacks",
            "committed",
            "stacks_after",
            "allin_mask",
            "folded_mask",
            "scale",
        ],
        "target_key": TARGET_KEY,
        "config": {},
        "shards": [
            {
                "file": "shard_000000.pt",
                "examples": examples,
                "start": 0,
                "end": examples,
            }
        ],
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest))

    dataset = PregeneratedAllInDataset(tmp_path, async_shard_prefetch=True)
    dataset.prefetch_shard_for_row(2)
    try:
        wrapped_batch, wrapped_targets = dataset.get_wrapped_batch(
            2,
            4,
            device=torch.device("cpu"),
        )
    finally:
        dataset.close()

    expected_rows = torch.tensor([2, 0, 1, 2])
    torch.testing.assert_close(wrapped_batch.beliefs, beliefs[expected_rows])
    torch.testing.assert_close(wrapped_targets, targets[expected_rows])


def test_pregenerated_suit_permutation_changes_on_second_epoch() -> None:
    examples = 7
    first_epoch = _pregenerated_suit_permutation_idxs(
        global_start=0,
        batch_size=examples,
        dataset_examples=examples,
        seed=123,
        device=torch.device("cpu"),
    )
    second_epoch = _pregenerated_suit_permutation_idxs(
        global_start=examples,
        batch_size=examples,
        dataset_examples=examples,
        seed=123,
        device=torch.device("cpu"),
    )

    assert torch.all(first_epoch != second_epoch)


def test_pregenerated_player_permutations_are_valid_and_epoch_varying() -> None:
    examples = 7
    players = 4
    first_epoch = _pregenerated_player_permutations(
        global_start=0,
        batch_size=examples,
        dataset_examples=examples,
        players=players,
        seed=123,
        device=torch.device("cpu"),
    )
    second_epoch = _pregenerated_player_permutations(
        global_start=examples,
        batch_size=examples,
        dataset_examples=examples,
        players=players,
        seed=123,
        device=torch.device("cpu"),
    )

    expected_sorted = torch.arange(players).expand(examples, -1)
    torch.testing.assert_close(first_epoch.sort(dim=1).values, expected_sorted)
    torch.testing.assert_close(second_epoch.sort(dim=1).values, expected_sorted)
    assert torch.any(first_epoch != second_epoch)


def test_random_preflop_allin_batch_has_covering_caller() -> None:
    generator = torch.Generator(device="cpu").manual_seed(234)
    batch = make_random_preflop_allin_batch(
        512,
        players=4,
        bb=100,
        device="cpu",
        generator=generator,
    )

    live_mask = ~batch.folded_mask
    caller_mask = live_mask & ~batch.allin_mask
    assert not torch.any(batch.folded_mask & batch.allin_mask)
    assert torch.all(caller_mask.sum(dim=1) <= 1)

    torch.testing.assert_close(
        batch.committed[batch.allin_mask],
        batch.starting_stacks[batch.allin_mask],
    )
    torch.testing.assert_close(
        batch.stacks_after[batch.allin_mask],
        torch.zeros_like(batch.stacks_after[batch.allin_mask]),
    )

    max_allin_commit = torch.where(
        batch.allin_mask,
        batch.committed,
        torch.zeros_like(batch.committed),
    ).amax(dim=1, keepdim=True)
    torch.testing.assert_close(
        batch.committed[caller_mask],
        max_allin_commit.expand_as(batch.committed)[caller_mask],
    )
    assert torch.all(batch.stacks_after[caller_mask] > 0.0)


def test_random_preflop_allin_batch_marks_tied_deepest_live_players_allin() -> None:
    generator = torch.Generator(device="cpu").manual_seed(235)
    batch = make_random_preflop_allin_batch(
        128,
        players=4,
        bb=100,
        min_stack_bb=10,
        mid_stack_bb=10,
        max_stack_bb=10,
        device="cpu",
        generator=generator,
    )

    live_mask = ~batch.folded_mask
    assert torch.all(batch.allin_mask == live_mask)
    torch.testing.assert_close(
        batch.committed[live_mask],
        batch.starting_stacks[live_mask],
    )
    torch.testing.assert_close(
        batch.stacks_after[live_mask],
        torch.zeros_like(batch.stacks_after[live_mask]),
    )


def test_random_preflop_allin_batch_caps_folded_dead_money() -> None:
    generator = torch.Generator(device="cpu").manual_seed(321)
    batch = make_random_preflop_allin_batch(
        512,
        players=4,
        bb=100,
        folded_commit_max_frac=1.0,
        device="cpu",
        generator=generator,
    )

    live_mask = ~batch.folded_mask
    max_live_commit = torch.where(
        live_mask,
        batch.committed,
        torch.zeros_like(batch.committed),
    ).amax(dim=1, keepdim=True)
    assert torch.all(
        batch.committed[batch.folded_mask]
        <= max_live_commit.expand_as(batch.committed)[batch.folded_mask]
    )


def test_preflop_allin_model_shapes_and_prenorm_blocks() -> None:
    generator = torch.Generator(device="cpu").manual_seed(456)
    batch = make_random_preflop_allin_batch(
        3,
        players=4,
        device="cpu",
        generator=generator,
    )
    model = PreflopAllInEquityModel(
        players=4,
        hidden_dim=64,
        hand_dim=32,
        num_layers=2,
    )
    out = model(
        batch.beliefs,
        batch.starting_stacks,
        batch.committed,
        batch.stacks_after,
        batch.allin_mask,
        batch.folded_mask,
    )

    assert out.shape == (3, 4, NUM_HANDS)
    assert torch.isfinite(out).all()
    assert model.film_rank == 64
    assert model.value_film_hand_proj.weight.shape == (64, 32)
    assert model.value_film_state.weight.shape == (64, 64)
    blocks = [m for m in model.modules() if isinstance(m, _LeakyRMSBlock)]
    assert len(blocks) == 2
    assert all(isinstance(block.norm, torch.nn.RMSNorm) for block in blocks)
    assert all(isinstance(block.activation, torch.nn.LeakyReLU) for block in blocks)


def test_preflop_allin_model_configurable_film_rank() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=4,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(458),
    )
    model = PreflopAllInEquityModel(
        players=4,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        film_rank=7,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(459))

    out = model(
        batch.beliefs,
        batch.starting_stacks,
        batch.committed,
        batch.stacks_after,
        batch.allin_mask,
        batch.folded_mask,
    )

    assert out.shape == (2, 4, NUM_HANDS)
    assert model.value_film_hand_proj.weight.shape == (7, 32)
    assert model.value_film_state.weight.shape == (7, 64)
    torch.testing.assert_close(
        model.value_film_state.weight,
        torch.zeros_like(model.value_film_state.weight),
    )
    torch.testing.assert_close(
        model.value_bias.bias,
        torch.full_like(model.value_bias.bias, OUTPUT_HEAD_INIT_BIAS),
    )
    torch.testing.assert_close(
        model.value_scale.weight.norm(dim=1),
        torch.full((64,), OUTPUT_HEAD_INIT_SCALE),
    )


def test_preflop_allin_model_can_disable_film_branch() -> None:
    model = PreflopAllInEquityModel(
        players=4,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        film_rank=0,
    )

    assert model.film_rank == 0
    assert not hasattr(model, "value_film_hand_proj")
    assert not hasattr(model, "value_film_state")


def test_preflop_allin_model_dense_belief_residual_is_configurable() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(460),
    )
    model = PreflopAllInEquityModel(
        players=3,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        film_rank=0,
        dense_belief_residual=True,
    )

    assert model.dense_belief_residual is True
    assert hasattr(model, "dense_belief_proj")
    out = model(
        batch.beliefs,
        batch.starting_stacks,
        batch.committed,
        batch.stacks_after,
        batch.allin_mask,
        batch.folded_mask,
    )

    assert out.shape == (2, 3, NUM_HANDS)
    assert torch.isfinite(out).all()


def test_preflop_allin_model_card_and_blocker_features_are_combo_exact() -> None:
    model = PreflopAllInEquityModel(
        players=2,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        film_rank=0,
    )
    beliefs = torch.zeros(1, 2, NUM_HANDS)
    hero_combo = combo_index(0, 13)
    opp_combo = combo_index(0, 1)
    beliefs[0, 0, hero_combo] = 1.0
    beliefs[0, 1, opp_combo] = 1.0

    card_mass, bucket_features = model._range_mass_features(beliefs)
    blocker_features = model._blocker_features(
        beliefs,
        card_mass,
        folded_mask=torch.zeros(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(card_mass[0, 0, 0], torch.tensor(1.0))
    torch.testing.assert_close(card_mass[0, 0, 13], torch.tensor(1.0))
    torch.testing.assert_close(card_mass[0, 1, 0], torch.tensor(1.0))
    torch.testing.assert_close(card_mass[0, 1, 1], torch.tensor(1.0))
    assert bucket_features.shape == (1, 2, 20)
    torch.testing.assert_close(
        blocker_features[0, 0, hero_combo, 1],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        blocker_features[0, 1, opp_combo, 0],
        torch.tensor(0.0),
    )


def test_preflop_allin_model_max_eligible_to_win_feature() -> None:
    committed = torch.tensor(
        [
            [100.0, 200.0, 50.0, 0.0],
            [100.0, 100.0, 100.0, 10.0],
        ]
    )
    folded_mask = torch.tensor(
        [
            [False, False, True, True],
            [False, False, False, True],
        ]
    )

    max_eligible = PreflopAllInEquityModel._max_eligible_to_win(
        committed,
        folded_mask,
    )

    expected = torch.tensor(
        [
            [250.0, 350.0, 0.0, 0.0],
            [310.0, 310.0, 310.0, 0.0],
        ]
    )
    torch.testing.assert_close(max_eligible, expected)


def test_preflop_allin_model_hard_codes_folded_values() -> None:
    generator = torch.Generator(device="cpu").manual_seed(457)
    batch = make_random_preflop_allin_batch(
        3,
        players=4,
        device="cpu",
        generator=generator,
        min_allin_players=3,
    )
    model = PreflopAllInEquityModel(
        players=4,
        hidden_dim=64,
        hand_dim=32,
        num_layers=2,
    )
    out = model(
        batch.beliefs,
        batch.starting_stacks,
        batch.committed,
        batch.stacks_after,
        batch.allin_mask,
        batch.folded_mask,
    )

    folded_value = (
        batch.stacks_after - batch.starting_stacks
    ) / batch.starting_stacks.mean(dim=1, keepdim=True).clamp_min(1.0)
    expected = folded_value[:, :, None].expand_as(out)
    torch.testing.assert_close(out[batch.folded_mask], expected[batch.folded_mask])


def test_preflop_allin_sampler_small_smoke() -> None:
    generator = torch.Generator(device="cpu").manual_seed(789)
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        bb=100,
        device="cpu",
        generator=generator,
    )
    values, diagnostics = estimate_preflop_allin_values(
        batch,
        board_samples=4,
        tuple_samples=2,
        tuple_tries=2,
        board_chunk=2,  # cur_boards > 1 exercises the segmented-sum accumulation
        hand_chunk=256,
        generator=generator,
    )

    assert values.shape == (2, 3, NUM_HANDS)
    assert torch.isfinite(values).all()
    assert diagnostics["target_seconds"] >= 0.0
    assert diagnostics["target_boards_per_second"] > 0.0


def test_preflop_allin_sampler_compute_stats_false_matches_values() -> None:
    def run(compute_stats: bool):
        generator = torch.Generator(device="cpu").manual_seed(789)
        batch = make_random_preflop_allin_batch(
            2,
            players=3,
            bb=100,
            device="cpu",
            generator=generator,
        )
        return estimate_preflop_allin_values(
            batch,
            board_samples=2,
            tuple_samples=2,
            tuple_tries=2,
            board_chunk=1,
            hand_chunk=256,
            generator=generator,
            compute_stats=compute_stats,
        )

    values, diagnostics = run(True)
    values_no_stats, diagnostics_no_stats = run(False)

    # Skipping diagnostics must not change the estimated values.
    torch.testing.assert_close(values_no_stats, values)
    assert diagnostics_no_stats == {}
    assert diagnostics["target_seconds"] >= 0.0


def test_preflop_allin_sampler_uses_exact_table_for_two_live_players() -> None:
    generator = torch.Generator(device="cpu").manual_seed(987)
    batch = make_random_preflop_allin_batch(
        2,
        players=2,
        bb=100,
        device="cpu",
        generator=generator,
    )
    values_a, diagnostics_a = estimate_preflop_allin_values(
        batch,
        sample_count=1,
        board_samples=1,
        tuple_samples=None,
        generator=generator,
    )
    values_b, diagnostics_b = estimate_preflop_allin_values(
        batch,
        sample_count=17,
        board_samples=3,
        tuple_samples=None,
        generator=generator,
    )

    assert values_a.shape == (2, 2, NUM_HANDS)
    assert torch.isfinite(values_a).all()
    torch.testing.assert_close(values_a, values_b)
    assert diagnostics_a["target_exact_two_player_rows"] == 2.0
    assert diagnostics_a["target_mc_rows"] == 0.0
    assert diagnostics_b["target_exact_two_player_rows"] == 2.0
    assert diagnostics_b["target_mc_rows"] == 0.0


def test_preflop_allin_targets_award_folded_dead_money_to_live_players() -> None:
    players = 3
    beliefs = torch.full((1, players, NUM_HANDS), 1.0 / NUM_HANDS)
    base = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=torch.full((1, players), 100.0),
        committed=torch.tensor([[100.0, 100.0, 0.0]]),
        stacks_after=torch.tensor([[0.0, 0.0, 100.0]]),
        allin_mask=torch.tensor([[True, True, False]]),
        folded_mask=torch.tensor([[False, False, True]]),
        scale=torch.tensor([100.0]),
    )
    with_dead_money = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=base.starting_stacks,
        committed=torch.tensor([[100.0, 100.0, 20.0]]),
        stacks_after=torch.tensor([[0.0, 0.0, 80.0]]),
        allin_mask=base.allin_mask,
        folded_mask=base.folded_mask,
        scale=base.scale,
    )

    base_values, _ = estimate_preflop_allin_values(
        base,
        sample_count=1,
        board_samples=1,
        tuple_samples=None,
        compute_stats=False,
    )
    dead_money_values, _ = estimate_preflop_allin_values(
        with_dead_money,
        sample_count=1,
        board_samples=1,
        tuple_samples=None,
        compute_stats=False,
    )

    live_delta = (
        (dead_money_values[:, :2] - base_values[:, :2]) * beliefs[:, :2]
    ).sum()
    folded_delta = (
        (dead_money_values[:, 2] - base_values[:, 2]) * beliefs[:, 2]
    ).sum()
    torch.testing.assert_close(live_delta, torch.tensor(0.2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(folded_delta, torch.tensor(-0.2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(live_delta + folded_delta, torch.tensor(0.0), atol=1e-5, rtol=1e-5)
