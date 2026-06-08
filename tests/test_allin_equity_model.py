import importlib.util
import json
from pathlib import Path

import pytest
import torch

from p2.allin import (
    PreflopAllIn169EquityModel,
    PreflopAllIn169TransformerModel,
    PreflopAllInBatch,
    PreflopAllInEquityModel,
    PreflopAllInTransformerModel,
    estimate_preflop_allin_values,
    estimate_preflop_allin_values_169,
    make_random_preflop_allin_batch,
)
from p2.allin.pregenerate import PregenerateConfig, _data_config
from p2.allin.model import (
    OUTPUT_HEAD_INIT_BIAS,
    OUTPUT_HEAD_INIT_SCALE,
    PLAYER_FEATURE_DIM,
    POT_LAYER_SCALAR_FEATURE_DIM,
    _LeakyRMSBlock,
)
from p2.allin.oracle import PreflopAllIn169Oracle
from p2.allin.sampler import NUM_CANONICAL_FULL_BOARDS
from p2.allin.train import (
    AllInTrainConfig,
    _build_model,
    _data_config_from_train,
    _eligible_pot_share_to_values,
    _evaluate_pregenerated_dataset,
    _predict_allin_values,
    _pregenerated_player_permutations,
    _pregenerated_suit_permutation_idxs,
    _values_to_eligible_pot_share,
)
from p2.allin.training_data import (
    AllInDataGenConfig,
    FEATURE_KEYS,
    MANIFEST_NAME,
    TARGET_KEY,
    PregeneratedAllInDataset,
    batch_to_tensors,
    generate_allin_training_chunk,
    permute_allin_batch_by_players,
    permute_allin_batch_by_suit,
)
from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    collapse_1326_to_169,
    combo_index,
    combo_suit_permutation_inverse_tensor,
)
from p2.env.rules import rank_hands


def _load_allin_169_converter():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "convert_allin_dataset_to_169.py"
    )
    spec = importlib.util.spec_from_file_location(
        "convert_allin_dataset_to_169",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert_allin_dataset_to_169


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


def test_random_preflop_allin_batch_supports_native_169_mode() -> None:
    generator = torch.Generator(device="cpu").manual_seed(124)
    batch = make_random_preflop_allin_batch(
        8,
        players=3,
        bb=100,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=generator,
    )

    assert batch.hand_dim == PREFLOP_HANDS
    assert batch.beliefs.shape == (8, 3, PREFLOP_HANDS)
    torch.testing.assert_close(
        batch.beliefs.sum(dim=-1),
        torch.ones(8, 3),
        rtol=1e-6,
        atol=1e-6,
    )

    with pytest.raises(ValueError, match="hand_dim"):
        make_random_preflop_allin_batch(1, hand_dim=170)


def test_random_preflop_allin_batch_respects_min_allin_players() -> None:
    generator = torch.Generator(device="cpu").manual_seed(125)
    batch = make_random_preflop_allin_batch(
        64,
        players=6,
        min_allin_players=4,
        device="cpu",
        generator=generator,
    )

    assert torch.all((~batch.folded_mask).sum(dim=1) >= 4)


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


def test_allin_eval_logs_mse_mae_by_live_player_count(tmp_path) -> None:
    players = 4
    examples = 4
    folded_mask = torch.tensor(
        [
            [False, False, True, True],
            [False, True, False, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    beliefs = torch.full((examples, players, NUM_HANDS), 1.0 / NUM_HANDS)
    batch = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=torch.ones(examples, players),
        committed=torch.zeros(examples, players),
        stacks_after=torch.ones(examples, players),
        allin_mask=~folded_mask,
        folded_mask=folded_mask,
        scale=torch.ones(examples),
    )
    row_targets = torch.tensor([1.0, 3.0, 2.0, -4.0])
    targets = row_targets[:, None, None].expand(-1, players, NUM_HANDS).contiguous()
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

    class ZeroModel(torch.nn.Module):
        def forward(
            self,
            beliefs,
            starting_stacks,
            committed,
            stacks_after,
            allin_mask,
            folded_mask,
        ):
            return torch.zeros_like(beliefs)

    dataset = PregeneratedAllInDataset(tmp_path)
    try:
        metrics = _evaluate_pregenerated_dataset(
            ZeroModel(),
            dataset,
            batch_size=2,
            device=torch.device("cpu"),
        )
    finally:
        dataset.close()

    assert metrics["eval/mse"] == 7.5
    assert metrics["eval/mae"] == 2.5
    assert metrics["eval/mae_p95"] == 4.0
    assert metrics["eval/mae_p99"] == 4.0
    assert metrics["eval/live_players/2/examples"] == 2.0
    assert metrics["eval/live_players/2/mse"] == 5.0
    assert metrics["eval/live_players/2/mae"] == 2.0
    assert metrics["eval/live_players/3/examples"] == 1.0
    assert metrics["eval/live_players/3/mse"] == 4.0
    assert metrics["eval/live_players/3/mae"] == 2.0
    assert metrics["eval/live_players/4/examples"] == 1.0
    assert metrics["eval/live_players/4/mse"] == 16.0
    assert metrics["eval/live_players/4/mae"] == 4.0

    dataset = PregeneratedAllInDataset(tmp_path)
    try:
        filtered = _evaluate_pregenerated_dataset(
            ZeroModel(),
            dataset,
            batch_size=2,
            device=torch.device("cpu"),
            min_live_players=3,
        )
    finally:
        dataset.close()

    assert filtered["eval/examples"] == 2.0
    assert filtered["eval/raw_examples"] == 4.0
    assert filtered["eval/skipped_examples"] == 2.0
    assert filtered["eval/min_live_players"] == 3.0
    assert filtered["eval/mse"] == 10.0
    assert filtered["eval/mae"] == 3.0
    assert "eval/live_players/2/examples" not in filtered
    assert filtered["eval/live_players/3/examples"] == 1.0
    assert filtered["eval/live_players/4/examples"] == 1.0


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
    assert PLAYER_FEATURE_DIM == 8
    assert model.player_proj[0].in_features == PLAYER_FEATURE_DIM
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


def test_preflop_allin_model_dense_output_residual_is_configurable() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(463),
    )
    model = PreflopAllInEquityModel(
        players=3,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        film_rank=0,
        dense_output_residual=True,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(464))

    assert model.dense_output_residual is True
    assert model.dense_output_proj.weight.shape == (NUM_HANDS, 64)
    torch.testing.assert_close(
        model.dense_output_proj.weight,
        torch.zeros_like(model.dense_output_proj.weight),
    )
    torch.testing.assert_close(
        model.dense_output_proj.bias,
        torch.zeros_like(model.dense_output_proj.bias),
    )
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


def test_preflop_allin_transformer_model_shapes_and_tokens() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(461),
    )
    model = PreflopAllInTransformerModel(
        players=3,
        hidden_dim=64,
        hand_dim=32,
        num_layers=2,
        film_rank=7,
        transformer_heads=4,
        dense_belief_residual=True,
        dense_output_residual=True,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(462))

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
    assert model.context_proj[0].in_features == 3 * PLAYER_FEATURE_DIM
    assert model.pot_layer_proj[0].in_features == POT_LAYER_SCALAR_FEATURE_DIM + 6
    assert len(model.encoder) == 2
    assert model.encoder[0].num_heads == 4
    assert hasattr(model, "dense_belief_proj")
    assert model.dense_output_proj.weight.shape == (NUM_HANDS, 64)
    assert not hasattr(model, "player_proj")


def test_preflop_allin_transformer_masks_ineligible_pot_attention() -> None:
    model = PreflopAllInTransformerModel(
        players=3,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        transformer_heads=4,
    )
    beliefs = torch.full((1, 3, NUM_HANDS), 1.0 / NUM_HANDS)
    committed = torch.tensor([[100.0, 50.0, 25.0]])
    folded_mask = torch.zeros(1, 3, dtype=torch.bool)
    scale = torch.ones(1, 1)

    layer_features, eligible = model._pot_layer_features(
        beliefs,
        committed,
        folded_mask,
        scale,
    )
    mask = model._token_attention_mask(eligible)

    assert layer_features.shape == (1, 3, POT_LAYER_SCALAR_FEATURE_DIM + 6)
    assert mask.shape == (1, 1, 7, 7)
    assert torch.isfinite(mask[0, 0, 1, 4:7]).all()
    assert torch.isfinite(mask[0, 0, 2, 4:6]).all()
    assert torch.isneginf(mask[0, 0, 2, 6])
    assert torch.isfinite(mask[0, 0, 3, 4])
    assert torch.isneginf(mask[0, 0, 3, 5:7]).all()


def test_preflop_allin_transformer_can_mask_folded_player_tokens() -> None:
    model = PreflopAllInTransformerModel(
        players=3,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        transformer_heads=4,
        mask_folded_player_tokens=True,
    )
    player_pot_eligible = torch.ones(1, 3, 3, dtype=torch.bool)
    folded_mask = torch.tensor([[False, True, False]])

    mask = model._token_attention_mask(player_pot_eligible, folded_mask)

    assert torch.isfinite(mask[0, 0, :, 1]).all()
    assert torch.isneginf(mask[0, 0, :, 2]).all()
    assert torch.isfinite(mask[0, 0, :, 3]).all()


def test_preflop_allin_transformer_pot_bitmasks_follow_player_permutation() -> None:
    players = 4
    model = PreflopAllInTransformerModel(
        players=players,
        hidden_dim=64,
        hand_dim=32,
        num_layers=1,
        transformer_heads=4,
    )
    beliefs = torch.full((1, players, NUM_HANDS), 1.0 / NUM_HANDS)
    batch = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=torch.tensor([[120.0, 90.0, 60.0, 30.0]]),
        committed=torch.tensor([[100.0, 50.0, 25.0, 10.0]]),
        stacks_after=torch.tensor([[20.0, 40.0, 35.0, 20.0]]),
        allin_mask=torch.tensor([[True, True, True, False]]),
        folded_mask=torch.tensor([[False, False, False, True]]),
        scale=torch.ones(1),
    )
    targets = torch.zeros(1, players, NUM_HANDS)
    permutation = torch.tensor([[2, 0, 3, 1]])

    permuted_batch, _ = permute_allin_batch_by_players(
        batch,
        targets,
        player_permutations=permutation,
    )
    base_features, base_eligible = model._pot_layer_features(
        batch.beliefs,
        batch.committed,
        batch.folded_mask,
        batch.starting_stacks.mean(dim=1, keepdim=True),
    )
    permuted_features, permuted_eligible = model._pot_layer_features(
        permuted_batch.beliefs,
        permuted_batch.committed,
        permuted_batch.folded_mask,
        permuted_batch.starting_stacks.mean(dim=1, keepdim=True),
    )

    bit_index = permutation[:, None, :].expand(-1, players, -1)
    torch.testing.assert_close(
        permuted_features[..., :POT_LAYER_SCALAR_FEATURE_DIM],
        base_features[..., :POT_LAYER_SCALAR_FEATURE_DIM],
    )
    torch.testing.assert_close(
        permuted_features[
            ...,
            POT_LAYER_SCALAR_FEATURE_DIM : POT_LAYER_SCALAR_FEATURE_DIM + players,
        ],
        torch.gather(
            base_features[
                ...,
                POT_LAYER_SCALAR_FEATURE_DIM : POT_LAYER_SCALAR_FEATURE_DIM + players,
            ],
            2,
            bit_index,
        ),
    )
    torch.testing.assert_close(
        permuted_features[..., POT_LAYER_SCALAR_FEATURE_DIM + players :],
        torch.gather(
            base_features[..., POT_LAYER_SCALAR_FEATURE_DIM + players :],
            2,
            bit_index,
        ),
    )
    assert torch.equal(permuted_eligible, torch.gather(base_eligible, 2, bit_index))


def test_allin_train_config_builds_transformer_model() -> None:
    cfg = AllInTrainConfig(
        players=3,
        model_type="player_transformer",
        hidden_dim=64,
        hand_dim=32,
        layers=1,
        transformer_heads=4,
        dense_output_residual=True,
        mask_folded_player_tokens=True,
    )

    model = _build_model(cfg)

    assert isinstance(model, PreflopAllInTransformerModel)
    assert model.transformer_heads == 4
    assert model.dense_output_residual is True
    assert model.mask_folded_player_tokens is True


def test_allin_train_config_passes_min_allin_players_to_data_config() -> None:
    cfg = _data_config_from_train(
        AllInTrainConfig(players=6, min_allin_players=4)
    )

    assert cfg.players == 6
    assert cfg.min_allin_players == 4


def test_eligible_pot_share_target_round_trips_live_values() -> None:
    batch = make_random_preflop_allin_batch(
        8,
        players=4,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(465),
    )
    targets = torch.randn(8, 4, NUM_HANDS) * 0.1
    folded_value = (batch.stacks_after - batch.starting_stacks) / batch.scale[
        :, None
    ].clamp_min(1.0)
    targets = torch.where(
        batch.folded_mask[:, :, None],
        folded_value[:, :, None],
        targets,
    )

    share = _values_to_eligible_pot_share(targets, batch)
    round_trip = _eligible_pot_share_to_values(share, batch)

    torch.testing.assert_close(round_trip, targets)
    torch.testing.assert_close(
        share[batch.folded_mask],
        torch.zeros_like(share[batch.folded_mask]),
    )


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


def test_preflop_allin_oracle_net_model_uses_evaluator_scale_for_folded_values() -> None:
    class ZeroNetValueModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(
            self,
            beliefs: torch.Tensor,
            starting_stacks: torch.Tensor,
            committed: torch.Tensor,
            stacks_after: torch.Tensor,
            allin_mask: torch.Tensor,
            folded_mask: torch.Tensor,
            *,
            hardcode_folded_values: bool = True,
        ) -> torch.Tensor:
            assert hardcode_folded_values is False
            return torch.zeros_like(beliefs) + self.weight

    oracle = PreflopAllIn169Oracle(device=torch.device("cpu"))
    oracle._model = ZeroNetValueModel()
    oracle._model_target_mode = "net_value"
    beliefs = torch.full((1, 4, PREFLOP_HANDS), 1.0 / PREFLOP_HANDS)
    starting_stacks = torch.tensor([[100.0, 200.0, 300.0, 400.0]])
    committed = torch.tensor([[100.0, 200.0, 300.0, 25.0]])
    stacks_after = torch.tensor([[0.0, 0.0, 0.0, 375.0]])
    allin_mask = torch.tensor([[True, True, True, False]])
    folded_mask = torch.tensor([[False, False, False, True]])
    scale = torch.tensor([50.0])

    out = oracle.values(
        beliefs=beliefs,
        starting_stacks=starting_stacks,
        committed=committed,
        stacks_after=stacks_after,
        allin_mask=allin_mask,
        folded_mask=folded_mask,
        scale=scale,
        live_players=4,
    )

    expected_folded = (stacks_after - starting_stacks) / scale[:, None].clamp_min(1.0)
    torch.testing.assert_close(
        out[folded_mask],
        expected_folded[:, :, None].expand_as(out)[folded_mask],
    )
    torch.testing.assert_close(
        out[~folded_mask],
        torch.zeros_like(out[~folded_mask]),
    )


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


def test_preflop_allin_sampler_native_169_small_smoke() -> None:
    generator = torch.Generator(device="cpu").manual_seed(789)
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        bb=100,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=generator,
    )
    values, diagnostics = estimate_preflop_allin_values_169(
        batch,
        board_samples=3,
        tuple_samples=2,
        tuple_tries=2,
        board_chunk=1,
        generator=generator,
        use_exact_two_player=False,
    )

    assert values.shape == (2, 3, PREFLOP_HANDS)
    assert torch.isfinite(values).all()
    assert diagnostics["target_hand_dim"] == float(PREFLOP_HANDS)
    assert diagnostics["target_board_samples"] == 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_preflop_allin_sampler_native_169_uses_cuda_kernel_path() -> None:
    device = torch.device("cuda")
    batch = make_random_preflop_allin_batch(
        2,
        players=4,
        min_allin_players=3,
        hand_dim=PREFLOP_HANDS,
        device=device,
        generator=torch.Generator(device=device).manual_seed(7901),
    )

    values, diagnostics = estimate_preflop_allin_values_169(
        batch,
        sample_count=None,
        board_samples=2,
        tuple_samples=1,
        tuple_tries=1,
        board_chunk=2,
        generator=torch.Generator(device=device).manual_seed(7902),
        compute_stats=True,
        use_exact_two_player=False,
    )

    assert values.shape == (2, 4, PREFLOP_HANDS)
    assert torch.isfinite(values).all()
    assert diagnostics["target_hand_dim"] == float(PREFLOP_HANDS)
    assert "target_kernel_launch_seconds" in diagnostics


def test_preflop_allin_sampler_dispatches_native_169_exact_two_player() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=2,
        bb=100,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(788),
    )
    values, diagnostics = estimate_preflop_allin_values(
        batch,
        sample_count=1,
        board_samples=1,
        tuple_samples=None,
    )

    assert values.shape == (2, 2, PREFLOP_HANDS)
    assert torch.isfinite(values).all()
    assert diagnostics["target_hand_dim"] == float(PREFLOP_HANDS)
    assert diagnostics["target_exact_two_player_rows"] == 2.0


def test_preflop_allin_169_model_uses_hand_major_boundary() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        bb=100,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(790),
    )
    model = PreflopAllIn169EquityModel(
        players=3,
        hidden_dim=48,
        hand_dim=16,
        num_layers=1,
        film_rank=8,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(791))

    out = model(
        batch.beliefs.transpose(1, 2).contiguous(),
        batch.starting_stacks,
        batch.committed,
        batch.stacks_after,
        batch.allin_mask,
        batch.folded_mask,
    )

    assert out.shape == (2, PREFLOP_HANDS, 3)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match=r"\[batch, 169, 3\]"):
        model(
            batch.beliefs,
            batch.starting_stacks,
            batch.committed,
            batch.stacks_after,
            batch.allin_mask,
            batch.folded_mask,
        )


def test_preflop_allin_169_transformer_model_shapes_and_tokens() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        bb=100,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(795),
    )
    model = PreflopAllIn169TransformerModel(
        players=3,
        hidden_dim=64,
        hand_dim=32,
        num_layers=2,
        film_rank=7,
        transformer_heads=4,
        dense_belief_residual=True,
        dense_output_residual=True,
        mask_folded_player_tokens=True,
    )
    model.init_weights(torch.Generator(device="cpu").manual_seed(796))

    out = model(
        batch.beliefs.transpose(1, 2).contiguous(),
        batch.starting_stacks,
        batch.committed,
        batch.stacks_after,
        batch.allin_mask,
        batch.folded_mask,
    )

    assert out.shape == (2, PREFLOP_HANDS, 3)
    assert torch.isfinite(out).all()
    assert model.context_proj[0].in_features == 3 * PLAYER_FEATURE_DIM
    assert model.pot_layer_proj[0].in_features == POT_LAYER_SCALAR_FEATURE_DIM + 6
    assert model.bucket_mass_proj[0].in_features == 16
    assert model.dense_belief_proj[0].in_features == 3 * PREFLOP_HANDS
    assert model.dense_output_proj.weight.shape == (PREFLOP_HANDS, 64)
    assert len(model.encoder) == 2
    assert model.encoder[0].num_heads == 4
    assert model.mask_folded_player_tokens is True
    assert not hasattr(model, "card_mass_proj")


def test_allin_train_builds_native_169_model_and_predicts_player_major_loss_shape() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(792),
    )
    cfg = AllInTrainConfig(
        model_type="compact_169",
        range_hand_dim=PREFLOP_HANDS,
        players=3,
        hidden_dim=48,
        hand_dim=16,
        layers=1,
        film_rank=8,
    )

    model = _build_model(cfg)
    assert isinstance(model, PreflopAllIn169EquityModel)
    pred = _predict_allin_values(
        model,
        batch,
        target_mode="net_value",
    )

    assert pred.shape == (2, 3, PREFLOP_HANDS)
    assert torch.isfinite(pred).all()
    with pytest.raises(ValueError, match="range_hand_dim=169"):
        _build_model(AllInTrainConfig(range_hand_dim=PREFLOP_HANDS))


def test_allin_train_builds_native_169_transformer_model() -> None:
    batch = make_random_preflop_allin_batch(
        2,
        players=3,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=torch.Generator(device="cpu").manual_seed(797),
    )
    cfg = AllInTrainConfig(
        model_type="transformer_169",
        range_hand_dim=PREFLOP_HANDS,
        players=3,
        hidden_dim=64,
        hand_dim=32,
        layers=1,
        film_rank=8,
        transformer_heads=4,
        dense_output_residual=True,
        mask_folded_player_tokens=True,
    )

    model = _build_model(cfg)
    assert isinstance(model, PreflopAllIn169TransformerModel)
    assert isinstance(model, PreflopAllIn169EquityModel)
    assert model.transformer_heads == 4
    assert model.dense_output_residual is True
    pred = _predict_allin_values(
        model,
        batch,
        target_mode="net_value",
    )

    assert pred.shape == (2, 3, PREFLOP_HANDS)
    assert torch.isfinite(pred).all()


def test_generate_allin_training_chunk_can_emit_compact_169_targets() -> None:
    cfg = AllInDataGenConfig(
        players=3,
        range_hand_dim=PREFLOP_HANDS,
        board_samples=2,
        tuple_samples=1,
        tuple_tries=1,
        board_chunk=1,
        hand_chunk=512,
        use_exact_two_player=False,
    )

    tensors, diagnostics = generate_allin_training_chunk(
        2,
        cfg,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(793),
        compute_stats=True,
    )

    assert tensors["beliefs"].shape == (2, 3, PREFLOP_HANDS)
    assert tensors[TARGET_KEY].shape == (2, 3, PREFLOP_HANDS)
    torch.testing.assert_close(
        tensors["beliefs"].sum(dim=-1),
        torch.ones(2, 3),
        atol=1e-6,
        rtol=1e-6,
    )
    assert diagnostics["target_hand_dim"] == float(PREFLOP_HANDS)


def test_generate_allin_training_chunk_routes_169_batch_to_estimator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import p2.allin.training_data as training_data

    seen_hand_dims: list[int] = []

    def fake_estimator(batch: PreflopAllInBatch, **kwargs):
        del kwargs
        seen_hand_dims.append(batch.hand_dim)
        return torch.zeros_like(batch.beliefs), {"target_hand_dim": float(batch.hand_dim)}

    monkeypatch.setattr(training_data, "estimate_preflop_allin_values", fake_estimator)
    cfg = AllInDataGenConfig(
        players=6,
        min_allin_players=4,
        range_hand_dim=PREFLOP_HANDS,
    )

    tensors, diagnostics = training_data.generate_allin_training_chunk(
        2,
        cfg,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(794),
        compute_stats=True,
    )

    assert seen_hand_dims == [PREFLOP_HANDS]
    assert tensors["beliefs"].shape == (2, 6, PREFLOP_HANDS)
    assert tensors[TARGET_KEY].shape == (2, 6, PREFLOP_HANDS)
    assert torch.all((~tensors["folded_mask"]).sum(dim=1) >= 4)
    assert diagnostics["target_hand_dim"] == float(PREFLOP_HANDS)


def test_pregenerated_dataset_can_load_1326_data_as_compact_169(tmp_path) -> None:
    examples = 2
    players = 2
    beliefs = torch.zeros(examples, players, NUM_HANDS)
    beliefs[..., :10] = 1.0
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
    targets = torch.arange(
        examples * players * NUM_HANDS,
        dtype=torch.float32,
    ).view(examples, players, NUM_HANDS)
    batch = PreflopAllInBatch(
        beliefs=beliefs,
        starting_stacks=torch.ones(examples, players),
        committed=torch.zeros(examples, players),
        stacks_after=torch.ones(examples, players),
        allin_mask=torch.ones(examples, players, dtype=torch.bool),
        folded_mask=torch.zeros(examples, players, dtype=torch.bool),
        scale=torch.ones(examples),
    )
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

    dataset = PregeneratedAllInDataset(tmp_path, target_hands=PREFLOP_HANDS)
    compact_batch, compact_targets = dataset.get_batch(
        0,
        examples,
        device=torch.device("cpu"),
    )

    assert compact_batch.beliefs.shape == (examples, players, PREFLOP_HANDS)
    assert compact_targets.shape == (examples, players, PREFLOP_HANDS)
    torch.testing.assert_close(
        compact_batch.beliefs,
        collapse_1326_to_169(beliefs, reduction="sum"),
    )
    torch.testing.assert_close(
        compact_targets,
        collapse_1326_to_169(targets, reduction="mean"),
    )


def test_convert_allin_dataset_to_169_writes_compact_shards(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "converted"
    source.mkdir()
    examples = 2
    players = 3
    beliefs = torch.rand(examples, players, NUM_HANDS)
    beliefs = beliefs / beliefs.sum(dim=-1, keepdim=True)
    targets = torch.randn(examples, players, NUM_HANDS)
    tensors = {
        "beliefs": beliefs,
        "starting_stacks": torch.ones(examples, players),
        "committed": torch.zeros(examples, players),
        "stacks_after": torch.ones(examples, players),
        "allin_mask": torch.ones(examples, players, dtype=torch.bool),
        "folded_mask": torch.zeros(examples, players, dtype=torch.bool),
        "scale": torch.ones(examples),
        TARGET_KEY: targets,
    }
    torch.save(tensors, source / "shard_000000.pt")
    manifest = {
        "format": "p2.allin.training_data.v1",
        "examples": examples,
        "players": players,
        "hands": NUM_HANDS,
        "feature_keys": list(FEATURE_KEYS),
        "target_key": TARGET_KEY,
        "config": {"range_hand_dim": NUM_HANDS},
        "shards": [
            {
                "file": "shard_000000.pt",
                "examples": examples,
                "start": 0,
                "end": examples,
            }
        ],
    }
    (source / MANIFEST_NAME).write_text(json.dumps(manifest))

    convert_allin_dataset_to_169 = _load_allin_169_converter()
    converted_manifest = convert_allin_dataset_to_169(source, output)
    converted = torch.load(output / "shard_000000.pt", map_location="cpu")

    assert converted_manifest["hands"] == PREFLOP_HANDS
    assert converted_manifest["config"]["range_hand_dim"] == PREFLOP_HANDS
    assert converted_manifest["conversion"]["source_hands"] == NUM_HANDS
    assert converted_manifest["conversion"]["target_hands"] == PREFLOP_HANDS
    assert converted["beliefs"].shape == (examples, players, PREFLOP_HANDS)
    assert converted[TARGET_KEY].shape == (examples, players, PREFLOP_HANDS)
    torch.testing.assert_close(
        converted["beliefs"],
        collapse_1326_to_169(beliefs, reduction="sum"),
    )
    torch.testing.assert_close(
        converted[TARGET_KEY],
        collapse_1326_to_169(targets, reduction="mean"),
    )


def test_pregenerate_all_boards_config_uses_samples_per_board() -> None:
    cfg = _data_config(
        PregenerateConfig(
            all_boards=True,
            samples_per_board=3,
            sample_count=50_000,
            board_samples=256,
            tuple_samples=0,
        )
    )

    assert cfg.all_boards is True
    assert cfg.sample_count is None
    assert cfg.board_samples == NUM_CANONICAL_FULL_BOARDS
    assert cfg.tuple_samples == 3
    assert cfg.board_ranks_path


def test_pregenerate_config_passes_native_169_hand_dim() -> None:
    cfg = _data_config(
        PregenerateConfig(
            players=6,
            min_allin_players=4,
            range_hand_dim=PREFLOP_HANDS,
        )
    )

    assert cfg.players == 6
    assert cfg.min_allin_players == 4
    assert cfg.range_hand_dim == PREFLOP_HANDS


def test_pregenerate_all_boards_requires_samples_per_board() -> None:
    with pytest.raises(ValueError, match="--all-boards requires"):
        _data_config(
            PregenerateConfig(all_boards=True, samples_per_board=0, tuple_samples=0)
        )


def test_preflop_allin_sampler_exhaustive_boards_uses_fixed_board_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import p2.allin.sampler as sampler

    boards = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
            [0, 1, 2, 3, 6],
            [0, 1, 2, 3, 7],
        ],
        dtype=torch.long,
    )

    def fail_random_boards(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("random board sampler should not run")

    monkeypatch.setattr(sampler, "NUM_FULL_BOARDS", boards.shape[0])
    monkeypatch.setattr(sampler, "_full_boards", lambda device: boards.to(device))
    monkeypatch.setattr(sampler, "_sample_full_boards", fail_random_boards)

    generator = torch.Generator(device="cpu").manual_seed(789)
    batch = make_random_preflop_allin_batch(
        1,
        players=3,
        bb=100,
        device="cpu",
        generator=generator,
    )
    values, diagnostics = sampler.estimate_preflop_allin_values(
        batch,
        sample_count=None,
        tuple_samples=1,
        tuple_tries=1,
        board_chunk=2,
        hand_chunk=512,
        generator=generator,
        use_exact_two_player=False,
        exhaustive_boards=True,
    )

    assert values.shape == (1, 3, NUM_HANDS)
    assert torch.isfinite(values).all()
    assert diagnostics["target_board_samples"] == float(boards.shape[0])
    assert diagnostics["target_tuple_samples"] == 1.0
    assert diagnostics["target_exhaustive_boards"] == 1.0


def test_preflop_allin_sampler_exhaustive_boards_can_use_cached_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import p2.allin.sampler as sampler

    boards = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 5],
        ],
        dtype=torch.long,
    )
    scores, _ = rank_hands(boards.int())
    sorted_scores, sorted_idx = scores.sort(dim=1)
    group_start = torch.ones_like(sorted_scores, dtype=torch.bool)
    group_start[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    sorted_dense = torch.cumsum(group_start.to(torch.int32), dim=1)
    dense = torch.empty_like(sorted_dense)
    dense.scatter_(1, sorted_idx, sorted_dense)
    cache = sampler.CanonicalBoardRankCache(
        path=Path("/tmp/fake-board-ranks.bin"),
        ranks=dense.to(torch.uint16).numpy(),
        boards=boards,
        weights=torch.tensor([1.0, 3.0], dtype=torch.float32),
        metadata={},
    )

    def fail_board_source(*args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("raw exhaustive board source should not run")

    monkeypatch.setattr(
        sampler,
        "_load_canonical_board_rank_cache",
        lambda path: cache,
    )
    monkeypatch.setattr(sampler, "_full_boards", fail_board_source)
    monkeypatch.setattr(sampler, "_rank_hands", fail_board_source)

    generator = torch.Generator(device="cpu").manual_seed(790)
    batch = make_random_preflop_allin_batch(
        1,
        players=3,
        bb=100,
        hand_dim=PREFLOP_HANDS,
        device="cpu",
        generator=generator,
    )
    values, diagnostics = sampler.estimate_preflop_allin_values(
        batch,
        sample_count=None,
        board_samples=boards.shape[0],
        tuple_samples=2,
        tuple_tries=2,
        board_chunk=1,
        hand_chunk=512,
        generator=generator,
        use_exact_two_player=False,
        exhaustive_boards=True,
        board_ranks_path="/tmp/fake-board-ranks.bin",
    )

    assert values.shape == (1, 3, PREFLOP_HANDS)
    assert torch.isfinite(values).all()
    assert diagnostics["target_board_samples"] == float(boards.shape[0])
    assert diagnostics["target_cached_board_ranks"] == 1.0


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
    folded_delta = ((dead_money_values[:, 2] - base_values[:, 2]) * beliefs[:, 2]).sum()
    torch.testing.assert_close(live_delta, torch.tensor(0.2), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(folded_delta, torch.tensor(-0.2), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        live_delta + folded_delta, torch.tensor(0.0), atol=1e-5, rtol=1e-5
    )
