import torch

from p2.allin import (
    PreflopAllInEquityModel,
    estimate_preflop_allin_values,
    make_random_preflop_allin_batch,
)
from p2.allin.model import _LeakyRMSBlock
from p2.env.card_utils import NUM_HANDS


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
    assert torch.all(batch.allin_mask.sum(dim=1) >= 2)
    torch.testing.assert_close(batch.scale, batch.starting_stacks.mean(dim=1))


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
    blocks = [m for m in model.modules() if isinstance(m, _LeakyRMSBlock)]
    assert len(blocks) == 2
    assert all(isinstance(block.norm, torch.nn.RMSNorm) for block in blocks)
    assert all(isinstance(block.activation, torch.nn.LeakyReLU) for block in blocks)


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
