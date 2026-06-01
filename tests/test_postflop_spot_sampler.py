from __future__ import annotations

import torch

from p2.env.card_utils import board_allowed_hands
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.search.postflop_spot_sampler import (
    BOARD_TEXTURE_DRY,
    BOARD_TEXTURE_MONOTONE,
    BOARD_TEXTURE_PAIRED,
    BOARD_TEXTURE_STRAIGHT_HEAVY,
    BOARD_TEXTURE_TWO_TONE,
    _board_texture_labels,
    _board_texture_target_matches,
    _recursive_strength_beliefs,
    sample_end_of_street_chance_roots,
    sample_flop_start_roots,
    sample_postflop_start_roots,
    sample_river_start_roots,
    sample_turn_start_roots,
)


def _assert_street_start_root(pbs, *, street: int, board_cards: int) -> None:
    batch_size = pbs.env.N
    assert pbs.beliefs.shape == (batch_size, 2, 1326)
    assert torch.equal(
        pbs.env.street, torch.full((batch_size,), street, dtype=torch.long)
    )
    assert torch.equal(
        pbs.env.actions_this_round, torch.zeros(batch_size, dtype=torch.long)
    )
    assert torch.equal(pbs.env.to_act, 1 - pbs.env.button)
    assert not pbs.env.done.any()
    assert not pbs.env.has_folded.any()
    assert not pbs.env.is_allin.any()
    assert torch.equal(pbs.env.committed, torch.zeros(batch_size, 2, dtype=torch.long))
    assert torch.equal(pbs.env.pot, pbs.env.chips_placed.sum(dim=1))
    assert torch.equal(pbs.env.stacks + pbs.env.chips_placed, pbs.env.starting_stacks)
    assert (pbs.env.pot >= 2 * pbs.env.bb).all()

    board = pbs.env.board_indices
    assert board.shape == (batch_size, 5)
    assert (board[:, :board_cards] >= 0).all()
    assert (board[:, board_cards:] == -1).all()
    assert all(row.unique().numel() == board_cards for row in board[:, :board_cards])

    if street == 1:
        assert torch.equal(
            pbs.env.last_board_indices,
            torch.full((batch_size, 5), -1, dtype=torch.long),
        )
    else:
        assert torch.equal(
            pbs.env.last_board_indices[:, : board_cards - 1],
            board[:, : board_cards - 1],
        )
        assert torch.equal(
            pbs.env.last_board_indices[:, board_cards - 1 :],
            torch.full((batch_size, 6 - board_cards), -1, dtype=torch.long),
        )

    allowed = board_allowed_hands(board)
    assert torch.equal(pbs.beliefs > 0, allowed[:, None, :].expand_as(pbs.beliefs))
    torch.testing.assert_close(
        pbs.beliefs.sum(dim=-1),
        torch.ones(batch_size, 2),
        atol=1e-6,
        rtol=0.0,
    )
    assert pbs.env.legal_bins_mask().any(dim=1).all()


def test_sample_postflop_start_roots_builds_board_legal_street_start_pbs():
    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=2,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        float_dtype=torch.float32,
    )
    generator = torch.Generator(device=device).manual_seed(123)

    for street, board_cards, sampler in (
        (1, 3, sample_flop_start_roots),
        (2, 4, sample_turn_start_roots),
        (3, 5, sample_river_start_roots),
    ):
        pbs = sampler(env, batch_size=8, generator=generator)
        _assert_street_start_root(pbs, street=street, board_cards=board_cards)


def test_sample_postflop_start_roots_rejects_unsupported_street():
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
    )

    try:
        sample_postflop_start_roots(env, batch_size=1, street=0)
    except ValueError as exc:
        assert "street must be" in str(exc)
    else:
        raise AssertionError("Expected unsupported street to raise ValueError")


def test_sample_postflop_start_roots_randomizes_board_legal_beliefs():
    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        float_dtype=torch.float32,
    )

    random_pbs = sample_river_start_roots(
        env,
        batch_size=4,
        generator=torch.Generator(device=device).manual_seed(17),
    )
    uniform_pbs = sample_postflop_start_roots(
        env,
        batch_size=4,
        street=3,
        generator=torch.Generator(device=device).manual_seed(17),
        randomize_beliefs=False,
    )

    allowed = board_allowed_hands(random_pbs.env.board_indices)
    legal_counts = allowed.sum(dim=-1).to(torch.float32)
    random_allowed_values = random_pbs.beliefs.masked_select(
        allowed[:, None, :].expand_as(random_pbs.beliefs)
    )
    assert random_allowed_values.unique().numel() > 16

    expected_uniform = torch.where(
        allowed[:, None, :],
        (1.0 / legal_counts)[:, None, None],
        torch.zeros_like(uniform_pbs.beliefs),
    )
    torch.testing.assert_close(uniform_pbs.beliefs, expected_uniform)


def test_recursive_strength_beliefs_are_board_legal_normalized_and_nonuniform():
    device = torch.device("cpu")
    board = torch.tensor(
        [
            [0, 14, 28, 42, 51],
            [12, 25, 38, 3, 16],
        ],
        device=device,
        dtype=torch.long,
    )
    allowed = board_allowed_hands(board)
    beliefs = _recursive_strength_beliefs(
        board,
        allowed,
        num_players=2,
        generator=torch.Generator(device=device).manual_seed(7),
    )

    assert beliefs.shape == (2, 2, 1326)
    assert torch.equal(beliefs > 0, allowed[:, None, :].expand_as(beliefs))
    torch.testing.assert_close(
        beliefs.sum(dim=-1),
        torch.ones(2, 2),
        atol=1e-6,
        rtol=0.0,
    )
    allowed_values = beliefs.masked_select(allowed[:, None, :].expand_as(beliefs))
    assert allowed_values.unique().numel() > 128
    assert not torch.allclose(beliefs[:, 0], beliefs[:, 1])


def test_sample_postflop_start_roots_stratifies_board_textures():
    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        float_dtype=torch.float32,
    )

    pbs = sample_river_start_roots(
        env,
        batch_size=256,
        generator=torch.Generator(device=device).manual_seed(41),
        randomize_beliefs=False,
    )

    labels = _board_texture_labels(pbs.env.board_indices)
    observed = set(labels.unique().tolist())
    assert {BOARD_TEXTURE_PAIRED, BOARD_TEXTURE_MONOTONE}.issubset(observed)
    target_coverage = _board_texture_target_matches(pbs.env.board_indices).any(dim=0)
    assert target_coverage[BOARD_TEXTURE_PAIRED]
    assert target_coverage[BOARD_TEXTURE_MONOTONE]
    assert target_coverage[BOARD_TEXTURE_TWO_TONE]
    assert target_coverage[BOARD_TEXTURE_STRAIGHT_HEAVY]
    assert target_coverage[BOARD_TEXTURE_DRY]


def test_sample_postflop_start_roots_uses_conservative_spot_templates():
    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        float_dtype=torch.float32,
    )

    pbs = sample_river_start_roots(
        env,
        batch_size=64,
        generator=torch.Generator(device=device).manual_seed(29),
        randomize_beliefs=False,
    )

    assert pbs.env.pot.unique().numel() > 8
    assert set(pbs.env.actions_last_round.unique().tolist()).issubset({2, 3, 5})
    assert pbs.env.actions_last_round.unique().numel() >= 2
    assert torch.equal(pbs.env.committed, torch.zeros(64, 2, dtype=torch.long))
    assert torch.equal(pbs.env.pot, pbs.env.chips_placed.sum(dim=1))
    assert torch.equal(pbs.env.stacks + pbs.env.chips_placed, pbs.env.starting_stacks)
    assert pbs.env.legal_bins_mask().any(dim=1).all()


def test_sample_end_of_street_chance_roots_builds_pre_chance_beliefs():
    device = torch.device("cpu")
    env = HUNLTensorEnv(
        num_envs=2,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        float_dtype=torch.float32,
    )
    generator = torch.Generator(device=device).manual_seed(321)

    for closed_street, next_street, board_cards in (
        (0, 1, 3),
        (1, 2, 4),
        (2, 3, 5),
    ):
        sample = sample_end_of_street_chance_roots(
            env,
            batch_size=7,
            closed_street=closed_street,
            generator=generator,
        )

        assert sample.closed_street == closed_street
        _assert_street_start_root(
            sample.pbs, street=next_street, board_cards=board_cards
        )
        assert sample.pre_chance_beliefs.shape == (7, 2, 1326)

        previous_board = sample.pbs.env.last_board_indices
        previous_allowed = board_allowed_hands(previous_board)
        assert torch.equal(
            sample.pre_chance_beliefs > 0,
            previous_allowed[:, None, :].expand_as(sample.pre_chance_beliefs),
        )
        torch.testing.assert_close(
            sample.pre_chance_beliefs.sum(dim=-1),
            torch.ones(7, 2),
            atol=1e-6,
            rtol=0.0,
        )


def test_sample_end_of_street_chance_roots_rejects_unsupported_street():
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
    )

    try:
        sample_end_of_street_chance_roots(env, batch_size=1, closed_street=3)
    except ValueError as exc:
        assert "closed_street must be" in str(exc)
    else:
        raise AssertionError("Expected unsupported closed street to raise ValueError")
