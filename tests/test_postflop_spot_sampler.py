from __future__ import annotations

import torch

from p2.env.card_utils import board_allowed_hands
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.search.postflop_spot_sampler import (
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
