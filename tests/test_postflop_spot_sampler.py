from __future__ import annotations

import torch

from p2.env.card_utils import board_allowed_hands
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.search.postflop_spot_sampler import sample_river_start_roots


def test_sample_river_start_roots_builds_board_legal_street_start_pbs():
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

    pbs = sample_river_start_roots(env, batch_size=8, generator=generator)

    assert pbs.env.N == 8
    assert pbs.beliefs.shape == (8, 2, 1326)
    assert torch.equal(pbs.env.street, torch.full((8,), 3, dtype=torch.long))
    assert torch.equal(pbs.env.actions_this_round, torch.zeros(8, dtype=torch.long))
    assert torch.equal(pbs.env.to_act, 1 - pbs.env.button)
    assert not pbs.env.done.any()
    assert not pbs.env.has_folded.any()
    assert not pbs.env.is_allin.any()

    board = pbs.env.board_indices
    assert board.shape == (8, 5)
    assert (board >= 0).all()
    assert all(row.unique().numel() == 5 for row in board)
    assert torch.equal(pbs.env.last_board_indices[:, :4], board[:, :4])
    assert torch.equal(
        pbs.env.last_board_indices[:, 4], torch.full((8,), -1, dtype=torch.long)
    )

    allowed = board_allowed_hands(board)
    assert torch.equal(pbs.beliefs > 0, allowed[:, None, :].expand_as(pbs.beliefs))
    torch.testing.assert_close(
        pbs.beliefs.sum(dim=-1), torch.ones(8, 2), atol=1e-6, rtol=0.0
    )
    assert pbs.env.legal_bins_mask().any(dim=1).all()
