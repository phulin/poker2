from __future__ import annotations

import pytest
import torch

from p2.env.pbs_env import PBSEnv


def _make_envs(num_envs: int = 4) -> tuple[PBSEnv, object]:
    from p2.env.triton_pbs_env import TritonPBSEnv

    button = torch.zeros(num_envs, dtype=torch.long, device="cuda")
    deck = torch.tensor(
        [[10, 11, 12, 13, 14]] * num_envs,
        dtype=torch.long,
        device="cuda",
    )
    base = PBSEnv(
        num_envs=num_envs,
        num_players=3,
        starting_stack=40,
        sb=5,
        bb=10,
        device="cuda",
    )
    triton_env = TritonPBSEnv(
        num_envs=num_envs,
        num_players=3,
        starting_stack=40,
        sb=5,
        bb=10,
        device="cuda",
    )
    base.reset(force_button=button, force_deck=deck)
    triton_env.reset(force_button=button, force_deck=deck)
    return base, triton_env


def _assert_envs_equal(base: PBSEnv, triton_env: PBSEnv) -> None:
    fields = (
        "deck",
        "deck_pos",
        "button",
        "street",
        "to_act",
        "last_to_act",
        "pot",
        "min_raise",
        "last_aggressive_amount",
        "actions_this_round",
        "actions_last_round",
        "stacks",
        "starting_stacks",
        "scale",
        "committed",
        "chips_placed",
        "has_folded",
        "is_allin",
        "acted_this_round",
        "done",
        "winner",
        "winners",
        "board_indices",
        "last_board_indices",
        "board_onehot",
    )
    for field in fields:
        assert torch.equal(getattr(base, field), getattr(triton_env, field)), field


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_pbs_env_matches_pbs_env_mixed_steps_exactly() -> None:
    pytest.importorskip("triton")
    base, triton_env = _make_envs()
    allin = base.num_bet_bins - 1
    scripts = (
        torch.tensor([1, 4, 0, allin], dtype=torch.long, device="cuda"),
        torch.tensor([1, 1, 0, allin], dtype=torch.long, device="cuda"),
        torch.tensor([1, 1, -1, allin], dtype=torch.long, device="cuda"),
    )

    _assert_envs_equal(base, triton_env)
    for actions in scripts:
        base_amounts, base_mask = base.legal_bins_amounts_and_mask()
        triton_amounts, triton_mask = triton_env.legal_bins_amounts_and_mask()
        assert torch.equal(base_amounts, triton_amounts)
        assert torch.equal(base_mask, triton_mask)

        expected = base.step_bins(actions)
        actual = triton_env.step_bins(actions)
        for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
            assert torch.equal(expected_tensor, actual_tensor)
        _assert_envs_equal(base, triton_env)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_pbs_env_direct_step_matches_pbs_env_exactly() -> None:
    pytest.importorskip("triton")
    base, triton_env = _make_envs()
    amounts, _ = base.legal_bins_amounts_and_mask()
    actions = torch.tensor([1, 2, 0, 3], dtype=torch.long, device="cuda")
    bet_amounts = torch.tensor(
        [0, amounts[1, 2].item(), 0, 0],
        dtype=torch.long,
        device="cuda",
    )

    expected = base.step(actions, bet_amounts)
    actual = triton_env.step(actions, bet_amounts)
    for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
        assert torch.equal(expected_tensor, actual_tensor)
    _assert_envs_equal(base, triton_env)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_pbs_env_legal_kernel_after_state_changes() -> None:
    pytest.importorskip("triton")
    base, triton_env = _make_envs()
    first_actions = torch.tensor(
        [1, 4, 0, base.num_bet_bins - 1], dtype=torch.long, device="cuda"
    )
    base.step_bins(first_actions)
    triton_env.step_bins(first_actions)

    expected_amounts, expected_mask = base.legal_bins_amounts_and_mask()
    actual_amounts, actual_mask = triton_env.legal_bins_amounts_and_mask()
    assert torch.equal(expected_amounts, actual_amounts)
    assert torch.equal(expected_mask, actual_mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_pbs_env_copy_and_gather_preserve_class_and_state() -> None:
    pytest.importorskip("triton")
    _, triton_env = _make_envs()
    actions = torch.tensor([1, 4, 0, triton_env.num_bet_bins - 1], device="cuda")
    triton_env.step_bins(actions)

    gathered = triton_env.gather_rows(torch.tensor([3, 1, 0], device="cuda"))
    assert gathered.__class__.__name__ == "TritonPBSEnv"
    assert torch.equal(gathered.pot, triton_env.pot[torch.tensor([3, 1, 0], device="cuda")])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_pbs_env_reset_subset_matches_pbs_env() -> None:
    pytest.importorskip("triton")
    from p2.env.triton_pbs_env import TritonPBSEnv

    base = PBSEnv(num_envs=5, num_players=4, starting_stack=100, device="cuda")
    triton_env = TritonPBSEnv(num_envs=5, num_players=4, starting_stack=100, device="cuda")
    button = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long, device="cuda")
    deck = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25],
        ],
        dtype=torch.long,
        device="cuda",
    )
    base.reset(force_button=button, force_deck=deck)
    triton_env.reset(force_button=button, force_deck=deck)
    _assert_envs_equal(base, triton_env)

    ids = torch.tensor([1, 3], dtype=torch.long, device="cuda")
    subset_button = torch.tensor([2, 1], dtype=torch.long, device="cuda")
    subset_deck = torch.tensor(
        [[30, 31, 32, 33, 34], [35, 36, 37, 38, 39]],
        dtype=torch.long,
        device="cuda",
    )
    base.reset(env_indices=ids, force_button=subset_button, force_deck=subset_deck)
    triton_env.reset(env_indices=ids, force_button=subset_button, force_deck=subset_deck)
    _assert_envs_equal(base, triton_env)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_pbs_env_copy_state_from_and_repeat_match_pbs_env() -> None:
    pytest.importorskip("triton")
    base, triton_env = _make_envs()
    actions = torch.tensor([1, 4, 0, triton_env.num_bet_bins - 1], device="cuda")
    base.step_bins(actions)
    triton_env.step_bins(actions)

    base_dst = PBSEnv(
        num_envs=3, num_players=3, starting_stack=40, sb=5, bb=10, device="cuda"
    )
    triton_dst = triton_env.__class__(
        num_envs=3, num_players=3, starting_stack=40, sb=5, bb=10, device="cuda"
    )
    src = torch.tensor([3, 1, 0], device="cuda")
    dst = torch.arange(3, device="cuda")
    base_dst.copy_state_from(base, src, dst)
    triton_dst.copy_state_from(triton_env, src, dst)
    _assert_envs_equal(base_dst, triton_dst)

    repeats = torch.tensor([2, 0, 1, 3], device="cuda")
    base_repeated = base.repeat_interleave(repeats, output_size=int(repeats.sum().item()))
    triton_repeated = triton_env.repeat_interleave(
        repeats, output_size=int(repeats.sum().item())
    )
    _assert_envs_equal(base_repeated, triton_repeated)
