from __future__ import annotations

import torch

from p2.env.card_utils import NUM_HANDS, combo_index
from p2.env.nl_env import NLEnv
from p2.env.pbs_env import PBSEnv
from p2.search.cfr_evaluator import PublicBeliefState

STREET_TO_INT = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
    "showdown": 4,
}


def _make_pbs(
    num_envs: int = 1,
    num_players: int = 3,
    starting_stack: int = 1000,
) -> PBSEnv:
    env = PBSEnv(
        num_envs=num_envs,
        num_players=num_players,
        starting_stack=starting_stack,
        sb=5,
        bb=10,
        device="cpu",
    )
    force_button = torch.zeros(num_envs, dtype=torch.long)
    force_deck = torch.tensor(
        [[10, 11, 12, 13, 14]] * num_envs,
        dtype=torch.long,
    )
    env.reset(force_button=force_button, force_deck=force_deck)
    return env


def _make_ref(num_players: int = 3, starting_stack: int = 1000) -> NLEnv:
    env = NLEnv(
        num_players=num_players,
        starting_stack=starting_stack,
        sb=5,
        bb=10,
        seed=123,
        deal_hole_cards=False,
    )
    env.reset(force_button=0, force_deck=[10, 11, 12, 13, 14])
    return env


def test_public_belief_state_clones_pbs_env_type():
    env = _make_pbs(num_envs=2, num_players=3)
    beliefs = torch.full((2, 3, NUM_HANDS), 1.0 / NUM_HANDS)

    pbs = PublicBeliefState.from_proto(env, beliefs, num_envs=2)

    assert isinstance(pbs.env, PBSEnv)
    assert pbs.env.num_players == 3
    assert pbs.beliefs.shape == (2, 3, NUM_HANDS)


def _assert_public_state_matches(ref: NLEnv, pbs: PBSEnv, idx: int = 0) -> None:
    state = ref.state
    assert state is not None
    assert int(pbs.button[idx]) == state.button
    assert int(pbs.street[idx]) == STREET_TO_INT[state.street]
    assert int(pbs.to_act[idx]) == state.to_act
    assert int(pbs.pot[idx]) == state.pot
    assert int(pbs.min_raise[idx]) == state.min_raise
    assert bool(pbs.done[idx]) is state.terminal

    expected_stacks = torch.tensor([p.stack for p in state.players], dtype=torch.long)
    expected_committed = torch.tensor(
        [p.committed for p in state.players], dtype=torch.long
    )
    expected_folded = torch.tensor(
        [p.has_folded for p in state.players], dtype=torch.bool
    )
    expected_allin = torch.tensor([p.is_allin for p in state.players], dtype=torch.bool)
    expected_acted = torch.tensor(
        [p.acted_this_round for p in state.players], dtype=torch.bool
    )

    torch.testing.assert_close(pbs.stacks[idx].cpu(), expected_stacks)
    torch.testing.assert_close(pbs.committed[idx].cpu(), expected_committed)
    torch.testing.assert_close(pbs.has_folded[idx].cpu(), expected_folded)
    torch.testing.assert_close(pbs.is_allin[idx].cpu(), expected_allin)
    torch.testing.assert_close(pbs.acted_this_round[idx].cpu(), expected_acted)

    board = torch.full((5,), -1, dtype=torch.long)
    if state.board:
        board[: len(state.board)] = torch.tensor(state.board, dtype=torch.long)
    torch.testing.assert_close(pbs.board_indices[idx].cpu(), board)

    if state.winners:
        expected_winners = torch.zeros(len(state.players), dtype=torch.bool)
        expected_winners[list(state.winners)] = True
        torch.testing.assert_close(pbs.winners[idx].cpu(), expected_winners)


def _assert_legal_bins_match(ref: NLEnv, pbs: PBSEnv) -> None:
    ref_amounts, ref_mask = ref.legal_bins_amounts_and_mask()
    pbs_amounts, pbs_mask = pbs.legal_bins_amounts_and_mask()
    pbs_subset_amounts = pbs.legal_bins_amounts_for(
        torch.tensor([0], dtype=torch.long, device=pbs.device)
    )
    torch.testing.assert_close(
        pbs_amounts[0].cpu(),
        torch.tensor(ref_amounts, dtype=torch.long),
    )
    torch.testing.assert_close(pbs_subset_amounts[0].cpu(), pbs_amounts[0].cpu())
    torch.testing.assert_close(
        pbs_mask[0].cpu(),
        torch.tensor(ref_mask, dtype=torch.bool),
    )


def _step_and_match(ref: NLEnv, pbs: PBSEnv, action_bin: int) -> torch.Tensor:
    _assert_legal_bins_match(ref, pbs)
    _, ref_rewards, _, _ = ref.step_bins(action_bin)
    pbs_rewards, _, _ = pbs.step_bins(torch.tensor([action_bin], dtype=torch.long))
    _assert_public_state_matches(ref, pbs)
    torch.testing.assert_close(
        pbs_rewards[0].cpu(),
        torch.tensor(ref_rewards, dtype=torch.float32),
    )
    return pbs_rewards[0].cpu()


def test_nl_env_reset_deals_multiway_hole_cards_and_showdown_winner():
    env = NLEnv(num_players=3, starting_stack=1000, sb=5, bb=10, deal_hole_cards=True)
    state = env.reset(
        force_button=0,
        force_deck=[12, 25, 11, 24, 10, 23, 0, 2, 4, 6, 8],
    )

    assert state.to_act == 0
    assert [player.hole_cards for player in state.players] == [
        [12, 25],
        [11, 24],
        [10, 23],
    ]

    for _ in range(12):
        state, rewards, done, _ = env.step_bins(1)
        if done:
            break

    assert state.street == "showdown"
    assert state.terminal is True
    assert state.winner == 0
    assert rewards[0] > rewards[1]
    assert rewards[0] > rewards[2]


def test_nl_env_private_showdown_rewards_side_pots():
    env = NLEnv(num_players=3, starting_stack=100, sb=0, bb=0, deal_hole_cards=True)
    state = env.reset(
        force_button=0,
        force_starting_stacks=[40, 100, 100],
        force_deck=[12, 25, 11, 24, 10, 23, 0, 14, 28, 42, 7],
    )
    assert [player.starting_stack for player in state.players] == [40, 100, 100]

    allin = env.num_bet_bins - 1
    rewards = [0.0, 0.0, 0.0]
    for _ in range(3):
        state, rewards, done, _ = env.step_bins(allin)
        if done:
            break

    assert state.terminal is True
    assert state.winner == 0
    assert rewards == [2.0, 0.2, -1.0]


def test_pbs_env_reset_is_public_only_and_multiway():
    env = _make_pbs(num_envs=4, num_players=4)

    assert env.stacks.shape == (4, 4)
    assert env.committed.shape == (4, 4)
    assert env.board_indices.shape == (4, 5)
    assert not hasattr(env, "hole_indices")
    assert not hasattr(env, "hole_onehot")
    torch.testing.assert_close(env.button, torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(env.to_act, torch.full((4,), 3, dtype=torch.long))
    torch.testing.assert_close(env.committed[:, 1], torch.full((4,), 5))
    torch.testing.assert_close(env.committed[:, 2], torch.full((4,), 10))


def test_pbs_env_reset_samples_each_player_stack_independently():
    rng = torch.Generator(device="cpu").manual_seed(7)
    env = PBSEnv(
        num_envs=128,
        num_players=4,
        starting_stack=1000,
        sb=5,
        bb=10,
        stack_mode="weighted_uniform_bb",
        min_stack_bb=4,
        mid_stack_bb=8,
        max_stack_bb=12,
        rng=rng,
        device="cpu",
    )
    env.reset(force_button=torch.zeros(128, dtype=torch.long))

    assert env.starting_stacks.shape == (128, 4)
    assert torch.all(env.starting_stacks >= 40)
    assert torch.all(env.starting_stacks <= 120)
    assert (env.starting_stacks[:, 0] != env.starting_stacks[:, 1]).any()


def test_pbs_env_matches_nl_env_call_down_to_turn():
    ref = _make_ref()
    pbs = _make_pbs()
    _assert_public_state_matches(ref, pbs)

    for action_bin in [1, 1, 1]:
        _step_and_match(ref, pbs, action_bin)
    assert ref.state is not None
    assert ref.state.street == "flop"
    torch.testing.assert_close(
        pbs.board_indices[0].cpu(),
        torch.tensor([10, 11, 12, -1, -1]),
    )

    for action_bin in [1, 1, 1]:
        _step_and_match(ref, pbs, action_bin)
    assert ref.state.street == "turn"
    torch.testing.assert_close(
        pbs.board_indices[0].cpu(),
        torch.tensor([10, 11, 12, 13, -1]),
    )


def test_pbs_env_matches_nl_env_raise_call_close_round():
    ref = _make_ref()
    pbs = _make_pbs()

    for action_bin in [4, 1, 1]:
        _step_and_match(ref, pbs, action_bin)

    assert ref.state is not None
    assert ref.state.street == "flop"
    torch.testing.assert_close(pbs.committed[0].cpu(), torch.zeros(3, dtype=torch.long))
    assert int(pbs.pot[0]) == ref.state.pot


def test_pbs_env_can_force_heads_up_preflop_flop_transition():
    pbs = _make_pbs(num_players=4)
    pbs.force_heads_up_preflop_flop = True

    for action_bin in [1, 1, 1, 1]:
        _, new_streets, _ = pbs.step_bins(torch.tensor([action_bin], dtype=torch.long))

    assert int(new_streets[0]) == 1
    assert int(pbs.street[0]) == 1
    assert int((~pbs.has_folded[0]).sum()) == 2
    torch.testing.assert_close(
        pbs.has_folded[0].cpu(),
        torch.tensor([False, True, False, True]),
    )
    torch.testing.assert_close(pbs.committed[0].cpu(), torch.zeros(4, dtype=torch.long))
    assert int(pbs.pot[0]) == 40


def test_pbs_env_matches_nl_env_multiway_fold_terminal():
    ref = _make_ref()
    pbs = _make_pbs()

    _step_and_match(ref, pbs, 0)
    assert ref.state is not None
    assert ref.state.terminal is False
    _step_and_match(ref, pbs, 0)
    assert ref.state.terminal is True
    assert ref.state.winner == 2
    assert int(pbs.winner[0]) == 2


def test_pbs_env_matches_nl_env_public_allin_split_runout():
    ref = _make_ref(starting_stack=40)
    pbs = _make_pbs(starting_stack=40)
    allin_bin = pbs.num_bet_bins - 1

    rewards = torch.zeros(3)
    for _ in range(3):
        rewards = _step_and_match(ref, pbs, allin_bin)

    assert ref.state is not None
    assert ref.state.terminal is True
    assert ref.state.street == "showdown"
    assert ref.state.winners == (0, 1, 2)
    torch.testing.assert_close(
        pbs.board_indices[0].cpu(), torch.tensor([10, 11, 12, 13, 14])
    )
    torch.testing.assert_close(rewards, torch.zeros(3))


def test_pbs_env_vectorized_mixed_rows_match_independent_references():
    refs = [_make_ref(), _make_ref()]
    pbs = _make_pbs(num_envs=2)

    scripts = ([1, 1, 1], [4, 1, 1])
    for step_idx in range(3):
        bins = torch.tensor(
            [scripts[0][step_idx], scripts[1][step_idx]], dtype=torch.long
        )
        pbs.step_bins(bins)
        for env_idx, ref in enumerate(refs):
            ref.step_bins(scripts[env_idx][step_idx])
            _assert_public_state_matches(ref, pbs, env_idx)


def test_pbs_expected_showdown_rewards_use_marginal_beliefs_and_side_pots():
    env = PBSEnv(
        num_envs=1,
        num_players=3,
        starting_stack=100,
        sb=0,
        bb=0,
        device="cpu",
    )
    env.reset(
        force_button=torch.tensor([0]),
        force_deck=torch.tensor([[0, 14, 28, 42, 7]], dtype=torch.long),
    )
    env.board_indices[0] = torch.tensor([0, 14, 28, 42, 7])
    env.starting_stacks[0] = torch.tensor([40, 100, 100])
    env.stacks[0] = 0
    env.chips_placed[0] = torch.tensor([40, 100, 100])
    env.pot[0] = 240
    env.has_folded[0] = False

    beliefs = torch.zeros(1, 3, 1326)
    beliefs[0, 0, combo_index(12, 25)] = 1.0  # AA
    beliefs[0, 1, combo_index(11, 24)] = 1.0  # KK
    beliefs[0, 2, combo_index(10, 23)] = 1.0  # QQ

    rewards = env.expected_showdown_rewards(beliefs)

    torch.testing.assert_close(
        rewards[0],
        torch.tensor([2.0, 0.2, -1.0]),
    )


def test_pbs_expected_showdown_rewards_conserve_chips_under_product_marginals():
    env = PBSEnv(
        num_envs=2,
        num_players=4,
        starting_stack=100,
        sb=0,
        bb=0,
        device="cpu",
    )
    env.reset(
        force_button=torch.zeros(2, dtype=torch.long),
        force_deck=torch.tensor(
            [[0, 14, 28, 42, 7], [1, 15, 29, 43, 8]],
            dtype=torch.long,
        ),
    )
    env.board_indices[:] = env.deck
    env.starting_stacks[:] = torch.tensor(
        [[40, 80, 120, 120], [50, 50, 100, 200]],
        dtype=torch.long,
    )
    env.chips_placed[:] = torch.tensor(
        [[40, 80, 120, 120], [50, 50, 75, 200]],
        dtype=torch.long,
    )
    env.stacks[:] = env.starting_stacks - env.chips_placed
    env.pot[:] = env.chips_placed.sum(dim=1)
    env.has_folded[:] = False

    generator = torch.Generator(device="cpu").manual_seed(11)
    beliefs = torch.rand(2, 4, 1326, generator=generator)
    beliefs = beliefs / beliefs.sum(dim=2, keepdim=True)

    rewards = env.expected_showdown_rewards(beliefs)
    net_chips = rewards * env.starting_stacks.to(torch.float32)

    torch.testing.assert_close(
        net_chips.sum(dim=1),
        torch.zeros(2),
        atol=2.0e-4,
        rtol=1.0e-5,
    )
