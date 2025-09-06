from __future__ import annotations

from typing import List

from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.hunl_vector_env import HUNLVectorEnv
from alphaholdem.encoding.action_mapping import bin_to_action
from alphaholdem.env.types import Action


def _states_equal(a, b) -> bool:
    # Compare key scalar fields and player states; ignore env references
    if a.button != b.button:
        return False
    if a.street != b.street:
        return False
    if a.pot != b.pot:
        return False
    if a.to_act != b.to_act:
        return False
    if a.small_blind != b.small_blind or a.big_blind != b.big_blind:
        return False
    if (
        a.min_raise != b.min_raise
        or a.last_aggressive_amount != b.last_aggressive_amount
    ):
        return False
    if a.terminal != b.terminal or a.winner != b.winner:
        return False
    if a.board != b.board:
        return False
    for i in (0, 1):
        pa, pb = a.players[i], b.players[i]
        if (
            pa.stack != pb.stack
            or pa.committed != pb.committed
            or pa.hole_cards != pb.hole_cards
            or pa.has_folded != pb.has_folded
            or pa.is_allin != pb.is_allin
        ):
            return False
    return True


def _drive_one_hand(env: HUNLEnv, policy_pick_first: bool = True) -> None:
    s = env.reset(seed=123)
    steps = 0
    while not s.terminal and steps < 200:
        legal_bins = env.legal_action_bins(num_bet_bins=8)
        # simple deterministic policy: pick first legal bin
        bin_idx = legal_bins[0]
        action = bin_to_action(bin_idx, s, num_bet_bins=8)
        s, _, _, _ = env.step(action)
        steps += 1


def test_vector_env_matches_single_env_basic():
    single = HUNLEnv(starting_stack=1000, sb=5, bb=10)
    vec = HUNLVectorEnv(num_envs=1, starting_stack=1000, sb=5, bb=10)

    s_single = single.reset(seed=123)
    s_vec = vec.reset(seed=123)[0]

    assert _states_equal(s_single, s_vec)

    # Roll out with a deterministic policy
    steps = 0
    while not s_single.terminal and steps < 200:
        legal_bins_single = single.legal_action_bins(num_bet_bins=8)
        legal_bins_vec = vec.legal_action_bins(num_bet_bins=8)[0]
        assert legal_bins_single == legal_bins_vec

        bin_idx = legal_bins_single[0]
        a_single = bin_to_action(bin_idx, s_single, num_bet_bins=8)
        a_vec = a_single  # same action for vector env

        s_single, r_single, d_single, _ = single.step(a_single)
        next_states, rewards, dones, _ = vec.step([a_vec])
        s_vec = next_states[0]

        assert r_single == rewards[0]
        assert d_single == dones[0]
        assert _states_equal(s_single, s_vec)

        steps += 1

    assert s_single.terminal and s_vec.terminal


def test_vector_env_two_envs_match_two_singles():
    # Seeds chosen to match vector fan-out logic (seed + i*9973)
    single0 = HUNLEnv(starting_stack=1000, sb=5, bb=10)
    single1 = HUNLEnv(starting_stack=1000, sb=5, bb=10)
    vec = HUNLVectorEnv(num_envs=2, starting_stack=1000, sb=5, bb=10)

    s0 = single0.reset(seed=42)
    s1 = single1.reset(seed=42 + 9973)
    states = vec.reset(seed=42)
    assert _states_equal(s0, states[0])
    assert _states_equal(s1, states[1])

    # Step until both done with a deterministic but slightly varied policy
    steps = 0
    while (not s0.terminal or not s1.terminal) and steps < 200:
        actions: List = []
        # Env 0
        if not s0.terminal:
            bins0 = single0.legal_action_bins(num_bet_bins=8)
            idx0 = bins0[0]
            a0 = bin_to_action(idx0, s0, num_bet_bins=8)
        else:
            a0 = Action("check")  # unused when terminal
        # Env 1
        if not s1.terminal:
            bins1 = single1.legal_action_bins(num_bet_bins=8)
            idx1 = bins1[-1]
            a1 = bin_to_action(idx1, s1, num_bet_bins=8)
        else:
            a1 = Action("check")

        # Step singles
        if not s0.terminal:
            s0, r0, d0, _ = single0.step(a0)
        if not s1.terminal:
            s1, r1, d1, _ = single1.step(a1)

        # Step vector
        ns, rs, ds, _ = vec.step([a0, a1])
        vs0, vs1 = ns[0], ns[1]

        assert _states_equal(s0, vs0)
        assert _states_equal(s1, vs1)

        steps += 1
