from __future__ import annotations

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv


def _make_env(
    N=1,
    starting_stack=1000,
    sb=5,
    bb=10,
    bet_bins=None,
    device=None,
    seed=123,
):
    if bet_bins is None:
        bet_bins = [0.5, 0.75, 1.0, 1.5, 2.0]

    # Create RNG
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    env = HUNLTensorEnv(
        num_envs=N,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        rng=rng,
    )
    env.reset()
    return env


def test_n1_reset_and_shapes():
    env = _make_env(N=1)
    assert env.N == 1
    assert env.deck.shape == (1, 9)
    assert env.deck_pos.shape == (1,)
    assert env.stacks.shape == (1, 2)
    assert env.hole_onehot.shape == (1, 2, 2, 4, 13)
    assert env.board_onehot.shape == (1, 5, 4, 13)
    assert env.to_act.item() in (0, 1)
    # legal mask basics
    mask = env.legal_bins_mask()
    assert mask.shape == (1, 8)
    # Check/Call must be legal at start (SB posted, to_act is SB)
    assert mask[0, 1].item() is True
    # All-in always legal if chips remain
    assert mask[0, 7].item() is True


def test_n1_round_closure_and_deal():
    env = _make_env(N=1)
    # Force two checks to close preflop and deal flop
    for _ in range(2):
        mask = env.legal_bins_mask()
        assert mask[0, 1]
        r, d, _, _ = env.step_bins(torch.tensor([1]))
        assert not d.item()
    # After two actions with equal committed, street should be flop
    assert env.street.item() in (1, 2, 3)  # progressed
    # If on flop now, there should be 3 board cards set
    if env.street.item() >= 1:
        assert int(env.board_onehot[0].sum().item()) >= 3


def test_n1_min_raise_and_amounts():
    env = _make_env(N=1)
    # Make a small opening bet from SB on preflop when check allowed
    amounts, mask = env.legal_bins_amounts_and_mask()
    for b in range(2, 7):
        if mask[0, b]:
            amt = amounts[0, b].item()
            assert amt >= env.bb
            assert amt < env.stacks[0, env.to_act.item()].item()
            opp = 1 - env.to_act.item()
            assert amt <= env.stacks[0, opp].item()


def test_n1_allin_and_auto_runout_rewards():
    env = _make_env(N=1)
    # Drive an all-in quickly: choose all-in bin
    steps = 0
    while not env.done.item() and steps < 20:
        _, mask = env.legal_bins_amounts_and_mask()
        # prefer all-in when legal
        action = 7 if mask[0, 7] else 1
        r, d, _, _ = env.step_bins(torch.tensor([action]))
        steps += 1
    assert env.done.item() is True
    # Reward must be finite and scaled
    assert torch.isfinite(r).all()


def test_n1_action_history_logging():
    env = _make_env(N=1)
    env.to_act.zero_()  # we go first.
    first_legal_mask = env.legal_bins_mask()[0].cpu()
    # quarter pot (floor(15 * 0.25) = 3) is below min raise (5).
    assert torch.equal(
        first_legal_mask, torch.tensor([1, 1, 0, 1, 1, 1, 1, 1], dtype=torch.bool)
    )
    env.step_bins(torch.tensor([1]))  # we call.
    second_legal_mask = env.legal_bins_mask()[0].cpu()
    # can't fold, not facing a bet
    assert torch.equal(
        second_legal_mask, torch.tensor([0, 1, 1, 1, 1, 1, 1, 1], dtype=torch.bool)
    )
    env.step_bins(torch.tensor([2]))  # opp bets half pot.
    assert isinstance(env.get_action_history(), torch.Tensor)
    hist = env.get_action_history()
    assert hist.shape[-1] == 8
    print(hist[0, 0, 0].cpu())
    assert torch.equal(
        hist[0, 0, 0].cpu(),
        torch.tensor(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        hist[0, 0, 1].cpu(),
        torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.all(hist[0, 0, 2:] == 0)


def test_n2_independence_and_mixed_actions():
    env = _make_env(N=2, seed=777)
    # Step env0 check/call, env1 all-in (if legal)
    mask = env.legal_bins_mask()
    a0 = 1
    a1 = 7 if mask[1, 7] else 1
    r, d, _, _ = env.step_bins(torch.tensor([a0, a1]))
    # Ensure states progressed independently
    assert env.actions_this_round[0] >= 1
    assert env.actions_this_round[1] >= 1
    # If env1 went all-in, it should either be done or auto-runout will complete later
    if a1 == 7:
        assert env.is_allin[1].any().item() is True


def test_n2_reset_done_partial():
    env = _make_env(N=2, seed=42)
    # Force env0 to all-in, env1 to check once
    mask = env.legal_bins_mask()
    r, d, _, _ = env.step_bins(torch.tensor([7 if mask[0, 7] else 1, 1]))
    # Advance until any done
    steps = 0
    while (~env.done).any() and steps < 50:
        mask = env.legal_bins_mask()
        a = torch.where(mask[:, 7], torch.tensor(7), torch.tensor(1))
        r, d, _, _ = env.step_bins(a)
        steps += 1
    # Reset only done envs
    before_done = env.done.clone()
    env.reset_done()
    after_done = env.done.clone()
    # Done envs should be reset to not done; others unchanged
    for i in range(env.N):
        if before_done[i].item():
            assert after_done[i].item() is False
        else:
            assert after_done[i].item() is False  # still running


def test_step_bins_negative_one_no_action():
    """Test that environments with -1 in step_bins are not stepped."""
    env = _make_env(N=3, seed=123)

    # Record initial state before stepping
    initial_actions_this_round = env.actions_this_round.clone()
    initial_to_act = env.to_act.clone()
    initial_acted_since_reset = env.acted_since_reset.clone()

    # Step with -1 for middle environment (index 1), valid actions for others
    mask = env.legal_bins_mask()
    actions = torch.tensor(
        [1, -1, 1], device=env.device
    )  # check/call for envs 0,2, no action for env 1
    r, d, _, _ = env.step_bins(actions)

    # Environment 1 (index 1) should not have been stepped
    assert env.actions_this_round[1] == initial_actions_this_round[1]
    assert env.to_act[1] == initial_to_act[1]
    assert env.acted_since_reset[1] == initial_acted_since_reset[1]

    # Environments 0 and 2 should have been stepped
    assert env.actions_this_round[0] > initial_actions_this_round[0]
    assert env.actions_this_round[2] > initial_actions_this_round[2]
    assert env.acted_since_reset[0].item() is True
    assert env.acted_since_reset[2].item() is True

    # Rewards should be zero for environment 1 (no action taken)
    assert r[1] == 0.0

    # Test with all -1 (no environments should be stepped)
    env2 = _make_env(N=2, seed=456)
    initial_actions_2 = env2.actions_this_round.clone()
    initial_to_act_2 = env2.to_act.clone()

    actions_all_negative = torch.tensor([-1, -1], device=env2.device)
    r2, d2, _, _ = env2.step_bins(actions_all_negative)

    # No environments should have been stepped
    assert torch.equal(env2.actions_this_round, initial_actions_2)
    assert torch.equal(env2.to_act, initial_to_act_2)
    assert torch.equal(
        r2, torch.zeros(2, device=env2.device)
    )  # All rewards should be zero


def test_blinds_correctly_entered_after_reset():
    """Test that blinds are correctly deducted from player stacks and added to pot after reset."""
    # Test with different blind sizes
    sb, bb = 25, 50
    starting_stack = 1000

    env = HUNLTensorEnv(
        num_envs=3,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=[0.5, 1.0, 1.5],
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    env.reset()

    # Check that pot contains the blinds
    expected_pot = sb + bb  # 25 + 50 = 75
    assert torch.all(
        env.pot == expected_pot
    ), f"Expected pot {expected_pot}, got {env.pot}"

    # Check that committed amounts are correct based on button position
    for i in range(env.N):
        sb_player = env.button[i].item()
        bb_player = 1 - sb_player

        assert (
            env.committed[i, sb_player] == sb
        ), f"Env {i}: Expected SB player {sb_player} committed {sb}, got {env.committed[i, sb_player]}"
        assert (
            env.committed[i, bb_player] == bb
        ), f"Env {i}: Expected BB player {bb_player} committed {bb}, got {env.committed[i, bb_player]}"

    # Check that stacks are reduced by the blinds
    expected_stack_after_blinds = starting_stack - sb  # SB player
    expected_stack_bb = starting_stack - bb  # BB player

    # The button determines who is SB/BB, so we need to check based on button position
    for i in range(env.N):
        sb_player = env.button[i].item()
        bb_player = 1 - sb_player

        assert (
            env.stacks[i, sb_player] == expected_stack_after_blinds
        ), f"Env {i}: Expected SB player stack {expected_stack_after_blinds}, got {env.stacks[i, sb_player]}"
        assert (
            env.stacks[i, bb_player] == expected_stack_bb
        ), f"Env {i}: Expected BB player stack {expected_stack_bb}, got {env.stacks[i, bb_player]}"

    # Test with different blind sizes
    sb2, bb2 = 10, 20
    env2 = HUNLTensorEnv(
        num_envs=2,
        starting_stack=2000,
        sb=sb2,
        bb=bb2,
        bet_bins=[0.5, 1.0],
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    env2.reset()

    # Check pot and committed amounts
    expected_pot2 = sb2 + bb2  # 10 + 20 = 30
    assert torch.all(
        env2.pot == expected_pot2
    ), f"Expected pot {expected_pot2}, got {env2.pot}"

    # Check committed amounts based on button position
    for i in range(env2.N):
        sb_player = env2.button[i].item()
        bb_player = 1 - sb_player

        assert (
            env2.committed[i, sb_player] == sb2
        ), f"Env {i}: Expected SB player {sb_player} committed {sb2}, got {env2.committed[i, sb_player]}"
        assert (
            env2.committed[i, bb_player] == bb2
        ), f"Env {i}: Expected BB player {bb_player} committed {bb2}, got {env2.committed[i, bb_player]}"

    # Check stacks
    expected_stack_sb2 = 2000 - sb2  # 1990
    expected_stack_bb2 = 2000 - bb2  # 1980

    for i in range(env2.N):
        sb_player = env2.button[i].item()
        bb_player = 1 - sb_player

        assert (
            env2.stacks[i, sb_player] == expected_stack_sb2
        ), f"Env {i}: Expected SB player stack {expected_stack_sb2}, got {env2.stacks[i, sb_player]}"
        assert (
            env2.stacks[i, bb_player] == expected_stack_bb2
        ), f"Env {i}: Expected BB player stack {expected_stack_bb2}, got {env2.stacks[i, bb_player]}"


def test_reward_assignment_perspective():
    """Test that rewards are assigned based on the acting player's perspective, not always player 0."""
    # Test with different scenarios where player 0 and player 1 win

    # Scenario 1: Player 0 wins at showdown
    env1 = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=25,
        bb=50,
        bet_bins=[0.5, 1.0],
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    # Set up a scenario where player 0 has a better hand
    env1.reset()
    # Force both players to check/call to showdown
    for _ in range(8):  # Enough steps to reach showdown
        mask = env1.legal_bins_mask()
        if mask[0, 1]:  # Can check/call
            r, d, _, _ = env1.step_bins(torch.tensor([1]))
            if d.item():
                break

    # Check that reward is positive if player 0 won, negative if player 1 won
    final_reward = r[0].item()
    winner = env1.winner[0].item()

    if winner == 0:  # Player 0 won
        assert final_reward > 0, f"Player 0 won but reward is negative: {final_reward}"
    elif winner == 1:  # Player 1 won
        assert final_reward < 0, f"Player 1 won but reward is positive: {final_reward}"
    elif winner == -1:  # Tie
        assert (
            abs(final_reward) < 0.01
        ), f"Tie but reward is not near zero: {final_reward}"

    # Scenario 2: Test with multiple environments to ensure consistency
    env2 = HUNLTensorEnv(
        num_envs=3,
        starting_stack=2000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0],
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    env2.reset()

    # Force all environments to showdown
    for _ in range(10):
        mask = env2.legal_bins_mask()
        actions = []
        for i in range(3):
            if mask[i, 1]:  # Can check/call
                actions.append(1)
            else:
                actions.append(-1)

        r, d, _, _ = env2.step_bins(torch.tensor(actions))

        if d.all():
            break

    # Verify reward signs match winner assignments
    for i in range(3):
        reward = r[i].item()
        winner = env2.winner[i].item()

        if winner == 0:  # Player 0 won
            assert reward > 0, f"Env {i}: Player 0 won but reward is negative: {reward}"
        elif winner == 1:  # Player 1 won
            assert reward < 0, f"Env {i}: Player 1 won but reward is positive: {reward}"
        elif winner == -1:  # Tie
            assert (
                abs(reward) < 0.01
            ), f"Env {i}: Tie but reward is not near zero: {reward}"


def test_reward_calculation_consistency():
    """Test that reward calculation is consistent regardless of who was acting last."""
    # Create multiple environments and force them to showdown
    env = HUNLTensorEnv(
        num_envs=4,
        starting_stack=1500,
        sb=30,
        bb=60,
        bet_bins=[0.5, 1.0, 1.5],
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    env.reset()

    # Track the last acting player for each environment
    last_acting_players = []

    # Force all environments to showdown
    for step in range(12):
        mask = env.legal_bins_mask()
        actions = []
        current_acting = []

        for i in range(4):
            if mask[i, 1]:  # Can check/call
                actions.append(1)
                current_acting.append(env.to_act[i].item())
            else:
                actions.append(-1)
                current_acting.append(-1)

        # Record who was acting before the step
        if step == 0:
            last_acting_players = [env.to_act[i].item() for i in range(4)]

        r, d, _, _ = env.step_bins(torch.tensor(actions))

        if d.all():
            break

    # Verify that rewards are calculated from player 0's perspective consistently
    # regardless of who was acting last
    for i in range(4):
        reward = r[i].item()
        winner = env.winner[i].item()
        last_actor = last_acting_players[i]

        # The reward should always be from player 0's perspective
        if winner == 0:  # Player 0 won
            assert (
                reward > 0
            ), f"Env {i}: Player 0 won but reward is negative: {reward} (last actor: {last_actor})"
        elif winner == 1:  # Player 1 won
            assert (
                reward < 0
            ), f"Env {i}: Player 1 won but reward is positive: {reward} (last actor: {last_actor})"
        elif winner == -1:  # Tie
            assert (
                abs(reward) < 0.01
            ), f"Env {i}: Tie but reward is not near zero: {reward} (last actor: {last_actor})"


def test_fold_reward_assignment():
    """Test that fold rewards are assigned correctly based on the folding player's perspective."""
    env = HUNLTensorEnv(
        num_envs=2,
        starting_stack=1000,
        sb=25,
        bb=50,
        bet_bins=[0.5, 1.0],
        device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
    )

    env.reset()

    # Test 1: Player 0 folds in env 0 (should get negative reward)
    mask = env.legal_bins_mask()
    if mask[0, 0]:  # Can fold
        r, d, _, _ = env.step_bins(
            torch.tensor([0, -1])
        )  # Player 0 folds in env 0, no action in env 1
        assert (
            r[0].item() < 0
        ), f"Player 0 folded but got positive reward: {r[0].item()}"
        assert d[0].item(), "Environment 0 should be done after player 0 folds"
        assert not d[1].item(), "Environment 1 should not be done"

    # Reset and test player 1 folding in env 0
    env.reset()

    # Test 2: Player 1 folds in env 0 (should get negative reward)
    # We need to wait until player 1 is to act in env 0
    for _ in range(3):  # Try a few steps to get player 1 to act
        mask = env.legal_bins_mask()
        if env.to_act[0].item() == 1 and mask[0, 0]:  # Player 1 can fold in env 0
            r, d, _, _ = env.step_bins(
                torch.tensor([0, -1])
            )  # Player 1 folds in env 0, no action in env 1
            assert (
                r[0].item() < 0
            ), f"Player 1 folded but got positive reward: {r[0].item()}"
            assert d[0].item(), "Environment 0 should be done after player 1 folds"
            break
        elif mask[0, 1]:  # Can check/call
            env.step_bins(
                torch.tensor([1, -1])
            )  # Check/call in env 0, no action in env 1
        else:
            break
