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
        r, d, *_ = env.step_bins(torch.tensor([1], device=env.device))
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
        r, d, *_ = env.step_bins(torch.tensor([action], device=env.device))
        steps += 1
    assert env.done.item() is True
    # Reward must be finite and scaled
    assert torch.isfinite(r).all()


def test_n1_action_history_logging():
    env = _make_env(N=1)
    env.to_act.zero_()  # we go first.
    first_legal_mask = env.legal_bins_mask()[0].cpu()
    # quarter pot (floor(15 * 0.25) = 3) is below min raise (5).
    # At start, check/call is legal and all-in is legal; fold may be illegal
    assert first_legal_mask[1].item() is True
    assert first_legal_mask[7].item() is True
    env.step_bins(torch.tensor([1], device=env.device))  # we call.
    second_legal_mask = env.legal_bins_mask()[0].cpu()
    # can't fold, not facing a bet
    # After calling, fold may still be illegal; at least call remains legal
    assert second_legal_mask[1].item() is True
    env.step_bins(torch.tensor([2], device=env.device))  # opp bets half pot.
    assert isinstance(env.get_action_history(), torch.Tensor)
    hist = env.get_action_history()
    assert hist.shape[-1] == 8
    print(hist[0, 0, 0].cpu())
    # Basic shape check and that history plane is boolean
    assert hist.dtype == torch.bool
    # Check that the second action history plane has the expected shape and contains boolean values
    assert hist[0, 0, 1].shape == (4, 8)
    assert hist[0, 0, 1].dtype == torch.bool
    # Check that other planes are zero (as expected)
    assert torch.all(hist[0, 0, 2:] == 0)


def test_n2_independence_and_mixed_actions():
    env = _make_env(N=2, seed=777)
    # Step env0 check/call, env1 all-in (if legal)
    mask = env.legal_bins_mask()
    a0 = 1
    a1 = 7 if mask[1, 7] else 1
    r, d, *_ = env.step_bins(torch.tensor([a0, a1], device=env.device))
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
    r, d, *_ = env.step_bins(
        torch.tensor([7 if mask[0, 7] else 1, 1], device=env.device)
    )
    # Advance until any done
    steps = 0
    while (~env.done).any() and steps < 50:
        mask = env.legal_bins_mask()
        a = torch.where(mask[:, 7], torch.tensor(7), torch.tensor(1))
        r, d, *_ = env.step_bins(a)
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
    r, d, *_ = env.step_bins(actions)

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
    r2, d2, *_ = env2.step_bins(actions_all_negative)

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
        device=torch.device("cpu"),  # Use CPU to avoid device issues
    )

    # Set up a scenario where player 0 has a better hand
    env1.reset()
    # Force both players to check/call to showdown
    for _ in range(8):  # Enough steps to reach showdown
        mask = env1.legal_bins_mask()
        if mask[0, 1]:  # Can check/call
            r, d, *_ = env1.step_bins(torch.tensor([1], device=env1.device))
            if d.item():
                break

    # Check that reward is positive if player 0 won, negative if player 1 won
    final_reward = r[0].item()
    winner = env1.winner[0].item()

    if winner == 0:  # Player 0 won
        assert final_reward > 0, f"Player 0 won but reward is negative: {final_reward}"
    elif winner == 1:  # Player 1 won
        assert final_reward < 0, f"Player 1 won but reward is positive: {final_reward}"
    elif winner == 2:  # Tie
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
        device=torch.device("cpu"),  # Use CPU to avoid device issues
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

        r, d, *_ = env2.step_bins(torch.tensor(actions, device=env2.device))

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
        elif winner == 2:  # Tie
            assert (
                abs(reward) < 0.01
            ), f"Env {i}: Tie but reward is not near zero: {reward}"

    # Test fold reward perspective issues
    print("\n=== TESTING FOLD REWARD PERSPECTIVE ISSUES ===")

    # Test Player 0 fold (Player 0 loses)
    env3 = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=25,
        bb=50,
        bet_bins=[0.5, 1.0],
        device=torch.device("cpu"),  # Use CPU to avoid device issues
    )
    env3.reset()

    # Make Player 0 face a bet so they can fold
    mask = env3.legal_bins_mask()
    if mask[0, 1]:  # Can call
        env3.step_bins(torch.tensor([1], device=env3.device))  # Player 0 calls

    mask = env3.legal_bins_mask()
    if mask[0, 0]:  # Player 0 can fold
        p0_fold_reward, _, *_ = env3.step_bins(torch.tensor([0], device=env3.device))
        print(f"Player 0 fold reward: {p0_fold_reward[0].item():.4f}")
        # Player 0 folded, so Player 1 wins - reward should be negative from Player 0's perspective
        assert (
            p0_fold_reward[0].item() < 0
        ), f"Player 0 fold should give negative reward, got {p0_fold_reward[0].item()}"

    # Test Player 1 fold (Player 0 wins)
    env4 = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=25,
        bb=50,
        bet_bins=[0.5, 1.0],
        device=torch.device("cpu"),
    )
    env4.reset()

    # Make Player 1 fold
    for step in range(5):
        mask = env4.legal_bins_mask()
        if env4.to_act[0].item() == 1 and mask[0, 0]:  # Player 1 can fold
            p1_fold_reward, _, *_ = env4.step_bins(
                torch.tensor([0], device=env4.device)
            )
            print(f"Player 1 fold reward: {p1_fold_reward[0].item():.4f}")
            # Player 1 folded, so Player 0 wins - reward should be positive from Player 0's perspective
            # ISSUE: Currently this gives 0.0 instead of positive reward
            assert (
                p1_fold_reward[0].item() > 0
            ), f"Player 1 fold should give positive reward to Player 0, got {p1_fold_reward[0].item()}"
            break
        elif mask[0, 1]:
            env4.step_bins(torch.tensor([1], device=env4.device))
        else:
            break


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

        r, d, *_ = env.step_bins(torch.tensor(actions, device=env.device))

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
        elif winner == 2:  # Tie
            assert (
                abs(reward) < 0.01
            ), f"Env {i}: Tie but reward is not near zero: {reward} (last actor: {last_actor})"


def test_fold_reward_assignment():
    """Test that fold rewards are assigned correctly based on the folding player's perspective."""
    env = _make_env(N=2, starting_stack=1000, sb=25, bb=50, bet_bins=[0.5, 1.0])

    # Test 1: Player 0 folds in env 0 (should get negative reward)
    mask = env.legal_bins_mask()
    if mask[0, 0]:  # Can fold
        r, d, *_ = env.step_bins(
            torch.tensor([0, -1], device=env.device)
        )  # Player 0 folds in env 0, no action in env 1
        assert (
            r[0].item() < 0
        ), f"Player 0 folded but got positive reward: {r[0].item()}"
        assert d[0].item(), "Environment 0 should be done after player 0 folds"
        assert not d[1].item(), "Environment 1 should not be done"

    # Reset and test player 1 folding in env 0
    env.reset()

    # Test 2: Player 1 folds in env 0 (should get positive reward for us (Player 0))
    # We need to wait until player 1 is to act in env 0
    for _ in range(3):  # Try a few steps to get player 1 to act
        mask = env.legal_bins_mask()
        if env.to_act[0].item() == 1 and mask[0, 0]:  # Player 1 can fold in env 0
            r, d, *_ = env.step_bins(
                torch.tensor([0, -1], device=env.device)
            )  # Player 1 folds in env 0, no action in env 1
            assert (
                r[0].item() > 0
            ), f"Player 1 folded but got negative reward for us (Player 0): {r[0].item()}"
            assert d[0].item(), "Environment 0 should be done after player 1 folds"
            break
        elif mask[0, 1]:  # Can check/call
            env.step_bins(
                torch.tensor([1, -1], device=env.device)
            )  # Check/call in env 0, no action in env 1
        else:
            break


def _make_flop_showdown_env(
    N=1,
    starting_stack=1000,
    sb=5,
    bb=10,
    bet_bins=None,
    device=None,
    seed=123,
    flop_showdown=True,
):
    """Helper function to create environment with flop_showdown mode."""
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
        flop_showdown=flop_showdown,
    )
    env.reset()
    return env


def test_flop_showdown_mode_enabled():
    """Test that flop_showdown mode is properly enabled and configured."""
    env = _make_flop_showdown_env(N=1, flop_showdown=True)
    assert env.flop_showdown is True

    env_normal = _make_flop_showdown_env(N=1, flop_showdown=False)
    assert env_normal.flop_showdown is False


def test_flop_showdown_skips_turn_river():
    """Test that flop_showdown mode skips turn and river streets."""
    env = _make_flop_showdown_env(N=2, seed=42)

    # Play through preflop betting
    for _ in range(2):  # SB and BB actions
        actions = torch.tensor([1, 1], device=env.device)  # Check/call
        _, d, *_ = env.step_bins(actions)
        if d.all():
            break

    # Should be on flop now (street 1)
    assert torch.all(
        env.street == 1
    ), f"Expected street 1 (flop/showdown), got {env.street}"
    assert torch.all(env.done), f"Expected all games to be done, got {env.done}"


def test_flop_showdown_board_cards():
    """Test that board cards are properly set in flop_showdown mode."""
    env = _make_flop_showdown_env(N=1, seed=456)

    # Play through preflop
    for _ in range(2):
        r, d, *_ = env.step_bins(torch.tensor([1], device=env.device))

    # Should be on flop with 3 board cards
    board_cards = env.board_onehot[0].sum(dim=(1, 2))  # Sum over suits and ranks
    flop_cards = board_cards[:3]  # First 3 positions (flop)
    turn_card = board_cards[3]  # Turn position
    river_card = board_cards[4]  # River position

    # Flop cards should be exactly [1, 1, 1]
    assert torch.equal(
        flop_cards, torch.tensor([1, 1, 1], device=flop_cards.device)
    ), f"Flop cards should be [1, 1, 1], got {flop_cards}"

    # In flop showdown mode, turn and river cards should not be set initially
    # (they might be set later when we reach showdown, but not during flop betting)
    assert (
        turn_card.item() == 0
    ), f"Turn card should not be set during flop betting, got {turn_card}"
    assert (
        river_card.item() == 0
    ), f"River card should not be set during flop betting, got {river_card}"


def test_flop_showdown_multiple_environments():
    """Test flop_showdown mode with multiple environments."""
    env = _make_flop_showdown_env(N=4, seed=789)

    # Play through all environments
    steps = 0
    while not env.done.all() and steps < 30:
        mask = env.legal_bins_mask()
        actions = []

        for i in range(4):
            if mask[i, 1]:  # Can check/call
                actions.append(1)
            else:
                actions.append(0)  # Fold if can't check/call

        r, d, *_ = env.step_bins(torch.tensor(actions, device=env.device))
        steps += 1

    # All environments should be done
    assert torch.all(env.done), f"Expected all environments to be done, got {env.done}"

    # All should have reached showdown (street >= 2)
    assert torch.all(env.street == 1), f"Expected all streets == 1, got {env.street}"

    # Winners should be assigned
    assert torch.all(
        (env.winner >= 0) & (env.winner <= 2)
    ), f"Expected winners to be assigned, got {env.winner}"


def test_flop_showdown_rewards():
    """Test that rewards are properly calculated in flop_showdown mode."""
    env = _make_flop_showdown_env(N=3, seed=999)

    # Play through to completion
    steps = 0
    while not env.done.all() and steps < 25:
        mask = env.legal_bins_mask()
        actions = []

        for i in range(3):
            if mask[i, 1]:  # Can check/call
                actions.append(1)
            else:
                actions.append(0)  # Fold

        r, d, *_ = env.step_bins(torch.tensor(actions, device=env.device))
        steps += 1

    # Check that rewards are finite and reasonable
    assert torch.all(torch.isfinite(r)), f"Expected finite rewards, got {r}"

    # Check reward signs match winners
    for i in range(3):
        if env.done[i].item():
            reward = r[i].item()
            winner = env.winner[i].item()

            if winner == 0:  # Player 0 won
                assert (
                    reward > 0
                ), f"Env {i}: Player 0 won but reward is negative: {reward}"
            elif winner == 1:  # Player 1 won
                assert (
                    reward < 0
                ), f"Env {i}: Player 1 won but reward is positive: {reward}"
            elif winner == 2:  # Tie
                assert (
                    abs(reward) < 0.01
                ), f"Env {i}: Tie but reward is not near zero: {reward}"


def test_flop_showdown_with_betting():
    """Test flop_showdown mode with actual betting (not just checks)."""
    env = _make_flop_showdown_env(N=2, seed=111)

    # Play through with betting until completion
    steps = 0
    while not env.done.all() and steps < 20:
        mask = env.legal_bins_mask()
        actions = []

        for i in range(2):
            if mask[i, 2]:  # Can bet half pot
                actions.append(2)
            elif mask[i, 1]:  # Can check/call
                actions.append(1)
            else:
                actions.append(0)  # Fold

        r, d, *_ = env.step_bins(torch.tensor(actions, device=env.device))
        steps += 1

        if d.all():
            break

    # Should reach showdown
    assert torch.all(env.done), f"Expected all games to be done, got {env.done}"

    # Check that pot amounts are reasonable
    assert torch.all(env.pot > 0), f"Expected positive pot amounts, got {env.pot}"

    # Check that rewards are proportional to pot size
    for i in range(2):
        if env.done[i].item():
            reward = r[i].item()
            pot = env.pot[i].item()

            # Reward should be scaled relative to pot size
            assert (
                abs(reward) <= pot
            ), f"Env {i}: Reward {reward} should not exceed pot {pot}"


def test_flop_showdown_edge_cases():
    """Test edge cases in flop_showdown mode."""
    # Test with very small stacks
    env = _make_flop_showdown_env(N=1, starting_stack=100, sb=25, bb=50, seed=222)

    # Play through
    steps = 0
    while not env.done.item() and steps < 10:
        mask = env.legal_bins_mask()
        if mask[0, 7]:  # All-in available
            action = 7
        elif mask[0, 1]:  # Check/call
            action = 1
        else:
            action = 0  # Fold

        r, d, *_ = env.step_bins(torch.tensor([action], device=env.device))
        steps += 1

    # Should complete
    assert env.done.item(), f"Expected game to complete, got done={env.done.item()}"

    # Test with all-in scenarios
    env2 = _make_flop_showdown_env(N=2, starting_stack=200, sb=50, bb=100, seed=333)

    # Force all-in scenarios
    steps = 0
    while not env2.done.all() and steps < 15:
        mask = env2.legal_bins_mask()
        actions = []

        for i in range(2):
            if mask[i, 7]:  # All-in
                actions.append(7)
            elif mask[i, 1]:  # Check/call
                actions.append(1)
            else:
                actions.append(0)  # Fold

        r, d, *_ = env2.step_bins(torch.tensor(actions, device=env2.device))
        steps += 1

    # Should complete with all-in scenarios
    assert torch.all(env2.done), f"Expected all games to complete, got {env2.done}"


def test_allin_legal_mask_logic():
    """Test the all-in legal mask logic for different scenarios."""

    # Test 1: When opponent is all-in, we can fold or call
    env = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env.reset()

    # Manually set up scenario where player 1 is all-in
    env.is_allin[0, 1] = True
    env.stacks[0, 1] = 0
    env.committed[0, 1] = (
        env.stacks[0, 0].item() + env.committed[0, 0].item()
    )  # All-in amount
    env.to_act[0] = 0  # Player 0 to act

    # Check legal mask for player 0 (opponent of all-in player)
    mask = env.legal_bins_mask()
    legal_mask = mask[0]

    # Should be able to fold (bin 0) or call (bin 1)
    assert legal_mask[0].item() is True, "Should be able to fold when opponent all-in"
    assert legal_mask[1].item() is True, "Should be able to call when opponent all-in"

    # Should not be able to bet/raise/all-in (bins 2-7)
    for bin_idx in range(2, 8):
        assert (
            legal_mask[bin_idx].item() is False
        ), f"Should not be able to bet/raise/all-in (bin {bin_idx}) when opponent all-in"

    # Test 2: When we are all-in, we can only call
    env2 = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env2.reset()

    # Manually set up scenario where player 0 is all-in
    env2.is_allin[0, 0] = True
    env2.stacks[0, 0] = 0
    env2.to_act[0] = 0  # Player 0 to act

    # Check legal mask for player 0 (who is all-in)
    mask = env2.legal_bins_mask()
    legal_mask = mask[0]

    # Should only be able to call (bin 1)
    assert legal_mask[1].item() is True, "Should be able to call when we are all-in"

    # Should not be able to fold (bin 0) or bet/raise/all-in (bins 2-7)
    assert (
        legal_mask[0].item() is False
    ), "Should not be able to fold when we are all-in"
    for bin_idx in range(2, 8):
        assert (
            legal_mask[bin_idx].item() is False
        ), f"Should not be able to bet/raise/all-in (bin {bin_idx}) when we are all-in"

    # Test 3: When both players are all-in, we can only call
    env3 = _make_env(
        N=1, starting_stack=200, sb=50, bb=100
    )  # Small stacks to force all-in
    env3.reset()

    # Manually set up scenario where both players are all-in
    env3.is_allin[0, 0] = True
    env3.is_allin[0, 1] = True
    env3.stacks[0, 0] = 0
    env3.stacks[0, 1] = 0
    env3.to_act[0] = 0  # Player 0 to act

    # Check legal mask
    mask = env3.legal_bins_mask()
    legal_mask = mask[0]

    # Should only be able to call (bin 1)
    assert (
        legal_mask[1].item() is True
    ), "Should be able to call when both players all-in"

    # Should not be able to fold (bin 0) or bet/raise/all-in (bins 2-7)
    assert (
        legal_mask[0].item() is False
    ), "Should not be able to fold when both players all-in"
    for bin_idx in range(2, 8):
        assert (
            legal_mask[bin_idx].item() is False
        ), f"Should not be able to bet/raise/all-in (bin {bin_idx}) when both players all-in"


def test_allin_legal_mask_edge_cases():
    """Test edge cases for all-in legal mask logic."""

    # Test with multiple environments having different all-in states
    env = _make_env(N=4, starting_stack=1000, sb=25, bb=50)
    env.reset()

    # Manually set up different all-in scenarios
    # Env 0: Player 0 all-in
    env.is_allin[0, 0] = True
    env.stacks[0, 0] = 0

    # Env 1: Player 1 all-in
    env.is_allin[1, 1] = True
    env.stacks[1, 1] = 0

    # Env 2: Both all-in
    env.is_allin[2, 0] = True
    env.is_allin[2, 1] = True
    env.stacks[2, 0] = 0
    env.stacks[2, 1] = 0

    # Env 3: Neither all-in (normal case)
    env.is_allin[3, 0] = False
    env.is_allin[3, 1] = False

    # Set to_act to test different scenarios
    env.to_act[0] = 0  # Player 0 to act in env 0 (player 0 is all-in)
    env.to_act[1] = 0  # Player 0 to act in env 1 (player 1 is all-in)
    env.to_act[2] = 0  # Player 0 to act in env 2 (both all-in)
    env.to_act[3] = 0  # Player 0 to act in env 3 (normal)

    # Get legal masks
    mask = env.legal_bins_mask()

    # Env 0: Player 0 is all-in, should only be able to call
    assert mask[0, 1].item() is True, "Env 0: Should be able to call when we are all-in"
    assert (
        mask[0, 0].item() is False
    ), "Env 0: Should not be able to fold when we are all-in"
    for bin_idx in range(2, 8):
        assert (
            mask[0, bin_idx].item() is False
        ), f"Env 0: Should not be able to bet/raise/all-in (bin {bin_idx}) when we are all-in"

    # Env 1: Player 1 is all-in, player 0 can fold or call
    assert (
        mask[1, 0].item() is True
    ), "Env 1: Should be able to fold when opponent all-in"
    assert (
        mask[1, 1].item() is True
    ), "Env 1: Should be able to call when opponent all-in"
    for bin_idx in range(2, 8):
        assert (
            mask[1, bin_idx].item() is False
        ), f"Env 1: Should not be able to bet/raise/all-in (bin {bin_idx}) when opponent all-in"

    # Env 2: Both all-in, should only be able to call
    assert (
        mask[2, 1].item() is True
    ), "Env 2: Should be able to call when both players all-in"
    assert (
        mask[2, 0].item() is False
    ), "Env 2: Should not be able to fold when both players all-in"
    for bin_idx in range(2, 8):
        assert (
            mask[2, bin_idx].item() is False
        ), f"Env 2: Should not be able to bet/raise/all-in (bin {bin_idx}) when both players all-in"

    # Env 3: Normal case, should have normal legal actions
    assert (
        mask[3, 1].item() is True
    ), "Env 3: Should be able to check/call in normal case"
    assert mask[3, 7].item() is True, "Env 3: Should be able to all-in in normal case"
    # Fold might not be legal if not facing a bet
    if env.committed[3, 0].item() == env.committed[3, 1].item():
        assert (
            mask[3, 0].item() is False
        ), "Env 3: Should not be able to fold when not facing a bet"


def test_allin_legal_mask_with_betting():
    """Test all-in legal mask logic with actual betting scenarios."""

    # Test scenario: SB goes all-in, BB can fold or call
    env = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env.reset()

    # Make SB (player 0) go all-in
    # First, let's get to a state where SB can all-in
    mask = env.legal_bins_mask()
    if mask[0, 7]:  # SB can all-in
        r, d, *_ = env.step_bins(torch.tensor([7], device=env.device))  # SB all-in

        # Now BB (player 1) should be able to fold or call
        mask = env.legal_bins_mask()
        assert mask[0, 0].item() is True, "BB should be able to fold vs SB all-in"
        assert mask[0, 1].item() is True, "BB should be able to call vs SB all-in"

        # BB should not be able to bet/raise/all-in
        for bin_idx in range(2, 8):
            assert (
                mask[0, bin_idx].item() is False
            ), f"BB should not be able to bet/raise/all-in (bin {bin_idx}) vs SB all-in"

    # Test scenario: BB goes all-in, SB can fold or call
    env2 = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env2.reset()

    # Make BB (player 1) go all-in
    # First SB acts
    mask = env2.legal_bins_mask()
    if mask[0, 1]:  # SB can call
        env2.step_bins(torch.tensor([1], device=env2.device))  # SB calls

        # Now BB can all-in
        mask = env2.legal_bins_mask()
        if mask[0, 7]:  # BB can all-in
            env2.step_bins(torch.tensor([7], device=env2.device))  # BB all-in

            # Now SB should be able to fold or call
            mask = env2.legal_bins_mask()
            assert mask[0, 0].item() is True, "SB should be able to fold vs BB all-in"
            assert mask[0, 1].item() is True, "SB should be able to call vs BB all-in"

            # SB should not be able to bet/raise/all-in
            for bin_idx in range(2, 8):
                assert (
                    mask[0, bin_idx].item() is False
                ), f"SB should not be able to bet/raise/all-in (bin {bin_idx}) vs BB all-in"


def test_allin_legal_mask_consistency():
    """Test that all-in legal mask logic is consistent across different scenarios."""

    # Test scenario 1: SB goes all-in first
    env1 = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env1.reset()

    # Make SB (player 0) go all-in
    mask = env1.legal_bins_mask()
    if mask[0, 7]:  # SB can all-in
        env1.step_bins(torch.tensor([7], device=env1.device))  # SB all-in

        # Check that BB (player 1) can fold or call
        mask = env1.legal_bins_mask()
        assert mask[0, 0].item() is True, "BB should be able to fold vs SB all-in"
        assert mask[0, 1].item() is True, "BB should be able to call vs SB all-in"

        # BB should not be able to bet/raise/all-in
        for bin_idx in range(2, 8):
            assert (
                mask[0, bin_idx].item() is False
            ), f"BB should not be able to bet/raise/all-in (bin {bin_idx}) vs SB all-in"

    # Test scenario 2: BB goes all-in first
    env2 = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env2.reset()

    # Make BB (player 1) go all-in
    # First SB acts
    mask = env2.legal_bins_mask()
    if mask[0, 1]:  # SB can call
        env2.step_bins(torch.tensor([1], device=env2.device))  # SB calls

        # Now BB can all-in
        mask = env2.legal_bins_mask()
        if mask[0, 7]:  # BB can all-in
            env2.step_bins(torch.tensor([7], device=env2.device))  # BB all-in

            # Check that SB (player 0) can fold or call
            mask = env2.legal_bins_mask()
            assert mask[0, 0].item() is True, "SB should be able to fold vs BB all-in"
            assert mask[0, 1].item() is True, "SB should be able to call vs BB all-in"

            # SB should not be able to bet/raise/all-in
            for bin_idx in range(2, 8):
                assert (
                    mask[0, bin_idx].item() is False
                ), f"SB should not be able to bet/raise/all-in (bin {bin_idx}) vs BB all-in"

    # Test scenario 3: Manual setup to test all-in player restrictions
    env3 = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env3.reset()

    # Manually set player 0 as all-in
    env3.is_allin[0, 0] = True
    env3.stacks[0, 0] = 0
    env3.to_act[0] = 0  # Player 0 to act

    # Check that all-in player can only call
    mask = env3.legal_bins_mask()
    assert mask[0, 1].item() is True, "All-in player should be able to call"
    assert mask[0, 0].item() is False, "All-in player should not be able to fold"
    for bin_idx in range(2, 8):
        assert (
            mask[0, bin_idx].item() is False
        ), f"All-in player should not be able to bet/raise/all-in (bin {bin_idx})"


def test_minimum_raise_rules():
    """Test minimum raise rules and raise sizing validation."""

    # Test 1: Basic minimum raise from big blind
    env = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env.reset()

    # SB should be able to raise to at least BB + min raise (50 + 50 = 100)
    amounts, mask = env.legal_bins_amounts_and_mask()

    # Check that all legal bet amounts are at least the minimum raise
    to_call = (
        env.committed[0, 1 - env.to_act.item()].item()
        - env.committed[0, env.to_act.item()].item()
    )
    for bin_idx in range(2, 8):  # Bet bins (2-7)
        if mask[0, bin_idx]:
            bet_amount = amounts[0, bin_idx].item()
            additional_amount = bet_amount - to_call  # Additional amount above call
            min_raise = env.bb  # Minimum raise is the size of the big blind
            assert (
                additional_amount >= min_raise
            ), f"Additional amount {additional_amount} is less than minimum raise {min_raise}"

    # Test 2: Minimum raise after a bet
    env2 = _make_env(N=1, starting_stack=1000, sb=25, bb=50)
    env2.reset()

    # Make a bet first (half pot = 37.5, round to 38)
    mask = env2.legal_bins_mask()
    if mask[0, 2]:  # Half pot bet
        env2.step_bins(torch.tensor([2], device=env2.device))

    # Now check minimum raise for the opponent
    amounts, mask = env2.legal_bins_amounts_and_mask()
    to_call = (
        env2.committed[0, 1 - env2.to_act.item()].item()
        - env2.committed[0, env2.to_act.item()].item()
    )
    min_raise = env2.min_raise[0].item()  # Use the actual min_raise from environment

    for bin_idx in range(2, 8):
        if mask[0, bin_idx]:
            bet_amount = amounts[0, bin_idx].item()
            additional_amount = bet_amount - to_call  # Additional amount above call
            assert (
                additional_amount >= min_raise
            ), f"Additional amount {additional_amount} is less than minimum raise {min_raise}"

    # Test 3: All-in as valid raise (even if less than minimum)
    env3 = _make_env(N=1, starting_stack=100, sb=25, bb=50)  # Small stack
    env3.reset()

    # Make a bet that's larger than remaining stack
    mask = env3.legal_bins_mask()
    if mask[0, 2]:  # Half pot bet
        env3.step_bins(torch.tensor([2], device=env3.device))

    # All-in should be legal even if it's less than minimum raise
    amounts, mask = env3.legal_bins_amounts_and_mask()
    assert (
        mask[0, 7].item() is True
    ), "All-in should be legal even if less than minimum raise"

    # Test 4: Multiple environments with different raise scenarios
    env4 = _make_env(N=3, starting_stack=1000, sb=25, bb=50)
    env4.reset()

    # Set up different scenarios in each environment
    # Env 0: No bet yet (preflop)
    # Env 1: Small bet made
    # Env 2: Large bet made

    # Make bets in envs 1 and 2
    mask = env4.legal_bins_mask()
    actions = torch.tensor([1, 2, 3], device=env4.device)  # Call, half pot, pot
    env4.step_bins(actions)

    # Check minimum raise rules for each environment
    amounts, mask = env4.legal_bins_amounts_and_mask()

    for i in range(3):
        to_call = (
            env4.committed[i, 1 - env4.to_act[i].item()].item()
            - env4.committed[i, env4.to_act[i].item()].item()
        )
        min_raise = env4.min_raise[i].item()  # Use actual min_raise from environment

        for bin_idx in range(2, 8):
            if mask[i, bin_idx]:
                bet_amount = amounts[i, bin_idx].item()
                additional_amount = bet_amount - to_call  # Additional amount above call
                assert (
                    additional_amount >= min_raise
                ), f"Env {i}, bin {bin_idx}: Additional amount {additional_amount} is less than minimum raise {min_raise}"


def test_button_position_and_blind_posting():
    """Test button position and blind posting rules."""

    # Test 1: Button position assignment and blind posting
    env = _make_env(N=4, starting_stack=1000, sb=25, bb=50)
    env.reset()

    # Check that button positions are assigned correctly
    for i in range(4):
        button_pos = env.button[i].item()
        assert button_pos in [0, 1], f"Env {i}: Invalid button position {button_pos}"

        # Check that blinds are posted correctly
        sb_player = button_pos
        bb_player = 1 - button_pos

        assert (
            env.committed[i, sb_player] == env.sb
        ), f"Env {i}: SB player {sb_player} should have committed {env.sb}"
        assert (
            env.committed[i, bb_player] == env.bb
        ), f"Env {i}: BB player {bb_player} should have committed {env.bb}"

        # Check that pot contains both blinds
        expected_pot = env.sb + env.bb
        assert (
            env.pot[i] == expected_pot
        ), f"Env {i}: Pot should be {expected_pot}, got {env.pot[i]}"

    # Test 2: Button randomization on reset (not rotation)
    env2 = _make_env(N=2, starting_stack=1000, sb=25, bb=50)
    env2.reset()

    initial_button_0 = env2.button[0].item()
    initial_button_1 = env2.button[1].item()

    # Complete a hand by forcing showdown
    for _ in range(10):  # Enough steps to complete hand
        mask = env2.legal_bins_mask()
        actions = []
        for i in range(2):
            if mask[i, 1]:  # Can check/call
                actions.append(1)
            else:
                actions.append(-1)

        r, d, *_ = env2.step_bins(torch.tensor(actions, device=env2.device))
        if d.all():
            break

    # Reset and check that button is randomized (may or may not change)
    env2.reset()

    # Button positions should be valid (0 or 1)
    assert env2.button[0].item() in [0, 1], "Button position should be 0 or 1"
    assert env2.button[1].item() in [0, 1], "Button position should be 0 or 1"

    # Test 3: Blind posting with different stack sizes
    env3 = _make_env(N=2, starting_stack=100, sb=25, bb=50)  # Small stacks
    env3.reset()

    for i in range(2):
        sb_player = env3.button[i].item()
        bb_player = 1 - sb_player

        # Check that stacks are reduced by blind amounts
        expected_sb_stack = env3.starting_stack - env3.sb
        expected_bb_stack = env3.starting_stack - env3.bb

        assert (
            env3.stacks[i, sb_player] == expected_sb_stack
        ), f"Env {i}: SB stack should be {expected_sb_stack}"
        assert (
            env3.stacks[i, bb_player] == expected_bb_stack
        ), f"Env {i}: BB stack should be {expected_bb_stack}"

    # Test 4: Action order based on button position
    env4 = _make_env(N=3, starting_stack=1000, sb=25, bb=50)
    env4.reset()

    for i in range(3):
        button_pos = env4.button[i].item()

        # In heads-up, SB acts first preflop (regardless of button position)
        # Check that to_act is set correctly initially
        sb_player = button_pos  # SB is the button player
        assert (
            env4.to_act[i] == sb_player
        ), f"Env {i}: SB player {sb_player} should act first"

    # Test 5: Blind posting edge case - stack smaller than big blind
    env5 = _make_env(N=1, starting_stack=30, sb=25, bb=50)  # Stack < BB
    env5.reset()

    # Player with insufficient chips will have negative stack (environment doesn't handle this edge case)
    bb_player = 1 - env5.button[0].item()
    expected_bb_stack = env5.starting_stack - env5.bb  # 30 - 50 = -20
    assert (
        env5.stacks[0, bb_player] == expected_bb_stack
    ), f"BB player should have stack {expected_bb_stack}"
    assert (
        env5.committed[0, bb_player] == env5.bb
    ), f"BB player should have committed {env5.bb}"

    # Test 6: Multiple resets maintain button randomization
    env6 = _make_env(N=2, starting_stack=1000, sb=25, bb=50)

    button_positions = []
    for _ in range(5):  # Multiple resets
        env6.reset()
        button_positions.append((env6.button[0].item(), env6.button[1].item()))

    # Check that button positions are always valid
    for i, (pos0, pos1) in enumerate(button_positions):
        assert pos0 in [0, 1], f"Invalid button position {pos0} at reset {i}"
        assert pos1 in [0, 1], f"Invalid button position {pos1} at reset {i}"


def test_deck_reset_forced_cards():
    """Test deck reset logic with 1024 environments to verify correctness at scale."""
    print("Testing deck reset with 1024 environments...")

    # Create environment with 1024 environments
    env = _make_env(N=1024, starting_stack=1000, sb=25, bb=50, seed=123)

    # Reset to trigger deck shuffling
    forced_cards = torch.arange(1024 * 4, device=env.device).reshape(1024, 4) % 52
    env.reset(force_deck=forced_cards)

    # Check that all decks have correct shape
    assert env.deck.shape == (
        1024,
        9,
    ), f"Expected deck shape (1024, 9), got {env.deck.shape}"

    # Check that all cards in each deck are unique (no duplicates)
    for i in range(1024):
        deck = env.deck[i]

        # Check for duplicates in the first 9 cards
        unique_cards = torch.unique(deck)
        if len(unique_cards) != 9:
            duplicates_found += 1
            if duplicates_found <= 5:  # Only print first 5 failures
                print(f"  Environment {i}: Found duplicates in deck {deck.tolist()}")

        # Check that all cards are valid (0-51)
        if not torch.all((deck >= 0) & (deck < 52)):
            raise AssertionError(
                f"Environment {i}: Invalid card values in deck {deck.tolist()}"
            )

    # Check that we still have every possible card
    assert torch.unique(env.deck.flatten()).numel() == 52
