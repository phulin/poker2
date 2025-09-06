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
    env = HUNLTensorEnv(
        num_envs=N,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        bet_bins=bet_bins,
        device=device,
        seed=seed,
    )
    env.reset(seed=seed)
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
    mask = env.legal_action_bins_mask()
    assert mask.shape == (1, 8)
    # Check/Call must be legal at start (SB posted, to_act is SB)
    assert mask[0, 1].item() is True
    # All-in always legal if chips remain
    assert mask[0, 7].item() is True


def test_n1_round_closure_and_deal():
    env = _make_env(N=1)
    # Force two checks to close preflop and deal flop
    for _ in range(2):
        mask = env.legal_action_bins_mask()
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
    amounts = env.bin_amounts()
    # bins 2..6 are preset; ensure any legal bet is >= bb and < stack and <= opp stack
    mask = env.legal_action_bins_mask()
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
        mask = env.legal_action_bins_mask()
        # prefer all-in when legal
        action = 7 if mask[0, 7] else 1
        r, d, _, _ = env.step_bins(torch.tensor([action]))
        steps += 1
    assert env.done.item() is True
    # Reward must be finite and scaled
    assert torch.isfinite(r).all()


def test_n1_action_history_logging():
    env = _make_env(N=1)
    # Ensure history tensor allocated after first step
    mask = env.legal_action_bins_mask()
    r, d, _, _ = env.step_bins(torch.tensor([1]))
    assert isinstance(env.get_action_history(), torch.Tensor)
    hist = env.get_action_history()
    assert hist.shape[-1] == 8
    # Next-legal mask should be written at row 3 for next slot
    next_mask = env.legal_action_bins_mask().to(torch.float32)
    round_idx = env.street.clamp(min=0, max=3).item()
    slot_idx = min(env.actions_this_round.item(), env.history_slots - 1)
    assert torch.equal(hist[0, round_idx, slot_idx, 3], next_mask[0])


def test_n1_bin_deduplication_unique_amounts():
    env = _make_env(N=1)
    amounts = env.bin_amounts()[0]
    mask = env.legal_action_bins_mask()[0]
    seen = set()
    for b in range(2, 7):
        if mask[b]:
            amt = int(amounts[b].item())
            assert amt not in seen
            seen.add(amt)


def test_n2_independence_and_mixed_actions():
    env = _make_env(N=2, seed=777)
    # Step env0 check/call, env1 all-in (if legal)
    mask = env.legal_action_bins_mask()
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
    mask = env.legal_action_bins_mask()
    r, d, _, _ = env.step_bins(torch.tensor([7 if mask[0, 7] else 1, 1]))
    # Advance until any done
    steps = 0
    while (~env.done).any() and steps < 50:
        mask = env.legal_action_bins_mask()
        a = torch.where(mask[:, 7], torch.tensor(7), torch.tensor(1))
        r, d, _, _ = env.step_bins(a)
        steps += 1
    # Reset only done envs
    before_done = env.done.clone()
    env.reset_done(seed=99)
    after_done = env.done.clone()
    # Done envs should be reset to not done; others unchanged
    for i in range(env.N):
        if before_done[i].item():
            assert after_done[i].item() is False
        else:
            assert after_done[i].item() is False  # still running
