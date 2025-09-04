from __future__ import annotations

from typing import List
import torch

from ..env.types import GameState, Action
from ..core.config_loader import get_config
from typing import Optional


def bin_to_action(bin_idx: int, game_state: GameState, num_bet_bins: int) -> Action:
    """Convert discrete bin index to concrete Action with direct legality clamping.

    Avoids constructing full legal action lists. We compute the target action
    and clamp amounts to be executable by env.step given current to_call/stack.
    """
    me = game_state.to_act
    opp = 1 - me
    me_p = game_state.players[me]
    opp_p = game_state.players[opp]
    to_call = max(0, opp_p.committed - me_p.committed)
    stack = me_p.stack

    # Direct mapping for special actions
    if (
        bin_idx == 0
    ):  # fold (only meaningful when to_call>0; env will accept regardless)
        return Action("fold")

    # Check/Call
    if bin_idx == 1:
        if to_call > 0:
            return Action("call", amount=min(to_call, stack))
        else:
            return Action("check")

    # All-in
    if bin_idx == (num_bet_bins - 1):
        return Action("allin", amount=stack)

    # Bet/Raise bins -> compute target and clamp
    target = _bin_to_target_action(bin_idx, game_state, num_bet_bins)
    if target.kind == "bet":
        # Valid when no bet to call and stack > 0
        if stack <= 0:
            return Action("check")
        amt = max(1, min(target.amount, stack))
        return Action("bet", amount=int(amt))
    elif target.kind == "raise":
        # Need stack > to_call to raise; otherwise call or all-in handled above
        if stack <= to_call:
            # Cannot raise; fallback to call
            return Action("call", amount=min(to_call, stack))
        # Ensure strictly greater than to_call and within stack
        min_raise_total = to_call + 1
        amt = max(min_raise_total, min(target.amount, stack))
        return Action("raise", amount=int(amt))
    else:
        # Fallbacks already handled; default to check/call behavior
        if to_call > 0:
            return Action("call", amount=min(to_call, stack))
        return Action("check")


def _bin_to_target_action(
    bin_idx: int, game_state: GameState, num_bet_bins: int
) -> Action:
    """Convert bin to target Action (may not be legal) using total_committed reference.
    For raises, include the call amount; for bets, use pure fraction of total_committed.
    """
    me = game_state.to_act
    opp = 1 - me
    to_call = game_state.players[opp].committed - game_state.players[me].committed
    total_committed = (
        game_state.pot
        + game_state.players[0].committed
        + game_state.players[1].committed
    )

    if bin_idx == 1:  # check/call
        if to_call > 0:
            return Action("call", amount=to_call)
        else:
            return Action("check")
    elif bin_idx >= 2 and bin_idx < (num_bet_bins - 1):
        # Read multipliers from config; fall back to defaults if needed
        cfg = get_config()
        bins = cfg.bet_bins
        # Map indices 2..(2+len(bins)-1) to bins[0..len(bins)-1]
        idx = bin_idx - 2
        idx = max(0, min(idx, len(bins) - 1))
        mult = bins[idx]
        base = int(total_committed * mult) if total_committed > 0 else 0
        if to_call > 0:
            amount = to_call + base
            return Action("raise", amount=amount)
        else:
            amount = base
            return Action("bet", amount=amount)
    elif bin_idx == (num_bet_bins - 1):  # all-in
        stack = game_state.players[me].stack
        return Action("allin", amount=stack)
    else:
        return Action("fold")


def get_legal_mask(
    game_state: GameState, num_bet_bins: int, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Get legal action mask for current state.

    Uses env.legal_action_bins if available to avoid Action construction.
    """
    mask = torch.zeros(num_bet_bins, dtype=torch.float32, device=device)
    env = game_state.env
    bins = env.legal_action_bins(num_bet_bins)
    mask[bins] = 1.0
    return mask


def _action_to_bin_idx(
    action: Action, game_state: GameState, num_bet_bins: int
) -> int | None:
    """Map Action to discrete bin index using total_committed reference."""
    if action.kind == "fold":
        return 0
    elif action.kind in ["check", "call"]:
        return 1

    total_committed = (
        game_state.pot
        + game_state.players[0].committed
        + game_state.players[1].committed
    )
    if total_committed == 0:
        return 1

    if action.kind == "bet":
        fraction = action.amount / total_committed
    elif action.kind == "raise":
        me = game_state.to_act
        opp = 1 - me
        to_call = game_state.players[opp].committed - game_state.players[me].committed
        raise_part = max(0, action.amount - max(0, to_call))
        fraction = raise_part / total_committed if total_committed > 0 else 0.0
    elif action.kind == "allin":
        return num_bet_bins - 1
    else:
        return None

    # Determine closest configured bin
    cfg = get_config(None)
    bins = cfg.bet_bins
    # Choose nearest multiplier index
    nearest = min(range(len(bins)), key=lambda i: abs(fraction - bins[i]))
    return 2 + nearest
