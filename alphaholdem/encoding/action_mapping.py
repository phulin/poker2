from __future__ import annotations

from typing import List
import torch

from ..env.types import GameState, Action
from ..core.config_loader import get_config
from typing import Optional


def bin_to_action(bin_idx: int, game_state: GameState, num_bet_bins: int) -> Action:
    """Convert discrete bin index to concrete Action with snapping."""
    legal_actions = game_state.env.legal_actions() if hasattr(game_state, "env") else []

    # Direct mapping for special actions
    if bin_idx == 0:  # fold
        return Action("fold")

    # For betting actions, find closest legal match
    target_action = _bin_to_target_action(bin_idx, game_state, num_bet_bins)

    # Find closest legal action
    best_action = None
    best_distance = float("inf")

    for legal in legal_actions:
        if legal.kind == target_action.kind:
            if legal.kind in ["check", "call"]:
                return legal  # exact match
            else:
                # For bet/raise/allin, find closest amount
                dist = abs(legal.amount - target_action.amount)
                if dist < best_distance:
                    best_distance = dist
                    best_action = legal

    # Fallback: return first legal action that's not fold
    for legal in legal_actions:
        if legal.kind != "fold":
            return legal

    # Last resort: fold
    return Action("fold")


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
        return 7
    else:
        return None

    # Determine closest configured bin
    cfg = get_config(None)
    bins = cfg.bet_bins
    # Choose nearest multiplier index
    nearest = min(range(len(bins)), key=lambda i: abs(fraction - bins[i]))
    return 2 + nearest
