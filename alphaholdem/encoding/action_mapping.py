from __future__ import annotations

from typing import List
import torch

from ..env.types import GameState, Action


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
    """Convert bin to target Action (may not be legal)."""
    pot = game_state.pot
    to_call = 0
    if len(game_state.players) >= 2:
        me = game_state.to_act
        opp = 1 - me
        to_call = game_state.players[opp].committed - game_state.players[me].committed

    if bin_idx == 1:  # check/call
        if to_call > 0:
            return Action("call", amount=to_call)
        else:
            return Action("check")
    elif bin_idx == 2:  # 1/2 pot
        amount = max(to_call, int(pot * 0.5))
        return Action("bet", amount=amount)
    elif bin_idx == 3:  # 3/4 pot
        amount = max(to_call, int(pot * 0.75))
        return Action("bet", amount=amount)
    elif bin_idx == 4:  # pot
        amount = max(to_call, pot)
        return Action("bet", amount=amount)
    elif bin_idx == 5:  # 1.5x pot
        amount = max(to_call, int(pot * 1.5))
        return Action("bet", amount=amount)
    elif bin_idx == 6:  # 2x pot
        amount = max(to_call, pot * 2)
        return Action("bet", amount=amount)
    elif bin_idx == 7:  # all-in
        me = game_state.to_act
        stack = game_state.players[me].stack
        return Action("allin", amount=stack)
    else:
        return Action("fold")


def get_legal_mask(game_state: GameState, num_bet_bins: int) -> torch.Tensor:
    """Get legal action mask for current state."""
    legal_actions = game_state.env.legal_actions() if hasattr(game_state, "env") else []
    mask = torch.zeros(num_bet_bins, dtype=torch.float32)

    for action in legal_actions:
        bin_idx = _action_to_bin_idx(action, game_state, num_bet_bins)
        if bin_idx is not None:
            mask[bin_idx] = 1.0
    return mask


def _action_to_bin_idx(
    action: Action, game_state: GameState, num_bet_bins: int
) -> int | None:
    """Map Action to discrete bin index."""
    if action.kind == "fold":
        return 0
    elif action.kind in ["check", "call"]:
        return 1
    elif action.kind == "bet":
        pot = game_state.pot
        if pot == 0:
            return 1  # check
        fraction = action.amount / pot
        if fraction <= 0.6:
            return 2  # 1/2 pot
        elif fraction <= 0.8:
            return 3  # 3/4 pot
        elif fraction <= 1.2:
            return 4  # pot
        elif fraction <= 1.8:
            return 5  # 1.5x pot
        else:
            return 6  # 2x pot
    elif action.kind == "raise":
        pot = game_state.pot
        if pot == 0:
            return 4  # pot-sized
        fraction = action.amount / pot
        if fraction <= 1.2:
            return 4  # pot
        elif fraction <= 1.8:
            return 5  # 1.5x pot
        else:
            return 6  # 2x pot
    elif action.kind == "allin":
        return 7  # all-in
    return None
