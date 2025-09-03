from __future__ import annotations

from typing import Any, List
import torch

from ..core.interfaces import Encoder
from ..core.registry import register_action_encoder
from ..env.types import GameState, Action


NUM_BET_BINS = 9


@register_action_encoder("actions_hu_v1")
class ActionsHUEncoderV1(Encoder):
    def __init__(self, history_actions_per_round: int = 6):
        self.history_actions_per_round = history_actions_per_round

    def encode_cards(self, game_state: Any, seat: int) -> Any:
        raise NotImplementedError("Use card encoder for cards tensor")

    def encode_actions(self, game_state: Any, seat: int, num_bet_bins: int = NUM_BET_BINS) -> Any:
        rounds = ["preflop", "flop", "turn", "river"]
        channels: List[torch.Tensor] = []
        for _ in rounds:
            for _ in range(self.history_actions_per_round):
                channels.append(torch.zeros((4, num_bet_bins), dtype=torch.float32))
        
        # Fill current legal actions for the next action slot in current round
        legal = self._get_legal_mask(game_state, seat, num_bet_bins)
        round_idx = max(0, rounds.index(game_state.street)) if game_state.street in rounds else 0
        ch_idx = round_idx * self.history_actions_per_round + (self.history_actions_per_round - 1)
        mat = channels[ch_idx]
        mat[3, :] = legal
        channels[ch_idx] = mat
        return torch.stack(channels, dim=0)

    def _get_legal_mask(self, game_state: GameState, seat: int, num_bet_bins: int = NUM_BET_BINS) -> torch.Tensor:
        """Generate legal action mask for current state."""
        legal_actions = game_state.env.legal_actions() if hasattr(game_state, 'env') else []
        mask = torch.zeros(num_bet_bins, dtype=torch.float32)
        
        # Map legal actions to bins
        for action in legal_actions:
            bin_idx = self._action_to_bin(action, game_state, num_bet_bins)
            if bin_idx is not None:
                mask[bin_idx] = 1.0
        return mask

    def _action_to_bin(self, action: Action, game_state: GameState, num_bet_bins: int = NUM_BET_BINS) -> int | None:
        """Map Action to discrete bin index."""
        if action.kind == "fold":
            return 0
        elif action.kind == "check":
            return 1
        elif action.kind == "call":
            return 1  # check/call share bin
        elif action.kind == "bet":
            # Map bet amount to pot fraction
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
            # Similar mapping for raises
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
