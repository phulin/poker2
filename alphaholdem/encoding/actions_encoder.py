from __future__ import annotations

from typing import Any, List, Optional
import torch

from ..core.interfaces import Encoder
from ..core.registry import register_action_encoder
from ..env.types import GameState, Action
from ..encoding.action_mapping import _action_to_bin_idx


@register_action_encoder("actions_hu_v1")
class ActionsHUEncoderV1(Encoder):
    def __init__(self, history_actions_per_round: int = 6):
        self.history_actions_per_round = history_actions_per_round

    def encode_cards(
        self, game_state: Any, seat: int, device: Optional[torch.device] = None
    ) -> Any:
        raise NotImplementedError("Use card encoder for cards tensor")

    def encode_actions(
        self,
        game_state: Any,
        seat: int,
        num_bet_bins: int,
        device: Optional[torch.device] = None,
    ) -> Any:
        rounds = ["preflop", "flop", "turn", "river"]
        channels: List[torch.Tensor] = []
        for _ in rounds:
            for _ in range(self.history_actions_per_round):
                channels.append(
                    torch.zeros((4, num_bet_bins), dtype=torch.float32, device=device)
                )

        # Populate historical planes per round: player-specific and sum
        if hasattr(game_state, "action_history") and game_state.action_history:
            slots = self.history_actions_per_round
            for street_i, street in enumerate(rounds):
                events = [e for e in game_state.action_history if e[0] == street]
                events = events[-slots:]
                for idx, evt in enumerate(reversed(events)):
                    _, actor, kind, amount, _, _ = evt
                    bin_idx = _action_to_bin_idx(
                        Action(kind, amount), game_state, num_bet_bins
                    )
                    if bin_idx is None:
                        continue
                    ch = street_i * slots + (slots - 1 - idx)
                    mat = channels[ch]
                    mat[actor, bin_idx] = 1.0
                    mat[2, bin_idx] = 1.0  # sum plane
                    channels[ch] = mat

        # Fill current legal actions for the next action slot in current round
        legal = self._get_legal_mask(game_state, seat, num_bet_bins, device=device)
        round_idx = (
            max(0, rounds.index(game_state.street))
            if game_state.street in rounds
            else 0
        )
        ch_idx = round_idx * self.history_actions_per_round + (
            self.history_actions_per_round - 1
        )
        mat = channels[ch_idx]
        # Ensure legal mask lives on same device and dtype
        if legal.device != mat.device:
            legal = legal.to(mat.device)
        mat[3, :] = legal
        channels[ch_idx] = mat
        return torch.stack(channels, dim=0)

    def _get_legal_mask(
        self,
        game_state: GameState,
        seat: int,
        num_bet_bins: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate legal action mask for current state."""
        mask = torch.zeros(num_bet_bins, dtype=torch.float32, device=device)
        if hasattr(game_state, "env") and game_state.env is not None:
            legal_actions = game_state.env.legal_actions()
            for action in legal_actions:
                bin_idx = self._action_to_bin(action, game_state, num_bet_bins)
                if bin_idx is not None:
                    mask[bin_idx] = 1.0
        else:
            # Fallback when no env is attached (e.g., in unit tests): allow basic options
            # fold (0), check/call (1), pot-sized (4) if available, and all-in (last bin)
            mask[0] = 1.0
            mask[1] = 1.0
            if num_bet_bins > 4:
                mask[4] = 1.0
            mask[num_bet_bins - 1] = 1.0
        return mask

    def _action_to_bin(
        self, action: Action, game_state: GameState, num_bet_bins: int
    ) -> int | None:
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
