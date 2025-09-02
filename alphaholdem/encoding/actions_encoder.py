from __future__ import annotations

from typing import Any, List
import torch

from ..core.interfaces import Encoder
from ..core.registry import register_action_encoder


@register_action_encoder("actions_hu_v1")
class ActionsHUEncoderV1(Encoder):
    def __init__(self, history_actions_per_round: int = 6):
        self.history_actions_per_round = history_actions_per_round

    def encode_cards(self, game_state: Any, seat: int) -> Any:
        raise NotImplementedError("Use card encoder for cards tensor")

    def encode_actions(self, game_state: Any, seat: int, num_bet_bins: int) -> Any:
        rounds = ["preflop", "flop", "turn", "river"]
        channels: List[torch.Tensor] = []
        for _ in rounds:
            for _ in range(self.history_actions_per_round):
                channels.append(torch.zeros((4, num_bet_bins), dtype=torch.float32))
        legal = torch.zeros((num_bet_bins,), dtype=torch.float32)
        round_idx = max(0, rounds.index(game_state.street)) if game_state.street in rounds else 0
        ch_idx = round_idx * self.history_actions_per_round + (self.history_actions_per_round - 1)
        mat = channels[ch_idx]
        mat[3, :] = legal
        channels[ch_idx] = mat
        return torch.stack(channels, dim=0)
