"""Agent snapshot for storing model states and statistics."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch.nn as nn


class AgentSnapshot:
    """Represents a snapshot of an agent at a specific training step."""

    model: nn.Module
    step: int
    elo: float
    total_rewards: float
    games_played: int
    wins: int
    losses: int
    draws: int
    data: Optional[Any]

    def __init__(
        self,
        model: nn.Module,
        step: int,
        elo: float = 1200.0,
        data: Optional[Any] = None,
    ):
        self.model = copy.deepcopy(model)
        self.step = step
        self.elo = elo
        self.total_rewards = 0.0
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.data = data  # Additional data that can be filled by the pool

    def get_win_rate(self) -> float:
        """Get win rate of this snapshot."""
        total_games = self.games_played
        if total_games == 0:
            return 0.5
        return self.wins / total_games

    def get_expected_reward(self) -> float:
        """Get expected reward of this snapshot."""
        if self.games_played == 0:
            return 0.0
        return self.total_rewards / self.games_played

    def update_stats(self, result: str) -> None:
        """Update game statistics."""
        self.games_played += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:  # draw
            self.draws += 1
