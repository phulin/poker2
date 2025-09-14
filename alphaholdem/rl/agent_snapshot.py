"""Agent snapshot for storing model states and statistics."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
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
    model_dtype: torch.dtype

    def __init__(
        self,
        model: nn.Module,
        step: int,
        elo: float = 1200.0,
        data: Optional[Any] = None,
        model_dtype: torch.dtype = torch.float32,
    ):
        self.model_dtype = model_dtype
        if model is not None:
            self.model = copy.deepcopy(model).to(model_dtype)
        else:
            self.model = None

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

    def get_model_for_inference(self, device: torch.device) -> nn.Module:
        """Get the model for inference. Convert bfloat16 models back to float32 for inference."""
        if self.model is None:
            raise ValueError("Cannot get model for inference: model is None")

        if (
            self.model_dtype == torch.bfloat16
            and next(self.model.parameters()).dtype == torch.bfloat16
        ):
            # Convert bfloat16 model back to float32 for inference to maintain precision
            # This is the standard approach for mixed precision inference
            model = self.model.to(torch.float32)
        else:
            model = self.model

        return model.to(device)

    def convert_input_to_model_dtype(self, structured_data):
        """Convert input data to match the model's dtype for inference."""
        # No conversion needed since we convert the model to float32 for inference
        return structured_data
