"""Abstract base class for opponent pools with ELO calculation functionality.

This module provides the OpponentPool abstract base class that all opponent pool
implementations inherit from. It handles common ELO calculation methods while
allowing each pool to implement its own sampling and management strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

import torch

from alphaholdem.rl.agent_snapshot import AgentSnapshot
from alphaholdem.rl.elo_calculator import ELOCalculator


class OpponentPool(ABC):
    """
    Abstract base class for opponent pools with ELO calculation functionality.

    This class provides common ELO calculation methods that all opponent pools
    can inherit and use, while allowing each pool to implement its own
    sampling and management strategies.
    """

    def __init__(self, k_factor: float = 32.0):
        """
        Initialize the opponent pool with ELO calculation.

        Args:
            k_factor: ELO K-factor for rating changes
        """
        self.k_factor = k_factor
        self.current_elo = 1200.0  # Starting ELO rating
        self.elo_calculator = ELOCalculator(k_factor)

    @abstractmethod
    def sample(self, k: int = 1) -> Iterable[Any]:
        """Sample k opponents from the pool."""
        ...

    @abstractmethod
    def add_snapshot(
        self,
        model: Any,
        step: int,
        rating: Optional[float] = None,
        is_exploiter: bool = False,
    ) -> None:
        """Add a new snapshot to the pool."""
        ...

    @abstractmethod
    def should_add_snapshot(
        self, current_step: int, kl_divergence: float = 0.0
    ) -> bool:
        """Determine if a new snapshot should be added to the pool."""
        ...

    @abstractmethod
    def get_pool_stats(self) -> dict:
        """Get statistics about the opponent pool."""
        ...

    def update_elo_after_game(
        self, opponent: Any, result: str, k_factor: Optional[float] = None
    ) -> None:
        """
        Update ELO ratings after a game.

        Args:
            opponent: The opponent that was played against
            result: 'win', 'loss', or 'draw'
            k_factor: ELO K-factor for rating changes (uses instance default if None)
        """

        original_current_elo = self.current_elo

        # Update current ELO
        self.current_elo = self.elo_calculator.update_elo_after_game(
            self.current_elo, opponent, result, k_factor
        )

        # Update opponent ELO (opposite change)
        opponent.elo = self.elo_calculator.update_elo_after_game(
            opponent.elo,
            type("Opponent", (), {"elo": original_current_elo})(),
            self._reverse_result(result),
            k_factor,
        )

    def update_elo_batch_vectorized(
        self,
        opponent: Any,
        rewards: torch.Tensor,
    ) -> None:
        """
        Vectorized ELO update for a single opponent over multiple games.

        Args:
            opponent: The opponent that was played against
            rewards: Tensor of rewards from multiple games against this opponent
        """
        # Store original current ELO for opponent calculation (needed for ELO conservation)
        original_current_elo = self.current_elo

        # Update current ELO using the calculator
        self.current_elo = self.elo_calculator.update_elo_batch_vectorized(
            self.current_elo, opponent, rewards
        )

        # Update opponent ELO (opposite change)
        # Use ORIGINAL current ELO for opponent calculation to maintain ELO conservation
        temp_snapshot = AgentSnapshot(None, -1, original_current_elo)
        opponent.elo = self.elo_calculator.update_elo_batch_vectorized(
            opponent.elo, temp_snapshot, -rewards
        )

        wins = (rewards > 0).sum().item()
        losses = (rewards < 0).sum().item()

        # rewards on the opponent are from their perspective
        opponent.total_rewards += -rewards.sum().item()
        opponent.wins += wins
        opponent.losses += losses
        opponent.draws += rewards.numel() - wins - losses
        opponent.games_played += rewards.numel()

    def _reverse_result(self, result: str) -> str:
        """Reverse the result for the opponent's perspective."""
        if result == "win":
            return "loss"
        elif result == "loss":
            return "win"
        else:  # draw
            return "draw"
