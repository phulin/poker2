from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from .opponent_pool import OpponentPool
from ..utils.profiling import profile
from .agent_snapshot import AgentSnapshot


class FixedOpponentPool(OpponentPool):
    """
    Fixed opponent pool that maintains a single opponent snapshot.

    This is useful for exploiter training where we want to train against
    a specific fixed opponent (like the current model) rather than a
    diverse pool of opponents.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize Fixed opponent pool.

        Args:
            k_factor: ELO K-factor for rating changes (unused but kept for interface compatibility)
            use_mixed_precision: Whether to store models in bfloat16 for memory efficiency
        """
        # Initialize parent class with ELO calculation
        super().__init__(k_factor=k_factor)

        self.use_mixed_precision = use_mixed_precision
        self.snapshots: List[AgentSnapshot] = []

    def sample(self, k: int = 1) -> List[AgentSnapshot]:
        """
        Sample k opponents from the pool.

        Since this is a fixed pool with at most 1 opponent, we return
        the same opponent k times if it exists, or empty list if no opponent.

        Args:
            k: Number of opponents to sample

        Returns:
            List of sampled opponent snapshots (same opponent repeated k times)
        """
        if not self.snapshots:
            return []

        # Return the single opponent k times
        return [self.snapshots[0] for _ in range(k)]

    def add_snapshot(
        self,
        model: Any,
        step: int,
        rating: Optional[float] = None,
        is_exploiter: bool = False,
    ) -> None:
        """
        Add a new snapshot to the pool, replacing any existing snapshot.

        Args:
            model: The model to snapshot
            step: Training step
            rating: ELO rating of the agent
            is_exploiter: Whether this snapshot is an exploiter
        """
        # Create new snapshot
        model_dtype = torch.bfloat16 if self.use_mixed_precision else torch.float32
        new_snapshot = AgentSnapshot(
            model=model,
            step=step,
            elo=rating if rating is not None else self.current_elo,
            model_dtype=model_dtype,
            is_exploiter=is_exploiter,
        )

        # Reduce memory footprint of snapshot models
        if model is not None:
            for p in new_snapshot.model.parameters():
                p.requires_grad = False

        # Replace existing snapshot (fixed pool only holds one)
        self.snapshots = [new_snapshot]

    def should_add_snapshot(
        self,
        current_step: int,
        kl_divergence: float = 0.0,
    ) -> bool:
        """
        Determine if a new snapshot should be added to the pool.

        For a fixed pool, we always add new snapshots since we want to
        keep the opponent up-to-date.

        Args:
            current_step: Current training step
            kl_divergence: KL divergence from the last training step (unused)

        Returns:
            True (always add new snapshots to keep opponent current)
        """
        return True

    def get_best_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the snapshot with the highest ELO rating (the only snapshot)."""
        if not self.snapshots:
            return None
        return self.snapshots[0]

    def get_last_admitted_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the most recently admitted snapshot (the only snapshot)."""
        if not self.snapshots:
            return None
        return self.snapshots[0]

    def get_pool_stats(self) -> dict:
        """Get statistics about the opponent pool."""
        if not self.snapshots:
            return {
                "pool_size": 0,
                "avg_elo": 0.0,
                "min_elo": 0.0,
                "max_elo": 0.0,
                "current_elo": self.current_elo,
                "best_snapshot_step": 0,
                "best_snapshot_elo": 0.0,
            }

        snapshot = self.snapshots[0]
        return {
            "pool_size": 1,
            "avg_elo": snapshot.elo,
            "min_elo": snapshot.elo,
            "max_elo": snapshot.elo,
            "current_elo": self.current_elo,
            "best_snapshot_step": snapshot.step,
            "best_snapshot_elo": snapshot.elo,
        }

    def set_fixed_opponent(
        self, model: Any, step: int, rating: Optional[float] = None
    ) -> None:
        """
        Set the fixed opponent for this pool.

        This is a convenience method that clears any existing opponent
        and sets a new one.

        Args:
            model: The model to use as the fixed opponent
            step: Training step
            rating: ELO rating of the agent
        """
        self.add_snapshot(model, step, rating)

    def update_elo_batch_vectorized(
        self,
        opponent: AgentSnapshot,
        rewards: torch.Tensor,
    ) -> None:
        """
        Vectorized ELO update for a single opponent over multiple games.

        Args:
            opponent: The opponent that was fought
            rewards: Tensor of rewards [num_games] where >0 = win, <0 = loss, =0 = draw
        """
        if opponent is None or rewards.numel() == 0:
            return

        # Use parent class batch update (no additional logic needed for fixed pool)
        super().update_elo_batch_vectorized(opponent, rewards)

    def clear(self) -> None:
        """Clear all snapshots from the pool."""
        self.snapshots.clear()

    def __len__(self) -> int:
        """Return the number of snapshots in the pool (0 or 1)."""
        return len(self.snapshots)

    def __bool__(self) -> bool:
        """Return True if the pool has an opponent."""
        return len(self.snapshots) > 0
