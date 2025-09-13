from __future__ import annotations

import copy
import random
from typing import Any, List, Optional

import torch
import torch.nn as nn

from ..core.interfaces import OpponentPool
from ..utils.profiling import profile
from .agent_snapshot import AgentSnapshot
from .elo_calculator import ELOCalculator


class KBestOpponentPool(OpponentPool):
    """
    K-Best opponent pool as described in AlphaHoldem paper.

    Maintains a pool of K best historical versions of the agent and samples
    opponents from this pool to ensure diverse training data and prevent
    the agent from getting trapped in local minima.
    """

    def __init__(
        self,
        k: int = 5,
        min_elo_diff: float = 50.0,
        min_step_diff: int = 300,
        k_factor: float = 32.0,
    ):
        """
        Initialize K-Best opponent pool.

        Args:
            k: Number of best opponents to maintain in pool
            min_elo_diff: Minimum ELO difference to consider for pool updates
            min_step_diff: Minimum step difference before considering for pool updates
            k_factor: ELO K-factor for rating changes
        """
        self.k = k
        self.min_elo_diff = min_elo_diff
        self.min_step_diff = min_step_diff
        self.k_factor = k_factor
        self.snapshots: List[AgentSnapshot] = []
        self.current_elo = 1200.0  # Starting ELO rating
        self.elo_calculator = ELOCalculator(k_factor)

    def sample(self, k: int = 1) -> List[AgentSnapshot]:
        """
        Sample k opponents from the pool.

        Args:
            k: Number of opponents to sample

        Returns:
            List of sampled opponent snapshots
        """
        if not self.snapshots:
            return []

        # If we have fewer snapshots than requested, return all
        if len(self.snapshots) <= k:
            return self.snapshots.copy()

        # Sample with replacement, weighted by ELO rating
        # Higher ELO opponents are more likely to be selected
        weights = [snapshot.elo for snapshot in self.snapshots]
        total_weight = sum(weights)

        if total_weight == 0:
            # If all ELOs are 0, sample uniformly
            return random.sample(self.snapshots, min(k, len(self.snapshots)))

        # Normalize weights
        normalized_weights = [w / total_weight for w in weights]

        # Sample with replacement
        sampled_indices = random.choices(
            range(len(self.snapshots)), weights=normalized_weights, k=k
        )

        return [self.snapshots[i] for i in sampled_indices]

    def add_snapshot(
        self, model: Any, step: int, rating: Optional[float] = None
    ) -> None:
        """
        Add a new snapshot to the pool.

        Args:
            agent: The agent to snapshot (should have a model attribute)
            rating: ELO rating of the agent
        """
        # Create new snapshot
        new_snapshot = AgentSnapshot(
            model=model,
            step=step,
            elo=rating if rating is not None else self.current_elo,
        )

        # Reduce memory footprint of snapshot models on accelerators
        if model is not None:
            # Disable gradients for snapshot models
            for p in new_snapshot.model.parameters():
                p.requires_grad = False

        # Add to snapshots list
        self.snapshots.append(new_snapshot)

        # Sort by ELO rating (descending)
        self.snapshots.sort(key=lambda x: x.elo, reverse=True)

        # Keep only the top K snapshots
        if len(self.snapshots) > self.k:
            self.snapshots = self.snapshots[: self.k]

    def update_elo_after_game(
        self, opponent: AgentSnapshot, result: str, k_factor: Optional[float] = None
    ):
        """
        Update ELO ratings after a game.

        Args:
            opponent: The opponent that was played against
            result: 'win', 'loss', or 'draw'
            k_factor: ELO K-factor for rating changes (uses instance default if None)
        """
        # Update current ELO
        self.current_elo = self.elo_calculator.update_elo_after_game(
            self.current_elo, opponent, result, k_factor
        )

        # Update opponent ELO (opposite change)
        opponent.elo = self.elo_calculator.update_elo_after_game(
            opponent.elo,
            AgentSnapshot(opponent.model, opponent.step, self.current_elo),
            "loss" if result == "win" else "win" if result == "loss" else "draw",
            k_factor,
        )

        # Update opponent stats
        opponent.update_stats(result)

        # Re-sort snapshots by ELO
        self.snapshots.sort(key=lambda x: x.elo, reverse=True)

    @profile
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

        # Update current ELO using the calculator
        self.current_elo = self.elo_calculator.update_elo_batch_vectorized(
            self.current_elo, opponent, rewards
        )

        # Update opponent ELO (opposite change)
        # Create a temporary snapshot with current ELO for opponent calculation
        temp_snapshot = AgentSnapshot(opponent.model, opponent.step, self.current_elo)
        opponent.elo = self.elo_calculator.update_elo_batch_vectorized(
            opponent.elo, temp_snapshot, -rewards
        )

        # Re-sort snapshots by ELO
        self.snapshots.sort(key=lambda x: x.elo, reverse=True)

    def get_best_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the snapshot with the highest ELO rating."""
        if not self.snapshots:
            return None
        return self.snapshots[0]

    def get_last_admitted_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the most recently admitted snapshot."""
        if not self.snapshots:
            return None
        return max(self.snapshots, key=lambda s: s.step)

    def get_pool_stats(self) -> dict:
        """Get statistics about the opponent pool."""
        if not self.snapshots:
            return {
                "pool_size": 0,
                "avg_elo": 0.0,
                "min_elo": 0.0,
                "max_elo": 0.0,
                "current_elo": self.current_elo,
            }

        elos = [snapshot.elo for snapshot in self.snapshots]
        return {
            "pool_size": len(self.snapshots),
            "avg_elo": sum(elos) / len(elos),
            "min_elo": min(elos),
            "max_elo": max(elos),
            "current_elo": self.current_elo,
            "best_snapshot_step": self.snapshots[0].step,
            "best_snapshot_elo": self.snapshots[0].elo,
        }

    def should_add_snapshot(
        self,
        current_step: int,
        kl_divergence: float = 0.0,
    ) -> bool:
        """
        Determine if a new snapshot should be added to the pool.

        Args:
            current_step: Current training step
            kl_divergence: KL divergence from the last training step (unused for K-Best)

        Returns:
            True if the snapshot should be added
        """
        if len(self.snapshots) < self.k:
            return True

        latest_step = max(snapshot.step for snapshot in self.snapshots)
        if abs(current_step - latest_step) < self.min_step_diff:
            return False

        # Check if new ELO is significantly different from existing snapshots and enough time has passed
        for snapshot in self.snapshots:
            if abs(self.current_elo - snapshot.elo) >= self.min_elo_diff:
                return True

        return False

    def cleanup_old_snapshots(self, max_age_steps: int = 10000):
        """
        Remove snapshots that are too old.

        Args:
            max_age_steps: Maximum age in training steps
        """
        current_step = getattr(self, "current_step", 0)
        self.snapshots = [
            snapshot
            for snapshot in self.snapshots
            if (current_step - snapshot.step) <= max_age_steps
        ]

    def save_pool(self, path: str):
        """Save the opponent pool to disk."""
        pool_data = {
            "current_elo": self.current_elo,
            "snapshots": [],
        }

        for snapshot in self.snapshots:
            snapshot_data = {
                "step": snapshot.step,
                "elo": snapshot.elo,
                "games_played": snapshot.games_played,
                "wins": snapshot.wins,
                "losses": snapshot.losses,
                "draws": snapshot.draws,
                "model_state_dict": snapshot.model.state_dict(),
            }
            pool_data["snapshots"].append(snapshot_data)

        torch.save(pool_data, path)

    def load_pool(self, path: str, model_class):
        """Load the opponent pool from disk."""
        pool_data = torch.load(path)

        self.current_elo = pool_data["current_elo"]
        self.snapshots = []

        for snapshot_data in pool_data["snapshots"]:
            # Recreate model
            model = model_class()  # You'll need to pass the model class
            model.load_state_dict(snapshot_data["model_state_dict"])

            snapshot = AgentSnapshot(
                model=model, step=snapshot_data["step"], elo=snapshot_data["elo"]
            )
            snapshot.games_played = snapshot_data["games_played"]
            snapshot.wins = snapshot_data["wins"]
            snapshot.losses = snapshot_data["losses"]
            snapshot.draws = snapshot_data["draws"]

            self.snapshots.append(snapshot)
