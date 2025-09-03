from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn

from ..core.interfaces import OpponentPool


class AgentSnapshot:
    """Represents a snapshot of an agent at a specific training step."""

    def __init__(self, model: nn.Module, step: int, elo: float = 1200.0):
        self.model = copy.deepcopy(model)
        self.step = step
        self.elo = elo
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def get_win_rate(self) -> float:
        """Get win rate of this snapshot."""
        total_games = self.games_played
        if total_games == 0:
            return 0.5
        return self.wins / total_games

    def update_stats(self, result: str):
        """Update game statistics."""
        self.games_played += 1
        if result == "win":
            self.wins += 1
        elif result == "loss":
            self.losses += 1
        else:  # draw
            self.draws += 1


class KBestOpponentPool(OpponentPool):
    """
    K-Best opponent pool as described in AlphaHoldem paper.

    Maintains a pool of K best historical versions of the agent and samples
    opponents from this pool to ensure diverse training data and prevent
    the agent from getting trapped in local minima.
    """

    def __init__(self, k: int = 5, min_elo_diff: float = 50.0):
        """
        Initialize K-Best opponent pool.

        Args:
            k: Number of best opponents to maintain in pool
            min_elo_diff: Minimum ELO difference to consider for pool updates
        """
        self.k = k
        self.min_elo_diff = min_elo_diff
        self.snapshots: List[AgentSnapshot] = []
        self.current_elo = 1200.0  # Starting ELO rating

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

    def add_snapshot(self, agent: Any, rating: float) -> None:
        """
        Add a new snapshot to the pool.

        Args:
            agent: The agent to snapshot (should have a model attribute)
            rating: ELO rating of the agent
        """
        # Create new snapshot
        model = agent.model if agent is not None else None
        new_snapshot = AgentSnapshot(
            model=model,
            step=getattr(agent, "episode_count", 0) if agent is not None else 0,
            elo=rating,
        )

        # Add to snapshots list
        self.snapshots.append(new_snapshot)

        # Sort by ELO rating (descending)
        self.snapshots.sort(key=lambda x: x.elo, reverse=True)

        # Keep only the top K snapshots
        if len(self.snapshots) > self.k:
            self.snapshots = self.snapshots[: self.k]

        # Update current ELO
        self.current_elo = rating

    def update_elo_after_game(
        self, opponent: AgentSnapshot, result: str, k_factor: float = 32.0
    ):
        """
        Update ELO ratings after a game.

        Args:
            opponent: The opponent that was played against
            result: 'win', 'loss', or 'draw'
            k_factor: ELO K-factor for rating changes
        """
        # Calculate expected score
        expected_score = 1.0 / (1.0 + 10 ** ((opponent.elo - self.current_elo) / 400.0))

        # Calculate actual score
        if result == "win":
            actual_score = 1.0
        elif result == "loss":
            actual_score = 0.0
        else:  # draw
            actual_score = 0.5

        # Calculate ELO change
        elo_change = k_factor * (actual_score - expected_score)

        # Update current ELO
        self.current_elo += elo_change

        # Update opponent ELO (opposite change)
        opponent.elo -= elo_change

        # Update opponent stats
        opponent.update_stats(result)

        # Re-sort snapshots by ELO
        self.snapshots.sort(key=lambda x: x.elo, reverse=True)

    def get_best_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the snapshot with the highest ELO rating."""
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

    def should_add_snapshot(self, new_elo: float) -> bool:
        """
        Determine if a new snapshot should be added to the pool.

        Args:
            new_elo: ELO rating of the new snapshot

        Returns:
            True if the snapshot should be added
        """
        if len(self.snapshots) < self.k:
            return True

        # Check if new ELO is significantly different from existing snapshots
        for snapshot in self.snapshots:
            if abs(new_elo - snapshot.elo) >= self.min_elo_diff:
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
            "k": self.k,
            "min_elo_diff": self.min_elo_diff,
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

        self.k = pool_data["k"]
        self.min_elo_diff = pool_data["min_elo_diff"]
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
