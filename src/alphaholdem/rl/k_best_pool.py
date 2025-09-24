from __future__ import annotations

import copy
import random
from typing import Any, List, Optional

import torch
import torch.nn as nn

from alphaholdem.rl.agent_snapshot import AgentSnapshot
from alphaholdem.rl.opponent_pool import OpponentPool
from alphaholdem.utils.profiling import profile


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
        use_mixed_precision: bool = False,
    ):
        """
        Initialize K-Best opponent pool.

        Args:
            k: Number of best opponents to maintain in pool
            min_elo_diff: Minimum ELO difference to consider for pool updates
            min_step_diff: Minimum step difference before considering for pool updates
            k_factor: ELO K-factor for rating changes
            use_mixed_precision: Whether to store models in bfloat16 for memory efficiency
        """
        # Initialize parent class with ELO calculation
        super().__init__(k_factor=k_factor)

        self.k = k
        self.min_elo_diff = min_elo_diff
        self.min_step_diff = min_step_diff
        self.use_mixed_precision = use_mixed_precision
        self.snapshots: List[AgentSnapshot] = []

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
        self,
        model: Any,
        step: int,
        rating: Optional[float] = None,
        is_exploiter: bool = False,
    ) -> None:
        """
        Add a new snapshot to the pool.

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
        Update ELO ratings after a game with K-Best specific sorting.

        Args:
            opponent: The opponent that was played against
            result: 'win', 'loss', or 'draw'
            k_factor: ELO K-factor for rating changes (uses instance default if None)
        """
        # Use parent class ELO calculation
        super().update_elo_after_game(opponent, result, k_factor)

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
        Vectorized ELO update for a single opponent over multiple games with K-Best sorting.

        Args:
            opponent: The opponent that was fought
            rewards: Tensor of rewards [num_games] where >0 = win, <0 = loss, =0 = draw
        """
        if opponent is None or rewards.numel() == 0:
            return

        # Use parent class batch update
        super().update_elo_batch_vectorized(opponent, rewards)

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
            "use_mixed_precision": self.use_mixed_precision,
            "snapshots": [],
        }

        for snapshot in self.snapshots:
            # Ensure model is in the correct dtype before saving state dict
            model_state_dict = snapshot.model.state_dict()
            # Convert all tensors to the stored dtype to ensure consistency
            if snapshot.model_dtype != torch.float32:
                model_state_dict = {
                    k: v.to(snapshot.model_dtype) if isinstance(v, torch.Tensor) else v
                    for k, v in model_state_dict.items()
                }

            snapshot_data = {
                "step": snapshot.step,
                "elo": snapshot.elo,
                "games_played": snapshot.games_played,
                "wins": snapshot.wins,
                "losses": snapshot.losses,
                "draws": snapshot.draws,
                "model_state_dict": model_state_dict,
                "model_dtype": snapshot.model_dtype,
            }
            pool_data["snapshots"].append(snapshot_data)

        torch.save(pool_data, path)

    def load_pool(self, path: str, model_class):
        """Load the opponent pool from disk."""
        pool_data = torch.load(path)

        self.current_elo = pool_data["current_elo"]
        # Handle backward compatibility for older pool files
        self.use_mixed_precision = pool_data.get("use_mixed_precision", False)
        self.snapshots = []

        for snapshot_data in pool_data["snapshots"]:
            # Handle backward compatibility for older snapshots
            model_dtype = snapshot_data.get("model_dtype", torch.float32)
            # Handle legacy use_mixed_precision field
            if (
                "use_mixed_precision" in snapshot_data
                and "model_dtype" not in snapshot_data
            ):
                model_dtype = (
                    torch.bfloat16
                    if snapshot_data["use_mixed_precision"]
                    else torch.float32
                )

            # Recreate model
            model = model_class()  # You'll need to pass the model class
            model.load_state_dict(snapshot_data["model_state_dict"])
            # Convert model to the stored dtype
            model = model.to(model_dtype)

            snapshot = AgentSnapshot(
                model=model,
                step=snapshot_data["step"],
                elo=snapshot_data["elo"],
                model_dtype=model_dtype,
            )
            snapshot.games_played = snapshot_data["games_played"]
            snapshot.wins = snapshot_data["wins"]
            snapshot.losses = snapshot_data["losses"]
            snapshot.draws = snapshot_data["draws"]

            self.snapshots.append(snapshot)
