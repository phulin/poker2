"""DRED (Diverse Recent Experience Dataset) opponent pool implementation.

This implements a more sophisticated opponent pool that considers:
- ELO ratings for skill-based sampling
- Age-based decay to prefer recent opponents
- Difficulty estimation via Beta distribution
- Diversity against recent opponents
- Weak opponent floor to ensure training diversity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn as nn

from ..core.interfaces import OpponentPool
from ..utils.profiling import profile
from ..utils.kl_divergence import compute_kl_divergence
from .agent_snapshot import AgentSnapshot
from .elo_calculator import ELOCalculator
from .kmedoids import SimpleKMedoids


@dataclass
class DREDSnapshotData:
    """Data associated with a DRED snapshot."""

    age: int = 0
    alpha: float = 1.0
    beta: float = 1.0


class DREDPool(OpponentPool):
    """
    DRED opponent pool implementation.

    Maintains a diverse pool of opponents using:
    - ELO-based skill weighting
    - Age-based decay (prefer recent opponents)
    - Difficulty estimation (Beta distribution)
    - Diversity against recent opponents
    - Weak opponent floor for training diversity
    """

    def __init__(
        self,
        max_size: int = 100,
        beta: float = 0.02,
        lam: float = 0.003,
        tau: float = 1.0,
        eta: float = 40.0,
        gamma: float = 0.5,
        p_curr: float = 0.5,
        weak_floor: float = 0.1,
        k_recent: int = 20,
        k_factor: float = 32.0,
        embedding_dim: int = 128,
        use_mixed_precision: bool = False,
    ):
        """
        Initialize DRED opponent pool.

        Args:
            max_size: Maximum number of snapshots to maintain
            beta: ELO weighting factor (higher = more weight on high ELO)
            lam: Age decay factor (higher = faster decay of old opponents)
            tau: Difficulty weighting factor
            eta: Difficulty peak sharpness (higher = sharper peak at 0.5)
            gamma: Diversity weighting factor
            p_curr: Probability of sampling current agent vs pool
            weak_floor: Minimum weight for weak opponents (bottom decile)
            k_recent: Number of recent opponents to consider for diversity
            k_factor: ELO K-factor for rating changes
            embedding_dim: Dimension of opponent embeddings
            use_mixed_precision: Whether to store models in bfloat16 for memory efficiency
        """
        self.max_size = max_size
        self.beta = beta
        self.lam = lam
        self.tau = tau
        self.eta = eta
        self.gamma = gamma
        self.p_curr = p_curr
        self.weak_floor = weak_floor
        self.k_recent = k_recent
        self.k_factor = k_factor
        self.embedding_dim = embedding_dim
        self.use_mixed_precision = use_mixed_precision

        self.snapshots: List[AgentSnapshot] = []
        self.current_elo = 1200.0
        self.elo_calculator = ELOCalculator(k_factor)
        self.recent_opponents: List[torch.Tensor] = (
            []
        )  # Store embeddings of last K opponents

        # Track last admitted snapshot for step-based admission
        self.last_admitted_step: int = -1

        # Initialize embedding generator (simple MLP)
        self._init_embedding_generator()

    def _init_embedding_generator(self):
        """Initialize a simple MLP to generate opponent embeddings."""
        self.embedding_generator = nn.Sequential(
            nn.Linear(1, 64),  # Input: ELO rating
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim),
        )

        # Disable gradients for embedding generator
        for p in self.embedding_generator.parameters():
            p.requires_grad = False

    def _generate_embedding(self, snapshot: AgentSnapshot) -> torch.Tensor:
        """Generate embedding for a snapshot based on its characteristics."""
        # Simple embedding based on ELO rating
        elo_tensor = torch.tensor([[snapshot.elo]], dtype=torch.float32)
        with torch.no_grad():
            embedding = self.embedding_generator(elo_tensor).squeeze()

        # Add some noise based on step and stats for diversity
        noise = torch.randn(self.embedding_dim) * 0.1
        embedding = embedding + noise * (
            snapshot.step / 10000.0
        )  # More noise for older snapshots

        return embedding

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

        # Sample with DRED weighting
        weights = self._calculate_weights()

        # Sample with replacement using DRED weights
        sampled_indices = torch.multinomial(weights, num_samples=k, replacement=True)

        # Update recent opponents for diversity calculation
        for idx in sampled_indices:
            snapshot = self.snapshots[idx]
            embedding = self._generate_embedding(snapshot)
            self.recent_opponents.append(embedding)

            # Keep only last k_recent
            if len(self.recent_opponents) > self.k_recent:
                self.recent_opponents.pop(0)

        return [self.snapshots[i.item()] for i in sampled_indices]

    def _calculate_weights(self) -> Optional[torch.Tensor]:
        """Calculate DRED sampling weights for all snapshots."""
        if not self.snapshots:
            raise ValueError("No snapshots available")

        n = len(self.snapshots)

        # ELO-based skill weighting
        elos = torch.tensor([s.elo for s in self.snapshots], dtype=torch.float32)
        skill_weights = torch.exp(self.beta * elos)

        # Age-based decay
        ages = torch.tensor([s.data.age for s in self.snapshots], dtype=torch.float32)
        age_weights = torch.exp(-self.lam * ages)

        # Difficulty estimation via Beta distribution
        # Use win/loss ratios to estimate difficulty
        difficulties = []
        for s in self.snapshots:
            if s.games_played > 0:
                # Convert to Beta parameters (simplified)
                alpha = s.wins + 1
                beta_param = s.losses + 1
                p = alpha / (alpha + beta_param)
            else:
                p = 0.5  # Default difficulty

            # Peak difficulty near 0.5 (balanced opponents)
            diff_weight = torch.exp(torch.tensor(-self.eta * (p - 0.5) ** 2))
            difficulties.append(diff_weight)

        difficulty_weights = torch.stack(difficulties)

        # Diversity vs recent opponents
        if self.recent_opponents:
            recent_embeddings = torch.stack(self.recent_opponents[-self.k_recent :])
            snapshot_embeddings = torch.stack(
                [self._generate_embedding(s) for s in self.snapshots]
            )

            # Calculate average distance to recent opponents
            distances = torch.norm(
                snapshot_embeddings[:, None, :] - recent_embeddings[None, :, :], dim=-1
            ).mean(dim=1)

            diversity_weights = 1.0 + self.gamma * distances
        else:
            diversity_weights = torch.ones(n)

        # Combine all weights
        weights = (
            skill_weights
            * age_weights
            * (0.5 + self.tau * difficulty_weights)
            * diversity_weights
        )

        # Apply weak opponent floor
        elo_ranks = torch.argsort(elos)
        weak_idx = elo_ranks[: max(1, n // 10)]  # Bottom decile
        floor_weight = self.weak_floor * weights.sum() / len(weak_idx)
        weights[weak_idx] = torch.maximum(weights[weak_idx], floor_weight)

        # Normalize
        return weights / weights.sum()

    def add_snapshot(
        self, model: Any, step: int, rating: Optional[float] = None
    ) -> None:
        """
        Add a new snapshot to the pool.

        Args:
            model: The model to snapshot
            step: Training step
            rating: ELO rating of the agent
        """
        # Create new snapshot
        model_dtype = torch.bfloat16 if self.use_mixed_precision else torch.float32
        new_snapshot = AgentSnapshot(
            model=model,
            step=step,
            elo=rating if rating is not None else self.current_elo,
            data=DREDSnapshotData(),  # DRED-specific data
            model_dtype=model_dtype,
        )

        # Reduce memory footprint of snapshot models
        if model is not None:
            for p in new_snapshot.model.parameters():
                p.requires_grad = False

        # Add to snapshots list
        self.snapshots.append(new_snapshot)

        # Update tracking for admission criteria (only when actually adding)
        self.last_admitted_step = step

        # Age all other snapshots
        for snapshot in self.snapshots:
            if snapshot is not new_snapshot:
                snapshot.data.age += 1

        # Prune if over capacity
        if len(self.snapshots) > self.max_size:
            self._prune()

    def _prune(self):
        """Prune snapshots using DRED strategy."""
        n = len(self.snapshots)

        # Keep top ELO 10%
        top_count = max(1, n // 10)
        elos = torch.tensor([s.elo for s in self.snapshots])
        sorted_indices = torch.argsort(-elos)
        top_indices = sorted_indices[:top_count]
        remaining_indices = sorted_indices[top_count:]

        keep_indices = set(top_indices.tolist())

        if remaining_indices.numel() > 1:
            # Generate embeddings for remaining snapshots
            embeddings = torch.stack(
                [self._generate_embedding(self.snapshots[i]) for i in remaining_indices]
            )

            # Cluster count (up to 30 clusters)
            k_clusters = min(30, remaining_indices.numel())

            if k_clusters > 1:
                # Use our custom k-medoids implementation
                kmedoids = SimpleKMedoids(n_clusters=k_clusters, random_state=0).fit(
                    embeddings
                )

                # Add medoid indices to keep set
                for medoid_idx in kmedoids.medoid_indices_:
                    keep_indices.add(remaining_indices[medoid_idx])
            else:
                # If only one cluster, keep all remaining indices
                keep_indices.update(remaining_indices)

        # Final reservoir sampling if still over budget
        if len(keep_indices) > self.max_size:
            keep_indices = set(list(keep_indices)[: self.max_size])

        # Update snapshots list
        self.snapshots = [self.snapshots[i] for i in sorted(keep_indices)]

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

        # Update DRED-specific stats
        if result == "win":
            opponent.data.alpha += 1
        elif result == "loss":
            opponent.data.beta += 1

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

        # Update opponent stats with negated rewards (from opponent's perspective)
        num_wins = (-rewards > 0).sum().item()  # Opponent wins when we lose
        num_losses = (-rewards < 0).sum().item()  # Opponent loses when we win
        opponent.games_played += rewards.numel()
        opponent.wins += num_wins
        opponent.losses += num_losses
        opponent.draws += rewards.numel() - num_wins - num_losses

        # Update DRED-specific stats (from opponent's perspective)
        opponent.data.alpha += num_wins
        opponent.data.beta += num_losses

    def get_best_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the snapshot with the highest ELO rating."""
        if not self.snapshots:
            return None
        return max(self.snapshots, key=lambda s: s.elo)

    def get_last_admitted_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the most recently admitted snapshot."""
        if not self.snapshots:
            return None
        return max(self.snapshots, key=lambda s: s.step)

    def should_add_snapshot(
        self, current_step: int, kl_divergence: float = 0.0
    ) -> bool:
        """
        Determine if a new snapshot should be added to the pool.

        Args:
            current_step: Current training step
            kl_divergence: KL divergence from the last training step

        Returns:
            True if the snapshot should be added
        """

        # Always add if current ELO is significantly higher than best snapshot
        best_snapshot = self.get_best_snapshot()
        if best_snapshot is None or self.current_elo > best_snapshot.elo + 10:
            return True

        # Check if enough steps have passed since last admission
        steps_since_last = current_step - self.last_admitted_step
        if steps_since_last >= 10:
            return True

        # Check if KL divergence is significant enough
        kl_threshold = 0.1  # Threshold for significant policy change
        if kl_divergence > kl_threshold:
            return True

        return False

    def get_pool_stats(self) -> dict:
        """Get statistics about the opponent pool."""
        if not self.snapshots:
            return {
                "pool_size": 0,
                "avg_elo": 0.0,
                "min_elo": 0.0,
                "max_elo": 0.0,
                "current_elo": self.current_elo,
                "avg_age": 0.0,
                "avg_difficulty": 0.0,
            }

        elos = [snapshot.elo for snapshot in self.snapshots]
        ages = [snapshot.data.age for snapshot in self.snapshots]

        # Calculate average difficulty
        difficulties = []
        for snapshot in self.snapshots:
            if snapshot.games_played > 0:
                alpha = snapshot.data.alpha
                beta_param = snapshot.data.beta
                p = alpha / (alpha + beta_param)
                difficulties.append(p)
            else:
                difficulties.append(0.5)

        return {
            "pool_size": len(self.snapshots),
            "avg_elo": sum(elos) / len(elos),
            "min_elo": min(elos),
            "max_elo": max(elos),
            "current_elo": self.current_elo,
            "avg_age": sum(ages) / len(ages),
            "avg_difficulty": sum(difficulties) / len(difficulties),
            "recent_opponents_count": len(self.recent_opponents),
        }
