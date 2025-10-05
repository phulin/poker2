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
from typing import Any, List, Optional, Union

import torch

from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.rl.agent_snapshot import AgentSnapshot
from alphaholdem.rl.kmedoids import SimpleKMedoids
from alphaholdem.rl.opponent_pool import OpponentPool


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
            use_mixed_precision: Whether to store models in bfloat16 for memory efficiency
        """
        # Initialize parent class with ELO calculation
        super().__init__(k_factor=k_factor)

        self.max_size = max_size
        self.beta = beta
        self.lam = lam
        self.tau = tau
        self.eta = eta
        self.gamma = gamma
        self.p_curr = p_curr
        self.weak_floor = weak_floor
        self.k_recent = k_recent
        self.use_mixed_precision = use_mixed_precision

        self.snapshots: List[AgentSnapshot] = []

        # Track last admitted snapshot for step-based admission
        self.last_admitted_step: int = -1

        # Last batch of data for getting embeddings when pruning
        self.last_batch_data: Union[CNNEmbeddingData, StructuredEmbeddingData] = None

    def set_last_batch_data(
        self, last_batch_data: Union[CNNEmbeddingData, StructuredEmbeddingData]
    ) -> None:
        """Set the last batch of data for getting KL divergence when pruning."""
        self.last_batch_data = last_batch_data

    def _generate_embedding(
        self,
        snapshot: AgentSnapshot,
        sample_batch: Union[CNNEmbeddingData, StructuredEmbeddingData],
    ) -> torch.Tensor:
        """Generate embedding for a snapshot based on its characteristics."""
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=sample_batch.device.type,
                dtype=snapshot.model_dtype,
                enabled=self.use_mixed_precision,
            ),
        ):
            snapshot.model.to(sample_batch.device)
            model_outputs = snapshot.model(sample_batch)
            snapshot.model.to("cpu")
            probs = torch.softmax(model_outputs.policy_logits, dim=-1)
            all_info = torch.cat([probs, model_outputs.value.unsqueeze(-1)], dim=-1)
            return all_info.flatten().detach()

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

        # Defensive: ensure weights form a valid multinomial distribution
        if (
            not torch.isfinite(weights).all()
            or (weights < 0).any()
            or weights.sum() <= 0
        ):
            print("Warning: Invalid weights in DRED pool")
            print(weights)
            # Fallback to uniform sampling
            weights = torch.ones(len(self.snapshots)) / len(self.snapshots)

        # Sample with replacement using DRED weights
        sampled_indices = torch.multinomial(weights, num_samples=k, replacement=True)

        return [self.snapshots[i.item()] for i in sampled_indices]

    def _calculate_weights(self) -> torch.Tensor:
        """Calculate DRED sampling weights for all snapshots."""
        if not self.snapshots:
            raise ValueError("No snapshots available")

        n = len(self.snapshots)

        # ELO-based skill weighting
        elos = torch.tensor([s.elo for s in self.snapshots], dtype=torch.float32)
        # Clamp ELO values to prevent extreme exponentials
        elos_clamped = torch.clamp(elos, min=-1000, max=1000)
        skill_weights = torch.exp(self.beta * elos_clamped)

        # Age-based decay
        ages = torch.tensor([s.data.age for s in self.snapshots], dtype=torch.float32)
        # Clamp ages to prevent extreme exponentials
        ages_clamped = torch.clamp(ages, min=0, max=1000)
        age_weights = torch.exp(-self.lam * ages_clamped)

        # Difficulty estimation via Beta distribution
        # Use win/loss ratios to estimate difficulty
        difficulties = []
        for s in self.snapshots:
            if s.games_played > 0:
                # Convert to Beta parameters (simplified)
                alpha = s.wins + 1
                beta_param = s.losses + 1
                # Ensure denominator is never zero
                total = alpha + beta_param
                if total > 0:
                    p = alpha / total
                else:
                    p = 0.5  # Fallback
            else:
                p = 0.5  # Default difficulty

            # Clamp p to valid range and prevent extreme exponentials
            p_tensor = torch.tensor(p, dtype=torch.float32)
            p_clamped = torch.clamp(p_tensor, min=0.001, max=0.999)
            diff_exp = torch.clamp(
                -self.eta * (p_clamped - 0.5) ** 2, min=-100, max=100
            )
            diff_weight = torch.exp(diff_exp)
            difficulties.append(diff_weight)

        difficulty_weights = torch.stack(difficulties)

        # Combine all weights
        weights = skill_weights * age_weights * (0.5 + self.tau * difficulty_weights)

        # Apply weak opponent floor
        elo_ranks = torch.argsort(elos)
        weak_idx = elo_ranks[: max(1, n // 10)]  # Bottom decile
        weights_sum = weights.sum()
        if weights_sum > 0:
            floor_weight = self.weak_floor * weights_sum / len(weak_idx)
            weights[weak_idx] = torch.maximum(weights[weak_idx], floor_weight)

        # Ensure weights are non-negative and finite
        weights = torch.clamp(weights, min=0.0)

        # Check for NaN or infinite values
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print(f"Warning: Invalid weights detected in DRED pool")
            print(f"Skill weights: {skill_weights}")
            print(f"Age weights: {age_weights}")
            print(f"Difficulty weights: {difficulty_weights}")
            print(f"Combined weights: {weights}")
            # Fallback to uniform weights
            weights = torch.ones(n) / n

        # Normalize
        weights_sum = weights.sum()
        if weights_sum > 0:
            return weights / weights_sum
        else:
            # Fallback to uniform weights if sum is zero
            return torch.ones(n) / n

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
            data=DREDSnapshotData(),  # DRED-specific data
            model_dtype=model_dtype,
            is_exploiter=is_exploiter,
        )

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
        """Prune snapshots using DRED strategy while preserving last 5 exploiters."""
        n = len(self.snapshots)

        print("=> Pool size exceeded. Pruning...")

        # First, identify and preserve the last 5 exploiters
        exploiter_indices = []
        for i, snapshot in enumerate(self.snapshots):
            if snapshot.is_exploiter:
                exploiter_indices.append(i)

        # Sort exploiters by step (most recent first) and keep last 5
        if exploiter_indices:
            exploiter_steps = [(i, self.snapshots[i].step) for i in exploiter_indices]
            exploiter_steps.sort(key=lambda x: x[1], reverse=True)  # Most recent first
            keep_exploiters = set(i for i, _ in exploiter_steps[:5])  # Keep last 5
        else:
            keep_exploiters = set()

        # Keep top ELO 10%
        top_count = max(1, n // 10)
        elos = torch.tensor([s.elo for s in self.snapshots])
        sorted_indices = torch.argsort(-elos)
        top_indices = sorted_indices[:top_count]
        remaining_indices = sorted_indices[top_count:]

        keep_indices = set(top_indices.tolist())

        # Always keep the preserved exploiters
        keep_indices.update(keep_exploiters)

        if remaining_indices.numel() > 1:
            # Generate embeddings for remaining snapshots (excluding preserved exploiters)
            non_exploiter_remaining = [
                i for i in remaining_indices if i not in keep_exploiters
            ]

            if non_exploiter_remaining and self.last_batch_data is not None:
                embeddings = torch.stack(
                    [
                        self._generate_embedding(
                            self.snapshots[i], self.last_batch_data
                        )
                        for i in non_exploiter_remaining
                    ]
                )

                # Cluster count (up to 30 clusters)
                k_clusters = min(
                    30,
                    len(non_exploiter_remaining),
                    max(0, self.max_size - len(keep_indices)),
                )

                if k_clusters > 1:
                    # Use our custom k-medoids implementation
                    kmedoids = SimpleKMedoids(
                        n_clusters=k_clusters, random_state=0
                    ).fit(embeddings.float())

                    # Add medoid indices to keep set
                    for medoid_idx in kmedoids.medoid_indices_:
                        keep_indices.add(non_exploiter_remaining[medoid_idx])
                else:
                    # If only one cluster, keep all remaining non-exploiter indices
                    keep_indices.update(non_exploiter_remaining)

        # Ensure we always keep the weakest opponent when we have room to encourage diversity
        target_size = min(self.max_size, n)
        if len(keep_indices) < target_size:
            weakest_idx = torch.argmin(elos).item()
            keep_indices.add(weakest_idx)

        # Ensure we retain enough snapshots to fill the pool budget
        if len(keep_indices) < target_size:
            for idx_tensor in sorted_indices:
                idx = idx_tensor.item()
                if idx not in keep_indices:
                    keep_indices.add(idx)
                if len(keep_indices) >= target_size:
                    break

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
        # Use parent class ELO calculation
        super().update_elo_after_game(opponent, result, k_factor)

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
        Vectorized ELO update for a single opponent over multiple games with DRED-specific stats.

        Args:
            opponent: The opponent that was fought
            rewards: Tensor of rewards [num_games] where >0 = win, <0 = loss, =0 = draw
        """
        if opponent is None or rewards.numel() == 0:
            return

        # Use parent class batch update
        super().update_elo_batch_vectorized(opponent, rewards)

        # Update DRED-specific stats
        num_wins = (rewards > 0).sum().item()
        num_losses = (rewards < 0).sum().item()
        if opponent.data is not None:
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
        }

    def save_pool(self, path: str):
        """Save the DRED opponent pool to disk."""
        pool_data = {
            "current_elo": self.current_elo,
            "use_mixed_precision": self.use_mixed_precision,
            "max_size": self.max_size,
            "beta": self.beta,
            "lam": self.lam,
            "tau": self.tau,
            "eta": self.eta,
            "gamma": self.gamma,
            "p_curr": self.p_curr,
            "weak_floor": self.weak_floor,
            "k_recent": self.k_recent,
            "k_factor": self.k_factor,
            "last_admitted_step": self.last_admitted_step,
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
                "model_dtype": snapshot.model_dtype,
            }

            # Add DRED-specific data
            if hasattr(snapshot, "data") and snapshot.data is not None:
                snapshot_data["dred_age"] = snapshot.data.age
                snapshot_data["dred_alpha"] = snapshot.data.alpha
                snapshot_data["dred_beta"] = snapshot.data.beta

            pool_data["snapshots"].append(snapshot_data)

        torch.save(pool_data, path)

    def load_pool(self, path: str, model_class):
        """Load the DRED opponent pool from disk."""
        pool_data = torch.load(path)

        self.current_elo = pool_data["current_elo"]
        self.use_mixed_precision = pool_data.get("use_mixed_precision", False)
        self.max_size = pool_data.get("max_size", 100)
        self.beta = pool_data.get("beta", 0.02)
        self.lam = pool_data.get("lam", 0.003)
        self.tau = pool_data.get("tau", 1.0)
        self.eta = pool_data.get("eta", 40.0)
        self.gamma = pool_data.get("gamma", 0.5)
        self.p_curr = pool_data.get("p_curr", 0.5)
        self.weak_floor = pool_data.get("weak_floor", 0.1)
        self.k_recent = pool_data.get("k_recent", 20)
        self.k_factor = pool_data.get("k_factor", 32.0)
        self.last_admitted_step = pool_data.get("last_admitted_step", 0)

        self.snapshots = []
        for snapshot_data in pool_data["snapshots"]:
            # Create model instance
            model = model_class()
            model.load_state_dict(snapshot_data["model_state_dict"])

            # Get model dtype with backward compatibility
            model_dtype = snapshot_data.get("model_dtype", torch.float32)

            # Convert model to the stored dtype
            model = model.to(model_dtype)

            # Create snapshot
            snapshot = AgentSnapshot(
                model=model,
                step=snapshot_data["step"],
                elo=snapshot_data["elo"],
                model_dtype=model_dtype,
            )

            # Restore game statistics
            snapshot.games_played = snapshot_data["games_played"]
            snapshot.wins = snapshot_data["wins"]
            snapshot.losses = snapshot_data["losses"]
            snapshot.draws = snapshot_data["draws"]

            # Restore DRED-specific data
            if "dred_age" in snapshot_data:
                snapshot.data = DREDSnapshotData(
                    age=snapshot_data["dred_age"],
                    alpha=snapshot_data.get("dred_alpha", 1.0),
                    beta=snapshot_data.get("dred_beta", 1.0),
                )

            self.snapshots.append(snapshot)
