"""ELO rating system for poker agents."""

from __future__ import annotations

from typing import Optional

import torch

from alphaholdem.rl.agent_snapshot import AgentSnapshot


class ELOCalculator:
    """ELO rating calculator with magnitude-based scoring."""

    def __init__(self, k_factor: float = 32.0):
        """
        Initialize ELO calculator.

        Args:
            k_factor: ELO K-factor for rating changes
        """
        self.k_factor = k_factor

    def update_elo_after_game(
        self,
        current_elo: float,
        opponent: AgentSnapshot,
        result: str,
        k_factor: Optional[float] = None,
    ) -> float:
        """
        Update ELO ratings after a single game.

        Args:
            current_elo: Current ELO rating
            opponent: The opponent that was played against
            result: 'win', 'loss', or 'draw'
            k_factor: ELO K-factor for rating changes (uses instance default if None)

        Returns:
            New ELO rating
        """
        if k_factor is None:
            k_factor = self.k_factor

        # Calculate expected score
        expected_score = 1.0 / (1.0 + 10 ** ((opponent.elo - current_elo) / 400.0))

        # Calculate actual score
        if result == "win":
            actual_score = 1.0
        elif result == "loss":
            actual_score = 0.0
        else:  # draw
            actual_score = 0.5

        # Calculate ELO change
        elo_change = k_factor * (actual_score - expected_score)

        return current_elo + elo_change

    def update_elo_batch_vectorized(
        self,
        current_elo: float,
        opponent: AgentSnapshot,
        rewards: torch.Tensor,
    ) -> float:
        """
        Vectorized ELO update for a single opponent over multiple games.

        Args:
            current_elo: Current ELO rating
            opponent: The opponent that was fought
            rewards: Tensor of rewards [num_games] where >0 = win, <0 = loss, =0 = draw

        Returns:
            New ELO rating
        """
        if opponent is None or rewards.numel() == 0:
            return current_elo

        # Check for rewards outside [-1, 1] range and warn
        extreme_rewards = torch.abs(rewards) > 1.0
        if extreme_rewards.any():
            extreme_values = rewards[extreme_rewards]
            print(
                f"Warning: {extreme_rewards.numel()} rewards outside [-1, 1] range: {extreme_values.tolist()}"
            )

        # Scale rewards to [0, 1] range based on magnitude
        # Convert to actual scores: negative rewards -> 0, positive rewards -> 1, scaled by magnitude
        actual_scores = 0.5 + 0.5 * rewards

        # Calculate expected scores for each game
        expected_scores = 1.0 / (1.0 + 10 ** ((opponent.elo - current_elo) / 400.0))

        # Calculate ELO changes
        elo_changes = self.k_factor * (actual_scores - expected_scores)

        # Update current ELO (sum of all changes)
        total_elo_change = elo_changes.sum().item()

        # Note: Opponent stats are updated by the calling pool, not here
        # This prevents double-counting when both players' ELOs are updated

        return current_elo + total_elo_change
