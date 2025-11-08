from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from alphaholdem.core.structured_config import CFRType, SearchConfig
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.agent_snapshot import AgentSnapshot
from alphaholdem.rl.opponent_pool import OpponentPool
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
from alphaholdem.utils.profiling import profile


class PBSPool(OpponentPool):
    """
    Public Belief State (PBS) pool that evaluates candidates against existing pool members
    and only admits candidates with ELO higher than the worst existing model.

    This pool maintains a fixed-size collection of the best models as determined by
    ELO ratings from head-to-head evaluation.
    """

    def __init__(
        self,
        pool_size: int = 5,
        k_factor: float = 32.0,
        use_mixed_precision: bool = False,
        generator: torch.Generator | None = None,
    ):
        """
        Initialize PBS opponent pool.

        Args:
            pool_size: Maximum number of snapshots to maintain in the pool
            k_factor: ELO K-factor for rating changes
            use_mixed_precision: Whether to store models in bfloat16 for memory efficiency
            generator: Generator for reproducible randomness
        """
        super().__init__(k_factor=k_factor)

        self.pool_size = pool_size
        self.use_mixed_precision = use_mixed_precision
        self.generator = generator
        self.snapshots: List[AgentSnapshot] = []

        # Default evaluation config if not provided
        self.default_search_cfg: Optional[SearchConfig] = None
        self.default_bet_bins: Optional[List[float]] = None
        self.default_device: Optional[torch.device] = None

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
        weights = torch.tensor([snapshot.elo for snapshot in self.snapshots])

        if weights.sum() == 0:
            # If all ELOs are 0, sample uniformly
            samples = torch.randint(
                0, len(self.snapshots), (k,), generator=self.generator
            )
            return [self.snapshots[i] for i in samples.tolist()]

        # Normalize weights
        normalized_weights = weights / weights.sum()

        sampled_indices = torch.multinomial(
            normalized_weights,
            k,
            replacement=len(self.snapshots) < k,
            generator=self.generator,
        )
        return [self.snapshots[i] for i in sampled_indices.tolist()]

    def evaluate_model_against_pool(
        self,
        candidate_model: BetterFFN | RebelFFN,
        num_games_per_opponent: int = 50,
    ) -> float:
        """
        Evaluate a candidate model against all pool members using inline public-belief games.

        Args:
            candidate_model: Model to evaluate (RebelFFN or BetterFFN)
            num_games_per_opponent: Number of games to play against each pool member

        Returns:
            Final ELO rating of the candidate after playing against all pool members
        """
        if not self.snapshots:
            return self.current_elo

        # Default config for evaluation
        bet_bins: List[float] = [0.5, 1.5]
        device = (
            next(candidate_model.parameters()).device
            if any(p.requires_grad or p.is_leaf for p in candidate_model.parameters())
            else torch.device("cpu")
        )
        search_cfg = SearchConfig()
        search_cfg.depth = 1
        search_cfg.iterations = 1
        search_cfg.warm_start_iterations = 0
        search_cfg.cfr_type = CFRType.linear
        search_cfg.cfr_avg = True

        candidate_elo = self.current_elo

        for opponent in self.snapshots:
            opponent_model = opponent.model
            rewards = PBSPool._play_public_belief_games(
                candidate_model,
                opponent_model,
                num_games_per_opponent,
                bet_bins,
                self.generator,
                device,
                search_cfg,
            )

            # Update ELO based on batch results using a temporary snapshot
            temp_candidate = AgentSnapshot(
                model=candidate_model, step=-1, elo=candidate_elo
            )
            self.update_elo_batch_vectorized(temp_candidate, rewards)
            candidate_elo = temp_candidate.elo

        return candidate_elo

    def add_snapshot(
        self,
        model: BetterFFN | RebelFFN,
        step: int,
        rating: Optional[float] = None,
        is_exploiter: bool = False,
        evaluate_against_pool: bool = True,
        num_games_per_opponent: int = 100,
    ) -> bool:
        """
        Evaluate and potentially add a new snapshot to the pool.

        Args:
            model: The model to snapshot (RebelFFN or BetterFFN)
            step: Training step
            rating: Optional pre-computed ELO rating (skips evaluation if provided)
            is_exploiter: Whether this snapshot is an exploiter
            evaluate_against_pool: Whether to evaluate against pool members (if False, uses rating)
            num_games_per_opponent: Number of games per opponent if evaluating

        Returns:
            True if the snapshot was added to the pool, False otherwise
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

        # Disable gradients for snapshot models
        if model is not None:
            for p in new_snapshot.model.parameters():
                p.requires_grad = False

        # Evaluate against pool if requested and evaluation function is available
        if evaluate_against_pool and rating is None:
            new_snapshot.elo = self.evaluate_model_against_pool(
                model, num_games_per_opponent
            )

        # Determine if we should add this snapshot
        if len(self.snapshots) < self.pool_size:
            # Pool not full, always add
            self.snapshots.append(new_snapshot)
            self.snapshots.sort(key=lambda x: x.elo, reverse=True)
            return True

        # Pool is full - only add if ELO is higher than worst existing model
        worst_snapshot = min(self.snapshots, key=lambda x: x.elo)
        if new_snapshot.elo > worst_snapshot.elo:
            # Remove worst snapshot and add new one
            self.snapshots.remove(worst_snapshot)
            self.snapshots.append(new_snapshot)
            self.snapshots.sort(key=lambda x: x.elo, reverse=True)
            return True

        # Not added - ELO too low
        return False

    def update_elo_after_game(
        self, opponent: AgentSnapshot, result: str, k_factor: Optional[float] = None
    ) -> None:
        """
        Update ELO ratings after a game with PBS-specific sorting.

        Args:
            opponent: The opponent that was played against
            result: 'win', 'loss', or 'draw'
            k_factor: ELO K-factor for rating changes (uses instance default if None)
        """
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
        Vectorized ELO update for a single opponent over multiple games with PBS sorting.

        Args:
            opponent: The opponent that was fought
            rewards: Tensor of rewards [num_games] where >0 = win, <0 = loss, =0 = draw
        """
        if opponent is None or rewards.numel() == 0:
            return

        super().update_elo_batch_vectorized(opponent, rewards)

        # Re-sort snapshots by ELO
        self.snapshots.sort(key=lambda x: x.elo, reverse=True)

    def get_best_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the snapshot with the highest ELO rating."""
        if not self.snapshots:
            return None
        return self.snapshots[0]

    def get_worst_snapshot(self) -> Optional[AgentSnapshot]:
        """Get the snapshot with the lowest ELO rating."""
        if not self.snapshots:
            return None
        return min(self.snapshots, key=lambda x: x.elo)

    def get_pool_stats(self) -> dict:
        """Get statistics about the opponent pool."""
        if not self.snapshots:
            return {
                "pool_size": 0,
                "max_pool_size": self.pool_size,
                "avg_elo": 0.0,
                "min_elo": 0.0,
                "max_elo": 0.0,
                "current_elo": self.current_elo,
            }

        elos = [snapshot.elo for snapshot in self.snapshots]
        return {
            "pool_size": len(self.snapshots),
            "max_pool_size": self.pool_size,
            "avg_elo": sum(elos) / len(elos),
            "min_elo": min(elos),
            "max_elo": max(elos),
            "current_elo": self.current_elo,
            "best_snapshot_step": self.snapshots[0].step,
            "best_snapshot_elo": self.snapshots[0].elo,
            "worst_snapshot_step": min(self.snapshots, key=lambda x: x.elo).step,
            "worst_snapshot_elo": min(elos),
        }

    def should_add_snapshot(
        self, current_step: int, kl_divergence: float = 0.0
    ) -> bool:
        """
        Determine if a new snapshot should be evaluated for addition to the pool.

        For PBS pool, this is always True since evaluation happens in add_snapshot.

        Args:
            current_step: Current training step (unused for PBS)
            kl_divergence: KL divergence from the last training step (unused for PBS)

        Returns:
            True (always evaluate candidates)
        """
        return True

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
            "pool_size": self.pool_size,
            "current_elo": self.current_elo,
            "use_mixed_precision": self.use_mixed_precision,
            "snapshots": [],
        }

        for snapshot in self.snapshots:
            model_state_dict = snapshot.model.state_dict()
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
        self.pool_size = pool_data.get("pool_size", len(pool_data["snapshots"]))
        self.use_mixed_precision = pool_data.get("use_mixed_precision", False)
        self.snapshots = []

        for snapshot_data in pool_data["snapshots"]:
            model_dtype = snapshot_data.get("model_dtype", torch.float32)
            if (
                "use_mixed_precision" in snapshot_data
                and "model_dtype" not in snapshot_data
            ):
                model_dtype = (
                    torch.bfloat16
                    if snapshot_data["use_mixed_precision"]
                    else torch.float32
                )

            model = model_class()
            model.load_state_dict(snapshot_data["model_state_dict"])
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

        self.snapshots.sort(key=lambda x: x.elo, reverse=True)

    @staticmethod
    def _play_public_belief_games(
        model_a: BetterFFN | RebelFFN,
        model_b: nn.Module,
        num_games: int,
        bet_bins: List[float],
        generator: torch.Generator,
        device: torch.device,
        search_cfg: SearchConfig,
    ) -> torch.Tensor:
        """
        Play public-belief head-to-head games between two models.

        Each model uses its own beliefs for decision-making, but showdown uses
        two-prior averaging for scoring.

        Args:
            model_a: First model (will be candidate in evaluation)
            model_b: Second model (will be opponent in evaluation)
            num_games: Number of games to play
            bet_bins: Bet size bins
            device: Device for computation
            search_cfg: Search configuration for evaluators

        Returns:
            rewards: [num_games] tensor where >0 = model_a win, <0 = loss, =0 = draw
            (from model_a's perspective)
        """
        assert search_cfg.depth > 0

        env_proto = HUNLTensorEnv(
            num_envs=1,
            starting_stack=1000,
            sb=5,
            bb=10,
            default_bet_bins=bet_bins,
            device=device,
            float_dtype=torch.float32,
            flop_showdown=False,
        )

        # Create evaluators for both models
        def create_evaluator(model: BetterFFN | RebelFFN) -> RebelCFREvaluator:
            return RebelCFREvaluator(
                search_batch_size=1,
                env_proto=env_proto,
                model=model,
                bet_bins=bet_bins,
                max_depth=search_cfg.depth,
                cfr_iterations=search_cfg.iterations,
                device=device,
                float_dtype=torch.float32,
                warm_start_iterations=search_cfg.warm_start_iterations,
                cfr_type=search_cfg.cfr_type,
                cfr_avg=search_cfg.cfr_avg,
            )

        evaluator_a = create_evaluator(model_a)
        evaluator_b = create_evaluator(model_b)

        rewards = torch.zeros(num_games, device=device, dtype=torch.float32)

        # Play each game
        for game_idx in range(num_games):
            # Initialize fresh environment
            env = HUNLTensorEnv.from_proto(env_proto)
            env.reset()

            # Track which evaluator to use (alternate button)
            button = game_idx % 2
            env.button[0] = button

            # Initialize evaluators with root state
            roots = torch.tensor([0], device=device)
            evaluator_a.initialize_search(env, roots)
            evaluator_b.initialize_search(env, roots)

            # Play until terminal using full CFR at each decision
            max_iterations = 40
            iteration = 0
            while not env.done[0].item() and iteration < max_iterations:
                iteration += 1

                # Get current evaluator based on to_act
                to_act = env.to_act[0].item()
                current_eval = evaluator_a if to_act == 0 else evaluator_b
                # Perform one CFR self-play iteration to choose and apply an action
                current_eval.evaluate_cfr(training_mode=False)

                hand = torch.multinomial(
                    current_eval.beliefs[0, 0], num_samples=1, generator=generator
                ).item()
                probs = current_eval.policy_probs_avg[
                    1 : 1 + current_eval.num_actions, hand
                ]
                action = torch.multinomial(
                    probs, num_samples=1, generator=generator
                ).item()

                # Keep both evaluators in sync with the new root
                env_rewards, _, _ = env.step_bins(
                    torch.full((1,), action, device=device, dtype=torch.long)
                )
                evaluator_a.initialize_search(env, roots)
                evaluator_b.initialize_search(env, roots)

            assert env.done[0].item(), "Environment should be done"

            # Terminal state: compute reward
            # Check if fold or showdown
            if env.street[0].item() == 4:  # Showdown
                # Compute expected payoff by averaging each evaluator's showdown EV
                # Each evaluator computes EV from its own beliefs/perspective
                idx = torch.tensor([0], device=device)
                ev_a = evaluator_a._showdown_value(idx)  # tensor scalar
                ev_b = evaluator_b._showdown_value(idx)
                payoff = 0.5 * (float(ev_a.item()) + float(ev_b.item()))

                # Reward is from player 0's perspective
                rewards[game_idx] = payoff
            else:
                # Fold: deterministic payoff
                rewards[game_idx] = env_rewards[0].item()

        return rewards
