from __future__ import annotations

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from line_profiler import profile
except Exception:  # pragma: no cover

    def profile(f):
        return f


from alphaholdem.core.config_loader import get_config

from ..env.hunl_env import HUNLEnv
from ..encoding.cards_encoder import CardsPlanesV1
from ..encoding.actions_encoder import ActionsHUEncoderV1
from ..encoding.action_mapping import bin_to_action, get_legal_mask
from ..models.siamese_convnet import SiameseConvNetV1
from ..models.heads import CategoricalPolicyV1
from ..rl.replay import (
    ReplayBuffer,
    Transition,
    Trajectory,
    compute_gae_returns,
    prepare_ppo_batch,
)
from ..rl.losses import trinal_clip_ppo_loss
from ..rl.k_best_pool import KBestOpponentPool, AgentSnapshot
from ..core.config import RootConfig
from ..core.builders import build_components_from_config


class SelfPlayTrainer:
    def __init__(
        self,
        learning_rate: float = None,
        batch_size: int = None,
        mini_batch_size: int = None,
        num_epochs: int = None,
        gamma: float = None,
        gae_lambda: float = None,
        epsilon: float = None,
        delta1: float = None,
        value_coef: float = None,
        entropy_coef: float = None,
        grad_clip: float = None,
        k_best_pool_size: int = 5,
        min_elo_diff: float = 30.0,
        device: torch.device = None,
        config: Union[RootConfig, str, None] = None,
    ):
        self.cfg = get_config(config) if config is not None else get_config(None)

        # Use provided args if not None, else use config defaults
        self.batch_size = batch_size if batch_size is not None else self.cfg.batch_size
        self.mini_batch_size = (
            mini_batch_size if mini_batch_size is not None else self.cfg.mini_batch_size
        )
        self.num_epochs = num_epochs if num_epochs is not None else self.cfg.num_epochs
        self.gamma = gamma if gamma is not None else self.cfg.gamma
        self.gae_lambda = gae_lambda if gae_lambda is not None else self.cfg.gae_lambda
        self.epsilon = epsilon if epsilon is not None else self.cfg.ppo_eps
        self.delta1 = delta1 if delta1 is not None else self.cfg.ppo_delta1
        self.value_coef = value_coef if value_coef is not None else self.cfg.value_coef
        self.entropy_coef = (
            entropy_coef if entropy_coef is not None else self.cfg.entropy_coef
        )
        self.grad_clip = grad_clip if grad_clip is not None else self.cfg.grad_clip

        # Set device
        self.device = (
            device
            if device is not None
            else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )

        # Initialize components
        self.env = HUNLEnv(starting_stack=1000, sb=5, bb=10)

        # Config-driven components
        ce, ae, model, policy = build_components_from_config(self.cfg)
        self.cards_encoder = ce
        self.actions_encoder = ae
        self.model = model
        self.policy = policy

        # Ensure bins align with model output size to avoid mask/logit mismatch
        if hasattr(self.model, "policy_head") and hasattr(
            self.model.policy_head, "out_features"
        ):
            self.num_bet_bins = int(self.model.policy_head.out_features)
        else:
            self.num_bet_bins = len(self.cfg.bet_bins) + 3

        self.model.to(self.device)  # Move model to device
        self._initialize_weights()  # Initialize with better weights
        # Replay buffer capacity in steps is batch_size * replay_buffer_batches
        # Keep trajectory container capacity large; we'll trim by step count later
        self.replay_buffer = ReplayBuffer(
            capacity=self.batch_size
            * max(1, int(getattr(self.cfg, "replay_buffer_batches", 1)))
        )

        # K-Best opponent pool
        self.opponent_pool = KBestOpponentPool(
            k=k_best_pool_size, min_elo_diff=min_elo_diff
        )

        # Optimizer
        lr = learning_rate if learning_rate is not None else self.cfg.learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Training stats
        self.episode_count = 0
        self.total_reward = 0.0
        self.current_elo = 1200.0  # Starting ELO rating

    def _initialize_weights(self):
        """Initialize model weights to prevent dead neurons."""
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    @profile
    def collect_trajectory(
        self, opponent_snapshot: Optional[AgentSnapshot] = None
    ) -> Trajectory:
        """
        Collect a single trajectory from self-play or against an opponent.

        Args:
            opponent_snapshot: Optional opponent snapshot to play against.
                             If None, plays against self (self-play).
        """
        state = self.env.reset()
        transitions = []
        our_chips = 0
        opp_chips = 0
        max_steps = 50  # Safety limit to prevent infinite loops

        step_count = 0
        final_reward_from_our_perspective = 0.0

        while not state.terminal and step_count < max_steps:
            step_count += 1

            # Determine which player's turn it is
            current_player = state.to_act

            # Encode state for current player
            cards = self.cards_encoder.encode_cards(
                state, seat=current_player, device=self.device
            )
            actions_tensor = self.actions_encoder.encode_actions(
                state,
                seat=current_player,
                num_bet_bins=self.num_bet_bins,
                device=self.device,
            )

            # Get model prediction
            with torch.no_grad():
                autocast_enabled = (
                    self.device.type == "cuda" or self.device.type == "mps"
                )
                dtype = torch.bfloat16 if autocast_enabled else torch.float16
                cm = (
                    torch.autocast(
                        device_type=self.device.type,
                        dtype=dtype,
                    )
                    if autocast_enabled
                    else torch.amp.autocast("cpu", enabled=False)
                )
                with cm:
                    if opponent_snapshot is not None and current_player != 0:
                        logits, value = opponent_snapshot.model(
                            cards.unsqueeze(0), actions_tensor.unsqueeze(0)
                        )
                    else:
                        logits, value = self.model(
                            cards.unsqueeze(0), actions_tensor.unsqueeze(0)
                        )

                legal_mask = get_legal_mask(
                    state, self.num_bet_bins, device=self.device
                )

                # Sample action
                action_idx, log_prob = self.policy.action(logits.squeeze(0), legal_mask)

                # Convert to concrete action
                action = bin_to_action(action_idx, state, self.num_bet_bins)

            # Take step
            next_state, reward, done, _ = self.env.step(action)

            # Record transition (only for our player's actions)
            if current_player == 0:  # Our player
                transition = Transition(
                    observation=torch.cat(
                        [cards.flatten(), actions_tensor.flatten()]
                    ),  # Simplified obs
                    action=action_idx,
                    log_prob=log_prob,
                    value=float(value.item()),
                    reward=reward,
                    done=done,
                    legal_mask=legal_mask,
                    chips_placed=action.amount,
                    # Per-sample clipping bounds: δ2 = -opp_cum, δ3 = our_cum
                    delta2=float(-opp_chips),
                    delta3=float(our_chips + action.amount),
                )
                transitions.append(transition)
                cum_chips_me += action.amount
            else:
                cum_chips_opp += action.amount

                # If the opponent's action ended the hand, record the final reward for our last transition
                if done and transitions:
                    # The reward is from our perspective, so it's -reward if opp ended the hand
                    transitions[-1].reward = -reward
                    transitions[-1].done = True

            # Track final reward from our perspective if episode ends
            if done:
                final_reward_from_our_perspective = (
                    float(reward) if current_player == 0 else float(-reward)
                )

            state = next_state

        if step_count >= max_steps:
            print(
                f"Warning: Trajectory reached max steps ({max_steps}), forcing termination"
            )
            # Force termination by setting final reward
            if transitions:
                transitions[-1].reward = 0.0
                transitions[-1].done = True
            final_reward_from_our_perspective = 0.0

        return (
            Trajectory(
                transitions=transitions,
                our_chips_committed=our_chips,
                opp_chips_committed=opp_chips,
            ),
            final_reward_from_our_perspective,
        )

    # def update_model(self) -> dict:
    #     with profile(record_shapes=True) as prof:
    #         result = self.update_model_internal()
    #     print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    #     prof.export_chrome_trace("update_model.json")
    #     return result

    @profile
    def update_model(self) -> dict:
        """Perform PPO update on collected trajectories."""
        # Require enough samples (transitions) across buffer
        total_samples = sum(len(t.transitions) for t in self.replay_buffer.trajectories)
        if total_samples < self.batch_size:
            return {}

        # Use all trajectories available for better mixing
        trajectories = self.replay_buffer.sample_trajectories(self.batch_size)

        # Compute GAE using stored V(s_t) values collected during rollout
        for trajectory in trajectories:
            rewards = [t.reward for t in trajectory.transitions]
            values = [t.value for t in trajectory.transitions]
            values.append(0.0)  # bootstrap at terminal

            advantages, returns = compute_gae_returns(
                rewards, values, gamma=self.gamma, lambda_=self.gae_lambda
            )

            for i, transition in enumerate(trajectory.transitions):
                transition.advantage = advantages[i]
                transition.return_ = returns[i]

        # Prepare batch
        batch = prepare_ppo_batch(trajectories)

        # Move batch tensors to device FIRST for consistent indexing on device tensors
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        # Normalize advantages across the batch for stability (mean=0, std=1)
        adv = batch["advantages"]
        adv_mean = adv.mean()
        adv_std = adv.std().clamp_min(1e-8)
        batch["advantages"] = (adv - adv_mean) / adv_std

        # Subsample to target sample batch size if necessary (indexing on same device)
        num_samples = batch["actions"].shape[0]
        if num_samples > self.batch_size:
            idx = torch.randperm(num_samples, device=self.device)[: self.batch_size]

            def take(d):
                return d.index_select(0, idx) if isinstance(d, torch.Tensor) else d

            batch = {k: take(v) for k, v in batch.items()}

        # Per-sample value clipping bounds computed in prepare_ppo_batch
        # batch['delta2'] and batch['delta3'] are length-N tensors aligned with samples
        delta2_vec = batch["delta2"]
        delta3_vec = batch["delta3"]

        # Multiple epochs of updates
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clipfrac = 0.0
        total_explained_var = 0.0
        total_minibatches = 0
        for _ in range(self.num_epochs):
            N = batch["actions"].shape[0]
            order = torch.randperm(N, device=self.device)
            for start in range(0, N, self.mini_batch_size):
                mb_idx = order[start : start + self.mini_batch_size]

                observations = batch["observations"].index_select(0, mb_idx)
                cards = observations[:, : (6 * 4 * 13)].reshape(-1, 6, 4, 13)
                actions_tensor = observations[:, (6 * 4 * 13) :].reshape(
                    -1, 24, 4, self.num_bet_bins
                )

                logits, values = self.model(cards, actions_tensor)

                loss_dict = trinal_clip_ppo_loss(
                    logits=logits,
                    values=values,
                    actions=batch["actions"].index_select(0, mb_idx),
                    log_probs_old=batch["log_probs_old"].index_select(0, mb_idx),
                    advantages=batch["advantages"].index_select(0, mb_idx),
                    returns=batch["returns"].index_select(0, mb_idx),
                    legal_masks=batch["legal_masks"].index_select(0, mb_idx),
                    epsilon=self.epsilon,
                    delta1=self.delta1,
                    delta2=delta2_vec.index_select(0, mb_idx),
                    delta3=delta3_vec.index_select(0, mb_idx),
                    value_coef=self.value_coef,
                    entropy_coef=self.entropy_coef,
                )

                # Debugging metrics: approx KL, clipfrac, explained variance
                with torch.no_grad():
                    legal_mb = batch["legal_masks"].index_select(0, mb_idx)
                    masked_logits = logits.clone()
                    masked_logits[legal_mb == 0] = -1e9
                    log_probs_new = torch.log_softmax(masked_logits, dim=-1)
                    a_mb = batch["actions"].index_select(0, mb_idx)
                    logp_new = log_probs_new.gather(1, a_mb.unsqueeze(1)).squeeze(1)
                    logp_old = batch["log_probs_old"].index_select(0, mb_idx)
                    ratio = torch.exp(logp_new - logp_old)
                    approx_kl = (logp_old - logp_new).mean()
                    clip_low, clip_high = 1.0 - self.epsilon, 1.0 + self.epsilon
                    clipped = torch.clamp(ratio, clip_low, clip_high)
                    clipfrac = (torch.abs(clipped - ratio) > 1e-8).float().mean()
                    # Use the same target as the loss for EV: clipped returns
                    ret_mb = batch["returns"].index_select(0, mb_idx)
                    d2_mb = delta2_vec.index_select(0, mb_idx)
                    d3_mb = delta3_vec.index_select(0, mb_idx)
                    ret_clipped_mb = torch.clamp(ret_mb, d2_mb, d3_mb)
                    # Explained variance of value predictions against clipped targets
                    var_y = torch.var(ret_clipped_mb)
                    var_err = torch.var(ret_clipped_mb - values.detach())
                    explained_var = 1.0 - (var_err / (var_y + 1e-8))

                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

                total_loss += loss_dict["total_loss"].item()
                total_policy_loss += float(loss_dict["policy_loss"].item())
                total_value_loss += float(loss_dict["value_loss"].item())
                total_entropy += float(loss_dict["entropy"].item())
                total_approx_kl += float(approx_kl.item())
                total_clipfrac += float(clipfrac.item())
                total_explained_var += float(explained_var.item())
                total_minibatches += 1

        denom = max(1, total_minibatches)
        return {
            "avg_loss": total_loss / self.num_epochs,
            "num_trajectories": len(trajectories),
            "delta2_mean": float(delta2_vec.mean().item()),
            "delta3_mean": float(delta3_vec.mean().item()),
            "policy_loss": total_policy_loss / denom,
            "value_loss": total_value_loss / denom,
            "entropy": total_entropy / denom,
            "approx_kl": total_approx_kl / denom,
            "clipfrac": total_clipfrac / denom,
            "explained_var": total_explained_var / denom,
        }

    def train_step(self) -> dict:
        """
        Single training step: collect trajectories against K-Best opponents and update model.

        Args:
            num_trajectories: Number of trajectories to collect

        Returns:
            Dictionary with training statistics
        """
        # Target total steps in replay buffer
        target_steps = self.batch_size * max(self.cfg.replay_buffer_batches, 1)

        # Warmup: fill buffer to target_steps
        print(f"Warmup: filling replay buffer to {target_steps} steps...")
        i = 0
        while self.replay_buffer.num_steps() < target_steps:
            # Collect a batch of trajectories in parallel to amortize device calls
            sampled_opponent = self.opponent_pool.sample(k=1)
            opponent = sampled_opponent[0] if sampled_opponent else None
            trajectory, reward = self.collect_trajectory(opponent)
            if len(trajectory.transitions) == 0:
                # opponent went first and folded. no decisions to train on. skip
                continue
            i += 1
            self.replay_buffer.add_trajectory(trajectory)
            self.episode_count += 1
            self.total_reward += reward
            if opponent is not None:
                if reward > 0:
                    result = "win"
                elif reward < 0:
                    result = "loss"
                else:
                    result = "draw"
                self.opponent_pool.update_elo_after_game(opponent, result)
                self.current_elo = self.opponent_pool.current_elo

        # Update model
        update_stats = self.update_model()

        # Check if we should add current model to opponent pool
        if self.opponent_pool.should_add_snapshot(self.current_elo):
            self.opponent_pool.add_snapshot(self, self.current_elo)

        # After update, add one more batch worth of fresh steps
        added_steps = 0
        while added_steps < self.batch_size:
            sampled_opponent = self.opponent_pool.sample(k=1)
            opponent = sampled_opponent[0] if sampled_opponent else None
            trajectory, reward = self.collect_trajectory(opponent)
            if len(trajectory.transitions) == 0:
                continue
            self.replay_buffer.add_trajectory(trajectory)
            added_steps += len(trajectory.transitions)
            self.episode_count += 1
            self.total_reward += reward
            if opponent is not None:
                if reward > 0:
                    result = "win"
                elif reward < 0:
                    result = "loss"
                else:
                    result = "draw"
                self.opponent_pool.update_elo_after_game(opponent, result)
                self.current_elo = self.opponent_pool.current_elo

        # Trim buffer back to target_steps
        self.replay_buffer.trim_to_steps(target_steps)

        return {
            "episode_count": self.episode_count,
            "avg_reward": self.total_reward / max(1, self.episode_count),
            "current_elo": self.current_elo,
            "pool_stats": self.opponent_pool.get_pool_stats(),
            **update_stats,
        }

    def save_checkpoint(self, path: str, step: int) -> None:
        """Save model checkpoint and opponent pool."""
        # Serialize opponent pool inline
        pool_data = {
            "k": self.opponent_pool.k,
            "min_elo_diff": self.opponent_pool.min_elo_diff,
            "current_elo": self.opponent_pool.current_elo,
            "snapshots": [],
        }
        for snapshot in self.opponent_pool.snapshots:
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

        checkpoint = {
            "step": step,
            "episode_count": self.episode_count,
            "total_reward": self.total_reward,
            "current_elo": self.current_elo,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # Store replay buffer to resume training seamlessly
            "replay_buffer": self.replay_buffer,
            # Store opponent pool inline for single-file checkpoints
            "opponent_pool": pool_data,
            "config": {
                "num_bet_bins": self.num_bet_bins,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "epsilon": self.epsilon,
                "delta1": self.delta1,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "grad_clip": self.grad_clip,
            },
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint and opponent pool. Returns the step number."""
        # PyTorch 2.6 defaults to weights_only=True which blocks unpickling
        # custom classes like our ReplayBuffer. We trust our local checkpoints,
        # so explicitly allow full load.
        checkpoint = torch.load(path, weights_only=False, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.episode_count = checkpoint["episode_count"]
        self.total_reward = checkpoint["total_reward"]
        self.current_elo = checkpoint.get("current_elo", 1200.0)

        # Restore replay buffer if present
        if "replay_buffer" in checkpoint and checkpoint["replay_buffer"] is not None:
            self.replay_buffer = checkpoint["replay_buffer"]

        print(f"Checkpoint loaded from {path}")

        # Restore opponent pool from inline data if available; fallback to old separate file
        pool_data = checkpoint.get("opponent_pool")
        if pool_data is not None:
            from ..models.siamese_convnet import SiameseConvNetV1

            self.opponent_pool.k = pool_data.get("k", self.opponent_pool.k)
            self.opponent_pool.min_elo_diff = pool_data.get(
                "min_elo_diff", self.opponent_pool.min_elo_diff
            )
            self.opponent_pool.current_elo = pool_data.get(
                "current_elo", self.opponent_pool.current_elo
            )
            self.opponent_pool.snapshots = []
            for snapshot_data in pool_data.get("snapshots", []):
                model = SiameseConvNetV1()
                model.load_state_dict(snapshot_data["model_state_dict"])
                model.to(self.device)
                snapshot = AgentSnapshot(
                    model=model,
                    step=snapshot_data.get("step", 0),
                    elo=snapshot_data.get("elo", 1200.0),
                )
                snapshot.games_played = snapshot_data.get("games_played", 0)
                snapshot.wins = snapshot_data.get("wins", 0)
                snapshot.losses = snapshot_data.get("losses", 0)
                snapshot.draws = snapshot_data.get("draws", 0)
                self.opponent_pool.snapshots.append(snapshot)
            print("Opponent pool restored from checkpoint file")
        else:
            # Backward-compatibility path: load separate pool file if present
            pool_path = path.replace(".pt", "_pool.pt")
            try:
                from ..models.siamese_convnet import SiameseConvNetV1

                self.opponent_pool.load_pool(pool_path, SiameseConvNetV1)
                # Ensure snapshot models are on the correct device
                for snap in self.opponent_pool.snapshots:
                    snap.model.to(self.device)
                print(f"Opponent pool loaded from {pool_path}")
            except FileNotFoundError:
                print(
                    f"No opponent pool found at {pool_path}; starting with empty pool"
                )

        return checkpoint["step"]

    def evaluate_against_pool(self, num_games: int = 100) -> dict:
        """
        Evaluate current model against all opponents in the pool.

        Args:
            num_games: Number of games to play against each opponent

        Returns:
            Dictionary with evaluation results
        """
        if not self.opponent_pool.snapshots:
            return {"error": "No opponents in pool"}

        results = {}
        total_wins = 0
        total_games = 0

        for i, opponent in enumerate(self.opponent_pool.snapshots):
            wins = 0
            for _ in range(num_games):
                _, final_reward = self.collect_trajectory(opponent_snapshot=opponent)
                if final_reward > 0:
                    wins += 1

            win_rate = wins / num_games
            results[f"opponent_{i}_step_{opponent.step}"] = {
                "win_rate": win_rate,
                "opponent_elo": opponent.elo,
                "wins": wins,
                "total_games": num_games,
            }
            total_wins += wins
            total_games += num_games

        overall_win_rate = total_wins / total_games if total_games > 0 else 0.0

        return {
            "overall_win_rate": overall_win_rate,
            "total_games": total_games,
            "opponent_results": results,
            "pool_stats": self.opponent_pool.get_pool_stats(),
        }

    def get_preflop_range_grid(self, seat: int = 0, metric: str = "allin") -> str:
        """Get preflop range as a grid showing selected action probabilities for button play.

        metric: "allin" or "fold"
        """
        # Create a 13x13 grid representing all possible hole card combinations
        # Rows/cols: A, K, Q, J, T, 9, 8, 7, 6, 5, 4, 3, 2
        ranks = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]

        # Initialize grid with action probabilities
        grid = []
        header = "    " + " ".join(f"{rank:>2}" for rank in ranks)
        grid.append(header)
        grid.append("   " + "-" * 39)  # Separator line

        for i, rank1 in enumerate(ranks):
            row = [f"{rank1:>2} |"]
            for j, rank2 in enumerate(ranks):
                if i == j:
                    # Same rank (pairs) - always suited
                    card1, card2 = f"{rank1}s", f"{rank1}h"  # Suited pair
                elif i < j:
                    # Top-right triangle: suited hands (e.g., AKs, AQs)
                    card1, card2 = f"{rank1}s", f"{rank2}h"  # Suited
                else:
                    # Bottom-left triangle: off-suit hands (e.g., KAs, QAs)
                    card1, card2 = f"{rank2}s", f"{rank1}d"  # Off-suit

                # Get action probability for this hand
                prob = self._get_preflop_action_probability(card1, card2, seat, metric)
                row.append(f"{prob:>2}")

            grid.append(" ".join(row))

        return "\n".join(grid)

    def _get_preflop_action_probability(
        self, card1: str, card2: str, seat: int, metric: str
    ) -> str:
        """Get action probability for a specific hole card combination.

        metric: "allin" or "fold"
        """
        # Create a minimal game state for preflop small blind (first to act) play
        from ..env.types import GameState, PlayerState

        # Create players
        p0 = PlayerState(stack=self.cfg.stack)
        p1 = PlayerState(stack=self.cfg.stack)

        # In heads-up, the small blind is on the button and acts first preflop.
        # Make the analyzed seat both button and small blind, and set it to act.
        button = seat
        p_sb = button
        p_bb = 1 - p_sb

        # Post blinds
        p_states = [p0, p1]
        p_states[p_sb].stack -= self.cfg.sb  # small blind
        p_states[p_sb].committed += self.cfg.sb
        p_states[p_bb].stack -= self.cfg.bb  # big blind
        p_states[p_bb].committed += self.cfg.bb

        # Set hole cards for the seat we're analyzing
        p_states[seat].hole_cards = [
            self._card_str_to_int(card1),
            self._card_str_to_int(card2),
        ]

        # Create game state with the small blind (our seat) first to act
        state = GameState(
            button=button,
            street="preflop",
            deck=[],  # Not needed for this analysis
            board=[],
            pot=150,
            to_act=seat,
            small_blind=50,
            big_blind=100,
            min_raise=100,
            last_aggressive_amount=100,
            players=(p_states[0], p_states[1]),
            terminal=False,
        )

        # Encode state
        cards = self.cards_encoder.encode_cards(state, seat=seat)
        actions_tensor = self.actions_encoder.encode_actions(
            state, seat=seat, num_bet_bins=self.num_bet_bins
        )

        # Move tensors to device
        cards = cards.to(self.device)
        actions_tensor = actions_tensor.to(self.device)

        # Get model prediction
        with torch.no_grad():
            logits, _ = self.model(cards.unsqueeze(0), actions_tensor.unsqueeze(0))

            # Preflop small blind (first to act) legal actions based on to_call
            legal_mask = torch.zeros(self.num_bet_bins)
            me = seat
            opp = 1 - me
            to_call = state.players[opp].committed - state.players[me].committed
            if to_call > 0:
                # Facing a bet (the big blind): fold, call, raises, all-in
                legal_mask[0] = 1.0  # fold
                legal_mask[1] = 1.0  # call
            else:
                # No bet to call: check, bets, all-in
                legal_mask[1] = 1.0  # check
            # Enable all configured bet/raise bins (2..6) and all-in (7)
            for idx in range(2, self.num_bet_bins):
                legal_mask[idx] = 1.0
            legal_mask = legal_mask.to(self.device)  # Move legal mask to device

            # Apply legal mask
            masked_logits = logits.clone()
            masked_logits[0, legal_mask == 0] = -1e9

            # Get probabilities
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)

            # Select metric index
            if metric == "allin":
                idx = 7
            elif metric == "fold":
                idx = 0
            else:
                idx = 7

            selected_prob = probs[idx].item()

            # Convert to percentage and format
            percentage = round(selected_prob * 100)
            if percentage >= 100:
                return "██"  # Two filled boxes to keep 2-char width
            return f"{percentage:2d}"

    def _card_str_to_int(self, card_str: str) -> int:
        """Convert card string (e.g., 'As', 'Kh') to integer representation."""
        rank_map = {
            "A": 12,
            "K": 11,
            "Q": 10,
            "J": 9,
            "T": 8,
            "9": 7,
            "8": 6,
            "7": 5,
            "6": 4,
            "5": 3,
            "4": 2,
            "3": 1,
            "2": 0,
        }
        suit_map = {"s": 0, "h": 1, "d": 2, "c": 3}

        rank = card_str[0]
        suit = card_str[1]

        return rank_map[rank] * 4 + suit_map[suit]
