from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from alphaholdem.core.structured_config import Config, ValueHeadType
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp import RebelFFN
from alphaholdem.rl.losses import RebelSupervisedLoss
from alphaholdem.rl.rebel_replay import RebelReplayBuffer
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
from alphaholdem.utils.model_context import model_eval


@dataclass
class TrainerMetrics:
    step: int
    loss: Optional[float] = None
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    buffer_size: Optional[int] = None
    cfr_entropy: Optional[float] = None

    def update(self, dict: Dict[str, float]) -> None:
        self.step = dict.get("step", self.step)
        self.loss = dict.get("loss", self.loss)
        self.policy_loss = dict.get("policy_loss", self.policy_loss)
        self.value_loss = dict.get("value_loss", self.value_loss)
        self.entropy = dict.get("entropy", self.entropy)
        self.buffer_size = dict.get("buffer_size", self.buffer_size)
        self.cfr_entropy = dict.get("cfr_entropy", self.cfr_entropy)


class RebelCFRTrainer:
    """Trainer that couples DCFR search with a ReBeL-style FFN."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.float_dtype = torch.float32
        self.search_cfg = cfg.search
        self.bet_bins = cfg.env.bet_bins
        self.batch_size = cfg.train.batch_size
        self.replay_capacity = self.batch_size * max(1, cfg.train.replay_buffer_batches)
        self.buffer_device = torch.device("cpu")
        self.buffer_rng = torch.Generator(device=self.buffer_device)
        if hasattr(cfg, "seed") and cfg.seed is not None:
            self.rng.manual_seed(int(cfg.seed))
            self.buffer_rng.manual_seed(int(cfg.seed))
        self.num_actions = len(self.bet_bins) + 3
        self.num_players = 2
        self.belief_dim = RebelFeatureEncoder.belief_dim
        if cfg.model.num_actions != self.num_actions:
            print(
                f"[RebelCFRTrainer] Overriding model.num_actions "
                f"({cfg.model.num_actions}) -> {self.num_actions} "
                f"to match bet bin configuration."
            )
            cfg.model.num_actions = self.num_actions

        # Environment used to provide root states for CFR search
        self.env = HUNLTensorEnv(
            num_envs=self.batch_size,
            starting_stack=cfg.env.stack,
            sb=cfg.env.sb,
            bb=cfg.env.bb,
            default_bet_bins=self.bet_bins,
            device=self.device,
            float_dtype=self.float_dtype,
            flop_showdown=cfg.env.flop_showdown,
        )
        self.env.reset()
        self.reach_probs = torch.ones(
            self.batch_size, dtype=self.float_dtype, device=self.device
        )

        # Model
        self.model = RebelFFN(
            input_dim=cfg.model.input_dim,
            num_actions=self.num_actions,
            hidden_dim=cfg.model.hidden_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            detach_value_head=cfg.model.detach_value_head,
            num_players=self.num_players,
        ).to(self.device)
        self.model.init_weights(self.rng)

        # Optimizer & loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.train.learning_rate
        )
        self.loss_fn = RebelSupervisedLoss(
            policy_weight=1.0,
            value_weight=cfg.train.value_coef,
            entropy_coef=cfg.train.entropy_coef,
        )
        self.grad_clip = cfg.train.grad_clip

        # Replay buffer stores samples on CPU
        self.replay_buffer = RebelReplayBuffer(
            capacity=self.replay_capacity,
            feature_dim=cfg.model.input_dim,
            num_actions=self.num_actions,
            belief_dim=self.belief_dim,
            num_players=self.num_players,
            device=self.buffer_device,
            dtype=self.float_dtype,
        )
        self.cfr_manager = self._build_cfr_manager()

        # Feature encoder for belief computation
        self.feature_encoder = RebelFeatureEncoder(
            env=self.env,
            device=self.device,
            dtype=self.float_dtype,
        )

    def _build_cfr_manager(self) -> RebelCFREvaluator:
        return RebelCFREvaluator(
            search_batch_size=self.batch_size,
            env_proto=self.env,
            bet_bins=self.bet_bins,
            max_depth=self.cfg.search.depth,
            device=self.device,
            float_dtype=self.float_dtype,
            belief_samples=max(
                1,
                (
                    self.cfg.search.belief_samples
                    if hasattr(self.cfg.search, "belief_samples")
                    else 1
                ),
            ),
        )

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        eps = 1e-8
        norm = probs.clamp_min(eps)
        entropy = -(norm * norm.log()).sum(dim=-1).mean()
        return float(entropy.item())

    def _reset_done_envs(self) -> None:
        done = torch.where(self.env.done)[0]
        if done.numel() > 0:
            self.env.reset(done)
            self.reach_probs[done] = 1.0

    @torch.no_grad()
    def _self_play_iteration(self) -> Dict[str, float]:
        self._reset_done_envs()
        manager = self.cfr_manager

        # Initialize search with current environment state
        env_indices = torch.arange(self.batch_size, device=self.device)
        manager.initialize_search(self.env, env_indices)

        # Run CFR search with ReBeL-style strategy-conditioned belief updates
        with model_eval(self.model):
            result = manager.run_cfr_iterations(
                num_iterations=self.cfg.search.iterations, value_network=self.model
            )

        # Extract results
        valid_mask = ~self.env.done[env_indices]

        if not torch.any(valid_mask):
            return {"cfr_entropy": 0.0, "num_samples": 0}

        # Use sampled policy for action selection (safe search per ReBeL)
        policy_sampled = result.root_policy_sampled
        policy_avg = result.root_policy_avg

        # Convert to full action space if needed
        # For now, assume we're working with collapsed 4-action space
        policy_full = policy_sampled  # Simplified for integration

        valid_env_indices = env_indices[valid_mask]
        valid_policy = policy_full[valid_mask]

        num_valid = valid_env_indices.numel()

        # Generate features for replay buffer
        features = torch.zeros(
            num_valid,
            self.cfg.model.input_dim,
            dtype=self.float_dtype,
            device=self.device,
        )

        # Encode states for each player
        valid_to_act = self.env.to_act[valid_env_indices]
        for player in (0, 1):
            mask = valid_to_act == player
            if mask.any():
                player_indices = valid_env_indices[mask]
                # Use feature encoder to encode current states
                agents = torch.full(
                    (player_indices.numel(),),
                    player,
                    device=self.device,
                    dtype=torch.long,
                )

                # Get belief states from the CFR result
                beliefs = result.belief_states.get(
                    0,
                    torch.zeros(
                        player_indices.numel(),
                        2,
                        self.belief_dim,
                        device=self.device,
                        dtype=self.float_dtype,
                    ),
                )
                hero_beliefs = beliefs[mask, player] if beliefs.numel() > 0 else None
                opp_beliefs = beliefs[mask, 1 - player] if beliefs.numel() > 0 else None

                encoded = self.feature_encoder.encode(
                    player_indices, agents, hero_beliefs, opp_beliefs
                )
                features[mask] = encoded

        # Get hand values and weights from CFR result
        hand_values = result.training_targets.get(
            "values_depth_0",
            torch.zeros(
                num_valid,
                2,
                self.belief_dim,
                device=self.device,
                dtype=self.float_dtype,
            ),
        )
        hand_weights = torch.ones_like(hand_values)

        # Sample weights from CFR result
        sample_weights = torch.ones(
            num_valid, device=self.device, dtype=self.float_dtype
        )

        state_reach = self.reach_probs[valid_env_indices] * sample_weights

        self.replay_buffer.add_batch(
            features=features.detach().to(self.buffer_device),
            policy_targets=valid_policy.detach().to(self.buffer_device),
            value_targets=hand_values.detach().to(self.buffer_device),
            legal_masks=torch.ones(
                num_valid, self.num_actions, device=self.buffer_device, dtype=torch.bool
            ),
            acting_players=valid_to_act.detach().to(self.buffer_device),
            reach_weights=state_reach.detach().to(self.buffer_device),
            value_weights=hand_weights.detach().to(self.buffer_device),
        )

        # Epsilon-greedy action selection on sampled-iteration policy
        eps = float(self.cfg.train.cfr_action_epsilon)
        # Mix with uniform over legal actions
        legal_float = torch.ones(
            num_valid, self.num_actions, device=self.device, dtype=self.float_dtype
        )
        num_legal = legal_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        uniform = legal_float / num_legal
        mixed = (1.0 - eps) * valid_policy + eps * uniform
        mixed = mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-8)

        # Sample actions for self-play using mixed distribution
        sampled_bins = torch.multinomial(mixed, 1).squeeze(1)
        selected_probs = mixed.gather(1, sampled_bins.unsqueeze(1)).squeeze(1)
        self.reach_probs[valid_env_indices] = self.reach_probs[
            valid_env_indices
        ] * selected_probs.clamp_min(1e-8)

        actions_full = torch.full(
            (self.batch_size,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        actions_full[valid_env_indices] = sampled_bins
        self.env.step_bins(actions_full, bet_bins=self.bet_bins)
        self._reset_done_envs()

        cfr_entropy = self._compute_entropy(valid_policy)
        return {"cfr_entropy": cfr_entropy, "num_samples": num_valid}

    def _update_model(self) -> Optional[Dict[str, float]]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch_cpu = self.replay_buffer.sample(self.batch_size, self.buffer_rng)
        batch = batch_cpu.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(batch.features)
        if output.hand_values is None:
            raise ValueError("ReBeL model must return per-hand values.")
        loss_dict = self.loss_fn(
            logits=output.policy_logits,
            hand_values=output.hand_values,
            batch=batch,
        )
        loss = loss_dict["total_loss"]
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(loss_dict["policy_loss"].item()),
            "value_loss": float(loss_dict["value_loss"].item()),
            "entropy": float(loss_dict["entropy"].item()),
        }

    def train_step(self, step: int) -> TrainerMetrics:
        step_public = step + 1
        metrics = TrainerMetrics(step=step_public)

        iterations = max(1, self.cfg.train.episodes_per_step)
        entropy_weighted = 0.0
        total_samples = 0
        for _ in range(iterations):
            info = self._self_play_iteration()
            samples = info.get("num_samples", 0)
            if samples > 0:
                entropy_weighted += info["cfr_entropy"] * samples
                total_samples += samples

        update_info = self._update_model()

        metrics.buffer_size = len(self.replay_buffer)
        if total_samples > 0:
            metrics.cfr_entropy = entropy_weighted / total_samples
        if update_info is not None:
            metrics.update(update_info)

        return metrics

    def train(self, num_steps: Optional[int] = None) -> list[TrainerMetrics]:
        total_steps = num_steps or self.cfg.num_steps
        history: list[TrainerMetrics] = []

        for step in range(total_steps):
            metrics = self.train_step(step)
            history.append(metrics)

        return history

    def save_checkpoint(self, path: str, step: int) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt["step"])
