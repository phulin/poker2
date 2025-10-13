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
from alphaholdem.search.cfr_manager import CFRManager
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


class RebelCFRTrainer:
    """Trainer that couples DCFR search with a ReBeL-style FFN."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.float_dtype = torch.float32
        self.search_cfg = cfg.search
        self.bet_bins = cfg.env.bet_bins
        self.batch_size = cfg.train.batch_size
        self.replay_capacity = self.batch_size * max(1, cfg.train.replay_buffer_batches)
        self.buffer_device = torch.device("cpu")
        self.buffer_rng = torch.Generator(device=self.buffer_device)
        if hasattr(cfg, "seed") and cfg.seed is not None:
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
        value_head_type = (
            cfg.model.value_head_type
            if isinstance(cfg.model.value_head_type, ValueHeadType)
            else ValueHeadType(cfg.model.value_head_type)
        )
        self.model = RebelFFN(
            input_dim=cfg.model.input_dim,
            num_actions=self.num_actions,
            hidden_dim=cfg.model.hidden_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            value_head_type=value_head_type.value,
            value_head_num_quantiles=cfg.model.value_head_num_quantiles,
            detach_value_head=cfg.model.detach_value_head,
            belief_dim=self.belief_dim,
            num_players=self.num_players,
        ).to(self.device)
        cpu_rng = torch.Generator(device=torch.device("cpu"))
        self.model.init_weights(cpu_rng)

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

    def _build_cfr_manager(self) -> CFRManager:
        return CFRManager(
            batch_size=self.batch_size,
            env_proto=self.env,
            bet_bins=self.bet_bins,
            sequence_length=self.cfg.train.max_sequence_length,
            device=self.device,
            float_dtype=self.float_dtype,
            cfg=self.search_cfg,
            popart_normalizer=None,
            use_rebel_features=True,
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
        manager.reset_for_new_search()
        env_indices = torch.arange(self.batch_size, device=self.device)
        roots = manager.seed_roots(self.env, env_indices, src_tokens=None)

        with model_eval(self.model):
            result = manager.run_search(self.model)

        to_act_all = manager.env.to_act[roots]
        legal_full_all = manager.legal_mask_full(roots)
        done_mask = manager.env.done[roots]
        valid_mask = legal_full_all.any(dim=1) & (~done_mask)

        if not torch.any(valid_mask):
            return {"cfr_entropy": 0.0, "num_samples": 0}

        policy_collapsed_all = (
            result.root_policy_avg_collapsed
            if result.root_policy_avg_collapsed is not None
            else result.root_policy_collapsed
        )
        policy_full_all = CFRManager.expand_collapsed_to_full(
            policy_collapsed_all,
            legal_full_all,
        )
        policy_full_all = policy_full_all * legal_full_all.to(policy_full_all.dtype)
        sums = policy_full_all.sum(dim=1, keepdim=True)
        zero_rows = sums.squeeze(1) <= 0
        if zero_rows.any():
            legal_float = legal_full_all[zero_rows].to(policy_full_all.dtype)
            legal_sums = legal_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            policy_full_all[zero_rows] = legal_float / legal_sums
            sums = policy_full_all.sum(dim=1, keepdim=True)
        policy_full_all = policy_full_all / sums.clamp_min(1e-8)

        hand_values_all = (
            result.root_hand_values
            if result.root_hand_values is not None
            else torch.zeros(
                policy_full_all.shape[0],
                self.num_players,
                self.belief_dim,
                dtype=self.float_dtype,
                device=self.device,
            )
        )
        hand_weights_all = (
            result.root_hand_value_weights
            if result.root_hand_value_weights is not None
            else torch.zeros_like(hand_values_all)
        )
        sample_weights_all = (
            result.root_sample_weights
            if result.root_sample_weights is not None
            else torch.ones(
                policy_full_all.shape[0], dtype=self.float_dtype, device=self.device
            )
        )

        valid_env_indices = env_indices[valid_mask]
        valid_roots = roots[valid_mask]
        valid_to_act = to_act_all[valid_mask]
        valid_policy_full = policy_full_all[valid_mask]
        valid_policy_collapsed = policy_collapsed_all[valid_mask]
        valid_legal_full = legal_full_all[valid_mask]
        valid_hand_values = hand_values_all[valid_mask]
        valid_hand_weights = hand_weights_all[valid_mask]
        valid_sample_weights = sample_weights_all[valid_mask]

        num_valid = valid_env_indices.numel()
        features = torch.zeros(
            num_valid,
            self.cfg.model.input_dim,
            dtype=self.float_dtype,
            device=self.device,
        )

        for player in (0, 1):
            mask = valid_to_act == player
            if mask.any():
                idxs = valid_roots[mask]
                encoded = manager.encode_states(player, idxs)
                features[mask] = encoded

        state_reach = self.reach_probs[valid_env_indices] * valid_sample_weights

        self.replay_buffer.add_batch(
            features=features.detach().to(self.buffer_device),
            policy_targets=valid_policy_full.detach().to(self.buffer_device),
            value_targets=valid_hand_values.detach().to(self.buffer_device),
            legal_masks=valid_legal_full.detach().to(self.buffer_device),
            acting_players=valid_to_act.detach().to(self.buffer_device),
            reach_weights=state_reach.detach().to(self.buffer_device),
            value_weights=valid_hand_weights.detach().to(self.buffer_device),
        )

        # Sample actions for self-play using full policy distribution
        sampled_bins = torch.multinomial(valid_policy_full, 1).squeeze(1)
        selected_probs = valid_policy_full.gather(1, sampled_bins.unsqueeze(1)).squeeze(
            1
        )
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

        cfr_entropy = self._compute_entropy(valid_policy_collapsed)
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
            metrics.loss = update_info["loss"]
            metrics.policy_loss = update_info["policy_loss"]
            metrics.value_loss = update_info["value_loss"]
            metrics.entropy = update_info["entropy"]

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
