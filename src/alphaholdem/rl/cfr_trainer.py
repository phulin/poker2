from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from alphaholdem.core.structured_config import Config
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp import RebelFFN
from alphaholdem.rl.losses import RebelSupervisedLoss
from alphaholdem.rl.rebel_replay import RebelReplayBuffer
from alphaholdem.search.rebel_cfr_evaluator import T_WARM, RebelCFREvaluator
from alphaholdem.search.rebel_data_generator import RebelDataGenerator
from alphaholdem.utils.profiling import profile


class RebelCFRTrainer:
    """Trainer that couples DCFR search with a ReBeL-style FFN."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.float_dtype = torch.float32
        self.search_cfg = cfg.search
        self.bet_bins = cfg.env.bet_bins
        self.num_bet_bins = len(self.bet_bins) + 3
        self.batch_size = cfg.train.batch_size
        self.replay_capacity = self.batch_size * max(1, cfg.train.replay_buffer_batches)
        self.buffer_device = torch.device("cpu")
        self.buffer_rng = torch.Generator(device=self.buffer_device)
        if cfg.seed is not None:
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
        expected_input_dim = RebelFeatureEncoder.feature_dim
        if cfg.model.input_dim != expected_input_dim:
            print(
                f"[RebelCFRTrainer] Overriding model.input_dim "
                f"({cfg.model.input_dim}) -> {expected_input_dim} "
                f"to match feature encoder output."
            )
            cfg.model.input_dim = expected_input_dim

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

        self.value_buffer = RebelReplayBuffer(
            capacity=self.replay_capacity,
            feature_dim=cfg.model.input_dim,
            num_actions=self.num_actions,
            num_players=self.num_players,
            device=self.buffer_device,
            policy_targets=False,
        )
        self.policy_buffer = RebelReplayBuffer(
            capacity=self.replay_capacity,
            feature_dim=cfg.model.input_dim,
            num_actions=self.num_actions,
            num_players=self.num_players,
            device=self.buffer_device,
            policy_targets=True,
            value_targets=False,
        )

        # Model
        self.model = RebelFFN(
            input_dim=cfg.model.input_dim,
            num_actions=self.num_actions,
            hidden_dim=cfg.model.hidden_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            detach_value_head=cfg.model.detach_value_head,
            num_players=self.num_players,
        ).compile()
        cpu_rng = torch.Generator(device="cpu")
        if self.cfg.seed is not None:
            cpu_rng.manual_seed(self.cfg.seed)
        self.model.init_weights(cpu_rng)
        self.model.to(self.device)

        # Optimizer & loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
        self.loss_fn = RebelSupervisedLoss(
            policy_weight=1.0,
            value_weight=cfg.train.value_coef,
            entropy_coef=cfg.train.entropy_coef,
        )
        self.grad_clip = cfg.train.grad_clip

        self.cfr_evaluator = RebelCFREvaluator(
            search_batch_size=self.cfg.num_envs,
            env_proto=self.env,
            model=self.model,
            bet_bins=self.bet_bins,
            max_depth=max(1, self.cfg.search.depth),
            cfr_iterations=max(T_WARM + 1, self.cfg.search.iterations),
            device=self.device,
            float_dtype=self.float_dtype,
            generator=self.rng,
            warm_start_iterations=self.cfg.search.warm_start_iterations,
            linear_cfr=self.cfg.search.linear_cfr,
            cfr_avg=self.cfg.search.cfr_avg,
        )
        self.data_generator = RebelDataGenerator(
            env_proto=self.env,
            evaluator=self.cfr_evaluator,
            value_buffer=self.value_buffer,
            policy_buffer=self.policy_buffer,
        )

        # Feature encoder for belief computation
        self.feature_encoder = RebelFeatureEncoder(
            env=self.env,
            device=self.device,
            dtype=self.float_dtype,
        )

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        eps = 1e-8
        norm = probs.clamp_min(eps)
        entropy = -(norm * norm.log()).sum(dim=-1).mean()
        return float(entropy.item())

    @profile
    def _update_model(self) -> Optional[Dict[str, float]]:
        self.data_generator.generate_data()

        # TODO: think about how to interleave these/ratio in a smarter way.
        # Might need to use different sizes for the two buffers.
        value_batch = self.value_buffer.sample(
            self.batch_size, generator=self.buffer_rng
        ).to(self.device)
        policy_batch = self.policy_buffer.sample(
            self.batch_size, generator=self.buffer_rng
        ).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        value_loss, policy_loss = None, None
        for batch in [value_batch, policy_batch]:
            output = self.model(batch.features)
            loss_dict = self.loss_fn(
                logits=output.policy_logits,
                hand_values=output.hand_values,
                batch=batch,
            )
            loss = loss_dict["total_loss"]
            if batch is value_batch:
                value_loss = loss_dict["value_loss"]
            else:
                policy_loss = loss_dict["policy_loss"]
                entropy_loss = loss_dict["entropy"]
            loss.backward()

        grad_norm_unclipped = (
            sum(
                p.grad.norm(2) ** 2
                for p in self.model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )

        preflop = value_batch.statistics["street"] == 0
        flop = value_batch.statistics["street"] == 1
        turn = value_batch.statistics["street"] == 2
        river = value_batch.statistics["street"] == 3
        showdown = value_batch.statistics["street"] == 4

        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "value_buffer_size": len(self.value_buffer),
            "policy_buffer_size": len(self.policy_buffer),
            "grad_norm_unclipped": grad_norm_unclipped,
            "value_batch_street": {
                "preflop": preflop.float().mean().item(),
                "flop": flop.float().mean().item(),
                "turn": turn.float().mean().item(),
                "river": river.float().mean().item(),
                "showdown": showdown.float().mean().item(),
            },
            **self.cfr_evaluator.stats,
        }

    def train_step(self, step: int) -> Dict[str, any]:
        step_public = step + 1

        update_info = self._update_model()
        update_info["step"] = step_public

        return update_info

    def train(self, num_steps: Optional[int] = None) -> List[Dict[str, any]]:
        total_steps = num_steps or self.cfg.num_steps
        history: List[Dict[str, any]] = []

        for step in range(total_steps):
            update_info = self.train_step(step)
            history.append(update_info)

        return history

    def save_checkpoint(
        self, path: str, step: int, wandb_run_id: Optional[str] = None
    ) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng.get_state(),
            "step": step,
            # Store wandb run ID for resumption
            "wandb_run_id": wandb_run_id,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.cfg.wandb_run_id = ckpt["wandb_run_id"]
        # self.rng.set_state(ckpt["rng"].to(self.device))
        return int(ckpt["step"])
