from __future__ import annotations

import copy
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2.core.structured_config import Config, LrSchedule, ModelType
from p2.env.aggression_analyzer import AggressionAnalyzer
from p2.env.card_utils import NUM_HANDS, combo_suit_permutation_tensor
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp import RebelFFN
from p2.models.mlp.better_features import context_length
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.better_trm import BetterTRM
from p2.models.model_output import ModelOutput, TRMLatent
from p2.rl.losses import RebelSupervisedLoss
from p2.rl.pbs_pool import PBSPool
from p2.rl.rebel_batch import RebelBatch
from p2.rl.rebel_replay import RebelPolicyBuffer, RebelValueBuffer
from p2.search.cfr_evaluator import CFREvaluator
from p2.search.rebel_cfr_evaluator import T_WARM, RebelCFREvaluator
from p2.search.rebel_data_generator import RebelDataGenerator
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator
from p2.utils.ema_helper import EMAHelper
from p2.utils.profiling import profile

STREETS = ["preflop", "flop", "turn", "river", "showdown"]


@dataclass
class EMAContext:
    helper: EMAHelper
    model: BaseMLPModel


class RebelCFRTrainer:
    """Trainer that couples DCFR search with a ReBeL-style FFN."""

    cfr_evaluator: CFREvaluator

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.rng = torch.Generator(device=self.device)
        self.float_dtype = torch.float32
        self.search_cfg = cfg.search
        self.bet_bins = cfg.env.bet_bins
        self.num_bet_bins = len(self.bet_bins) + 3
        self.batch_size = cfg.train.batch_size
        self.buffer_device = torch.device("cpu")
        self.buffer_rng = torch.Generator(device=self.buffer_device)
        if cfg.seed is not None:
            self.rng.manual_seed(int(cfg.seed))
            self.buffer_rng.manual_seed(int(cfg.seed))
        self.num_actions = len(self.bet_bins) + 3
        self.num_players = 2

        if cfg.model.num_actions != self.num_actions:
            print(
                f"[RebelCFRTrainer] Overriding model.num_actions "
                f"({cfg.model.num_actions}) -> {self.num_actions} "
                f"to match bet bin configuration."
            )
            cfg.model.num_actions = self.num_actions

        # Environment used to provide root states for CFR search
        self.env = HUNLTensorEnv(
            num_envs=self.cfg.num_envs,
            starting_stack=cfg.env.stack,
            sb=cfg.env.sb,
            bb=cfg.env.bb,
            default_bet_bins=self.bet_bins,
            device=self.device,
            float_dtype=self.float_dtype,
            flop_showdown=cfg.env.flop_showdown,
        )
        self.env.reset()

        # Model
        if cfg.model.name == ModelType.better_ffn:
            self.model = BetterFFN(
                num_actions=self.num_actions,
                hidden_dim=cfg.model.hidden_dim,
                range_hidden_dim=cfg.model.range_hidden_dim,
                ffn_dim=cfg.model.ffn_dim,
                num_hidden_layers=cfg.model.num_hidden_layers,
                num_policy_layers=cfg.model.num_policy_layers,
                num_value_layers=cfg.model.num_value_layers,
                num_players=self.num_players,
                shared_trunk=cfg.model.shared_trunk,
                enforce_zero_sum=cfg.model.enforce_zero_sum,
                nonlinearity=cfg.model.nonlinearity,
            )
            num_context_features = context_length(self.num_players)
        elif cfg.model.name == ModelType.better_trm:
            self.model = BetterTRM(
                num_actions=self.num_actions,
                hidden_dim=cfg.model.hidden_dim,
                range_hidden_dim=cfg.model.range_hidden_dim,
                ffn_dim=cfg.model.ffn_dim,
                num_hidden_layers=cfg.model.num_hidden_layers,
                num_policy_layers=cfg.model.num_policy_layers,
                num_value_layers=cfg.model.num_value_layers,
                num_players=self.num_players,
                num_recursions=cfg.model.num_recursions,
                num_iterations=cfg.model.num_iterations,
                shared_trunk=cfg.model.shared_trunk,
                enforce_zero_sum=cfg.model.enforce_zero_sum,
                nonlinearity=cfg.model.nonlinearity,
            )
            num_context_features = context_length(self.num_players)
        else:
            self.model = RebelFFN(
                input_dim=cfg.model.input_dim,
                num_actions=self.num_actions,
                hidden_dim=cfg.model.hidden_dim,
                num_hidden_layers=cfg.model.num_hidden_layers,
                detach_value_head=cfg.model.detach_value_head,
                num_players=self.num_players,
                nonlinearity=cfg.model.nonlinearity,
                enforce_zero_sum=cfg.model.enforce_zero_sum,
            )
            num_context_features = 4

        cpu_rng = torch.Generator(device="cpu")
        if self.cfg.seed is not None:
            cpu_rng.manual_seed(self.cfg.seed)
        self.model.init_weights(cpu_rng)
        self.model.to(self.device)
        if self.device.type == "cuda" and cfg.model.compile:
            self.model.compile(dynamic=True)

        # data generation rate per training step
        self.K_value = max(1, self.batch_size // self.cfg.train.value_reuse_goal)
        # approximate number of policy samples when collecting K_value value samples
        policy_decimate = (
            self.num_actions / 2
        ) ** self.cfg.search.depth / self.cfg.train.policy_capacity_factor

        C_over_K = self.cfg.train.replay_buffer_batches
        value_capacity = C_over_K * self.K_value
        policy_capacity = value_capacity * self.cfg.train.policy_capacity_factor

        # Replay buffers
        self.value_buffer = RebelValueBuffer(
            capacity=value_capacity,
            num_actions=self.num_actions,
            num_players=self.num_players,
            num_context_features=num_context_features,
            device=self.buffer_device,
            generator=self.buffer_rng,
        )
        # Larger policy buffer since we store more samples there
        self.policy_buffer = RebelPolicyBuffer(
            capacity=policy_capacity,
            num_actions=self.num_actions,
            num_players=self.num_players,
            num_context_features=num_context_features,
            device=self.buffer_device,
            decimate=1.0 / policy_decimate,
            generator=self.buffer_rng,
        )

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
            permutation_weight=cfg.train.permutation_coef,
            num_players=self.num_players,
        )
        self.grad_clip = cfg.train.grad_clip

        # EMA setup
        self.ema_context = None
        if cfg.train.model_ema is not None:
            self.ema_context = EMAContext(
                helper=EMAHelper(mu=cfg.train.model_ema),
                model=copy.deepcopy(self.model),
            )
            self.ema_context.helper.register(self.ema_context.model)
            self.ema_context.helper.apply_to_module(self.ema_context.model)

        eval_model = (
            self.ema_context.model if self.ema_context is not None else self.model
        )

        if cfg.search.sparse:
            self.cfr_evaluator = SparseCFREvaluator(
                model=eval_model,
                device=self.device,
                cfg=cfg,
                generator=self.rng,
            )
        else:
            self.cfr_evaluator = RebelCFREvaluator(
                search_batch_size=self.cfg.num_envs,
                env_proto=self.env,
                model=eval_model,
                bet_bins=self.bet_bins,
                max_depth=max(1, self.cfg.search.depth),
                cfr_iterations=max(T_WARM + 1, self.cfg.search.iterations),
                device=self.device,
                float_dtype=self.float_dtype,
                generator=self.rng,
                num_supervisions=self.cfg.model.num_supervisions,
                warm_start_iterations=self.cfg.search.warm_start_iterations,
                warm_start_type=self.cfg.search.warm_start_type,
                warm_start_multiplier=self.cfg.search.warm_start_multiplier,
                cfr_type=self.cfg.search.cfr_type,
                cfr_avg=self.cfg.search.cfr_avg,
                cfr_plus=self.cfg.search.cfr_plus,
                dcfr_alpha=self.cfg.search.dcfr_alpha,
                dcfr_beta=self.cfg.search.dcfr_beta,
                dcfr_gamma=self.cfg.search.dcfr_gamma,
                dcfr_alpha_final=self.cfg.search.dcfr_alpha_final,
                dcfr_beta_final=self.cfg.search.dcfr_beta_final,
                dcfr_gamma_final=self.cfg.search.dcfr_gamma_final,
                dcfr_delay=self.cfg.search.dcfr_plus_delay,
                value_targets_from_final_policy=self.cfg.search.value_targets_from_final_policy,
            )
        self.data_generator = RebelDataGenerator(
            env_proto=self.env,
            evaluator=self.cfr_evaluator,
            value_buffer=self.value_buffer,
            policy_buffer=self.policy_buffer,
        )

        self.aggression_analyzer = AggressionAnalyzer(device=self.device)
        self.pbs_pool = PBSPool(pool_size=3, generator=self.rng)

    def _apply_schedules(self, step: int) -> None:
        """Apply learning rate and iteration count schedules."""
        total_steps = max(1, self.cfg.num_steps)
        t = min(1.0, max(0.0, step / float(total_steps)))

        # Learning rate schedule
        lr_start = float(self.cfg.train.learning_rate)
        lr_final = float(self.cfg.train.learning_rate_final)
        if self.cfg.train.lr_schedule == LrSchedule.cosine and lr_final != lr_start:
            lr_now = lr_final + 0.5 * (lr_start - lr_final) * (
                1.0 + math.cos(math.pi * t)
            )
        elif self.cfg.train.lr_schedule == LrSchedule.linear and lr_final != lr_start:
            lr_now = lr_start + (lr_final - lr_start) * t
        else:
            lr_now = lr_start

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_now

        # Iteration count schedule (linear interpolation)
        if self.cfg.search.iterations_final is not None:
            iterations_start = self.cfg.search.iterations
            iterations_final = self.cfg.search.iterations_final
            iterations_now = int(
                round(iterations_start + (iterations_final - iterations_start) * t)
            )
            # Ensure iterations is at least warm_start_iterations + 1
            iterations_now = max(
                self.cfg.search.warm_start_iterations + 1, iterations_now
            )
            self.cfr_evaluator.cfr_iterations = iterations_now

    def _compute_permutation_loss(
        self,
        value_output: ModelOutput,
        value_output_permuted: ModelOutput,
        suit_permutation_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute suit permutation consistency loss."""
        combo_permutations = combo_suit_permutation_tensor(device=self.device)[
            suit_permutation_idxs
        ]
        if (
            value_output.hand_values is None
            or value_output_permuted.hand_values is None
        ):
            raise ValueError("hand_values is None")
        hand_values_permuted_reversed = torch.gather(
            value_output_permuted.hand_values,
            2,
            combo_permutations[:, None, :].expand(-1, self.num_players, -1),
        )
        return F.mse_loss(value_output.hand_values, hand_values_permuted_reversed)

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        eps = 1e-8
        norm = probs.clamp_min(eps)
        entropy = -(norm * norm.log()).sum(dim=-1).mean()
        return float(entropy.item())

    def _compute_metrics(
        self,
        episodes: int,
        updates: int,
        step_stats: dict[str, float],
        value_batch: RebelBatch,
        policy_batch: RebelBatch,
        value_output: ModelOutput,
        policy_output: ModelOutput,
        value_loss_all: torch.Tensor,
        policy_loss_all: torch.Tensor,
        fresh_value_loss: float | None = None,
        fresh_value_batch: RebelBatch | None = None,
    ) -> dict[str, int | float | torch.Tensor | dict[str, int | float]]:
        grad_norm_clipped = torch.nn.utils.get_total_norm(
            p.grad for p in self.model.parameters() if p.grad is not None
        ).item()

        def by_street(
            tensor: torch.Tensor, batch=value_batch, street=None, weights=None
        ) -> dict[str, float]:
            if street is None:
                street = batch.features.street
            masks = {street_name: street == i for i, street_name in enumerate(STREETS)}
            if weights is not None:
                result = {
                    k: (tensor[mask] * weights[mask]).sum(dim=-1)
                    / weights[mask].sum(dim=-1).mean().item()
                    for k, mask in masks.items()
                }
            else:
                result = {k: tensor[mask].mean().item() for k, mask in masks.items()}
            return {k: v for k, v in result.items() if not math.isnan(v)}

        def street_count(street: torch.Tensor) -> dict[str, float]:
            return {
                street_name: (street == i).sum().item()
                for i, street_name in enumerate(STREETS)
            }

        value_buffer_streets_stats = street_count(
            self.value_buffer.features.street[: len(self.value_buffer)]
        )

        metrics: dict[str, int | float | torch.Tensor | dict[str, int | float]] = {
            "episodes": episodes,
            "updates": updates,
            "loss": step_stats["total_loss"] / episodes,
            "policy_loss": step_stats["policy_loss"] / episodes,
            "value_loss": step_stats["value_loss"] / episodes,
            "entropy_loss": step_stats["entropy_loss"] / episodes,
            "permutation_loss": step_stats["permutation_loss"] / episodes,
            "param_update_norm": step_stats["update_norm"] / episodes,
            "value_buffer": value_buffer_streets_stats,
            "value_buffer_size": len(self.value_buffer),
            "policy_buffer_size": len(self.policy_buffer),
            "value_buffer_mean_sample_count": (
                self.value_buffer.sample_count[: len(self.value_buffer)]
                .float()
                .mean()
                .item()
                if len(self.value_buffer) > 0
                else 0.0
            ),
            "value_buffer_target_mean_abs": (
                self.value_buffer.value_targets[: len(self.value_buffer)]
                * self.value_buffer.features.beliefs[: len(self.value_buffer)].view(
                    -1, 2, NUM_HANDS
                )
            )
            .abs()
            .sum(dim=2)
            .mean()
            .item(),
            "value_buffer_target_mean_abs_street": by_street(
                (
                    self.value_buffer.value_targets[: len(self.value_buffer)]
                    * self.value_buffer.features.beliefs[: len(self.value_buffer)].view(
                        -1, 2, NUM_HANDS
                    )
                )
                .abs()
                .sum(dim=2)
                .mean(dim=1),
                street=self.value_buffer.features.street[: len(self.value_buffer)],
            ),
            "policy_buffer_mean_sample_count": (
                self.policy_buffer.sample_count[: len(self.policy_buffer)]
                .float()
                .mean()
                .item()
                if len(self.policy_buffer) > 0
                else 0.0
            ),
            "grad_norm_clipped": grad_norm_clipped,
            "aggression_stats": {
                f"chunk_{i}": v
                for i, v in enumerate(
                    self.aggression_analyzer.analyze_batch(
                        policy_batch, max_batch_size=self.batch_size
                    )["group_avg_bets"].tolist()
                )
            },
            "value_batch_street": street_count(value_batch.features.street),
            "value_loss_street": by_street(value_loss_all),
            "policy_loss_street": by_street(policy_loss_all, batch=policy_batch),
            "value_mean_std": value_output.value.std(dim=0).mean()
            if value_output.value is not None
            else 0.0,
            **self.cfr_evaluator.stats,
        }

        if value_batch.value_targets is not None:
            metrics["batch_value_target_mean_abs"] = (
                value_batch.value_targets.abs().mean().item()
            )
            metrics["batch_value_target_std"] = value_batch.value_targets.std().item()

        # Calculate loss on fresh data
        if fresh_value_batch:
            with torch.no_grad():
                self.model.eval()
                fresh_value_batch = fresh_value_batch.to(self.device)
                fresh_model_output = self.model.repeat(
                    fresh_value_batch.features,
                    count=self.cfg.model.num_supervisions,
                    include_policy=False,
                )
                fresh_loss_dict = self.loss_fn(fresh_model_output, fresh_value_batch)
                metrics["fresh_value_loss"] = fresh_loss_dict["value_loss"]

                if self.ema_context is not None:
                    self.ema_context.model.eval()
                    fresh_model_avg_output = self.ema_context.model.repeat(
                        fresh_value_batch.features,
                        count=self.cfg.model.num_supervisions,
                        include_policy=False,
                    )
                    metrics["fresh_value_loss_avg"] = self.loss_fn(
                        fresh_model_avg_output, fresh_value_batch
                    )["value_loss"]

                    model_avg_output = self.ema_context.model.repeat(
                        value_batch.features,
                        count=self.cfg.model.num_supervisions,
                        include_policy=False,
                    )
                    metrics["value_loss_avg"] = self.loss_fn(
                        model_avg_output, value_batch
                    )["value_loss"]

        if (
            fresh_value_batch is not None
            and fresh_value_batch.value_targets is not None
        ):
            metrics["fresh_value_batch_street"] = street_count(
                fresh_value_batch.features.street
            )
            metrics["fresh_value_target_mean_abs"] = (
                (
                    fresh_value_batch.value_targets
                    * fresh_value_batch.features.beliefs.view(-1, 2, NUM_HANDS)
                )
                .abs()
                .sum(dim=2)
                .mean()
                .item()
            )
            metrics["fresh_value_target_mean_abs_street"] = by_street(
                (
                    fresh_value_batch.value_targets
                    * fresh_value_batch.features.beliefs.view(-1, 2, NUM_HANDS)
                )
                .abs()
                .sum(dim=2)
                .mean(dim=1),
                batch=fresh_value_batch,
            )
        return metrics

    def _get_stratify_streets(self, step: int) -> list[float] | None:
        configs = self.cfg.train.stratify_streets
        if not configs:
            return None

        # Flat until the first threshold
        if step < configs[0].threshold:
            return configs[0].probabilities

        # Find the two thresholds that bracket the current step
        for i in range(len(configs) - 1):
            if configs[i].threshold <= step < configs[i + 1].threshold:
                # Linear interpolation between thresholds
                lower_threshold = configs[i].threshold
                upper_threshold = configs[i + 1].threshold
                lower_probs = configs[i].probabilities
                upper_probs = configs[i + 1].probabilities

                # Compute interpolation weight (0 at lower, 1 at upper)
                alpha = (step - lower_threshold) / (upper_threshold - lower_threshold)

                # Linearly interpolate each probability
                interpolated = [
                    lower * (1 - alpha) + upper * alpha
                    for lower, upper in zip(lower_probs, upper_probs)
                ]
                return interpolated

        # Step >= last threshold, return last config's probabilities
        return configs[-1].probabilities

    def _supervise(
        self,
        value_batch: RebelBatch,
        policy_batch: RebelBatch,
        permuted_batch: RebelBatch,
        suit_permutations_idxs: torch.Tensor,
        value_latent: TRMLatent | None,
        policy_latent: TRMLatent | None,
        permuted_latent: TRMLatent | None,
    ) -> tuple[
        dict[str, float],
        ModelOutput,
        ModelOutput,
        ModelOutput,
        torch.Tensor,
        torch.Tensor,
    ]:
        self.optimizer.zero_grad()

        value_loss, policy_loss, entropy_loss = None, None, None
        value_loss_update, policy_loss_update = None, None
        permutation_loss = 0.0

        if isinstance(self.model, BetterTRM):
            value_output_orig = self.model(
                value_batch.features,
                include_policy=False,
                latent=value_latent,
            )
            # Run model on permuted inputs [model(permute(features))]
            value_output_permuted = self.model(
                permuted_batch.features,
                include_policy=False,
                latent=permuted_latent,
            )
        else:
            value_output_orig = self.model(value_batch.features, include_policy=False)
            value_output_permuted = self.model(
                permuted_batch.features, include_policy=False
            )

        loss_dict = self.loss_fn(value_output_permuted, permuted_batch)
        value_loss = loss_dict["value_loss"]
        value_loss_update = loss_dict["value_loss_all"]
        total_loss = loss_dict["total_loss"]

        permutation_loss_tensor = self._compute_permutation_loss(
            value_output_orig, value_output_permuted, suit_permutations_idxs
        )
        permutation_loss = permutation_loss_tensor.item()
        total_loss = (
            total_loss + self.loss_fn.permutation_weight * permutation_loss_tensor
        )

        if isinstance(self.model, BetterTRM):
            policy_output = self.model(
                policy_batch.features, include_policy=True, latent=policy_latent
            )
        else:
            policy_output = self.model(policy_batch.features, include_policy=True)
        loss_dict = self.loss_fn(policy_output, policy_batch)
        policy_loss = loss_dict["policy_loss"]
        policy_loss_update = loss_dict["policy_loss_all"]
        entropy_loss = loss_dict["entropy"]
        total_loss += loss_dict["total_loss"]

        total_loss.backward()

        assert all(
            p.grad.isfinite().all()
            for p in self.model.parameters()
            if p.grad is not None
        ), "NaN/Inf in model gradients"

        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Store parameters before optimizer step to compute update norm
        params_before = [p.clone() for p in self.model.parameters()]
        self.optimizer.step()

        # Update EMA if enabled
        if self.ema_context is not None:
            self.ema_context.helper.update(self.model)
            self.ema_context.helper.apply_to_module(self.ema_context.model)

        # Compute parameter update norm using torch utility
        updates = (
            p_after - p_before
            for p_before, p_after in zip(params_before, self.model.parameters())
        )
        update_norm = torch.nn.utils.get_total_norm(updates).item()

        return (
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy_loss": entropy_loss,
                "permutation_loss": permutation_loss,
                "total_loss": total_loss.item(),
                "update_norm": update_norm,
            },
            value_output_permuted,
            value_output_orig,
            policy_output,
            value_loss_update,
            policy_loss_update,
        )

    @profile
    def _update_model(
        self, step: int
    ) -> dict[str, int | float | torch.Tensor | dict[str, int | float]]:
        fresh_value_batch, fresh_policy_batch = self.data_generator.generate_data(
            self.K_value
        )

        # Warmup: make sure we have enough samples.
        while min(len(self.value_buffer), len(self.policy_buffer)) < self.batch_size:
            self.data_generator.generate_data(self.K_value)

        value_fullness = len(self.value_buffer) / self.value_buffer.capacity
        episodes = math.ceil(self.cfg.train.episodes_per_step * value_fullness)
        supervisions = (
            self.cfg.model.num_supervisions if isinstance(self.model, BetterTRM) else 1
        )
        updates = episodes * supervisions
        value_batch_all = []
        policy_batch_all = []
        value_output_all = []
        policy_output_all = []
        value_loss_update_all = []
        policy_loss_update_all = []
        step_stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "permutation_loss": 0.0,
            "total_loss": 0.0,
            "update_norm": 0.0,
        }
        stratify = self._get_stratify_streets(step)

        self.model.train()
        for episode in range(episodes):
            value_latent, policy_latent, permuted_latent = None, None, None
            # TODO: think about how to interleave these/ratio in a smarter way.
            # Might need to use different sizes for the two batches.
            value_batch = self.value_buffer.sample(
                self.batch_size, stratify_streets=stratify
            ).to(self.device)
            policy_batch = self.policy_buffer.sample(
                self.batch_size, stratify_streets=stratify
            ).to(self.device)

            # Sample suit permutations and apply to features/targets together.
            permuted_batch, suit_permutations_idxs = value_batch.with_permuted_targets(
                generator=self.rng, num_players=self.num_players
            )

            for _ in range(supervisions):
                (
                    episode_stats,
                    permuted_value_output,
                    value_output_orig,
                    policy_output,
                    value_loss_update,
                    policy_loss_update,
                ) = self._supervise(
                    value_batch,
                    policy_batch,
                    permuted_batch,
                    suit_permutations_idxs,
                    value_latent,
                    policy_latent,
                    permuted_latent,
                )
                value_latent = (
                    value_output_orig.latent.detach()
                    if value_output_orig.latent is not None
                    else None
                )
                policy_latent = (
                    policy_output.latent.detach()
                    if policy_output.latent is not None
                    else None
                )
                permuted_latent = (
                    permuted_value_output.latent.detach()
                    if permuted_value_output.latent is not None
                    else None
                )

            # Keep track of last supervision stats.
            for k in step_stats:
                step_stats[k] += episode_stats[k]

            # Append last batch/output for metrics.
            value_batch_all.append(permuted_batch)
            policy_batch_all.append(policy_batch)
            value_output_all.append(permuted_value_output)
            policy_output_all.append(policy_output)
            value_loss_update_all.append(value_loss_update)
            policy_loss_update_all.append(policy_loss_update)

        metrics = self._compute_metrics(
            episodes,
            updates,
            step_stats,
            RebelBatch.cat(value_batch_all),
            RebelBatch.cat(policy_batch_all),
            ModelOutput.cat(value_output_all),
            ModelOutput.cat(policy_output_all),
            torch.cat(value_loss_update_all),
            torch.cat(policy_loss_update_all),
            fresh_value_batch=fresh_value_batch,
        )

        return metrics

    def train_step(self, step: int) -> dict[str, Any]:
        step_public = step + 1

        # Apply schedules before training step
        self._apply_schedules(step)

        update_info = self._update_model(step)
        update_info["step"] = step_public
        update_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        update_info["cfr_iterations"] = self.cfr_evaluator.cfr_iterations

        return update_info

    def train(self, num_steps: int | None = None) -> list[dict[str, Any]]:
        total_steps = num_steps or self.cfg.num_steps
        history: list[dict[str, Any]] = []

        for step in range(total_steps):
            update_info = self.train_step(step)
            history.append(update_info)

        return history

    def evaluate_against_pool(self, min_games: int) -> float:
        return self.pbs_pool.evaluate_model_against_pool(self.model, min_games)

    def save_checkpoint(
        self,
        path: str,
        step: int,
        wandb_run_id: str | None = None,
        save_optimizer: bool = True,
        save_dtype: torch.dtype | None = None,
        batch: RebelBatch | None = None,
    ) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Convert model state to bfloat16 if requested
        model_state = self.model.state_dict()
        if save_dtype is not None:
            model_state = {
                k: v.to(save_dtype) if v.dtype.is_floating_point else v
                for k, v in model_state.items()
            }

        state = {
            "model": model_state,
            "step": step,
            "save_dtype": str(save_dtype) if save_dtype is not None else None,
            "config": asdict(self.cfg),
            # Store wandb run ID for resumption
            "wandb_run_id": wandb_run_id,
        }

        # Only save optimizer and RNG state if requested
        if save_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
            state["rng"] = self.rng.get_state()

        # Save EMA state if enabled (only save model_avg, shadow weights can be reconstructed)
        if self.ema_context is not None:
            model_avg_state = self.ema_context.model.state_dict()
            if save_dtype is not None:
                model_avg_state = {
                    k: v.to(save_dtype) if v.dtype.is_floating_point else v
                    for k, v in model_avg_state.items()
                }
            state["model_avg"] = model_avg_state

        # Save batch if provided (move to CPU for storage)
        if batch is not None:
            batch_cpu = batch.to(torch.device("cpu"))
            state["batch"] = batch_cpu

        torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Convert model state back to host dtype if it was saved in bfloat16
        save_dtype_str = ckpt.get("save_dtype")
        model_state = ckpt["model"]
        if save_dtype_str is not None and save_dtype_str != str(self.float_dtype):
            # Convert back to float32 for host dtype
            model_state = {
                k: v.to(self.float_dtype) if v.dtype.is_floating_point else v
                for k, v in model_state.items()
            }

        self.model.load_state_dict(model_state)

        # Load EMA state if it exists in checkpoint and EMA is enabled
        if "model_avg" in ckpt and self.ema_context is not None:
            # Convert model_avg state back to host dtype if needed
            model_avg_state = ckpt["model_avg"]
            if save_dtype_str is not None and save_dtype_str != str(self.float_dtype):
                model_avg_state = {
                    k: v.to(self.float_dtype) if v.dtype.is_floating_point else v
                    for k, v in model_avg_state.items()
                }
            self.ema_context.model.load_state_dict(model_avg_state)
            # Reconstruct ema_helper shadow weights from model_avg
            self.ema_context.helper.register(self.ema_context.model)
            # Update evaluator to use model_avg
            self.cfr_evaluator.model = self.ema_context.model
            self.cfr_evaluator.chance_helper.model = self.ema_context.model

        # Only load optimizer if it exists in checkpoint
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        self.cfg.wandb_run_id = ckpt.get("wandb_run_id")
        # if "rng" in ckpt:
        #     self.rng.set_state(ckpt["rng"].to(self.device))
        return int(ckpt["step"])
