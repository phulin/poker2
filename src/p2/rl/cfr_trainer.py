from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2.core.structured_config import Config, LrSchedule, ModelType
from p2.env.aggression_analyzer import AggressionAnalyzer
from p2.env.card_utils import (
    NUM_HANDS,
    combo_suit_permutation_tensor,
    suit_permutations_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp import RebelFFN
from p2.models.mlp.better_features import context_length
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.better_trm import BetterTRM
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput, TRMLatent
from p2.rl.losses import RebelSupervisedLoss
from p2.rl.optimizers import build_optimizer
from p2.rl.rebel_batch import RebelBatch
from p2.rl.trueskill_tracker import TrueSkillTracker
from p2.rl.rebel_replay import RebelPolicyBuffer, RebelValueBuffer
from p2.search.cfr_evaluator import CFREvaluator
from p2.search.rebel_cfr_evaluator import T_WARM, RebelCFREvaluator
from p2.search.rebel_data_generator import RebelDataGenerator
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator
from p2.utils.ema_helper import EMAHelper
from p2.utils.profiling import profile

STREETS = ["preflop", "flop", "turn", "river", "showdown"]


def _value_samples_per_step(batch_size: int, value_reuse_goal: float) -> int:
    if value_reuse_goal <= 0:
        raise ValueError(
            f"train.value_reuse_goal must be positive; got {value_reuse_goal}"
        )
    return max(1, int(round(batch_size / value_reuse_goal)))


def _scheduled_learning_rate(
    step: int,
    total_steps: int,
    lr_start: float,
    lr_final: float,
    lr_schedule: LrSchedule,
    warmup_steps: int = 0,
) -> float:
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return lr_start * float(step + 1) / float(warmup_steps)

    if warmup_steps > 0:
        decay_steps = max(1, total_steps - warmup_steps)
        t = min(1.0, max(0.0, (step - warmup_steps) / float(decay_steps)))
    else:
        t = min(1.0, max(0.0, step / float(max(1, total_steps))))

    if lr_schedule == LrSchedule.cosine and lr_final != lr_start:
        return lr_final + 0.5 * (lr_start - lr_final) * (1.0 + math.cos(math.pi * t))
    if lr_schedule == LrSchedule.linear and lr_final != lr_start:
        return lr_start + (lr_final - lr_start) * t
    return lr_start


def _compile_setting(cfg: Config) -> str:
    value = str(cfg.model.compile).strip().lower()
    if value in {"0", "false", "no", "none"}:
        return "off"
    if value in {"", "true", "yes", "1"}:
        return "default"
    if value not in {"off", "default", "max-autotune"}:
        raise ValueError(
            "model.compile must be one of: off, default, max-autotune; "
            f"got {cfg.model.compile!r}"
        )
    return value


def _compile_kwargs(cfg: Config) -> dict[str, object]:
    kwargs: dict[str, object] = {"dynamic": True}
    mode = _compile_setting(cfg)
    if mode == "max-autotune":
        kwargs["mode"] = mode
    return kwargs


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
        buffer_device = cfg.train.replay_buffer_device.lower()
        if buffer_device in {"cuda", "gpu", "device"} and device.type == "cuda":
            self.buffer_device = device
        elif buffer_device == "device":
            self.buffer_device = device
        else:
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
            randomize_stacks=cfg.env.randomize_stacks,
            stack_mode=cfg.env.stack_mode,
            min_stack_bb=cfg.env.min_stack_bb,
            mid_stack_bb=cfg.env.mid_stack_bb,
            max_stack_bb=cfg.env.max_stack_bb,
            high_stack_mass_ratio=cfg.env.high_stack_mass_ratio,
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
                board_interaction_dim=cfg.model.board_interaction_dim,
                policy_rank=cfg.model.policy_rank,
                policy_hand_bias_rank=cfg.model.policy_hand_bias_rank,
                policy_factor_scale=cfg.model.policy_factor_scale,
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
        if self.device.type == "cuda" and _compile_setting(cfg) != "off":
            self.model.compile_forward_modes(**_compile_kwargs(cfg))

        # data generation rate per training step
        self.K_value = _value_samples_per_step(
            self.batch_size, self.cfg.train.value_reuse_goal
        )
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
            depth_stratify_decimate=cfg.train.policy_depth_stratify_decimate,
            depth_stratify_sample=cfg.train.policy_depth_stratify_sample,
            depth_stratify_probs=cfg.train.policy_depth_stratify_probs,
            depth_stratify_buckets=cfg.search.depth,
        )

        # Optimizer & loss
        self.optimizer = build_optimizer(self.model, cfg.train, device)
        self.loss_fn = RebelSupervisedLoss(
            policy_weight=1.0,
            value_weight=cfg.train.value_coef,
            entropy_coef=cfg.train.entropy_coef,
            permutation_weight=cfg.train.permutation_coef,
            num_players=self.num_players,
            policy_node_weighting=cfg.train.policy_node_weighting,
            policy_loss_type=cfg.train.policy_loss_type,
        )
        self.loss_fn.to(self.device)
        if self.device.type == "cuda" and _compile_setting(cfg) != "off":
            self.loss_fn.compile_forward_modes(**_compile_kwargs(cfg))
        self.grad_clip = cfg.train.grad_clip

        # EMA setup. Shadow weights live in EMAHelper; at search/eval time we
        # rebind self.model's parameter .data to the shadow tensors via a
        # context manager. This keeps a single compiled module — no second
        # torch.compile pass and no duplicated parameter memory.
        self.ema_helper: EMAHelper | None = None
        if cfg.train.model_ema is not None:
            self.ema_helper = EMAHelper(mu=cfg.train.model_ema)
            self.ema_helper.register(self.model)

        eval_model = self.model

        if cfg.search.sparse:
            evaluator_cls: type[SparseCFREvaluator] = SparseCFREvaluator
            if cfg.search.sparse_fused:
                from p2.search.fused_sparse_cfr_evaluator import (
                    FusedSparseCFREvaluator,
                )

                evaluator_cls = FusedSparseCFREvaluator
            self.cfr_evaluator = evaluator_cls(
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
                allin_call_terminal_abstraction=self.cfg.search.allin_call_terminal_abstraction,
                preflop_allin_table_path=self.cfg.search.preflop_allin_table_path,
            )
        self.data_generator = RebelDataGenerator(
            env_proto=self.env,
            evaluator=self.cfr_evaluator,
            value_buffer=self.value_buffer,
            policy_buffer=self.policy_buffer,
        )

        self.aggression_analyzer = AggressionAnalyzer(device=self.device)

        # TrueSkill tracker. Reuses the live ``self.model`` as the candidate-side
        # compiled instance and creates a second compiled instance for the
        # opponent. Both sides bind weights via .data rebinding (no recompile).
        self.trueskill_tracker: TrueSkillTracker | None = None
        if cfg.trueskill.enabled:
            opponent_model = self._make_eval_twin()
            self.trueskill_tracker = TrueSkillTracker(
                cfg=cfg,
                candidate_model=self.model,
                opponent_model=opponent_model,
                device=self.device,
                generator=self.rng,
                trainer_evaluator=self.cfr_evaluator,
            )

    def _model_autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _make_eval_twin(self) -> nn.Module:
        """Create a second compiled model instance with the same architecture
        as ``self.model``, used as the opponent side for TrueSkill matchups."""
        cfg = self.cfg
        if cfg.model.name == ModelType.better_ffn:
            twin: nn.Module = BetterFFN(
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
                board_interaction_dim=cfg.model.board_interaction_dim,
                policy_rank=cfg.model.policy_rank,
                policy_hand_bias_rank=cfg.model.policy_hand_bias_rank,
                policy_factor_scale=cfg.model.policy_factor_scale,
                nonlinearity=cfg.model.nonlinearity,
            )
        elif cfg.model.name == ModelType.better_trm:
            twin = BetterTRM(
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
        else:
            twin = RebelFFN(
                input_dim=cfg.model.input_dim,
                num_actions=self.num_actions,
                hidden_dim=cfg.model.hidden_dim,
                num_hidden_layers=cfg.model.num_hidden_layers,
                detach_value_head=cfg.model.detach_value_head,
                num_players=self.num_players,
                nonlinearity=cfg.model.nonlinearity,
                enforce_zero_sum=cfg.model.enforce_zero_sum,
            )
        twin.to(self.device)
        twin.eval()
        for p in twin.parameters():
            p.requires_grad = False
        if self.device.type == "cuda" and _compile_setting(cfg) != "off":
            twin.compile_forward_modes(**_compile_kwargs(cfg))
        return twin

    def trueskill_snapshot_weights(self) -> dict[str, torch.Tensor]:
        """Return the weights to snapshot for TrueSkill. Prefers EMA shadow
        weights so we evaluate the same averaged model we eval against PBS."""
        if self.ema_helper is not None:
            return {k: v for k, v in self.ema_helper.shadow.items()}
        return {
            name: param.data
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _eval_swap(self):
        """Bind EMA shadow weights into self.model for the duration of the block."""
        if self.ema_helper is None:
            return nullcontext()
        return self.ema_helper.swapped(self.model)

    def _apply_schedules(self, step: int) -> None:
        """Apply learning rate and iteration count schedules."""
        total_steps = max(1, self.cfg.num_steps)
        t = min(1.0, max(0.0, step / float(total_steps)))

        # Learning rate schedule
        lr_start = float(self.cfg.train.learning_rate)
        lr_final = float(self.cfg.train.learning_rate_final)
        lr_now = _scheduled_learning_rate(
            step=step,
            total_steps=total_steps,
            lr_start=lr_start,
            lr_final=lr_final,
            lr_schedule=self.cfg.train.lr_schedule,
            warmup_steps=self.cfg.train.warmup_steps,
        )
        lr_scale = lr_now / lr_start if lr_start > 0.0 else 1.0
        policy_head_muon_lr = (
            float(self.cfg.train.policy_head_muon_learning_rate) * lr_scale
        )
        adamw_lr_start = (
            lr_start
            if self.cfg.train.adamw_learning_rate is None
            else float(self.cfg.train.adamw_learning_rate)
        )
        adamw_lr = adamw_lr_start * lr_scale

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            if param_group.get("lr_role") == "policy_head_muon":
                param_group["lr"] = policy_head_muon_lr
            elif param_group.get("lr_role") == "adamw":
                param_group["lr"] = adamw_lr
            else:
                param_group["lr"] = lr_now

        # Iteration count schedule (linear interpolation)
        if self.cfg.search.iterations_final is not None:
            iterations_start = self.cfg.search.iterations
            iterations_final = self.cfg.search.iterations_final
            iterations_now = int(
                round(iterations_start + (iterations_final - iterations_start) * t)
            )
        else:
            iterations_now = self.cfg.search.iterations

        # Derived schedules: warm_start_iterations and dcfr_plus_delay
        # scale with the current iteration budget. Tuned values:
        #   warm_start_iterations ≈ iterations / 20
        #   dcfr_plus_delay       ≈ iterations * 0.4
        warm_now = max(1, iterations_now // 20)
        delay_now = int(round(iterations_now * 0.4))
        iterations_now = max(warm_now + 1, iterations_now)
        self.cfr_evaluator.cfr_iterations = iterations_now
        self.cfr_evaluator.warm_start_iterations = warm_now
        self.cfr_evaluator.dcfr_delay = delay_now

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

    @torch.no_grad()
    def _compute_metrics(
        self,
        episodes: int,
        updates: int,
        step_stats: dict[str, float],
        value_batch: RebelBatch,
        policy_batch: RebelBatch,
        value_output: ModelOutput,
        policy_output: ModelOutput | None,
        value_loss_all: torch.Tensor,
        policy_loss_all: torch.Tensor,
        policy_target_model_kl_all: torch.Tensor,
        fresh_value_loss: float | None = None,
        fresh_value_batch: RebelBatch | None = None,
        fresh_policy_batch: RebelBatch | None = None,
    ) -> dict[str, int | float | torch.Tensor | dict[str, int | float]]:
        grad_norm_clipped = torch.nn.utils.get_total_norm(
            p.grad for p in self.model.parameters() if p.grad is not None
        ).item()

        def by_street(
            tensor: torch.Tensor, batch=value_batch, street=None, weights=None
        ) -> dict[str, float]:
            # Stack the per-street reductions and pull them across in a
            # single DtoH instead of one .item() per street. Empty streets
            # produce NaN which the dict comprehension below filters out.
            if street is None:
                street = batch.features.street
            names = list(STREETS)
            if weights is not None:
                vals = torch.stack(
                    [
                        (tensor[street == i] * weights[street == i]).sum()
                        / weights[street == i].sum().clamp(min=1e-12)
                        for i, _ in enumerate(names)
                    ]
                )
            else:
                vals = torch.stack(
                    [tensor[street == i].mean() for i, _ in enumerate(names)]
                )
            vals_cpu = vals.cpu().tolist()
            return {k: v for k, v in zip(names, vals_cpu) if not math.isnan(v)}

        def street_count(street: torch.Tensor) -> dict[str, float]:
            # Single fused DtoH for all streets.
            counts = torch.stack([(street == i).sum() for i, _ in enumerate(STREETS)])
            counts_cpu = counts.cpu().tolist()
            return {name: counts_cpu[i] for i, name in enumerate(STREETS)}

        def policy_node_weights(
            batch: RebelBatch, dtype: torch.dtype
        ) -> torch.Tensor | None:
            return self.loss_fn._policy_node_weights(batch, dtype)

        def reduce_by_masks(
            tensor: torch.Tensor,
            masks: list[torch.Tensor],
            names: list[str],
            weights: torch.Tensor | None = None,
        ) -> dict[str, float]:
            mask_float = torch.stack([mask.to(dtype=tensor.dtype) for mask in masks])
            if weights is not None:
                reduce_weights = mask_float * weights.to(dtype=tensor.dtype)[None, :]
            else:
                reduce_weights = mask_float
            denom = reduce_weights.sum(dim=1)
            numer = (reduce_weights * tensor[None, :]).sum(dim=1)
            nan = torch.full_like(denom, float("nan"))
            vals = torch.where(denom > 0, numer / denom.clamp(min=1e-12), nan)
            vals_cpu = vals.cpu().tolist()
            return {k: v for k, v in zip(names, vals_cpu) if not math.isnan(v)}

        def policy_metric_by_depth(
            tensor: torch.Tensor, batch: RebelBatch
        ) -> dict[str, float]:
            depth = batch.statistics.get("node_depth")
            if depth is None:
                return {}
            max_depth = int(getattr(self.cfg.search, "depth", 0))
            names = [f"depth_{i}" for i in range(max_depth + 1)]
            masks = [depth == i for i in range(max_depth + 1)]
            weights = policy_node_weights(batch, tensor.dtype)
            return reduce_by_masks(tensor, masks, names, weights=weights)

        def policy_metric_by_reach_bucket(
            tensor: torch.Tensor, batch: RebelBatch
        ) -> dict[str, float]:
            reach = batch.statistics.get("policy_node_reach")
            if reach is None:
                return {}
            reach = reach.to(dtype=tensor.dtype)
            names = [
                "ge_1e-1",
                "1e-2_to_1e-1",
                "1e-3_to_1e-2",
                "1e-4_to_1e-3",
                "lt_1e-4",
            ]
            masks = [
                reach >= 1e-1,
                (reach < 1e-1) & (reach >= 1e-2),
                (reach < 1e-2) & (reach >= 1e-3),
                (reach < 1e-3) & (reach >= 1e-4),
                reach < 1e-4,
            ]
            weights = policy_node_weights(batch, tensor.dtype)
            return reduce_by_masks(tensor, masks, names, weights=weights)

        value_buffer_streets_stats = street_count(
            self.value_buffer.features.street[: len(self.value_buffer)]
        )

        metrics: dict[str, int | float | torch.Tensor | dict[str, int | float]] = {
            "episodes": episodes,
            "updates": updates,
            "loss": step_stats["total_loss"] / episodes,
            "policy_loss": step_stats["policy_loss"] / episodes,
            "policy_target_entropy": step_stats["policy_target_entropy"] / episodes,
            "policy_target_model_kl": (step_stats["policy_target_model_kl"] / episodes),
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
            "policy_target_model_kl_depth": policy_metric_by_depth(
                policy_target_model_kl_all, policy_batch
            ),
            "policy_target_model_kl_reach_bucket": policy_metric_by_reach_bucket(
                policy_target_model_kl_all, policy_batch
            ),
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
                with self._model_autocast():
                    fresh_model_output = self.model.repeat(
                        fresh_value_batch.features,
                        count=self.cfg.model.num_supervisions,
                        include_policy=False,
                    )
                fresh_loss_dict = self.loss_fn.forward_value(
                    fresh_model_output, fresh_value_batch
                )
                # loss_fn returns device tensors; .item() lands once per
                # metric here (3 syncs/step in the EMA case, 1 otherwise).
                metrics["fresh_value_loss"] = fresh_loss_dict["value_loss"].item()

                if self.ema_helper is not None:
                    with self._eval_swap():
                        self.model.eval()
                        with self._model_autocast():
                            fresh_model_avg_output = self.model.repeat(
                                fresh_value_batch.features,
                                count=self.cfg.model.num_supervisions,
                                include_policy=False,
                            )
                        metrics["fresh_value_loss_avg"] = self.loss_fn.forward_value(
                            fresh_model_avg_output, fresh_value_batch
                        )["value_loss"].item()

                        with self._model_autocast():
                            model_avg_output = self.model.repeat(
                                value_batch.features,
                                count=self.cfg.model.num_supervisions,
                                include_policy=False,
                            )
                        metrics["value_loss_avg"] = self.loss_fn.forward_value(
                            model_avg_output, value_batch
                        )["value_loss"].item()

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

        if (
            fresh_policy_batch is not None
            and fresh_policy_batch.policy_targets is not None
        ):
            with torch.no_grad():
                self.model.eval()
                fresh_policy_batch = fresh_policy_batch.to(self.device)
                with self._model_autocast():
                    fresh_policy_output = self.model.repeat(
                        fresh_policy_batch.features,
                        count=self.cfg.model.num_supervisions,
                        include_policy=True,
                        include_value=False,
                    )
                fresh_policy_loss_dict = self.loss_fn.forward_policy(
                    fresh_policy_output, fresh_policy_batch
                )
                metrics["fresh_policy_target_model_kl"] = fresh_policy_loss_dict[
                    "target_model_kl"
                ].item()
                metrics["fresh_policy_target_model_kl_depth"] = policy_metric_by_depth(
                    fresh_policy_loss_dict["target_model_kl_all"],
                    fresh_policy_batch,
                )
                metrics["fresh_policy_target_model_kl_reach_bucket"] = (
                    policy_metric_by_reach_bucket(
                        fresh_policy_loss_dict["target_model_kl_all"],
                        fresh_policy_batch,
                    )
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

    # Per-parameter NaN/Inf grad check. Each parameter's `.all()` materializes
    # a Python bool → one host sync per parameter per supervision call. With
    # ~50 params × 10 episodes/step that's ~500 syncs/step purely for the
    # safety check. Off by default; flip on for debugging.
    CHECK_GRADS: bool = False

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
        dict[str, float | torch.Tensor],
        ModelOutput,
        ModelOutput,
        ModelOutput,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self.optimizer.zero_grad(set_to_none=True)

        value_loss, policy_loss, entropy_loss = None, None, None
        value_loss_update, policy_loss_update = None, None

        with self._model_autocast():
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
                value_count = len(value_batch)
                value_output_both = self.model(
                    MLPFeatures.cat([value_batch.features, permuted_batch.features]),
                    include_policy=False,
                )
                value_output_orig = value_output_both[:value_count]
                value_output_permuted = value_output_both[value_count:]

        loss_dict = self.loss_fn._call_forward_value(
            value_output_permuted, permuted_batch
        )
        value_loss = loss_dict["value_loss"]
        value_loss_update = loss_dict["value_loss_all"]
        total_loss = loss_dict["total_loss"]

        permutation_loss_tensor = self._compute_permutation_loss(
            value_output_orig, value_output_permuted, suit_permutations_idxs
        )
        total_loss = (
            total_loss + self.loss_fn.permutation_weight * permutation_loss_tensor
        )

        with self._model_autocast():
            if isinstance(self.model, BetterTRM):
                policy_output = self.model(
                    policy_batch.features,
                    include_policy=True,
                    include_value=False,
                    latent=policy_latent,
                )
            else:
                policy_output = self.model(
                    policy_batch.features,
                    include_policy=True,
                    include_value=False,
                )
        loss_dict = self.loss_fn._call_forward_policy(policy_output, policy_batch)
        policy_loss = loss_dict["policy_loss"]
        policy_loss_update = loss_dict["policy_loss_all"]
        policy_kl_update = loss_dict["target_model_kl_all"]
        target_entropy = loss_dict["target_entropy"]
        target_model_kl = loss_dict["target_model_kl"]
        entropy_loss = loss_dict["entropy"]
        total_loss = total_loss + loss_dict["total_loss"]

        total_loss.backward()

        if self.CHECK_GRADS:
            # Single fused reduction → one host sync instead of one per param.
            grad_finite = torch.stack(
                [
                    p.grad.isfinite().all()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
            ).all()
            assert grad_finite.item(), "NaN/Inf in model gradients"

        if self.grad_clip is not None and self.grad_clip > 0:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.get_total_norm(
                p.grad for p in self.model.parameters() if p.grad is not None
            )

        self.optimizer.step()

        # Update EMA if enabled. Shadow weights are the source of truth and
        # are bound into self.model on demand via _eval_swap().
        if self.ema_helper is not None:
            self.ema_helper.update(self.model)

        # Approximate per-step parameter update norm. Previously we cloned
        # every parameter pre-step and computed ||p_after - p_before||, which
        # cost a full param-sized DtoD copy *and* a sync per supervision; for
        # SGD this is exactly lr*||grad||, and for Adam it's a close-enough
        # proxy of effective step magnitude for monitoring purposes.
        current_lr = self.optimizer.param_groups[0]["lr"]
        update_norm = current_lr * grad_norm  # tensor, no sync

        # Tensors here, not floats — caller accumulates and syncs once at
        # end of step. permutation_loss is the raw (unweighted) tensor for
        # logging; total_loss already includes its weighted contribution.
        return (
            {
                "policy_loss": policy_loss.detach(),
                "policy_target_entropy": target_entropy.detach(),
                "policy_target_model_kl": target_model_kl.detach(),
                "value_loss": value_loss.detach(),
                "entropy_loss": entropy_loss.detach(),
                "permutation_loss": permutation_loss_tensor.detach(),
                "total_loss": total_loss.detach(),
                "update_norm": update_norm.detach(),
            },
            value_output_permuted,
            value_output_orig,
            policy_output,
            value_loss_update,
            policy_loss_update,
            policy_kl_update,
        )

    @profile
    def _update_model(
        self, step: int
    ) -> dict[str, int | float | torch.Tensor | dict[str, int | float]]:
        with self._eval_swap():
            fresh_value_batch, fresh_policy_batch = self.data_generator.generate_data(
                self.K_value, return_policy_batch=True
            )

            # Warmup: make sure we have enough samples.
            while (
                min(len(self.value_buffer), len(self.policy_buffer)) < self.batch_size
            ):
                self.data_generator.generate_data(
                    self.K_value,
                    return_value_batch=False,
                    return_policy_batch=False,
                )

        value_fullness = len(self.value_buffer) / self.value_buffer.capacity
        episodes = math.ceil(self.cfg.train.episodes_per_step * value_fullness)
        supervisions = (
            self.cfg.model.num_supervisions if isinstance(self.model, BetterTRM) else 1
        )
        updates = episodes * supervisions
        value_batch_all = []
        policy_batch_all = []
        value_output_all = []
        value_loss_update_all = []
        policy_loss_update_all = []
        policy_kl_update_all = []
        step_stats: dict[str, float] = {}
        # Tensor-valued accumulators kept on device until end-of-step. The
        # original code did a host sync per supervision for each of these
        # (loss_fn used to call .item() inline on policy/value/entropy/
        # permutation losses, plus _supervise on total_loss/update_norm);
        # accumulating on device and syncing once below collapses ~6
        # syncs/episode into 1 sync/step.
        tensor_stats: dict[str, torch.Tensor | None] = {
            "policy_loss": None,
            "policy_target_entropy": None,
            "policy_target_model_kl": None,
            "value_loss": None,
            "entropy_loss": None,
            "permutation_loss": None,
            "total_loss": None,
            "update_norm": None,
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
            policy_stratify = (
                None if self.cfg.train.policy_depth_stratify_sample else stratify
            )
            policy_batch = self.policy_buffer.sample(
                self.batch_size, stratify_streets=policy_stratify
            ).to(self.device)

            # Sample suit permutations and apply to features/targets together.
            suit_permutations_idxs = torch.randint(
                0,
                24,
                (len(value_batch),),
                generator=self.rng,
                device=self.device,
            )
            suit_permutations = suit_permutations_tensor(device=self.device)[
                suit_permutations_idxs
            ]
            permuted_batch, suit_permutations_idxs = value_batch.with_permuted_targets(
                suit_permutations=suit_permutations,
                suit_permutation_idxs=suit_permutations_idxs,
                num_players=self.num_players,
            )

            for _ in range(supervisions):
                (
                    episode_stats,
                    permuted_value_output,
                    value_output_orig,
                    policy_output,
                    value_loss_update,
                    policy_loss_update,
                    policy_kl_update,
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

            # All loss/norm stats are device tensors now; keep them on
            # device until the single end-of-step sync below.
            for k, acc in tensor_stats.items():
                v = episode_stats[k]
                tensor_stats[k] = v if acc is None else acc + v

            # Append last batch/output for metrics.
            value_batch_all.append(permuted_batch)
            policy_batch_all.append(policy_batch)
            value_output_all.append(permuted_value_output)
            value_loss_update_all.append(value_loss_update)
            policy_loss_update_all.append(policy_loss_update)
            policy_kl_update_all.append(policy_kl_update)

        # Single host sync to fold the device-side accumulators into the
        # float-keyed step_stats dict that _compute_metrics expects.
        if any(v is not None for v in tensor_stats.values()):
            keys = [k for k, v in tensor_stats.items() if v is not None]
            # Reshape each accumulator to 0-d so stack succeeds even when
            # individual losses come back as scalars vs 1-element tensors.
            stacked = (
                torch.stack([tensor_stats[k].reshape(()) for k in keys]).cpu().tolist()
            )
            for k, val in zip(keys, stacked):
                step_stats[k] = val
        for k in tensor_stats:
            step_stats.setdefault(k, 0.0)

        value_metric_tensors = [
            output.value for output in value_output_all if output.value is not None
        ]
        value_metric_output = ModelOutput(
            value=torch.cat(value_metric_tensors) if value_metric_tensors else None
        )
        metrics = self._compute_metrics(
            episodes,
            updates,
            step_stats,
            RebelBatch.cat(value_batch_all),
            RebelBatch.cat(policy_batch_all),
            value_metric_output,
            None,
            torch.cat(value_loss_update_all),
            torch.cat(policy_loss_update_all),
            torch.cat(policy_kl_update_all),
            fresh_value_batch=fresh_value_batch,
            fresh_policy_batch=fresh_policy_batch,
        )

        return metrics

    def train_step(self, step: int) -> dict[str, Any]:
        step_public = step + 1

        # Apply schedules before training step
        self._apply_schedules(step)

        update_info = self._update_model(step)
        update_info["step"] = step_public
        update_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        policy_head_lrs = [
            group["lr"]
            for group in self.optimizer.param_groups
            if group.get("lr_role") == "policy_head_muon"
        ]
        if policy_head_lrs:
            update_info["policy_head_muon_learning_rate"] = policy_head_lrs[0]
        adamw_lrs = [
            group["lr"]
            for group in self.optimizer.param_groups
            if group.get("lr_role") == "adamw"
        ]
        if adamw_lrs:
            update_info["adamw_learning_rate"] = adamw_lrs[0]
        update_info["cfr_iterations"] = self.cfr_evaluator.cfr_iterations

        return update_info

    def train(self, num_steps: int | None = None) -> list[dict[str, Any]]:
        total_steps = num_steps or self.cfg.num_steps
        history: list[dict[str, Any]] = []

        for step in range(total_steps):
            update_info = self.train_step(step)
            history.append(update_info)

        return history

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

        # Save EMA shadow weights if enabled.
        if self.ema_helper is not None:
            model_avg_state = dict(self.ema_helper.shadow)
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

        self.model.load_state_dict(model_state, strict=self.cfg.strict_model_loading)

        # Load EMA state if it exists in checkpoint and EMA is enabled.
        if "model_avg" in ckpt and self.ema_helper is not None:
            model_avg_state = ckpt["model_avg"]
            if save_dtype_str is not None and save_dtype_str != str(self.float_dtype):
                model_avg_state = {
                    k: v.to(self.float_dtype) if v.dtype.is_floating_point else v
                    for k, v in model_avg_state.items()
                }
            # Older checkpoints saved a full state_dict (params + buffers); keep
            # only the trainable-param keys that EMAHelper tracks.
            shadow_keys = set(self.ema_helper.shadow.keys())
            self.ema_helper.shadow = {
                k: v.to(self.ema_helper.shadow[k].dtype).clone()
                for k, v in model_avg_state.items()
                if k in shadow_keys
            }

        # Only load optimizer if it exists in checkpoint
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if isinstance(self.optimizer, torch.optim.AdamW):
                for param_group in self.optimizer.param_groups:
                    param_group["lr_role"] = "adamw"

        self.cfg.wandb_run_id = ckpt.get("wandb_run_id")
        # if "rng" in ckpt:
        #     self.rng.set_state(ckpt["rng"].to(self.device))
        return int(ckpt["step"])
