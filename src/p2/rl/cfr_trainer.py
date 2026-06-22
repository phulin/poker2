from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from p2.core.action_schedule import apply_action_schedule_to_config
from p2.core.structured_config import Config, LrSchedule, ModelType
from p2.env.aggression_analyzer import AggressionAnalyzer
from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    combo_suit_permutation_tensor,
    suit_permutations_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.models.mlp import RebelFFN
from p2.models.mlp.better_features import context_length, legacy_context_length
from p2.models.mlp.better_ffn import (
    BetterPolicyFFN,
    BetterPreflopPolicyFFN,
    BetterPreflopTransformerPolicyFFN,
    BetterPreflopTransformerValueFFN,
    BetterPreflopValueFFN,
    BetterSplitFFN,
    BetterStreetValueFFN,
)
from p2.models.mlp.better_trm import BetterTRM
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput, TRMLatent
from p2.models.street_model_registry import StreetModelRegistry
from p2.rl.losses import RebelSupervisedLoss
from p2.rl.optimizers import build_optimizer
from p2.rl.rebel_batch import RebelBatch
from p2.rl.trueskill_tracker import TrueSkillTracker
from p2.rl.rebel_replay import RebelPolicyBuffer, RebelValueBuffer
from p2.search.cfr_evaluator import CFREvaluator, PublicBeliefState
from p2.search.postflop_spot_sampler import (
    sample_postflop_legal_prefix_roots,
    sample_postflop_start_roots,
)
from p2.search.rebel_data_generator import RebelDataGenerator
from p2.search.rebel_data_source import (
    HybridRebelDataSource,
    LiveRebelDataSource,
    PregeneratedRebelDataSource,
    RebelDataSource,
)
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator
from p2.utils.ema_helper import EMAHelper
from p2.utils.profiling import profile
from p2.utils.rng import generator_state_for_set_state

STREETS = ["preflop", "flop", "turn", "river", "showdown"]
REPLAY_BUFFER_CHECKPOINT = "rebel_replay_buffers.pt"


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
    wsd_decay_fraction: float = 0.2,
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
    if lr_schedule == LrSchedule.wsd and lr_final != lr_start:
        decay_fraction = min(1.0, max(1e-6, float(wsd_decay_fraction)))
        stable_until = 1.0 - decay_fraction
        if t <= stable_until:
            return lr_start
        decay_t = min(1.0, (t - stable_until) / decay_fraction)
        return lr_final + 0.5 * (lr_start - lr_final) * (
            1.0 + math.cos(math.pi * decay_t)
        )
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

    def __init__(
        self, cfg: Config, device: torch.device, pregeneration_only: bool = False
    ) -> None:
        debug_init = bool(os.environ.get("P2_DEBUG_TRAINER_INIT"))

        def init_trace(message: str) -> None:
            if debug_init:
                print(f"[RebelCFRTrainer:init] {message}", flush=True)

        init_trace("start")
        self.cfg = cfg
        apply_action_schedule_to_config(cfg)
        if cfg.data.mode not in {"live", "pregenerated", "hybrid"}:
            raise NotImplementedError(
                "RebelCFRTrainer supports data.mode=live, "
                "data.mode=pregenerated, or data.mode=hybrid."
            )
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
        self.num_players = int(cfg.env.num_players)
        if self.num_players < 2:
            raise ValueError("env.num_players must be at least 2")
        if self.num_players != 2 and cfg.model.enforce_zero_sum:
            print(
                "[RebelCFRTrainer] Disabling model.enforce_zero_sum for multiway "
                "training; multiway constant-sum projection needs joint blockers."
            )
            cfg.model.enforce_zero_sum = False
        if self.num_players != 2 and cfg.data.live_root_source == "self_play":
            if cfg.model.name != ModelType.better_ffn:
                raise ValueError(
                    "Multiway preflop self-play requires model.name=better_ffn "
                    "so the evaluator can use compact 169-hand preflop models."
                )
            cfg.model.preflop_hand_dim = 169
        self.policy_extra_updates_per_step = cfg.train.policy_extra_updates_per_step
        if self.policy_extra_updates_per_step < 0:
            raise ValueError("train.policy_extra_updates_per_step must be >= 0")
        self.policy_extra_batch_size = (
            cfg.train.policy_extra_batch_size
            if cfg.train.policy_extra_batch_size is not None
            else self.batch_size
        )
        if self.policy_extra_batch_size <= 0:
            raise ValueError("train.policy_extra_batch_size must be positive")
        self.target_update_block_batches = int(
            cfg.train.target_update_block_batches or 0
        )
        if self.target_update_block_batches < 0:
            raise ValueError("train.target_update_block_batches must be >= 0")
        self.cfr_target_model: nn.Module | None = None
        self.cfr_target_model_step = 0

        if cfg.model.num_actions != self.num_actions:
            print(
                f"[RebelCFRTrainer] Overriding model.num_actions "
                f"({cfg.model.num_actions}) -> {self.num_actions} "
                f"to match bet bin configuration."
            )
            cfg.model.num_actions = self.num_actions

        # Environment used to provide root states for CFR search. Heads-up keeps
        # the private-card tensor env as the fast regression path unless the
        # model is the compact 169-hand preflop variant, which is a PBS model
        # even with two players.
        compact_preflop_self_play = (
            int(cfg.model.preflop_hand_dim) == PREFLOP_HANDS
            and cfg.data.live_root_source == "self_play"
        )
        if self.num_players == 2 and not compact_preflop_self_play:
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
        else:
            if cfg.search.sparse_fused and cfg.data.live_root_source != "self_play":
                raise ValueError(
                    "Multiway fused sparse CFR is only available for preflop "
                    "self-play roots."
                )
            self.env = PBSEnv(
                num_envs=self.cfg.num_envs,
                num_players=self.num_players,
                starting_stack=cfg.env.stack,
                sb=cfg.env.sb,
                bb=cfg.env.bb,
                default_bet_bins=self.bet_bins,
                device=self.device,
                float_dtype=self.float_dtype,
                randomize_stacks=cfg.env.randomize_stacks,
                stack_mode=cfg.env.stack_mode,
                min_stack_bb=cfg.env.min_stack_bb,
                mid_stack_bb=cfg.env.mid_stack_bb,
                max_stack_bb=cfg.env.max_stack_bb,
                high_stack_mass_ratio=cfg.env.high_stack_mass_ratio,
            )
        self.env.reset()
        init_trace("environment ready")

        # Model
        if cfg.model.name == ModelType.better_ffn:
            self.model = self._make_better_split_ffn()
            num_context_features = (
                legacy_context_length(self.num_players)
                if cfg.model.legacy_context_features
                else context_length(self.num_players)
            )
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
        init_trace("training model ready")

        # data generation rate per training step
        self.K_value = _value_samples_per_step(
            self.batch_size, self.cfg.train.value_reuse_goal
        )
        # approximate number of policy samples when collecting K_value value samples
        policy_decimate = (
            self.num_actions / 2
        ) ** self.cfg.search.depth / self.cfg.train.policy_capacity_factor

        self.stream_live_batches = False

        C_over_K = self.cfg.train.replay_buffer_batches
        value_capacity = int(math.ceil(C_over_K * self.K_value))
        policy_capacity = int(
            math.ceil(value_capacity * self.cfg.train.policy_capacity_factor)
        )
        replay_hand_dim = int(cfg.model.preflop_hand_dim)

        if pregeneration_only:
            self.value_buffer = None
            self.policy_buffer = None
        else:
            # Replay buffers
            self.value_buffer = RebelValueBuffer(
                capacity=value_capacity,
                num_actions=self.num_actions,
                num_players=self.num_players,
                num_context_features=num_context_features,
                device=self.buffer_device,
                hand_dim=replay_hand_dim,
                underfull_evict_fraction=cfg.train.replay_buffer_underfull_evict_fraction,
                generator=self.buffer_rng,
            )
            # Larger policy buffer since we store more samples there
            self.policy_buffer = RebelPolicyBuffer(
                capacity=policy_capacity,
                num_actions=self.num_actions,
                num_players=self.num_players,
                num_context_features=num_context_features,
                device=self.buffer_device,
                hand_dim=replay_hand_dim,
                decimate=1.0 / policy_decimate,
                underfull_evict_fraction=cfg.train.replay_buffer_underfull_evict_fraction,
                generator=self.buffer_rng,
                depth_stratify_decimate=cfg.train.policy_depth_stratify_decimate,
                depth_stratify_sample=cfg.train.policy_depth_stratify_sample,
                depth_stratify_probs=cfg.train.policy_depth_stratify_probs,
                depth_stratify_buckets=cfg.search.depth,
            )
        init_trace("replay buffers ready")

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
            policy_logit_l2_coef=cfg.train.policy_logit_l2_coef,
        )
        self.loss_fn.to(self.device)
        if self.device.type == "cuda" and _compile_setting(cfg) != "off":
            self.loss_fn.compile_forward_modes(**_compile_kwargs(cfg))
        self.grad_clip = cfg.train.grad_clip
        self.policy_grad_clip = cfg.train.policy_grad_clip
        self._compute_grad_clip_groups()

        # EMA setup. If enabled, the dedicated inference model is synced from
        # these shadow weights after training updates.
        self.ema_helper: EMAHelper | None = None
        if cfg.train.model_ema is not None:
            self.ema_helper = EMAHelper(mu=cfg.train.model_ema)
            self.ema_helper.register(self.model)

        self.inference_model = self._make_eval_twin(compile_model=False)
        self._sync_inference_model()
        if self.device.type == "cuda" and _compile_setting(cfg) != "off":
            self.inference_model.compile_forward_modes(**_compile_kwargs(cfg))
        init_trace("inference model ready")
        eval_model = self.inference_model
        if self.target_update_block_batches > 0:
            if cfg.search.street_model_checkpoints:
                raise ValueError(
                    "train.target_update_block_batches is incompatible with "
                    "search.street_model_checkpoints because the CFR evaluator "
                    "uses an external street model registry."
                )
            self.cfr_target_model = self._make_eval_twin(compile_model=False)
            self._sync_cfr_target_model(step=0)
            if self.device.type == "cuda" and _compile_setting(cfg) != "off":
                self.cfr_target_model.compile_forward_modes(**_compile_kwargs(cfg))
            eval_model = self.cfr_target_model
            init_trace(
                "block CFR target model ready "
                f"(block={self.target_update_block_batches})"
            )
        if cfg.search.street_model_checkpoints:
            if cfg.search.sparse_fused:
                raise NotImplementedError(
                    "search.sparse_fused does not yet support street_model_checkpoints; "
                    "use non-fused sparse CFR for street-dispatched resolving."
                )
            eval_model = self._load_street_model_registry(
                cfg.search.street_model_checkpoints
            )
        closing_leaf_model = None
        if cfg.search.closing_leaf_checkpoint is not None:
            closing_leaf_model = self._load_closing_leaf_model(
                cfg.search.closing_leaf_checkpoint
            )
        init_trace("closing leaf model ready")

        if not cfg.search.sparse:
            raise ValueError(
                "Dense RebelCFREvaluator has been removed; set search.sparse=true."
            )
        evaluator_cls: type[SparseCFREvaluator] = SparseCFREvaluator
        if compact_preflop_self_play:
            if cfg.search.sparse_fused:
                from p2.search.fused_preflop_sparse_cfr_evaluator import (
                    FusedPreflopSparseCFREvaluator,
                )

                evaluator_cls = FusedPreflopSparseCFREvaluator
            else:
                from p2.search.preflop_sparse_cfr_evaluator import (
                    PreflopSparseCFREvaluator,
                )

                evaluator_cls = PreflopSparseCFREvaluator
        elif cfg.search.sparse_fused:
            from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

            evaluator_cls = FusedSparseCFREvaluator
        self.cfr_evaluator = evaluator_cls(
            model=eval_model,
            device=self.device,
            cfg=cfg,
            generator=self.rng,
            closing_leaf_model=closing_leaf_model,
        )
        init_trace(f"evaluator ready ({evaluator_cls.__name__})")
        root_sampler = None
        random_street_sources = {
            "random_flop": 1,
            "random_turn": 2,
            "random_river": 3,
        }
        random_prefix_sources = {
            "random_flop_prefix": 1,
            "random_turn_prefix": 2,
            "random_river_prefix": 3,
        }
        if cfg.data.live_root_source in random_street_sources:
            street = random_street_sources[cfg.data.live_root_source]

            def root_sampler(batch_size: int) -> PublicBeliefState:
                return sample_postflop_start_roots(
                    self.env,
                    batch_size=batch_size,
                    street=street,
                    generator=self.rng,
                )

        elif cfg.data.live_root_source in random_prefix_sources:
            street = random_prefix_sources[cfg.data.live_root_source]

            def root_sampler(batch_size: int) -> PublicBeliefState:
                return sample_postflop_legal_prefix_roots(
                    self.env,
                    batch_size=batch_size,
                    street=street,
                    generator=self.rng,
                )

        elif cfg.data.live_root_source != "self_play":
            raise ValueError(
                "data.live_root_source must be 'self_play', 'random_flop', "
                "'random_turn', 'random_river', 'random_flop_prefix', "
                "'random_turn_prefix', or 'random_river_prefix'; "
                f"got {cfg.data.live_root_source!r}"
            )

        def make_pregenerated_source(
            value_buffer: RebelValueBuffer,
            policy_buffer: RebelPolicyBuffer,
            *,
            generator: torch.Generator,
        ) -> PregeneratedRebelDataSource:
            return PregeneratedRebelDataSource(
                cfg.data.pregenerated.datasets,
                value_buffer,
                policy_buffer,
                value_sample_count=cfg.data.pregenerated.value_batch_size
                or self.K_value,
                policy_sample_count=cfg.data.pregenerated.policy_batch_size
                or self.batch_size,
                num_players=self.num_players,
                num_actions=self.num_actions,
                context_length=num_context_features,
                model_name=cfg.model.name.value,
                action_schedule={
                    "bet_bins": list(cfg.env.bet_bins),
                    "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
                    "allin_by_depth": cfg.search.allin_by_depth,
                },
                street_support=[0, 1, 2, 3],
                generator=generator,
                shuffle=cfg.data.pregenerated.shuffle,
                direct_sample=cfg.data.pregenerated.direct_sample,
                pin_memory=cfg.data.pregenerated.pin_memory,
                async_shard_prefetch=cfg.data.pregenerated.async_shard_prefetch,
            )

        self.data_generator: RebelDataGenerator | None = None
        if cfg.data.mode in {"live", "hybrid"}:
            self.data_generator = RebelDataGenerator(
                env_proto=self.env,
                evaluator=self.cfr_evaluator,
                value_buffer=self.value_buffer,
                policy_buffer=self.policy_buffer,
                warmup=cfg.data.warmup_self_play_roots,
                root_sampler=root_sampler,
                include_pre_chance_value_batches=cfg.data.include_pre_chance_value_batches,
                store_replay=not pregeneration_only,
                sample_continuations=not pregeneration_only,
                record_batch_diag=not pregeneration_only,
            )
            init_trace("data generator ready")
            if pregeneration_only:
                self.data_source = None
                self.aggression_analyzer = AggressionAnalyzer(device=self.device)
                self.trueskill_tracker = None
                return
            if self.value_buffer is None or self.policy_buffer is None:
                raise RuntimeError("Live training requires replay buffers")
            self.data_source = LiveRebelDataSource(
                self.data_generator,
                self.value_buffer,
                self.policy_buffer,
                value_sample_count=self.K_value,
                max_return_policy_samples=self.batch_size,
            )
            if cfg.data.mode == "hybrid":
                holdout_rng = torch.Generator(device="cpu")
                if cfg.seed is not None:
                    holdout_rng.manual_seed(int(cfg.seed) + 1)
                holdout_value_buffer = RebelValueBuffer(
                    capacity=value_capacity,
                    num_actions=self.num_actions,
                    num_players=self.num_players,
                    num_context_features=num_context_features,
                    device=self.buffer_device,
                    hand_dim=replay_hand_dim,
                )
                holdout_policy_buffer = RebelPolicyBuffer(
                    capacity=policy_capacity,
                    num_actions=self.num_actions,
                    num_players=self.num_players,
                    num_context_features=num_context_features,
                    device=self.buffer_device,
                    hand_dim=replay_hand_dim,
                )
                holdout_source = make_pregenerated_source(
                    holdout_value_buffer,
                    holdout_policy_buffer,
                    generator=holdout_rng,
                )
                self.data_source = HybridRebelDataSource(
                    self.data_source, holdout_source
                )
        else:
            dataset_rng = torch.Generator(device="cpu")
            if cfg.seed is not None:
                dataset_rng.manual_seed(int(cfg.seed))
            self.data_source: RebelDataSource = make_pregenerated_source(
                self.value_buffer,
                self.policy_buffer,
                generator=dataset_rng,
            )

        self.aggression_analyzer = AggressionAnalyzer(device=self.device)

        # TrueSkill tracker reuses the dedicated inference model as the
        # candidate-side compiled instance and creates a second compiled
        # instance for the opponent.
        self.trueskill_tracker: TrueSkillTracker | None = None
        if cfg.trueskill.enabled:
            opponent_model = self._make_eval_twin()
            self.trueskill_tracker = TrueSkillTracker(
                cfg=cfg,
                candidate_model=self.inference_model,
                opponent_model=opponent_model,
                device=self.device,
                generator=self.rng,
                trainer_evaluator=self.cfr_evaluator,
            )

    def _model_autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _compute_grad_clip_groups(self) -> None:
        """Partition parameters into policy/value groups for separate gradient
        clipping. Only possible when the two heads have disjoint parameters
        (separate nets, or a TRM whose policy head detaches from the trunk).
        With a shared trunk we fall back to a single combined clip."""
        model = self.model
        if type(model) is BetterSplitFFN:
            self._grad_clip_policy_params = list(model.policy_model.parameters())
            self._grad_clip_value_params = list(model.value_model.parameters())
            self._split_grad_clip = True
            return
        if type(model) is BetterTRM and not model.shared_trunk:
            policy_params = list(model.policy_head.parameters())
            policy_ids = {id(p) for p in policy_params}
            self._grad_clip_policy_params = policy_params
            self._grad_clip_value_params = [
                p for p in model.parameters() if id(p) not in policy_ids
            ]
            self._split_grad_clip = True
            return
        self._grad_clip_policy_params = []
        self._grad_clip_value_params = []
        self._split_grad_clip = False

    @staticmethod
    def _clip_grad_group(
        params: list[nn.Parameter], clip: float | None
    ) -> torch.Tensor:
        if clip is not None and clip > 0:
            return nn.utils.clip_grad_norm_(params, clip)
        return torch.nn.utils.get_total_norm(
            p.grad for p in params if p.grad is not None
        )

    def _clip_grads(self) -> torch.Tensor:
        """Clip gradients and return the pre-clip total grad norm for logging.

        When policy and value have disjoint parameters each group is clipped with
        its own threshold (``policy_grad_clip`` vs ``grad_clip``); otherwise a
        single combined clip is applied across all parameters. In the policy-only
        path the value group has no gradients, so its norm is 0 and the combined
        norm collapses to the policy norm."""
        if self._split_grad_clip:
            policy_norm = self._clip_grad_group(
                self._grad_clip_policy_params, self.policy_grad_clip
            )
            value_norm = self._clip_grad_group(
                self._grad_clip_value_params, self.grad_clip
            )
            return torch.sqrt(policy_norm.square() + value_norm.square())
        return self._clip_grad_group(list(self.model.parameters()), self.grad_clip)

    def _make_better_split_ffn(self) -> BetterSplitFFN:
        cfg = self.cfg
        common = dict(
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
            nonlinearity=cfg.model.nonlinearity,
            legacy_context_features=cfg.model.legacy_context_features,
        )
        if int(cfg.model.preflop_hand_dim) == 169:
            preflop_model_type = str(cfg.model.preflop_model_type).strip().lower()
            if preflop_model_type in {"transformer", "tokens", "player_transformer"}:
                transformer_common = dict(
                    common,
                    transformer_heads=int(cfg.model.preflop_transformer_heads),
                )
                return BetterSplitFFN(
                    policy_model=BetterPreflopTransformerPolicyFFN(
                        num_actions=self.num_actions, **transformer_common
                    ),
                    value_model=BetterPreflopTransformerValueFFN(
                        num_actions=1,
                        value_heads=cfg.model.street_value_heads,
                        **transformer_common,
                    ),
                )
            if preflop_model_type not in {"ffn", "mlp", "compact"}:
                raise ValueError(
                    "model.preflop_model_type must be one of: ffn, transformer; "
                    f"got {cfg.model.preflop_model_type!r}"
                )
            return BetterSplitFFN(
                policy_model=BetterPreflopPolicyFFN(
                    num_actions=self.num_actions, **common
                ),
                value_model=BetterPreflopValueFFN(
                    num_actions=1,
                    value_heads=cfg.model.street_value_heads,
                    **common,
                ),
            )
        return BetterSplitFFN(
            policy_model=BetterPolicyFFN(num_actions=self.num_actions, **common),
            value_model=BetterStreetValueFFN(
                num_actions=1,
                value_heads=cfg.model.street_value_heads,
                **common,
            ),
        )

    def _make_eval_twin(self, compile_model: bool = True) -> nn.Module:
        """Create a second compiled model instance with the same architecture
        as ``self.model``, used as the opponent side for TrueSkill matchups."""
        cfg = self.cfg
        if cfg.model.name == ModelType.better_ffn:
            twin: nn.Module = self._make_better_split_ffn()
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
        if (
            self.device.type == "cuda"
            and self.num_players != 2
            and cfg.data.live_root_source == "self_play"
        ):
            twin.to(dtype=torch.bfloat16)
        twin.eval()
        for p in twin.parameters():
            p.requires_grad = False
        if (
            compile_model
            and self.device.type == "cuda"
            and _compile_setting(cfg) != "off"
        ):
            twin.compile_forward_modes(**_compile_kwargs(cfg))
        return twin

    def _load_closing_leaf_model(self, checkpoint_path: str) -> nn.Module:
        return self._load_frozen_eval_model(checkpoint_path)

    def _load_frozen_eval_model(self, checkpoint_path: str) -> nn.Module:
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model_component = checkpoint.get("model_component")
        model = self._make_eval_twin(compile_model=False)
        model_state = checkpoint["model"]
        model_state = {
            key: value.to(self.float_dtype) if value.dtype.is_floating_point else value
            for key, value in model_state.items()
        }
        if model_component == "value_model":
            if type(model) is not BetterSplitFFN:
                raise TypeError("value-only checkpoints require model.name=BetterFFN")
            model = model.value_model
        model.load_state_dict(model_state, strict=self.cfg.strict_model_loading)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        if self.device.type == "cuda" and _compile_setting(self.cfg) != "off":
            model.compile_forward_modes(**_compile_kwargs(self.cfg))
        return model

    def _load_street_model_registry(
        self, checkpoints_by_street: dict[str, str]
    ) -> StreetModelRegistry:
        models = {
            street: self._load_frozen_eval_model(path)
            for street, path in checkpoints_by_street.items()
        }
        registry = StreetModelRegistry(models)
        registry.to(self.device)
        registry.eval()
        return registry

    @torch.no_grad()
    def _sync_eval_model_from_training(self, target_model: nn.Module) -> None:
        """Copy train or EMA weights into a frozen eval twin."""
        source_params = dict(self.model.named_parameters())
        source_buffers = dict(self.model.named_buffers())
        ema_shadow = self.ema_helper.shadow if self.ema_helper is not None else {}

        for name, param in target_model.named_parameters():
            src = ema_shadow.get(name)
            if src is None:
                src = source_params[name].data
            if src.device != param.device or src.dtype != param.dtype:
                src = src.to(device=param.device, dtype=param.dtype)
            param.data.copy_(src)

        for name, buf in target_model.named_buffers():
            src = source_buffers.get(name)
            if src is None:
                continue
            if src.device != buf.device or src.dtype != buf.dtype:
                src = src.to(device=buf.device, dtype=buf.dtype)
            buf.copy_(src)
        target_model.eval()

    @torch.no_grad()
    def _sync_inference_model(self) -> None:
        """Copy train or EMA weights into the dedicated inference model."""
        self._sync_eval_model_from_training(self.inference_model)

    @torch.no_grad()
    def _sync_cfr_target_model(self, step: int) -> bool:
        """Promote train or EMA weights into the block-frozen CFR target model."""
        if self.cfr_target_model is None:
            return False
        self._sync_eval_model_from_training(self.cfr_target_model)
        self.cfr_target_model_step = int(step)
        return True

    def _maybe_promote_cfr_target_model(self, step: int) -> bool:
        if self.cfr_target_model is None or self.target_update_block_batches <= 0:
            return False
        if step <= 0 or step % self.target_update_block_batches != 0:
            return False
        return self._sync_cfr_target_model(step)

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
            wsd_decay_fraction=self.cfg.train.lr_wsd_decay_fraction,
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

        # dcfr_plus_delay scales with the current iteration budget
        # (dcfr_plus_delay ≈ iterations * 0.2). warm_start_iterations is held
        # constant at the configured value instead: river sweeps showed the
        # model_br warm-start seed weight (warm_start_iterations *
        # warm_start_multiplier) has an iteration-count-independent optimum
        # (~30), so scaling it with the iteration budget pushes it into the
        # harmful regime as iterations grow.
        configured_warm = int(self.cfg.search.warm_start_iterations)
        warm_floor = 0 if self.num_players != 2 else 1
        warm_now = max(warm_floor, configured_warm)
        delay_now = int(round(iterations_now * 0.2))
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
        hand_dim = value_output.hand_values.shape[-1]
        if hand_dim == PREFLOP_HANDS:
            return F.mse_loss(
                value_output.hand_values.float(),
                value_output_permuted.hand_values.float(),
            )
        if hand_dim != NUM_HANDS:
            raise ValueError(f"Unsupported permutation hand_dim={hand_dim}")
        hand_values_permuted_reversed = torch.gather(
            value_output_permuted.hand_values,
            2,
            combo_permutations[:, None, :].expand(-1, self.num_players, -1),
        )
        return F.mse_loss(
            value_output.hand_values.float(), hand_values_permuted_reversed.float()
        )

    def _backup_consistency_coef(self) -> float:
        return float(self.cfg.train.backup_consistency_coef or 0.0)

    def _backup_consistency_loss(
        self,
        batch: RebelBatch,
        value_output: ModelOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch.features.context.device
        zero = torch.zeros((), device=device)
        if self._backup_consistency_coef() <= 0.0:
            return zero, zero
        if type(self.model) is BetterTRM:
            return zero, zero
        if value_output.hand_values is None:
            return zero, zero

        stats = batch.statistics
        required = (
            "backup_child_context",
            "backup_child_street",
            "backup_child_to_act",
            "backup_child_board",
            "backup_child_beliefs",
            "backup_child_valid",
            "backup_actor",
        )
        if any(key not in stats for key in required):
            return zero, zero

        child_valid_all = stats["backup_child_valid"].to(device=device, dtype=torch.bool)
        eligible = child_valid_all.any(dim=1)
        sample_indices = torch.where(eligible)[0]
        if sample_indices.numel() == 0:
            return zero, zero

        sample_fraction = float(
            self.cfg.train.backup_consistency_sample_fraction or 0.0
        )
        sample_fraction = max(0.0, min(1.0, sample_fraction))
        if sample_fraction <= 0.0:
            return zero, zero
        if sample_fraction < 1.0:
            keep = (
                torch.rand(
                    sample_indices.numel(),
                    device=device,
                    generator=self.rng,
                )
                < sample_fraction
            )
            sample_indices = sample_indices[keep]
            if sample_indices.numel() == 0:
                return zero, zero

        max_samples = int(self.cfg.train.backup_consistency_max_samples or 0)
        if max_samples > 0 and sample_indices.numel() > max_samples:
            perm = torch.randperm(
                sample_indices.numel(),
                device=device,
                generator=self.rng,
            )[:max_samples]
            sample_indices = sample_indices[perm]

        child_valid = child_valid_all[sample_indices]
        sample_count = int(sample_indices.numel())
        action_count = int(child_valid.shape[1])
        hand_dim = int(batch.features.hand_dim)
        actor = stats["backup_actor"][sample_indices].long().clamp(
            min=0,
            max=max(self.num_players - 1, 0),
        )

        flat_child_valid = child_valid.reshape(-1)
        child_context = stats["backup_child_context"][sample_indices].reshape(
            sample_count * action_count,
            -1,
        )[flat_child_valid].float()
        child_features = MLPFeatures(
            context=child_context,
            street=stats["backup_child_street"][sample_indices].reshape(-1)[
                flat_child_valid
            ],
            to_act=stats["backup_child_to_act"][sample_indices].reshape(-1)[
                flat_child_valid
            ],
            board=stats["backup_child_board"][sample_indices].reshape(-1, 5)[
                flat_child_valid
            ],
            beliefs=stats["backup_child_beliefs"][sample_indices].reshape(
                sample_count * action_count,
                -1,
            )[flat_child_valid].float(),
            hand_dim=hand_dim,
        )
        parent_features = batch.features[sample_indices]
        selected_batch = batch[sample_indices]

        with torch.no_grad():
            with self._model_autocast():
                child_output = self.model(child_features, include_policy=False)
                policy_output = self.model(
                    parent_features,
                    include_policy=True,
                    include_value=False,
                )
        if child_output.hand_values is None or policy_output.policy_logits is None:
            return zero, zero
        if child_output.hand_values.shape[-1] != hand_dim:
            raise ValueError(
                "Backup-consistency child hand dimension mismatch: "
                f"expected {hand_dim}, got {child_output.hand_values.shape[-1]}"
            )

        child_values_flat = torch.zeros(
            sample_count * action_count,
            self.num_players,
            hand_dim,
            dtype=child_output.hand_values.dtype,
            device=device,
        )
        child_values_flat[flat_child_valid] = child_output.hand_values
        child_values = child_values_flat.view(
            sample_count,
            action_count,
            self.num_players,
            hand_dim,
        ).float()
        child_actor_values = child_values.gather(
            2,
            actor[:, None, None, None].expand(-1, action_count, 1, hand_dim),
        ).squeeze(2)

        policy_logits = policy_output.policy_logits.float()
        if policy_logits.shape[-1] != action_count or policy_logits.shape[1] != hand_dim:
            raise ValueError(
                "Backup-consistency policy shape mismatch: "
                f"got {tuple(policy_logits.shape)}, expected "
                f"[{sample_count}, {hand_dim}, {action_count}]"
            )
        legal = selected_batch.legal_masks[:, None, :].to(dtype=torch.bool)
        legal_logits = torch.where(
            legal,
            policy_logits,
            torch.full_like(policy_logits, -1.0e9),
        )
        action_probs = F.softmax(legal_logits, dim=-1) * legal.to(
            dtype=policy_logits.dtype
        )
        valid_action = child_valid[:, None, :]
        action_probs = action_probs * valid_action.to(dtype=action_probs.dtype)
        valid_policy_mass = action_probs.sum(dim=-1, keepdim=True)
        action_probs = action_probs / valid_policy_mass.clamp_min(1e-12)

        backup_target = (
            action_probs.transpose(1, 2) * child_actor_values
        ).sum(dim=1).detach()
        parent_values = value_output.hand_values[sample_indices].float()
        parent_actor_values = parent_values.gather(
            1,
            actor[:, None, None].expand(-1, 1, hand_dim),
        ).squeeze(1)

        if hand_dim == PREFLOP_HANDS:
            value_weights = self.loss_fn._compact_value_weights(
                selected_batch,
                parent_actor_values.dtype,
            )
        else:
            _, allowed_hands_float, unblocked_mass = self.loss_fn._base_weights(
                selected_batch
            )
            value_weights = self.loss_fn._value_weights(
                unblocked_mass,
                allowed_hands_float,
                live_mask=self.loss_fn._live_player_mask(selected_batch),
            )
        actor_weights = value_weights.gather(
            1,
            actor[:, None, None].expand(-1, 1, hand_dim),
        ).squeeze(1)
        min_policy_mass = float(
            self.cfg.train.backup_consistency_min_policy_mass or 0.0
        )
        valid_hand = valid_policy_mass.squeeze(-1) > min_policy_mass
        weights = actor_weights.to(dtype=parent_actor_values.dtype) * valid_hand.to(
            dtype=parent_actor_values.dtype
        )
        denom = weights.sum()
        sq_error = (parent_actor_values - backup_target).square()
        loss = (sq_error * weights).sum() / denom.clamp_min(1e-8)
        loss = torch.where(denom > 0, loss, zero)
        return loss, torch.tensor(float(sample_count), device=device)

    def _compute_entropy(self, probs: torch.Tensor) -> float:
        eps = 1e-8
        norm = probs.clamp_min(eps)
        entropy = -(norm * norm.log()).sum(dim=-1).mean()
        return float(entropy.item())

    @torch.no_grad()
    def _policy_argmax_metrics(
        self,
        output: ModelOutput,
        batch: RebelBatch,
        kl_all: torch.Tensor,
    ) -> dict[str, float]:
        if output.policy_logits is None or batch.policy_targets is None:
            return {}
        legal = batch.legal_masks[:, None, :]
        logits = output.policy_logits.float()
        masked_logits = torch.where(
            legal, logits, torch.full_like(logits, float("-inf"))
        )
        model_top = masked_logits.argmax(dim=-1)
        target_top = batch.policy_targets.argmax(dim=-1)
        _, _, _, actor_belief, opp_matchup = self.loss_fn._policy_weights(batch)
        hand_weights_unnormalized = actor_belief * opp_matchup
        hand_weights = hand_weights_unnormalized / hand_weights_unnormalized.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        node_correct_mass = (
            (model_top == target_top).to(dtype=hand_weights.dtype) * hand_weights
        ).sum(dim=-1)
        action_bins = torch.arange(
            batch.policy_targets.shape[-1],
            device=kl_all.device,
        )
        hand_weights_expanded = hand_weights[:, :, None]
        model_top_action_mass = (
            (model_top[:, :, None] == action_bins[None, None, :]).to(
                dtype=hand_weights.dtype
            )
            * hand_weights_expanded
        ).sum(dim=1)
        target_top_action_mass = (
            (target_top[:, :, None] == action_bins[None, None, :]).to(
                dtype=hand_weights.dtype
            )
            * hand_weights_expanded
        ).sum(dim=1)
        node_model_top = model_top_action_mass.argmax(dim=-1)
        node_target_top = target_top_action_mass.argmax(dim=-1)

        node_weights = self.loss_fn._policy_node_weights(batch, kl_all.dtype)
        if node_weights is None:
            reduce_weights = torch.ones_like(kl_all, dtype=kl_all.dtype)
        else:
            reduce_weights = node_weights.to(dtype=kl_all.dtype)
        denom = reduce_weights.sum().clamp(min=1e-12)
        weighted_argmax_accuracy = (
            node_correct_mass.to(dtype=kl_all.dtype) * reduce_weights
        ).sum() / denom

        correct_mask = node_model_top == node_target_top
        wrong_mask = ~correct_mask
        names = ["correct_top1", "wrong_top1"]
        mask_float = torch.stack(
            [
                correct_mask.to(dtype=kl_all.dtype),
                wrong_mask.to(dtype=kl_all.dtype),
            ]
        )

        def weighted_group_stats(
            weights: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            group_weights = mask_float * weights[None, :]
            group_denoms = group_weights.sum(dim=1)
            nan = torch.full_like(group_denoms, float("nan"))
            group_kl = torch.where(
                group_denoms > 0,
                (group_weights * kl_all[None, :]).sum(dim=1)
                / group_denoms.clamp(min=1e-12),
                nan,
            )
            total = weights.sum().clamp(min=1e-12)
            group_frac = group_denoms / total
            accuracy = (
                node_correct_mass.to(dtype=kl_all.dtype) * weights
            ).sum() / total
            return accuracy, group_kl, group_frac

        _, group_kl, group_mass = weighted_group_stats(reduce_weights)
        node_accuracy, _, node_frac = weighted_group_stats(torch.ones_like(kl_all))

        out = {
            "weighted_argmax_accuracy": weighted_argmax_accuracy.item(),
            "train_weighted_argmax_accuracy": weighted_argmax_accuracy.item(),
            "node_argmax_accuracy": node_accuracy.item(),
        }
        out.update(
            {
                f"kl_{key}": value
                for key, value in zip(names, group_kl.cpu().tolist())
                if not math.isnan(value)
            }
        )
        out.update(
            {
                f"mass_{key}": value
                for key, value in zip(names, group_mass.cpu().tolist())
                if value > 0.0
            }
        )
        out.update(
            {
                f"train_weight_frac_{key}": value
                for key, value in zip(names, group_mass.cpu().tolist())
                if value > 0.0
            }
        )
        out.update(
            {
                f"node_frac_{key}": value
                for key, value in zip(names, node_frac.cpu().tolist())
                if value > 0.0
            }
        )

        reach = batch.statistics.get("policy_node_reach")
        if reach is not None:
            reach_weights = reach.to(device=kl_all.device, dtype=kl_all.dtype).clamp(
                min=0.0
            )
            reach_accuracy, reach_kl, reach_mass = weighted_group_stats(reach_weights)
            out["reach_weighted_argmax_accuracy"] = reach_accuracy.item()
            out.update(
                {
                    f"reach_kl_{key}": value
                    for key, value in zip(names, reach_kl.cpu().tolist())
                    if not math.isnan(value)
                }
            )
            out.update(
                {
                    f"reach_mass_{key}": value
                    for key, value in zip(names, reach_mass.cpu().tolist())
                    if value > 0.0
                }
            )
        return out

    @staticmethod
    def _accumulate_ratio(
        state: dict[str, tuple[torch.Tensor, torch.Tensor]],
        key: str,
        numer: torch.Tensor,
        denom: torch.Tensor,
    ) -> None:
        numer = numer.detach()
        denom = denom.detach()
        if key in state:
            prev_numer, prev_denom = state[key]
            state[key] = (prev_numer + numer, prev_denom + denom)
        else:
            state[key] = (numer.clone(), denom.clone())

    @staticmethod
    def _bucket_sum_parts(
        values: torch.Tensor,
        buckets: torch.Tensor,
        num_buckets: int,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = values.reshape(values.shape[0], -1).mean(dim=1)
        buckets = buckets.reshape(-1).to(device=values.device, dtype=torch.long)
        valid = (buckets >= 0) & (buckets < num_buckets)
        buckets = buckets[valid]
        values = values[valid]
        if weights is None:
            weights = torch.ones_like(values)
        else:
            weights = weights.reshape(-1).to(device=values.device, dtype=values.dtype)[
                valid
            ]
        numer = torch.zeros(num_buckets, device=values.device, dtype=values.dtype)
        denom = torch.zeros_like(numer)
        numer.scatter_add_(0, buckets, values * weights)
        denom.scatter_add_(0, buckets, weights)
        return numer, denom

    @staticmethod
    def _bucket_weighted_loss_parts(
        weighted_losses: torch.Tensor,
        buckets: torch.Tensor,
        num_buckets: int,
        loss_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = weighted_losses.reshape(weighted_losses.shape[0], -1).sum(dim=1)
        weights = loss_weights.reshape(loss_weights.shape[0], -1).sum(dim=1)
        buckets = buckets.reshape(-1).to(device=values.device, dtype=torch.long)
        valid = (buckets >= 0) & (buckets < num_buckets)
        buckets = buckets[valid]
        values = values[valid]
        weights = weights.to(device=values.device, dtype=values.dtype)[valid]
        numer = torch.zeros(num_buckets, device=values.device, dtype=values.dtype)
        denom = torch.zeros_like(numer)
        numer.scatter_add_(0, buckets, values)
        denom.scatter_add_(0, buckets, weights)
        return numer, denom

    @staticmethod
    def _bucket_counts(
        buckets: torch.Tensor,
        num_buckets: int,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        buckets = buckets.reshape(-1).to(dtype=torch.long)
        valid = (buckets >= 0) & (buckets < num_buckets)
        return torch.bincount(buckets[valid], minlength=num_buckets)[:num_buckets].to(
            device=buckets.device,
            dtype=dtype,
        )

    def _new_train_metric_stream(self) -> dict[str, Any]:
        dtype = torch.float32
        device = self.device
        return {
            "value_batch_street": torch.zeros(len(STREETS), device=device, dtype=dtype),
            "ratios": {},
            "argmax_ratios": {},
            "aggression_bet_sums": torch.zeros(5, device=device, dtype=dtype),
            "aggression_counts": torch.zeros(5, device=device, dtype=dtype),
            "value_sum": None,
            "value_sumsq": None,
            "value_count": torch.zeros((), device=device, dtype=dtype),
        }

    @torch.no_grad()
    def _accumulate_train_metrics(
        self,
        state: dict[str, Any],
        value_batch: RebelBatch,
        policy_batch: RebelBatch,
        value_output: ModelOutput,
        policy_output: ModelOutput,
        value_loss_all: torch.Tensor,
        value_weights: torch.Tensor,
        policy_loss_all: torch.Tensor,
        policy_target_model_kl_all: torch.Tensor,
    ) -> None:
        ratios = state["ratios"]
        street_count = self._bucket_counts(
            value_batch.features.street,
            len(STREETS),
            dtype=state["value_batch_street"].dtype,
        )
        state["value_batch_street"] += street_count

        numer, denom = self._bucket_weighted_loss_parts(
            value_loss_all,
            value_batch.features.street,
            len(STREETS),
            value_weights,
        )
        self._accumulate_ratio(ratios, "value_loss_street", numer, denom)

        numer, denom = self._bucket_sum_parts(
            policy_loss_all,
            policy_batch.features.street,
            len(STREETS),
        )
        self._accumulate_ratio(ratios, "policy_loss_street", numer, denom)

        node_weights = self.loss_fn._policy_node_weights(
            policy_batch, policy_target_model_kl_all.dtype
        )
        depth = policy_batch.statistics.get("node_depth")
        if depth is not None:
            numer, denom = self._bucket_sum_parts(
                policy_target_model_kl_all,
                depth,
                int(self.cfg.search.depth) + 1,
                weights=node_weights,
            )
            self._accumulate_ratio(
                ratios,
                "policy_target_model_kl_depth",
                numer,
                denom,
            )

        reach = policy_batch.statistics.get("policy_node_reach")
        if reach is not None:
            reach = reach.to(
                device=policy_target_model_kl_all.device,
                dtype=policy_target_model_kl_all.dtype,
            )
            reach_bucket = torch.full_like(reach, 4, dtype=torch.long)
            reach_bucket = torch.where(reach >= 1e-4, 3, reach_bucket)
            reach_bucket = torch.where(reach >= 1e-3, 2, reach_bucket)
            reach_bucket = torch.where(reach >= 1e-2, 1, reach_bucket)
            reach_bucket = torch.where(reach >= 1e-1, 0, reach_bucket)
            numer, denom = self._bucket_sum_parts(
                policy_target_model_kl_all,
                reach_bucket,
                5,
                weights=node_weights,
            )
            self._accumulate_ratio(
                ratios,
                "policy_target_model_kl_reach_bucket",
                numer,
                denom,
            )

        self._accumulate_policy_argmax_metric_parts(
            state["argmax_ratios"],
            policy_output,
            policy_batch,
            policy_target_model_kl_all,
        )

        aggression = self.aggression_analyzer.analyze_batch(
            policy_batch, max_batch_size=self.batch_size
        )
        if aggression:
            counts = aggression["group_counts"].to(
                device=state["aggression_counts"].device,
                dtype=state["aggression_counts"].dtype,
            )
            avg_bets = aggression["group_avg_bets"].to(
                device=state["aggression_bet_sums"].device,
                dtype=state["aggression_bet_sums"].dtype,
            )
            state["aggression_counts"] += counts
            state["aggression_bet_sums"] += avg_bets * counts

        if value_output.value is not None:
            value = value_output.value.detach().float()
            value_sum = value.sum(dim=0)
            value_sumsq = (value * value).sum(dim=0)
            if state["value_sum"] is None:
                state["value_sum"] = value_sum.clone()
                state["value_sumsq"] = value_sumsq.clone()
            else:
                state["value_sum"] += value_sum
                state["value_sumsq"] += value_sumsq
            state["value_count"] += value.shape[0]

    @torch.no_grad()
    def _accumulate_policy_argmax_metric_parts(
        self,
        ratios: dict[str, tuple[torch.Tensor, torch.Tensor]],
        output: ModelOutput,
        batch: RebelBatch,
        kl_all: torch.Tensor,
    ) -> None:
        if output.policy_logits is None or batch.policy_targets is None:
            return
        legal = batch.legal_masks[:, None, :]
        logits = output.policy_logits.float()
        masked_logits = torch.where(
            legal, logits, torch.full_like(logits, float("-inf"))
        )
        model_top = masked_logits.argmax(dim=-1)
        target_top = batch.policy_targets.argmax(dim=-1)
        _, _, _, actor_belief, opp_matchup = self.loss_fn._policy_weights(batch)
        hand_weights_unnormalized = actor_belief * opp_matchup
        hand_weights = hand_weights_unnormalized / hand_weights_unnormalized.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        metric_dtype = kl_all.dtype
        node_correct_mass = (
            (model_top == target_top).to(dtype=hand_weights.dtype) * hand_weights
        ).sum(dim=-1)
        action_bins = torch.arange(batch.policy_targets.shape[-1], device=kl_all.device)
        hand_weights_expanded = hand_weights[:, :, None]
        model_top_action_mass = (
            (model_top[:, :, None] == action_bins[None, None, :]).to(
                dtype=hand_weights.dtype
            )
            * hand_weights_expanded
        ).sum(dim=1)
        target_top_action_mass = (
            (target_top[:, :, None] == action_bins[None, None, :]).to(
                dtype=hand_weights.dtype
            )
            * hand_weights_expanded
        ).sum(dim=1)
        node_model_top = model_top_action_mass.argmax(dim=-1)
        node_target_top = target_top_action_mass.argmax(dim=-1)

        node_weights = self.loss_fn._policy_node_weights(batch, metric_dtype)
        reduce_weights = (
            torch.ones_like(kl_all, dtype=metric_dtype)
            if node_weights is None
            else node_weights.to(dtype=metric_dtype)
        )
        correct_mass = node_correct_mass.to(dtype=metric_dtype)
        self._accumulate_ratio(
            ratios,
            "weighted_argmax_accuracy",
            (correct_mass * reduce_weights).sum(),
            reduce_weights.sum(),
        )
        self._accumulate_ratio(
            ratios,
            "train_weighted_argmax_accuracy",
            (correct_mass * reduce_weights).sum(),
            reduce_weights.sum(),
        )
        node_unit_weights = torch.ones_like(kl_all, dtype=metric_dtype)
        self._accumulate_ratio(
            ratios,
            "node_argmax_accuracy",
            correct_mass.sum(),
            node_unit_weights.sum(),
        )

        correct_mask = node_model_top == node_target_top
        group_ids = torch.where(
            correct_mask,
            torch.zeros_like(node_model_top, dtype=torch.long),
            torch.ones_like(node_model_top, dtype=torch.long),
        )
        names = ["correct_top1", "wrong_top1"]

        def add_group(prefix: str, weights: torch.Tensor, values: torch.Tensor) -> None:
            group_den = torch.zeros(2, device=values.device, dtype=values.dtype)
            group_num = torch.zeros_like(group_den)
            group_den.scatter_add_(0, group_ids, weights)
            group_num.scatter_add_(0, group_ids, weights * values)
            total = weights.sum()
            for idx, name in enumerate(names):
                self._accumulate_ratio(
                    ratios,
                    f"{prefix}{name}",
                    group_num[idx],
                    group_den[idx],
                )
                if prefix == "kl_":
                    self._accumulate_ratio(
                        ratios,
                        f"mass_{name}",
                        group_den[idx],
                        total,
                    )
                    self._accumulate_ratio(
                        ratios,
                        f"train_weight_frac_{name}",
                        group_den[idx],
                        total,
                    )

        add_group("kl_", reduce_weights, kl_all.to(dtype=metric_dtype))

        node_group_den = torch.zeros(2, device=kl_all.device, dtype=metric_dtype)
        node_group_den.scatter_add_(0, group_ids, node_unit_weights)
        node_total = node_unit_weights.sum()
        for idx, name in enumerate(names):
            self._accumulate_ratio(
                ratios,
                f"node_frac_{name}",
                node_group_den[idx],
                node_total,
            )

        reach = batch.statistics.get("policy_node_reach")
        if reach is not None:
            reach_weights = reach.to(device=kl_all.device, dtype=metric_dtype).clamp(
                min=0.0
            )
            self._accumulate_ratio(
                ratios,
                "reach_weighted_argmax_accuracy",
                (correct_mass * reach_weights).sum(),
                reach_weights.sum(),
            )
            group_den = torch.zeros(2, device=kl_all.device, dtype=metric_dtype)
            group_num = torch.zeros_like(group_den)
            group_den.scatter_add_(0, group_ids, reach_weights)
            group_num.scatter_add_(
                0, group_ids, reach_weights * kl_all.to(metric_dtype)
            )
            total = reach_weights.sum()
            for idx, name in enumerate(names):
                self._accumulate_ratio(
                    ratios,
                    f"reach_kl_{name}",
                    group_num[idx],
                    group_den[idx],
                )
                self._accumulate_ratio(
                    ratios,
                    f"reach_mass_{name}",
                    group_den[idx],
                    total,
                )

    def _finalize_train_metric_stream(
        self, state: dict[str, Any]
    ) -> dict[str, float | dict[str, float]]:
        def ratio_vector(key: str, names: list[str]) -> dict[str, float]:
            if key not in state["ratios"]:
                return {}
            numer, denom = state["ratios"][key]
            vals = torch.where(
                denom > 0,
                numer / denom.clamp(min=1e-12),
                torch.full_like(denom, float("nan")),
            )
            vals_cpu = vals.cpu().tolist()
            return {k: v for k, v in zip(names, vals_cpu) if not math.isnan(v)}

        def ratio_scalars(
            ratios: dict[str, tuple[torch.Tensor, torch.Tensor]],
        ) -> dict[str, float]:
            out: dict[str, float] = {}
            for key, (numer, denom) in ratios.items():
                if bool((denom > 0).item()):
                    out[key] = (numer / denom.clamp(min=1e-12)).item()
            return out

        street_counts = state["value_batch_street"].cpu().tolist()
        aggression_counts = state["aggression_counts"]
        aggression_avg = torch.where(
            aggression_counts > 0,
            state["aggression_bet_sums"] / aggression_counts.clamp(min=1e-12),
            torch.zeros_like(aggression_counts),
        )
        metrics: dict[str, float | dict[str, float]] = {
            "value_batch_street": {
                name: street_counts[i] for i, name in enumerate(STREETS)
            },
            "value_loss_street": ratio_vector("value_loss_street", list(STREETS)),
            "policy_loss_street": ratio_vector("policy_loss_street", list(STREETS)),
            "policy_target_model_kl_depth": ratio_vector(
                "policy_target_model_kl_depth",
                [
                    f"depth_{i}"
                    for i in range(int(self.cfg.search.depth) + 1)
                ],
            ),
            "policy_target_model_kl_reach_bucket": ratio_vector(
                "policy_target_model_kl_reach_bucket",
                [
                    "ge_1e-1",
                    "1e-2_to_1e-1",
                    "1e-3_to_1e-2",
                    "1e-4_to_1e-3",
                    "lt_1e-4",
                ],
            ),
            "policy_argmax_metrics": ratio_scalars(state["argmax_ratios"]),
            "aggression_stats": {
                f"chunk_{i}": v for i, v in enumerate(aggression_avg.cpu().tolist())
            },
        }

        if state["value_sum"] is not None and bool((state["value_count"] > 1).item()):
            count = state["value_count"]
            value_sum = state["value_sum"]
            value_sumsq = state["value_sumsq"]
            var = (value_sumsq - (value_sum * value_sum) / count) / (count - 1)
            metrics["value_mean_std"] = var.clamp_min(0.0).sqrt().mean().item()
        else:
            metrics["value_mean_std"] = 0.0

        return metrics

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
        value_weights: torch.Tensor,
        policy_loss_all: torch.Tensor,
        policy_target_model_kl_all: torch.Tensor,
        fresh_value_loss: float | None = None,
        fresh_value_batch: RebelBatch | None = None,
        fresh_policy_batch: RebelBatch | None = None,
        streamed_train_metrics: dict[str, Any] | None = None,
    ) -> dict[str, int | float | torch.Tensor | dict[str, int | float]]:
        grad_norm_clipped = torch.nn.utils.get_total_norm(
            p.grad for p in self.model.parameters() if p.grad is not None
        ).item()

        def by_street(
            tensor: torch.Tensor,
            batch=value_batch,
            street=None,
            weights=None,
            denominator_weights=None,
        ) -> dict[str, float]:
            # Stack the per-street reductions and pull them across in a
            # single DtoH instead of one .item() per street. Empty streets
            # produce NaN which the dict comprehension below filters out.
            if street is None:
                street = batch.features.street
            names = list(STREETS)
            if denominator_weights is not None:
                numers = torch.stack(
                    [tensor[street == i].sum() for i, _ in enumerate(names)]
                )
                denoms = torch.stack(
                    [
                        denominator_weights[street == i].sum()
                        for i, _ in enumerate(names)
                    ]
                )
                vals = torch.where(
                    denoms > 0,
                    numers / denoms.clamp(min=1e-12),
                    torch.full_like(denoms, float("nan")),
                )
            elif weights is not None:
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
            max_depth = int(self.cfg.search.depth)
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

        has_replay_buffers = (
            self.value_buffer is not None and self.policy_buffer is not None
        )
        value_buffer_streets_stats = (
            street_count(self.value_buffer.features.street[: len(self.value_buffer)])
            if has_replay_buffers
            else {}
        )
        if has_replay_buffers and len(self.value_buffer) > 0:
            value_buffer_hand_dim = self.value_buffer.features.hand_dim
            value_buffer_target_mean_abs_tensor = (
                (
                    self.value_buffer.value_targets[: len(self.value_buffer)]
                    * self.value_buffer.features.beliefs[: len(self.value_buffer)].view(
                        -1, self.num_players, value_buffer_hand_dim
                    )
                )
                .abs()
                .sum(dim=2)
            )
            value_buffer_target_mean_abs = (
                value_buffer_target_mean_abs_tensor.mean().item()
            )
            value_buffer_target_mean_abs_street = by_street(
                value_buffer_target_mean_abs_tensor.mean(dim=1),
                street=self.value_buffer.features.street[: len(self.value_buffer)],
            )
        else:
            value_buffer_target_mean_abs = 0.0
            value_buffer_target_mean_abs_street = {}
        if streamed_train_metrics is None:
            aggression_stats = {
                f"chunk_{i}": v
                for i, v in enumerate(
                    self.aggression_analyzer.analyze_batch(
                        policy_batch, max_batch_size=self.batch_size
                    )["group_avg_bets"].tolist()
                )
            }
            value_batch_street = street_count(value_batch.features.street)
            value_loss_street = by_street(
                value_loss_all, denominator_weights=value_weights
            )
            policy_loss_street = by_street(policy_loss_all, batch=policy_batch)
            policy_target_model_kl_depth = policy_metric_by_depth(
                policy_target_model_kl_all, policy_batch
            )
            policy_target_model_kl_reach_bucket = policy_metric_by_reach_bucket(
                policy_target_model_kl_all, policy_batch
            )
            policy_argmax_metrics = (
                self._policy_argmax_metrics(
                    policy_output, policy_batch, policy_target_model_kl_all
                )
                if policy_output is not None
                else {}
            )
            value_mean_std = (
                value_output.value.std(dim=0).mean()
                if value_output.value is not None
                else 0.0
            )
        else:
            aggression_stats = streamed_train_metrics["aggression_stats"]
            value_batch_street = streamed_train_metrics["value_batch_street"]
            value_loss_street = streamed_train_metrics["value_loss_street"]
            policy_loss_street = streamed_train_metrics["policy_loss_street"]
            policy_target_model_kl_depth = streamed_train_metrics[
                "policy_target_model_kl_depth"
            ]
            policy_target_model_kl_reach_bucket = streamed_train_metrics[
                "policy_target_model_kl_reach_bucket"
            ]
            policy_argmax_metrics = streamed_train_metrics["policy_argmax_metrics"]
            value_mean_std = streamed_train_metrics["value_mean_std"]

        metrics: dict[str, int | float | torch.Tensor | dict[str, int | float]] = {
            "episodes": episodes,
            "updates": updates,
            "optimizer_updates": updates + self.policy_extra_updates_per_step,
            "loss": step_stats["total_loss"] / episodes,
            "policy_loss": step_stats["policy_loss"] / episodes,
            "policy_target_entropy": step_stats["policy_target_entropy"] / episodes,
            "policy_model_entropy": step_stats["policy_model_entropy"] / episodes,
            "policy_entropy_gap": step_stats["policy_entropy_gap"] / episodes,
            "policy_target_model_kl": (step_stats["policy_target_model_kl"] / episodes),
            "policy_logit_l2": step_stats["policy_logit_l2"] / episodes,
            "value_loss": step_stats["value_loss"] / episodes,
            "entropy_loss": step_stats["entropy_loss"] / episodes,
            "permutation_loss": step_stats["permutation_loss"] / episodes,
            "backup_consistency_loss": (
                step_stats["backup_consistency_loss"] / episodes
            ),
            "backup_consistency_weighted_loss": (
                step_stats["backup_consistency_weighted_loss"] / episodes
            ),
            "backup_consistency_samples": (
                step_stats["backup_consistency_samples"] / episodes
            ),
            "param_update_norm": step_stats["update_norm"] / episodes,
            "value_buffer": value_buffer_streets_stats,
            "value_buffer_size": len(self.value_buffer) if has_replay_buffers else 0,
            "policy_buffer_size": len(self.policy_buffer) if has_replay_buffers else 0,
            "value_buffer_mean_sample_count": (
                self.value_buffer.sample_count[: len(self.value_buffer)]
                .float()
                .mean()
                .item()
                if has_replay_buffers and len(self.value_buffer) > 0
                else 0.0
            ),
            "value_buffer_target_mean_abs": value_buffer_target_mean_abs,
            "value_buffer_target_mean_abs_street": value_buffer_target_mean_abs_street,
            "policy_buffer_mean_sample_count": (
                self.policy_buffer.sample_count[: len(self.policy_buffer)]
                .float()
                .mean()
                .item()
                if has_replay_buffers and len(self.policy_buffer) > 0
                else 0.0
            ),
            "grad_norm_clipped": grad_norm_clipped,
            "aggression_stats": aggression_stats,
            "value_batch_street": value_batch_street,
            "value_loss_street": value_loss_street,
            "policy_loss_street": policy_loss_street,
            "policy_target_model_kl_depth": policy_target_model_kl_depth,
            "policy_target_model_kl_reach_bucket": policy_target_model_kl_reach_bucket,
            "policy_argmax_metrics": policy_argmax_metrics,
            "value_mean_std": value_mean_std,
            **self.cfr_evaluator.stats,
        }
        if self.policy_extra_updates_per_step > 0:
            extra_updates = self.policy_extra_updates_per_step
            metrics.update(
                {
                    "extra_policy_updates": extra_updates,
                    "extra_policy_batch_size": self.policy_extra_batch_size,
                    "extra_policy_loss": step_stats["extra_policy_loss"]
                    / extra_updates,
                    "extra_policy_target_entropy": step_stats[
                        "extra_policy_target_entropy"
                    ]
                    / extra_updates,
                    "extra_policy_model_entropy": step_stats[
                        "extra_policy_model_entropy"
                    ]
                    / extra_updates,
                    "extra_policy_entropy_gap": step_stats["extra_policy_entropy_gap"]
                    / extra_updates,
                    "extra_policy_target_model_kl": step_stats[
                        "extra_policy_target_model_kl"
                    ]
                    / extra_updates,
                    "extra_policy_logit_l2": step_stats["extra_policy_logit_l2"]
                    / extra_updates,
                    "extra_entropy_loss": step_stats["extra_entropy_loss"]
                    / extra_updates,
                    "extra_loss": step_stats["extra_total_loss"] / extra_updates,
                    "extra_param_update_norm": step_stats["extra_update_norm"]
                    / extra_updates,
                }
            )

        if value_batch.value_targets is not None:
            metrics["batch_value_target_mean_abs"] = (
                value_batch.value_targets.abs().mean().item()
            )
            metrics["batch_value_target_std"] = value_batch.value_targets.std().item()

        # Calculate loss on fresh data
        if fresh_value_batch:
            with torch.no_grad():
                self.inference_model.eval()
                fresh_value_batch = fresh_value_batch.to(self.device)
                with self._model_autocast():
                    fresh_model_output = self.inference_model.repeat(
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
                    metrics["fresh_value_loss_avg"] = metrics["fresh_value_loss"]
                    with self._model_autocast():
                        model_avg_output = self.inference_model.repeat(
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
            fresh_value_hand_dim = fresh_value_batch.features.hand_dim
            metrics["fresh_value_batch_street"] = street_count(
                fresh_value_batch.features.street
            )
            metrics["fresh_value_target_mean_abs"] = (
                (
                    fresh_value_batch.value_targets
                    * fresh_value_batch.features.beliefs.view(
                        -1, self.num_players, fresh_value_hand_dim
                    )
                )
                .abs()
                .sum(dim=2)
                .mean()
                .item()
            )
            metrics["fresh_value_target_mean_abs_street"] = by_street(
                (
                    fresh_value_batch.value_targets
                    * fresh_value_batch.features.beliefs.view(
                        -1, self.num_players, fresh_value_hand_dim
                    )
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
                self.inference_model.eval()
                fresh_policy_batch = fresh_policy_batch.to(self.device)
                with self._model_autocast():
                    fresh_policy_output = self.inference_model.repeat(
                        fresh_policy_batch.features,
                        count=self.cfg.model.num_supervisions,
                        include_policy=True,
                        include_value=False,
                    )
                fresh_policy_loss_dict = self.loss_fn.forward_policy(
                    fresh_policy_output, fresh_policy_batch
                )
                metrics["fresh_policy_loss"] = fresh_policy_loss_dict[
                    "policy_loss"
                ].item()
                metrics["fresh_policy_target_model_kl"] = fresh_policy_loss_dict[
                    "target_model_kl"
                ].item()
                metrics["fresh_policy_target_entropy"] = fresh_policy_loss_dict[
                    "target_entropy"
                ].item()
                metrics["fresh_policy_model_entropy"] = fresh_policy_loss_dict[
                    "model_entropy"
                ].item()
                metrics["fresh_policy_entropy_gap"] = fresh_policy_loss_dict[
                    "entropy_gap"
                ].item()
                metrics["fresh_policy_logit_l2"] = fresh_policy_loss_dict[
                    "policy_logit_l2"
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
                metrics["fresh_policy_target_entropy_reach_bucket"] = (
                    policy_metric_by_reach_bucket(
                        fresh_policy_loss_dict["target_entropy_all"],
                        fresh_policy_batch,
                    )
                )
                metrics["fresh_policy_model_entropy_reach_bucket"] = (
                    policy_metric_by_reach_bucket(
                        fresh_policy_loss_dict["model_entropy_all"],
                        fresh_policy_batch,
                    )
                )
                metrics["fresh_policy_entropy_gap_reach_bucket"] = (
                    policy_metric_by_reach_bucket(
                        fresh_policy_loss_dict["entropy_gap_all"],
                        fresh_policy_batch,
                    )
                )
                metrics["fresh_policy_argmax_metrics"] = self._policy_argmax_metrics(
                    fresh_policy_output,
                    fresh_policy_batch,
                    fresh_policy_loss_dict["target_model_kl_all"],
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
        torch.Tensor,
    ]:
        self.optimizer.zero_grad(set_to_none=True)

        value_loss, policy_loss, entropy_loss = None, None, None
        value_loss_update, policy_loss_update = None, None

        with self._model_autocast():
            if type(self.model) is BetterTRM:
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
        value_weights_update = loss_dict["value_weights"]
        total_loss = loss_dict["total_loss"]

        permutation_loss_tensor = self._compute_permutation_loss(
            value_output_orig, value_output_permuted, suit_permutations_idxs
        )
        total_loss = (
            total_loss + self.loss_fn.permutation_weight * permutation_loss_tensor
        )

        backup_consistency_loss, backup_consistency_samples = (
            self._backup_consistency_loss(value_batch, value_output_orig)
        )
        backup_consistency_weighted_loss = (
            self._backup_consistency_coef() * backup_consistency_loss
        )
        total_loss = total_loss + backup_consistency_weighted_loss

        with self._model_autocast():
            if type(self.model) is BetterTRM:
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
        policy_logit_l2 = loss_dict["policy_logit_l2"]
        model_entropy = loss_dict["model_entropy"]
        entropy_gap = loss_dict["entropy_gap"]
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

        grad_norm = self._clip_grads()

        self.optimizer.step()

        # Update EMA if enabled. The inference model is synced from shadow
        # weights once all training updates for this step are done.
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
                "policy_model_entropy": model_entropy.detach(),
                "policy_entropy_gap": entropy_gap.detach(),
                "policy_target_model_kl": target_model_kl.detach(),
                "policy_logit_l2": policy_logit_l2.detach(),
                "value_loss": value_loss.detach(),
                "entropy_loss": entropy_loss.detach(),
                "permutation_loss": permutation_loss_tensor.detach(),
                "backup_consistency_loss": backup_consistency_loss.detach(),
                "backup_consistency_weighted_loss": (
                    backup_consistency_weighted_loss.detach()
                ),
                "backup_consistency_samples": backup_consistency_samples.detach(),
                "total_loss": total_loss.detach(),
                "update_norm": update_norm.detach(),
            },
            value_output_permuted,
            value_output_orig,
            policy_output,
            value_loss_update,
            value_weights_update,
            policy_loss_update,
            policy_kl_update,
        )

    def _supervise_policy_only(
        self,
        policy_batch: RebelBatch,
        policy_latent: TRMLatent | None = None,
    ) -> dict[str, torch.Tensor]:
        self.optimizer.zero_grad(set_to_none=True)

        with self._model_autocast():
            if type(self.model) is BetterTRM:
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
        total_loss = loss_dict["total_loss"]
        total_loss.backward()

        if self.CHECK_GRADS:
            grad_finite = torch.stack(
                [
                    p.grad.isfinite().all()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
            ).all()
            assert grad_finite.item(), "NaN/Inf in model gradients"

        grad_norm = self._clip_grads()

        self.optimizer.step()

        if self.ema_helper is not None:
            self.ema_helper.update(self.model)

        current_lr = self.optimizer.param_groups[0]["lr"]
        update_norm = current_lr * grad_norm

        return {
            "policy_loss": loss_dict["policy_loss"].detach(),
            "policy_target_entropy": loss_dict["target_entropy"].detach(),
            "policy_model_entropy": loss_dict["model_entropy"].detach(),
            "policy_entropy_gap": loss_dict["entropy_gap"].detach(),
            "policy_target_model_kl": loss_dict["target_model_kl"].detach(),
            "policy_logit_l2": loss_dict["policy_logit_l2"].detach(),
            "entropy_loss": loss_dict["entropy"].detach(),
            "total_loss": total_loss.detach(),
            "update_norm": update_norm.detach(),
        }

    def train_value_batch(
        self,
        value_batch: RebelBatch,
        step: int,
        *,
        sync_inference_model: bool = True,
    ) -> dict[str, Any]:
        """Run one value-only supervised update for distillation workloads."""

        self._apply_schedules(step)
        value_batch = value_batch.to(self.device)
        self.model.train()

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

        self.optimizer.zero_grad(set_to_none=True)
        with self._model_autocast():
            if type(self.model) is BetterTRM:
                value_output_orig = self.model(
                    value_batch.features,
                    include_policy=False,
                )
                value_output_permuted = self.model(
                    permuted_batch.features,
                    include_policy=False,
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
        permutation_loss = self._compute_permutation_loss(
            value_output_orig, value_output_permuted, suit_permutations_idxs
        )
        total_loss = (
            loss_dict["total_loss"] + self.loss_fn.permutation_weight * permutation_loss
        )
        total_loss.backward()

        if self.CHECK_GRADS:
            grad_finite = torch.stack(
                [
                    p.grad.isfinite().all()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
            ).all()
            assert grad_finite.item(), "NaN/Inf in model gradients"

        grad_norm = self._clip_grads()
        self.optimizer.step()

        if self.ema_helper is not None:
            self.ema_helper.update(self.model)
        if sync_inference_model:
            self._sync_inference_model()

        current_lr = self.optimizer.param_groups[0]["lr"]
        update_norm = current_lr * grad_norm
        street = value_batch.statistics.get("street")
        evaluator_street = (
            float(street.float().mean().item()) if street is not None else None
        )
        total_loss_detached = total_loss.detach()
        value_loss = loss_dict["value_loss"].detach()
        return {
            "step": step + 1,
            "loss": float(total_loss_detached.item()),
            "total_loss": float(total_loss_detached.item()),
            "policy_loss": None,
            "value_loss": float(value_loss.item()),
            "permutation_loss": float(permutation_loss.detach().item()),
            "update_norm": float(update_norm.detach().item()),
            "local_exploitability": None,
            "local_exploitability_mbbg": None,
            "evaluator_street": evaluator_street,
            "learning_rate": current_lr,
        }

    @profile
    def _update_model(
        self, step: int
    ) -> dict[str, int | float | torch.Tensor | dict[str, int | float]]:
        debug = bool(os.environ.get("P2_DEBUG_TRAINER_INIT"))

        def trace(message: str) -> None:
            if debug:
                print(f"[RebelCFRTrainer:update] {message}", flush=True)

        trace(f"step {step} prepare_step start")
        fresh_value_batch, fresh_policy_batch = self.data_source.prepare_step(step)
        trace(
            f"step {step} prepare_step done "
            f"value={len(fresh_value_batch) if fresh_value_batch is not None else 0} "
            f"policy={len(fresh_policy_batch) if fresh_policy_batch is not None else 0}"
        )

        # Warmup: make sure we have enough samples.
        min_policy_samples = max(
            self.batch_size,
            (
                self.policy_extra_batch_size
                if self.policy_extra_updates_per_step > 0
                else self.batch_size
            ),
        )
        trace(
            f"step {step} ensure_min_samples start "
            f"value={self.batch_size} policy={min_policy_samples}"
        )
        self.data_source.ensure_min_samples(self.batch_size, min_policy_samples)
        trace("ensure_min_samples done")

        supervisions = (
            self.cfg.model.num_supervisions if type(self.model) is BetterTRM else 1
        )
        if self.stream_live_batches:
            episodes = 1
        else:
            if self.value_buffer is None:
                raise RuntimeError("Buffered data source requires value_buffer")
            value_fullness = len(self.value_buffer) / self.value_buffer.capacity
            episodes = math.ceil(self.cfg.train.episodes_per_step * value_fullness)
        updates = episodes * supervisions
        metric_value_batch = None
        metric_policy_batch = None
        metric_value_output = None
        metric_policy_output = None
        metric_value_loss_all = None
        metric_value_weights = None
        metric_policy_loss_all = None
        metric_policy_kl_all = None
        streamed_metric_state = self._new_train_metric_stream()
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
            "policy_model_entropy": None,
            "policy_entropy_gap": None,
            "policy_target_model_kl": None,
            "policy_logit_l2": None,
            "value_loss": None,
            "entropy_loss": None,
            "permutation_loss": None,
            "backup_consistency_loss": None,
            "backup_consistency_weighted_loss": None,
            "backup_consistency_samples": None,
            "total_loss": None,
            "update_norm": None,
        }
        extra_tensor_stats: dict[str, torch.Tensor | None] = {
            "policy_loss": None,
            "policy_target_entropy": None,
            "policy_model_entropy": None,
            "policy_entropy_gap": None,
            "policy_target_model_kl": None,
            "policy_logit_l2": None,
            "entropy_loss": None,
            "total_loss": None,
            "update_norm": None,
        }
        stratify = self._get_stratify_streets(step)

        self.model.train()
        for episode in range(episodes):
            trace(f"episode {episode} sample batches start")
            value_latent, policy_latent, permuted_latent = None, None, None
            # TODO: think about how to interleave these/ratio in a smarter way.
            # Might need to use different sizes for the two batches.
            value_batch = self.data_source.sample_value(
                self.batch_size, stratify_streets=stratify
            ).to(self.device)
            policy_stratify = (
                None if self.cfg.train.policy_depth_stratify_sample else stratify
            )
            policy_batch = self.data_source.sample_policy(
                self.batch_size, stratify_streets=policy_stratify
            ).to(self.device)
            trace(f"episode {episode} sample batches done")

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
                trace(f"episode {episode} supervise start")
                (
                    episode_stats,
                    permuted_value_output,
                    value_output_orig,
                    policy_output,
                    value_loss_update,
                    value_weights_update,
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
                trace(f"episode {episode} supervise done")
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

            # Keep the last bounded batch for metric paths that still need a
            # concrete batch; aggregate diagnostics stream into small tensors.
            metric_value_batch = permuted_batch
            metric_policy_batch = policy_batch
            metric_value_output = permuted_value_output
            metric_policy_output = policy_output
            metric_value_loss_all = value_loss_update
            metric_value_weights = value_weights_update
            metric_policy_loss_all = policy_loss_update
            metric_policy_kl_all = policy_kl_update
            self._accumulate_train_metrics(
                streamed_metric_state,
                permuted_batch,
                policy_batch,
                permuted_value_output,
                policy_output,
                value_loss_update,
                value_weights_update,
                policy_loss_update,
                policy_kl_update,
            )

        policy_stratify = (
            None if self.cfg.train.policy_depth_stratify_sample else stratify
        )
        for _ in range(self.policy_extra_updates_per_step):
            extra_policy_batch = self.data_source.sample_policy(
                self.policy_extra_batch_size,
                stratify_streets=policy_stratify,
            ).to(self.device)
            extra_stats = self._supervise_policy_only(extra_policy_batch)
            for k, acc in extra_tensor_stats.items():
                v = extra_stats[k]
                extra_tensor_stats[k] = v if acc is None else acc + v

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
        if any(v is not None for v in extra_tensor_stats.values()):
            keys = [k for k, v in extra_tensor_stats.items() if v is not None]
            stacked = (
                torch.stack([extra_tensor_stats[k].reshape(()) for k in keys])
                .cpu()
                .tolist()
            )
            for k, val in zip(keys, stacked):
                step_stats[f"extra_{k}"] = val
        for k in extra_tensor_stats:
            step_stats.setdefault(f"extra_{k}", 0.0)

        self._sync_inference_model()

        if (
            metric_value_batch is None
            or metric_policy_batch is None
            or metric_value_output is None
            or metric_policy_output is None
            or metric_value_loss_all is None
            or metric_value_weights is None
            or metric_policy_loss_all is None
            or metric_policy_kl_all is None
        ):
            raise RuntimeError("No training batch was available for metric reporting.")
        metrics = self._compute_metrics(
            episodes,
            updates,
            step_stats,
            metric_value_batch,
            metric_policy_batch,
            metric_value_output,
            metric_policy_output,
            metric_value_loss_all,
            metric_value_weights,
            metric_policy_loss_all,
            metric_policy_kl_all,
            fresh_value_batch=fresh_value_batch,
            fresh_policy_batch=fresh_policy_batch,
            streamed_train_metrics=self._finalize_train_metric_stream(
                streamed_metric_state
            ),
        )

        return metrics

    def train_step(self, step: int) -> dict[str, Any]:
        step_public = step + 1

        # Apply schedules before training step
        self._apply_schedules(step)
        cfr_target_promoted = self._maybe_promote_cfr_target_model(step)

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
        if self.target_update_block_batches > 0:
            update_info["cfr_target_update_block_batches"] = (
                self.target_update_block_batches
            )
            update_info["cfr_target_model_step"] = self.cfr_target_model_step
            update_info["cfr_target_block_index"] = (
                step // self.target_update_block_batches
            )
            update_info["cfr_target_promoted"] = float(cfr_target_promoted)

        return update_info

    def train(self, num_steps: int | None = None) -> list[dict[str, Any]]:
        total_steps = num_steps or self.cfg.num_steps
        history: list[dict[str, Any]] = []

        for step in range(total_steps):
            update_info = self.train_step(step)
            history.append(update_info)

        return history

    @staticmethod
    def replay_buffer_checkpoint_path(checkpoint_path: str) -> str:
        directory = os.path.dirname(checkpoint_path)
        if not directory:
            directory = "."
        return os.path.join(directory, REPLAY_BUFFER_CHECKPOINT)

    def save_replay_buffers(self, checkpoint_path: str, step: int) -> str:
        if self.value_buffer is None or self.policy_buffer is None:
            return ""
        path = self.replay_buffer_checkpoint_path(checkpoint_path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "step": step,
            "value_buffer": self.value_buffer.state_dict(),
            "policy_buffer": self.policy_buffer.state_dict(),
        }
        tmp_path = f"{path}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        return path

    def load_replay_buffers(self, checkpoint_path: str) -> bool:
        if self.value_buffer is None or self.policy_buffer is None:
            return False
        path = self.replay_buffer_checkpoint_path(checkpoint_path)
        if not os.path.exists(path):
            return False
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.value_buffer.load_state_dict(payload["value_buffer"])
        self.policy_buffer.load_state_dict(payload["policy_buffer"])
        return True

    def save_checkpoint(
        self,
        path: str,
        step: int,
        wandb_run_id: str | None = None,
        save_optimizer: bool = True,
        save_dtype: torch.dtype | None = None,
        batch: RebelBatch | None = None,
        metadata: dict[str, object] | None = None,
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
            "replay_buffer_checkpoint": (
                REPLAY_BUFFER_CHECKPOINT
                if self.value_buffer is not None and self.policy_buffer is not None
                else None
            ),
            # Store wandb run ID for resumption
            "wandb_run_id": wandb_run_id,
        }
        if metadata is not None:
            state["metadata"] = dict(metadata)

        # Only save optimizer and RNG state if requested
        if save_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
            state["rng"] = self.rng.get_state()
            state["buffer_rng"] = self.buffer_rng.get_state()

        # Save EMA shadow weights if enabled.
        if self.ema_helper is not None:
            model_avg_state = dict(self.ema_helper.shadow)
            if save_dtype is not None:
                model_avg_state = {
                    k: v.to(save_dtype) if v.dtype.is_floating_point else v
                    for k, v in model_avg_state.items()
                }
            state["model_avg"] = model_avg_state

        if self.cfr_target_model is not None:
            cfr_target_state = self.cfr_target_model.state_dict()
            if save_dtype is not None:
                cfr_target_state = {
                    k: v.to(save_dtype) if v.dtype.is_floating_point else v
                    for k, v in cfr_target_state.items()
                }
            state["cfr_target_model"] = cfr_target_state
            state["cfr_target_model_step"] = self.cfr_target_model_step
            state["cfr_target_update_block_batches"] = (
                self.target_update_block_batches
            )

        # Save batch if provided (move to CPU for storage)
        if batch is not None:
            batch_cpu = batch.to(torch.device("cpu"))
            state["batch"] = batch_cpu
        data_source_state = self.data_source.state_dict()
        state["data_source"] = data_source_state
        # Compatibility for checkpoints written before the data-source refactor.
        state["data_generator"] = data_source_state

        save_replay_buffers = bool(self.cfg.train.save_replay_buffers)
        if not save_replay_buffers:
            state["replay_buffer_checkpoint"] = None

        torch.save(state, path)
        if save_replay_buffers:
            self.save_replay_buffers(path, step)

    def save_value_checkpoint(
        self,
        path: str,
        step: int,
        wandb_run_id: str | None = None,
        save_optimizer: bool = True,
        save_dtype: torch.dtype | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if type(self.model) is not BetterSplitFFN:
            raise TypeError("value-only checkpoints require model.name=BetterFFN")
        value_model = self.model.value_model
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        model_state = value_model.state_dict()
        if save_dtype is not None:
            model_state = {
                k: v.to(save_dtype) if v.dtype.is_floating_point else v
                for k, v in model_state.items()
            }

        state = {
            "model": model_state,
            "model_component": "value_model",
            "step": step,
            "save_dtype": str(save_dtype) if save_dtype is not None else None,
            "config": asdict(self.cfg),
            "replay_buffer_checkpoint": None,
            "wandb_run_id": wandb_run_id,
        }
        if metadata is not None:
            state["metadata"] = dict(metadata)
        if save_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
            state["rng"] = self.rng.get_state()
            state["buffer_rng"] = self.buffer_rng.get_state()

        data_source_state = self.data_source.state_dict()
        state["data_source"] = data_source_state
        state["data_generator"] = data_source_state

        torch.save(state, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> int:
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

        if ckpt.get("model_component") == "value_model":
            if type(self.model) is not BetterSplitFFN:
                raise TypeError("value-only checkpoints require model.name=BetterFFN")
            self.model.value_model.load_state_dict(
                model_state, strict=self.cfg.strict_model_loading
            )
        else:
            self.model.load_state_dict(
                model_state, strict=self.cfg.strict_model_loading
            )

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

        if self.cfr_target_model is not None:
            cfr_target_state = ckpt.get("cfr_target_model")
            if cfr_target_state is not None:
                self.cfr_target_model.load_state_dict(
                    cfr_target_state, strict=self.cfg.strict_model_loading
                )
                self.cfr_target_model.eval()
                self.cfr_target_model_step = int(
                    ckpt.get("cfr_target_model_step", ckpt.get("step", 0))
                )
            else:
                self._sync_cfr_target_model(step=int(ckpt.get("step", 0)) + 1)

        # Only load optimizer if it exists in checkpoint
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if isinstance(self.optimizer, torch.optim.AdamW):
                for param_group in self.optimizer.param_groups:
                    param_group["lr_role"] = "adamw"

        self.cfg.wandb_run_id = ckpt.get("wandb_run_id")
        self.checkpoint_metadata = dict(ckpt.get("metadata", {}))
        if "rng" in ckpt:
            self.rng.set_state(generator_state_for_set_state(ckpt["rng"]))
        if "buffer_rng" in ckpt:
            self.buffer_rng.set_state(generator_state_for_set_state(ckpt["buffer_rng"]))
        self._sync_inference_model()
        data_source_state = ckpt.get("data_source", ckpt.get("data_generator"))
        if data_source_state is not None:
            self.data_source.load_state_dict(data_source_state)
        replay_loaded = self.load_replay_buffers(path)
        if replay_loaded:
            print(
                "Replay buffers restored from "
                f"{self.replay_buffer_checkpoint_path(path)}"
            )
        return int(ckpt["step"])
