from __future__ import annotations

import copy
from dataclasses import dataclass, field

from p2.core.structured_config import (
    Config,
    CurriculumConfig,
    DataConfig,
    EnvConfig,
    LrSchedule,
    ModelConfig,
    ModelType,
    NonlinearityType,
    PolicyLossType,
    PolicyNodeWeighting,
    PreflopBucketTrainingConfig,
    PreflopValidationConfig,
    PreflopModelType,
    RebelPregenerateConfig,
    SearchConfig,
    StratifyConfig,
    StreetValueHeads,
    TrainingConfig,
    TrueSkillConfig,
    ValidationSetConfig,
    ValueHeadType,
)


@dataclass(frozen=True)
class RebelRunConfig:
    num_steps: int
    num_envs: int
    device: str
    seed: int
    use_tensor_env: bool
    config: str | None
    log_interval: int


@dataclass(frozen=True)
class RebelCheckpointConfig:
    checkpoint_dir: str
    checkpoint_interval: int
    resume_from: str | None
    economize_checkpoints: bool
    strict_model_loading: bool


@dataclass(frozen=True)
class RebelLoggingConfig:
    use_wandb: bool
    wandb_project: str | None
    wandb_name: str | None
    wandb_tags: list[str]
    wandb_run_id: str | None


@dataclass(frozen=True)
class RebelTrainingConfig:
    learning_rate: float
    learning_rate_final: float
    lr_schedule: LrSchedule
    warmup_steps: int
    lr_wsd_decay_fraction: float
    batch_size: int
    episodes_per_step: int
    replay_buffer_batches: int
    target_update_block_batches: int
    save_replay_buffers: bool
    replay_buffer_device: str
    replay_buffer_underfull_evict_fraction: float
    value_reuse_goal: float
    policy_capacity_factor: int
    policy_depth_stratify_decimate: bool
    policy_depth_stratify_sample: bool
    policy_depth_stratify_probs: list[float] | None
    policy_node_weighting: PolicyNodeWeighting
    policy_loss_type: PolicyLossType
    policy_logit_l2_coef: float
    policy_extra_updates_per_step: int
    policy_extra_batch_size: int | None
    value_coef: float
    entropy_coef: float
    permutation_coef: float
    backup_consistency_coef: float
    backup_consistency_sample_fraction: float
    backup_consistency_max_samples: int
    backup_consistency_min_policy_mass: float
    value_multiresolution_coef: float
    value_bucket_aux_coef: float
    value_ordering_coef: float
    value_bucket_usage_coef: float
    value_bucket_entropy_coef: float
    value_action_summary_coef: float
    grad_clip: float
    policy_grad_clip: float
    optimizer: str
    weight_decay: float
    muon_momentum: float
    muon_nesterov: bool
    muon_eps: float
    muon_ns_steps: int
    muon_adjust_lr_fn: str | None
    muon_compile_step: bool
    normuon_beta1: float
    normuon_beta2: float
    normuon_eps: float
    adamw_learning_rate: float | None
    stratify_streets: list[StratifyConfig] = field(default_factory=list)
    model_ema: float | None = None

    @classmethod
    def from_training_config(cls, cfg: TrainingConfig) -> RebelTrainingConfig:
        return cls(
            learning_rate=cfg.learning_rate,
            learning_rate_final=cfg.learning_rate_final,
            lr_schedule=cfg.lr_schedule,
            warmup_steps=cfg.warmup_steps,
            lr_wsd_decay_fraction=cfg.lr_wsd_decay_fraction,
            batch_size=cfg.batch_size,
            episodes_per_step=cfg.episodes_per_step,
            replay_buffer_batches=cfg.replay_buffer_batches,
            target_update_block_batches=cfg.target_update_block_batches,
            save_replay_buffers=cfg.save_replay_buffers,
            replay_buffer_device=cfg.replay_buffer_device,
            replay_buffer_underfull_evict_fraction=(
                cfg.replay_buffer_underfull_evict_fraction
            ),
            value_reuse_goal=cfg.value_reuse_goal,
            policy_capacity_factor=cfg.policy_capacity_factor,
            policy_depth_stratify_decimate=cfg.policy_depth_stratify_decimate,
            policy_depth_stratify_sample=cfg.policy_depth_stratify_sample,
            policy_depth_stratify_probs=(
                None
                if cfg.policy_depth_stratify_probs is None
                else list(cfg.policy_depth_stratify_probs)
            ),
            policy_node_weighting=cfg.policy_node_weighting,
            policy_loss_type=cfg.policy_loss_type,
            policy_logit_l2_coef=cfg.policy_logit_l2_coef,
            policy_extra_updates_per_step=cfg.policy_extra_updates_per_step,
            policy_extra_batch_size=cfg.policy_extra_batch_size,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
            permutation_coef=cfg.permutation_coef,
            backup_consistency_coef=cfg.backup_consistency_coef,
            backup_consistency_sample_fraction=cfg.backup_consistency_sample_fraction,
            backup_consistency_max_samples=cfg.backup_consistency_max_samples,
            backup_consistency_min_policy_mass=cfg.backup_consistency_min_policy_mass,
            value_multiresolution_coef=cfg.value_multiresolution_coef,
            value_bucket_aux_coef=cfg.value_bucket_aux_coef,
            value_ordering_coef=cfg.value_ordering_coef,
            value_bucket_usage_coef=cfg.value_bucket_usage_coef,
            value_bucket_entropy_coef=cfg.value_bucket_entropy_coef,
            value_action_summary_coef=cfg.value_action_summary_coef,
            grad_clip=cfg.grad_clip,
            policy_grad_clip=cfg.policy_grad_clip,
            optimizer=cfg.optimizer,
            weight_decay=cfg.weight_decay,
            muon_momentum=cfg.muon_momentum,
            muon_nesterov=cfg.muon_nesterov,
            muon_eps=cfg.muon_eps,
            muon_ns_steps=cfg.muon_ns_steps,
            muon_adjust_lr_fn=cfg.muon_adjust_lr_fn,
            muon_compile_step=cfg.muon_compile_step,
            normuon_beta1=cfg.normuon_beta1,
            normuon_beta2=cfg.normuon_beta2,
            normuon_eps=cfg.normuon_eps,
            adamw_learning_rate=cfg.adamw_learning_rate,
            stratify_streets=copy.deepcopy(cfg.stratify_streets),
            model_ema=cfg.model_ema,
        )

    def to_training_config(self) -> TrainingConfig:
        cfg = TrainingConfig()
        cfg.learning_rate = self.learning_rate
        cfg.learning_rate_final = self.learning_rate_final
        cfg.lr_schedule = self.lr_schedule
        cfg.warmup_steps = self.warmup_steps
        cfg.lr_wsd_decay_fraction = self.lr_wsd_decay_fraction
        cfg.batch_size = self.batch_size
        cfg.episodes_per_step = self.episodes_per_step
        cfg.replay_buffer_batches = self.replay_buffer_batches
        cfg.target_update_block_batches = self.target_update_block_batches
        cfg.save_replay_buffers = self.save_replay_buffers
        cfg.replay_buffer_device = self.replay_buffer_device
        cfg.replay_buffer_underfull_evict_fraction = (
            self.replay_buffer_underfull_evict_fraction
        )
        cfg.value_reuse_goal = self.value_reuse_goal
        cfg.policy_capacity_factor = self.policy_capacity_factor
        cfg.policy_depth_stratify_decimate = self.policy_depth_stratify_decimate
        cfg.policy_depth_stratify_sample = self.policy_depth_stratify_sample
        cfg.policy_depth_stratify_probs = (
            None
            if self.policy_depth_stratify_probs is None
            else list(self.policy_depth_stratify_probs)
        )
        cfg.policy_node_weighting = self.policy_node_weighting
        cfg.policy_loss_type = self.policy_loss_type
        cfg.policy_logit_l2_coef = self.policy_logit_l2_coef
        cfg.policy_extra_updates_per_step = self.policy_extra_updates_per_step
        cfg.policy_extra_batch_size = self.policy_extra_batch_size
        cfg.value_coef = self.value_coef
        cfg.entropy_coef = self.entropy_coef
        cfg.permutation_coef = self.permutation_coef
        cfg.backup_consistency_coef = self.backup_consistency_coef
        cfg.backup_consistency_sample_fraction = self.backup_consistency_sample_fraction
        cfg.backup_consistency_max_samples = self.backup_consistency_max_samples
        cfg.backup_consistency_min_policy_mass = self.backup_consistency_min_policy_mass
        cfg.value_multiresolution_coef = self.value_multiresolution_coef
        cfg.value_bucket_aux_coef = self.value_bucket_aux_coef
        cfg.value_ordering_coef = self.value_ordering_coef
        cfg.value_bucket_usage_coef = self.value_bucket_usage_coef
        cfg.value_bucket_entropy_coef = self.value_bucket_entropy_coef
        cfg.value_action_summary_coef = self.value_action_summary_coef
        cfg.grad_clip = self.grad_clip
        cfg.policy_grad_clip = self.policy_grad_clip
        cfg.optimizer = self.optimizer
        cfg.weight_decay = self.weight_decay
        cfg.muon_momentum = self.muon_momentum
        cfg.muon_nesterov = self.muon_nesterov
        cfg.muon_eps = self.muon_eps
        cfg.muon_ns_steps = self.muon_ns_steps
        cfg.muon_adjust_lr_fn = self.muon_adjust_lr_fn
        cfg.muon_compile_step = self.muon_compile_step
        cfg.normuon_beta1 = self.normuon_beta1
        cfg.normuon_beta2 = self.normuon_beta2
        cfg.normuon_eps = self.normuon_eps
        cfg.adamw_learning_rate = self.adamw_learning_rate
        cfg.stratify_streets = copy.deepcopy(self.stratify_streets)
        cfg.model_ema = self.model_ema
        return cfg


@dataclass(frozen=True)
class RebelModelConfig:
    name: ModelType
    compile: str
    value_head_type: ValueHeadType
    detach_value_head: bool
    nonlinearity: NonlinearityType
    num_actions: int
    input_dim: int
    hidden_dim: int
    num_hidden_layers: int
    num_policy_layers: int
    num_value_layers: int
    range_hidden_dim: int
    ffn_dim: int
    shared_trunk: bool
    enforce_zero_sum: bool
    board_interaction_dim: int
    board_interaction_skip_out: bool
    board_interaction_gated: bool
    policy_rank: int
    policy_hand_bias_rank: int
    value_per_hand_residual: bool
    board_conditioned_hand_embedding_dim: int
    cross_range_rank: int
    card_token_value_head_dim: int
    context_range_stats: bool
    postflop_multi_token_trunk: bool
    belief_second_moment: bool
    value_strength_bucket_count: int
    value_strength_bucket_film: bool
    value_strength_bucket_relative: bool
    value_strength_bucket_board_only: bool
    value_strength_bucket_blockers: bool
    value_strength_bucket_coarse_residual: bool
    value_bucket_coarse_dim: int
    value_latent_bucket_count: int
    value_latent_bucket_dim: int
    value_exact_river_features: bool
    value_showdown_baseline: bool
    value_river_range_equity_baseline: bool
    value_river_range_equity_baseline_scale: float
    value_river_range_equity_pot_power: float
    value_river_range_equity_pos_scale: float
    value_river_range_equity_neg_scale: float
    value_river_range_equity_intercept: float
    value_river_range_equity_blockers: bool
    value_river_range_equity_rank_bins: int
    value_river_range_equity_feature_head: bool
    value_river_range_equity_trunk_context: bool
    value_river_range_equity_film_rank: int
    value_river_range_equity_film_hidden_dim: int
    value_output_init_scale: float
    value_action_summary_head: bool
    value_head_rank: int
    value_hand_basis_rank: int
    belief_low_rank_dim: int
    belief_low_rank_board_conditioned: bool
    belief_skip_matching_encoder: bool
    belief_linear_encoder: bool
    belief_board_film: bool
    belief_board_bilinear_rank: int
    belief_board_mass_features: bool
    street_value_heads: StreetValueHeads
    preflop_hand_dim: int
    preflop_model_type: PreflopModelType
    preflop_transformer_heads: int
    preflop_range_slot_moment_slots: int
    num_recursions: int
    num_iterations: int
    num_supervisions: int

    @classmethod
    def from_model_config(cls, cfg: ModelConfig) -> RebelModelConfig:
        return cls(
            name=cfg.name,
            compile=cfg.compile,
            value_head_type=cfg.value_head_type,
            detach_value_head=cfg.detach_value_head,
            nonlinearity=cfg.nonlinearity,
            num_actions=cfg.num_actions,
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            num_policy_layers=cfg.num_policy_layers,
            num_value_layers=cfg.num_value_layers,
            range_hidden_dim=cfg.range_hidden_dim,
            ffn_dim=cfg.ffn_dim,
            shared_trunk=cfg.shared_trunk,
            enforce_zero_sum=cfg.enforce_zero_sum,
            board_interaction_dim=cfg.board_interaction_dim,
            board_interaction_skip_out=cfg.board_interaction_skip_out,
            board_interaction_gated=cfg.board_interaction_gated,
            policy_rank=cfg.policy_rank,
            policy_hand_bias_rank=cfg.policy_hand_bias_rank,
            value_per_hand_residual=cfg.value_per_hand_residual,
            board_conditioned_hand_embedding_dim=(
                cfg.board_conditioned_hand_embedding_dim
            ),
            cross_range_rank=cfg.cross_range_rank,
            card_token_value_head_dim=cfg.card_token_value_head_dim,
            context_range_stats=cfg.context_range_stats,
            postflop_multi_token_trunk=cfg.postflop_multi_token_trunk,
            belief_second_moment=cfg.belief_second_moment,
            value_strength_bucket_count=cfg.value_strength_bucket_count,
            value_strength_bucket_film=cfg.value_strength_bucket_film,
            value_strength_bucket_relative=cfg.value_strength_bucket_relative,
            value_strength_bucket_board_only=cfg.value_strength_bucket_board_only,
            value_strength_bucket_blockers=cfg.value_strength_bucket_blockers,
            value_strength_bucket_coarse_residual=(
                cfg.value_strength_bucket_coarse_residual
            ),
            value_bucket_coarse_dim=cfg.value_bucket_coarse_dim,
            value_latent_bucket_count=cfg.value_latent_bucket_count,
            value_latent_bucket_dim=cfg.value_latent_bucket_dim,
            value_exact_river_features=cfg.value_exact_river_features,
            value_showdown_baseline=cfg.value_showdown_baseline,
            value_river_range_equity_baseline=(
                cfg.value_river_range_equity_baseline
            ),
            value_river_range_equity_baseline_scale=(
                cfg.value_river_range_equity_baseline_scale
            ),
            value_river_range_equity_pot_power=(
                cfg.value_river_range_equity_pot_power
            ),
            value_river_range_equity_pos_scale=(
                cfg.value_river_range_equity_pos_scale
            ),
            value_river_range_equity_neg_scale=(
                cfg.value_river_range_equity_neg_scale
            ),
            value_river_range_equity_intercept=(
                cfg.value_river_range_equity_intercept
            ),
            value_river_range_equity_blockers=(
                cfg.value_river_range_equity_blockers
            ),
            value_river_range_equity_rank_bins=(
                cfg.value_river_range_equity_rank_bins
            ),
            value_river_range_equity_feature_head=(
                cfg.value_river_range_equity_feature_head
            ),
            value_river_range_equity_trunk_context=(
                cfg.value_river_range_equity_trunk_context
            ),
            value_river_range_equity_film_rank=(
                cfg.value_river_range_equity_film_rank
            ),
            value_river_range_equity_film_hidden_dim=(
                cfg.value_river_range_equity_film_hidden_dim
            ),
            value_output_init_scale=cfg.value_output_init_scale,
            value_action_summary_head=cfg.value_action_summary_head,
            value_head_rank=cfg.value_head_rank,
            value_hand_basis_rank=cfg.value_hand_basis_rank,
            belief_low_rank_dim=cfg.belief_low_rank_dim,
            belief_low_rank_board_conditioned=(
                cfg.belief_low_rank_board_conditioned
            ),
            belief_skip_matching_encoder=cfg.belief_skip_matching_encoder,
            belief_linear_encoder=cfg.belief_linear_encoder,
            belief_board_film=cfg.belief_board_film,
            belief_board_bilinear_rank=cfg.belief_board_bilinear_rank,
            belief_board_mass_features=cfg.belief_board_mass_features,
            street_value_heads=cfg.street_value_heads,
            preflop_hand_dim=cfg.preflop_hand_dim,
            preflop_model_type=cfg.preflop_model_type,
            preflop_transformer_heads=cfg.preflop_transformer_heads,
            preflop_range_slot_moment_slots=cfg.preflop_range_slot_moment_slots,
            num_recursions=cfg.num_recursions,
            num_iterations=cfg.num_iterations,
            num_supervisions=cfg.num_supervisions,
        )

    def to_model_config(self) -> ModelConfig:
        cfg = ModelConfig()
        cfg.name = self.name
        cfg.compile = self.compile
        cfg.value_head_type = self.value_head_type
        cfg.detach_value_head = self.detach_value_head
        cfg.nonlinearity = self.nonlinearity
        cfg.num_actions = self.num_actions
        cfg.input_dim = self.input_dim
        cfg.hidden_dim = self.hidden_dim
        cfg.num_hidden_layers = self.num_hidden_layers
        cfg.num_policy_layers = self.num_policy_layers
        cfg.num_value_layers = self.num_value_layers
        cfg.range_hidden_dim = self.range_hidden_dim
        cfg.ffn_dim = self.ffn_dim
        cfg.shared_trunk = self.shared_trunk
        cfg.enforce_zero_sum = self.enforce_zero_sum
        cfg.board_interaction_dim = self.board_interaction_dim
        cfg.board_interaction_skip_out = self.board_interaction_skip_out
        cfg.board_interaction_gated = self.board_interaction_gated
        cfg.policy_rank = self.policy_rank
        cfg.policy_hand_bias_rank = self.policy_hand_bias_rank
        cfg.value_per_hand_residual = self.value_per_hand_residual
        cfg.board_conditioned_hand_embedding_dim = (
            self.board_conditioned_hand_embedding_dim
        )
        cfg.cross_range_rank = self.cross_range_rank
        cfg.card_token_value_head_dim = self.card_token_value_head_dim
        cfg.context_range_stats = self.context_range_stats
        cfg.postflop_multi_token_trunk = self.postflop_multi_token_trunk
        cfg.belief_second_moment = self.belief_second_moment
        cfg.value_strength_bucket_count = self.value_strength_bucket_count
        cfg.value_strength_bucket_film = self.value_strength_bucket_film
        cfg.value_strength_bucket_relative = self.value_strength_bucket_relative
        cfg.value_strength_bucket_board_only = self.value_strength_bucket_board_only
        cfg.value_strength_bucket_blockers = self.value_strength_bucket_blockers
        cfg.value_strength_bucket_coarse_residual = (
            self.value_strength_bucket_coarse_residual
        )
        cfg.value_bucket_coarse_dim = self.value_bucket_coarse_dim
        cfg.value_latent_bucket_count = self.value_latent_bucket_count
        cfg.value_latent_bucket_dim = self.value_latent_bucket_dim
        cfg.value_exact_river_features = self.value_exact_river_features
        cfg.value_showdown_baseline = self.value_showdown_baseline
        cfg.value_river_range_equity_baseline = (
            self.value_river_range_equity_baseline
        )
        cfg.value_river_range_equity_baseline_scale = (
            self.value_river_range_equity_baseline_scale
        )
        cfg.value_river_range_equity_pot_power = (
            self.value_river_range_equity_pot_power
        )
        cfg.value_river_range_equity_pos_scale = (
            self.value_river_range_equity_pos_scale
        )
        cfg.value_river_range_equity_neg_scale = (
            self.value_river_range_equity_neg_scale
        )
        cfg.value_river_range_equity_intercept = (
            self.value_river_range_equity_intercept
        )
        cfg.value_river_range_equity_blockers = (
            self.value_river_range_equity_blockers
        )
        cfg.value_river_range_equity_rank_bins = (
            self.value_river_range_equity_rank_bins
        )
        cfg.value_river_range_equity_feature_head = (
            self.value_river_range_equity_feature_head
        )
        cfg.value_river_range_equity_trunk_context = (
            self.value_river_range_equity_trunk_context
        )
        cfg.value_river_range_equity_film_rank = (
            self.value_river_range_equity_film_rank
        )
        cfg.value_river_range_equity_film_hidden_dim = (
            self.value_river_range_equity_film_hidden_dim
        )
        cfg.value_output_init_scale = self.value_output_init_scale
        cfg.value_action_summary_head = self.value_action_summary_head
        cfg.value_head_rank = self.value_head_rank
        cfg.value_hand_basis_rank = self.value_hand_basis_rank
        cfg.belief_low_rank_dim = self.belief_low_rank_dim
        cfg.belief_low_rank_board_conditioned = (
            self.belief_low_rank_board_conditioned
        )
        cfg.belief_skip_matching_encoder = self.belief_skip_matching_encoder
        cfg.belief_linear_encoder = self.belief_linear_encoder
        cfg.belief_board_film = self.belief_board_film
        cfg.belief_board_bilinear_rank = self.belief_board_bilinear_rank
        cfg.belief_board_mass_features = self.belief_board_mass_features
        cfg.street_value_heads = self.street_value_heads
        cfg.preflop_hand_dim = self.preflop_hand_dim
        cfg.preflop_model_type = self.preflop_model_type
        cfg.preflop_transformer_heads = self.preflop_transformer_heads
        cfg.preflop_range_slot_moment_slots = self.preflop_range_slot_moment_slots
        cfg.num_recursions = self.num_recursions
        cfg.num_iterations = self.num_iterations
        cfg.num_supervisions = self.num_supervisions
        return cfg


@dataclass(frozen=True)
class RebelExperimentConfig:
    run: RebelRunConfig
    checkpoint: RebelCheckpointConfig
    logging: RebelLoggingConfig
    train: RebelTrainingConfig
    model: RebelModelConfig
    env: EnvConfig
    search: SearchConfig
    trueskill: TrueSkillConfig
    data: DataConfig
    curriculum: CurriculumConfig
    rebel_pregenerate: RebelPregenerateConfig
    validation_set: ValidationSetConfig
    preflop_validation: PreflopValidationConfig
    preflop_buckets: PreflopBucketTrainingConfig

    @classmethod
    def from_trainer_config(cls, cfg: Config) -> RebelExperimentConfig:
        return cls(
            run=RebelRunConfig(
                num_steps=cfg.num_steps,
                num_envs=cfg.num_envs,
                device=cfg.device,
                seed=cfg.seed,
                use_tensor_env=cfg.use_tensor_env,
                config=cfg.config,
                log_interval=cfg.log_interval,
            ),
            checkpoint=RebelCheckpointConfig(
                checkpoint_dir=cfg.checkpoint_dir,
                checkpoint_interval=cfg.checkpoint_interval,
                resume_from=cfg.resume_from,
                economize_checkpoints=cfg.economize_checkpoints,
                strict_model_loading=cfg.strict_model_loading,
            ),
            logging=RebelLoggingConfig(
                use_wandb=cfg.use_wandb,
                wandb_project=cfg.wandb_project,
                wandb_name=cfg.wandb_name,
                wandb_tags=list(cfg.wandb_tags),
                wandb_run_id=cfg.wandb_run_id,
            ),
            train=RebelTrainingConfig.from_training_config(cfg.train),
            model=RebelModelConfig.from_model_config(cfg.model),
            env=copy.deepcopy(cfg.env),
            search=copy.deepcopy(cfg.search),
            trueskill=copy.deepcopy(cfg.trueskill),
            data=copy.deepcopy(cfg.data),
            curriculum=copy.deepcopy(cfg.curriculum),
            rebel_pregenerate=copy.deepcopy(cfg.rebel_pregenerate),
            validation_set=copy.deepcopy(cfg.validation_set),
            preflop_validation=copy.deepcopy(cfg.preflop_validation),
            preflop_buckets=copy.deepcopy(cfg.preflop_buckets),
        )

    def to_trainer_config(self) -> Config:
        return Config(
            num_steps=self.run.num_steps,
            checkpoint_interval=self.checkpoint.checkpoint_interval,
            checkpoint_dir=self.checkpoint.checkpoint_dir,
            device=self.run.device,
            use_tensor_env=self.run.use_tensor_env,
            num_envs=self.run.num_envs,
            use_wandb=self.logging.use_wandb,
            wandb_project=self.logging.wandb_project,
            wandb_name=self.logging.wandb_name,
            wandb_tags=list(self.logging.wandb_tags),
            wandb_run_id=self.logging.wandb_run_id,
            resume_from=self.checkpoint.resume_from,
            seed=self.run.seed,
            config=self.run.config,
            log_interval=self.run.log_interval,
            economize_checkpoints=self.checkpoint.economize_checkpoints,
            strict_model_loading=self.checkpoint.strict_model_loading,
            train=self.train.to_training_config(),
            model=self.model.to_model_config(),
            env=copy.deepcopy(self.env),
            search=copy.deepcopy(self.search),
            trueskill=copy.deepcopy(self.trueskill),
            data=copy.deepcopy(self.data),
            curriculum=copy.deepcopy(self.curriculum),
            rebel_pregenerate=copy.deepcopy(self.rebel_pregenerate),
            validation_set=copy.deepcopy(self.validation_set),
            preflop_validation=copy.deepcopy(self.preflop_validation),
            preflop_buckets=copy.deepcopy(self.preflop_buckets),
        )


__all__ = [
    "RebelCheckpointConfig",
    "RebelExperimentConfig",
    "RebelLoggingConfig",
    "RebelModelConfig",
    "RebelRunConfig",
    "RebelTrainingConfig",
]
