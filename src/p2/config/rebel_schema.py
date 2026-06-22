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
    PolicyLossType,
    PolicyNodeWeighting,
    PreflopBucketTrainingConfig,
    PreflopValidationConfig,
    RebelPregenerateConfig,
    SearchConfig,
    StratifyConfig,
    TrainingConfig,
    TrueSkillConfig,
    ValidationSetConfig,
)


@dataclass(frozen=True)
class RebelRunConfig:
    num_steps: int
    num_envs: int
    device: str
    seed: int
    use_tensor_env: bool
    config: str | None


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
    grad_clip: float
    policy_grad_clip: float
    optimizer: str
    weight_decay: float
    muon_momentum: float
    muon_nesterov: bool
    muon_eps: float
    muon_ns_steps: int
    muon_adjust_lr_fn: str | None
    normuon_beta1: float
    normuon_beta2: float
    normuon_eps: float
    policy_head_muon_learning_rate: float
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
            grad_clip=cfg.grad_clip,
            policy_grad_clip=cfg.policy_grad_clip,
            optimizer=cfg.optimizer,
            weight_decay=cfg.weight_decay,
            muon_momentum=cfg.muon_momentum,
            muon_nesterov=cfg.muon_nesterov,
            muon_eps=cfg.muon_eps,
            muon_ns_steps=cfg.muon_ns_steps,
            muon_adjust_lr_fn=cfg.muon_adjust_lr_fn,
            normuon_beta1=cfg.normuon_beta1,
            normuon_beta2=cfg.normuon_beta2,
            normuon_eps=cfg.normuon_eps,
            policy_head_muon_learning_rate=cfg.policy_head_muon_learning_rate,
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
        cfg.grad_clip = self.grad_clip
        cfg.policy_grad_clip = self.policy_grad_clip
        cfg.optimizer = self.optimizer
        cfg.weight_decay = self.weight_decay
        cfg.muon_momentum = self.muon_momentum
        cfg.muon_nesterov = self.muon_nesterov
        cfg.muon_eps = self.muon_eps
        cfg.muon_ns_steps = self.muon_ns_steps
        cfg.muon_adjust_lr_fn = self.muon_adjust_lr_fn
        cfg.normuon_beta1 = self.normuon_beta1
        cfg.normuon_beta2 = self.normuon_beta2
        cfg.normuon_eps = self.normuon_eps
        cfg.policy_head_muon_learning_rate = self.policy_head_muon_learning_rate
        cfg.adamw_learning_rate = self.adamw_learning_rate
        cfg.stratify_streets = copy.deepcopy(self.stratify_streets)
        cfg.model_ema = self.model_ema
        return cfg


@dataclass(frozen=True)
class RebelExperimentConfig:
    run: RebelRunConfig
    checkpoint: RebelCheckpointConfig
    logging: RebelLoggingConfig
    train: RebelTrainingConfig
    model: ModelConfig
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
            model=copy.deepcopy(cfg.model),
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
            economize_checkpoints=self.checkpoint.economize_checkpoints,
            strict_model_loading=self.checkpoint.strict_model_loading,
            train=self.train.to_training_config(),
            model=copy.deepcopy(self.model),
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
    "RebelRunConfig",
    "RebelTrainingConfig",
]
