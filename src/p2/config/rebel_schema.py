from __future__ import annotations

import copy
from dataclasses import dataclass

from p2.core.structured_config import (
    Config,
    CurriculumConfig,
    DataConfig,
    EnvConfig,
    ModelConfig,
    PreflopBucketTrainingConfig,
    PreflopValidationConfig,
    RebelPregenerateConfig,
    SearchConfig,
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
class RebelExperimentConfig:
    run: RebelRunConfig
    checkpoint: RebelCheckpointConfig
    logging: RebelLoggingConfig
    train: TrainingConfig
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
            train=copy.deepcopy(cfg.train),
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
            train=copy.deepcopy(self.train),
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
]
