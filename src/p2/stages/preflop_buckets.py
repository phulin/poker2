from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig

from p2.core.structured_config import Config


@dataclass(frozen=True)
class PreflopBucketExecutionConfig:
    command: str
    config_name: str
    config_overrides: tuple[str, ...]
    state_dataset: str
    base_checkpoint: str
    output_dir: str
    device: str
    seed: int
    depth: int
    cfr_iterations: int
    warm_start_iterations: int
    sparse_fused: bool
    compile: str | None
    belief_mode: str
    states_per_bucket: int
    train_batch_size: int
    cfr_batch_size: int
    actions_12_15_cfr_batch_size: int | None
    actions_8_11_cfr_batch_size: int | None
    actions_12_15_epochs: int
    validation_items: int
    validation_cfr_iterations: int
    validation_interval_steps: int
    validation_eval_batch_size: int
    replay_buffer_batches: int
    storage_dtype: str
    write_solved_shards: bool
    allow_partial: bool
    overwrite: bool
    progress_roots: int
    use_wandb: bool
    wandb_project: str
    wandb_name: str | None
    wandb_group: str | None
    wandb_tags: tuple[str, ...]
    student_init: str | None
    distill_batch_size: int
    checkpoint_12_15: str | None
    checkpoint_8_11: str | None
    checkpoint_4_7: str | None
    checkpoint_0_3: str | None


def load_base_config(
    *,
    repo_root: Path,
    config_name: str,
    overrides: tuple[str, ...],
) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(repo_root / "conf"),
        version_base=None,
    ):
        dict_config: DictConfig = hydra.compose(
            config_name=config_name,
            overrides=list(overrides),
        )
    return Config.from_dict_config(dict_config)


def build_run_config(
    base_cfg: Config,
    execution: PreflopBucketExecutionConfig,
    *,
    checkpoint_dir: Path,
    num_steps: int,
    num_envs: int | None = None,
) -> Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.device = execution.device
    cfg.num_envs = int(execution.cfr_batch_size if num_envs is None else num_envs)
    cfg.num_steps = max(1, int(num_steps))
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.use_wandb = execution.use_wandb
    cfg.wandb_project = execution.wandb_project
    cfg.wandb_name = execution.wandb_name
    cfg.wandb_tags = list(execution.wandb_tags)
    cfg.resume_from = None
    cfg.data.mode = "live"
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.data.include_pre_chance_value_batches = False
    cfg.train.batch_size = int(execution.train_batch_size)
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = max(1, int(execution.replay_buffer_batches))
    cfg.train.save_replay_buffers = False
    cfg.search.depth = int(execution.depth)
    cfg.search.iterations = int(execution.cfr_iterations)
    cfg.search.iterations_final = None
    cfg.search.warm_start_iterations = int(execution.warm_start_iterations)
    cfg.search.sparse = True
    cfg.search.sparse_fused = execution.sparse_fused
    if execution.compile is not None:
        cfg.model.compile = execution.compile
    return cfg


__all__ = [
    "PreflopBucketExecutionConfig",
    "build_run_config",
    "load_base_config",
]
