from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import DictConfig

from p2.core.structured_config import Config


@dataclass(frozen=True)
class PreflopBucketRunConfig:
    config_name: str
    config_overrides: tuple[str, ...]
    device: str
    cfr_batch_size: int
    use_wandb: bool
    wandb_project: str
    wandb_name: str | None
    wandb_tags: tuple[str, ...]
    train_batch_size: int
    replay_buffer_batches: int
    depth: int
    cfr_iterations: int
    warm_start_iterations: int
    sparse_fused: bool
    compile: str | None


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
    run_cfg: PreflopBucketRunConfig,
    *,
    checkpoint_dir: Path,
    num_steps: int,
    num_envs: int | None = None,
) -> Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.device = run_cfg.device
    cfg.num_envs = int(run_cfg.cfr_batch_size if num_envs is None else num_envs)
    cfg.num_steps = max(1, int(num_steps))
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.use_wandb = run_cfg.use_wandb
    cfg.wandb_project = run_cfg.wandb_project
    cfg.wandb_name = run_cfg.wandb_name
    cfg.wandb_tags = list(run_cfg.wandb_tags)
    cfg.resume_from = None
    cfg.data.mode = "live"
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.data.include_pre_chance_value_batches = False
    cfg.train.batch_size = int(run_cfg.train_batch_size)
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = max(1, int(run_cfg.replay_buffer_batches))
    cfg.train.save_replay_buffers = False
    cfg.search.depth = int(run_cfg.depth)
    cfg.search.iterations = int(run_cfg.cfr_iterations)
    cfg.search.iterations_final = None
    cfg.search.warm_start_iterations = int(run_cfg.warm_start_iterations)
    cfg.search.sparse = True
    cfg.search.sparse_fused = run_cfg.sparse_fused
    if run_cfg.compile is not None:
        cfg.model.compile = run_cfg.compile
    return cfg


__all__ = [
    "PreflopBucketRunConfig",
    "build_run_config",
    "load_base_config",
]
