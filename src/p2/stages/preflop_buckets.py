from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

from p2.core.structured_config import Config


@dataclass(frozen=True)
class PreflopBucketExecutionConfig:
    command: str
    state_dataset: str
    base_checkpoint: str
    output_dir: str
    presolve_bucket: str
    train_bucket: str | None
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
    snapshot_interval_steps: int
    use_wandb: bool
    wandb_project: str
    wandb_name: str | None
    wandb_group: str | None
    wandb_tags: tuple[str, ...]
    student_init: str | None
    student_init_from_base: bool
    bootstrap_distill_checkpoint: str | None
    bootstrap_distill_epochs: int
    bootstrap_distill_rows: int | None
    bootstrap_distill_batch_size: int | None
    bootstrap_distill_train_value: bool
    distill_batch_size: int
    distill_buckets: tuple[str, ...] | None
    distill_train_value: bool
    checkpoint_12_15: str | None
    checkpoint_8_11: str | None
    checkpoint_4_7: str | None
    checkpoint_0_3: str | None


def _model_scope_name(value: object) -> str:
    return str(getattr(value, "value", value))


def _validate_bucket_run_config(cfg: Config) -> None:
    if (
        _model_scope_name(cfg.search.model_scope) == "mixed_street"
        and cfg.search.closing_leaf_checkpoint is None
    ):
        raise ValueError(
            "Preflop bucket mixed_street solving requires "
            "search.closing_leaf_checkpoint. New-street leaves must use the "
            "explicit end-of-preflop cutoff model instead of silently falling "
            "back to the active same-street model."
        )


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
    cfg.train.batch_size = int(execution.train_batch_size)
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = max(1, int(execution.replay_buffer_batches))
    cfg.train.save_replay_buffers = False
    cfg.model.enforce_zero_sum = False
    cfg.model.board_interaction_dim = 0
    cfg.search.depth = int(execution.depth)
    cfg.search.iterations = int(execution.cfr_iterations)
    cfg.search.iterations_final = None
    cfg.search.warm_start_iterations = int(execution.warm_start_iterations)
    cfg.search.sparse = True
    cfg.search.sparse_fused = execution.sparse_fused
    if execution.compile is not None:
        cfg.model.compile = execution.compile
    _validate_bucket_run_config(cfg)
    return cfg


__all__ = [
    "PreflopBucketExecutionConfig",
    "build_run_config",
]
