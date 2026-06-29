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
    resume_from: str | None
    output_dir: str
    presolve_bucket: str
    train_bucket: str | None
    device: str
    seed: int
    depth: int
    actions_12_15_depth: int | None
    actions_8_11_depth: int | None
    actions_4_7_depth: int | None
    actions_0_3_depth: int | None
    cfr_iterations: int
    warm_start_iterations: int
    sparse_fused: bool
    compile: str | None
    belief_mode: str
    states_per_bucket: int
    train_batch_size: int
    policy_train_batch_size: int | None
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


def _bucket_depth(
    execution: PreflopBucketExecutionConfig,
    bucket_label: str | None,
) -> int:
    overrides = {
        "actions_12_15": execution.actions_12_15_depth,
        "actions_8_11": execution.actions_8_11_depth,
        "actions_4_7": execution.actions_4_7_depth,
        "actions_0_3": execution.actions_0_3_depth,
    }
    if bucket_label is not None and bucket_label in overrides:
        override = overrides[bucket_label]
        if override is not None:
            return max(1, int(override))
    return max(1, int(execution.depth))


def _resize_depth_schedule(values: object, depth: int) -> list[list[float]] | None:
    if values is None:
        return None
    schedule = [[float(x) for x in row] for row in values]
    if len(schedule) >= depth:
        return schedule[:depth]
    fill = next((row for row in reversed(schedule) if row), [])
    while len(schedule) < depth:
        schedule.append(list(fill))
    return schedule


def _resize_bool_schedule(values: object, depth: int) -> list[bool] | None:
    if values is None:
        return None
    schedule = [bool(x) for x in values]
    if len(schedule) >= depth:
        return schedule[:depth]
    fill = schedule[-1] if schedule else True
    while len(schedule) < depth:
        schedule.append(fill)
    return schedule


def _apply_bucket_depth(
    cfg: Config,
    execution: PreflopBucketExecutionConfig,
    bucket_label: str | None,
) -> None:
    depth = _bucket_depth(execution, bucket_label)
    cfg.search.depth = depth
    cfg.search.bet_bins_by_depth = _resize_depth_schedule(
        cfg.search.bet_bins_by_depth,
        depth,
    )
    cfg.search.allin_by_depth = _resize_bool_schedule(
        cfg.search.allin_by_depth,
        depth,
    )


def build_run_config(
    base_cfg: Config,
    execution: PreflopBucketExecutionConfig,
    *,
    checkpoint_dir: Path,
    num_steps: int,
    num_envs: int | None = None,
    bucket_label: str | None = None,
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
    cfg.trueskill.enabled = False
    cfg.data.mode = "live"
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.train.batch_size = int(execution.train_batch_size)
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = max(1, int(execution.replay_buffer_batches))
    cfg.train.save_replay_buffers = False
    cfg.model.enforce_zero_sum = False
    cfg.model.board_interaction_dim = 0
    _apply_bucket_depth(cfg, execution, bucket_label)
    cfg.search.iterations = int(execution.cfr_iterations)
    cfg.search.iterations_final = None
    cfg.search.warm_start_iterations = int(execution.warm_start_iterations)
    cfg.search.sparse = True
    cfg.search.sparse_fused = execution.sparse_fused
    cfg.model.compile = execution.compile if execution.compile is not None else "off"
    _validate_bucket_run_config(cfg)
    return cfg


__all__ = [
    "PreflopBucketExecutionConfig",
    "build_run_config",
]
