#!/usr/bin/env python3
"""Hydra entry point for preflop backward-induction bucket stages."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config, PreflopBucketTrainingConfig
from p2.stages import preflop_backward_induction as preflop_bi
from p2.stages.preflop_buckets import PreflopBucketExecutionConfig


def _required(value: str | None, field: str) -> str:
    if value is None or value == "":
        raise ValueError(f"preflop_buckets.{field} is required")
    return value


def _command_name(command: str) -> str:
    if command == "train_specialists":
        return "train-specialists"
    if command == "presolve_values":
        return "presolve-values"
    if command in {"train-specialists", "distill", "presolve-values"}:
        return command
    raise ValueError(
        "preflop_buckets.command must be 'train_specialists', "
        "'presolve_values', or 'distill'; "
        f"got {command!r}"
    )


def _execution_config_from_config(cfg: Config) -> PreflopBucketExecutionConfig:
    preflop = cfg.preflop_buckets
    command = _command_name(preflop.command)
    base_checkpoint = _required(preflop.base_checkpoint, "base_checkpoint")
    state_dataset = _required(preflop.state_dataset, "state_dataset")
    checkpoint_args = _distill_checkpoint_args(preflop)

    return PreflopBucketExecutionConfig(
        command=command,
        state_dataset=state_dataset,
        base_checkpoint=base_checkpoint,
        output_dir=preflop.output_dir,
        presolve_bucket=preflop.presolve_bucket,
        train_bucket=preflop.train_bucket,
        device=cfg.device,
        seed=cfg.seed,
        depth=preflop.depth,
        cfr_iterations=preflop.cfr_iterations,
        warm_start_iterations=preflop.warm_start_iterations,
        sparse_fused=preflop.sparse_fused,
        compile=preflop.compile,
        belief_mode=preflop.belief_mode,
        states_per_bucket=preflop.states_per_bucket,
        train_batch_size=preflop.train_batch_size,
        cfr_batch_size=preflop.cfr_batch_size,
        actions_12_15_cfr_batch_size=preflop.actions_12_15_cfr_batch_size,
        actions_8_11_cfr_batch_size=preflop.actions_8_11_cfr_batch_size,
        actions_12_15_epochs=preflop.actions_12_15_epochs,
        validation_items=preflop.validation_items,
        validation_cfr_iterations=preflop.validation_cfr_iterations,
        validation_interval_steps=preflop.validation_interval_steps,
        validation_eval_batch_size=preflop.validation_eval_batch_size,
        replay_buffer_batches=preflop.replay_buffer_batches,
        storage_dtype=preflop.storage_dtype,
        write_solved_shards=preflop.write_solved_shards,
        allow_partial=preflop.allow_partial,
        overwrite=preflop.overwrite,
        progress_roots=preflop.progress_roots,
        snapshot_interval_steps=preflop.snapshot_interval_steps,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        wandb_group=preflop.wandb_group,
        wandb_tags=cfg.wandb_tags,
        student_init=preflop.student_init,
        distill_batch_size=preflop.distill_batch_size,
        distill_buckets=(
            None
            if preflop.distill_buckets is None
            else tuple(preflop.distill_buckets)
        ),
        distill_train_value=preflop.distill_train_value,
        **checkpoint_args,
    )


def _distill_checkpoint_args(preflop: PreflopBucketTrainingConfig) -> dict[str, str | None]:
    checkpoints = preflop.distill_checkpoints
    return {
        "checkpoint_12_15": checkpoints.checkpoint_12_15,
        "checkpoint_8_11": checkpoints.checkpoint_8_11,
        "checkpoint_4_7": checkpoints.checkpoint_4_7,
        "checkpoint_0_3": checkpoints.checkpoint_0_3,
    }


def train_rebel_preflop_buckets(cfg: Config) -> None:
    execution = _execution_config_from_config(cfg)
    if execution.command == "train-specialists":
        preflop_bi.run_train_specialists(execution, base_template=cfg)
        return
    if execution.command == "presolve-values":
        preflop_bi.run_presolve_values(execution, base_template=cfg)
        return
    bucket_to_key = {
        "actions_12_15": "checkpoint_12_15",
        "actions_8_11": "checkpoint_8_11",
        "actions_4_7": "checkpoint_4_7",
        "actions_0_3": "checkpoint_0_3",
    }
    distill_buckets = cfg.preflop_buckets.distill_buckets
    required_buckets = (
        tuple(bucket_to_key) if distill_buckets is None else tuple(distill_buckets)
    )
    unknown = sorted(set(required_buckets) - set(bucket_to_key))
    if unknown:
        raise ValueError(f"unknown preflop distill bucket(s): {', '.join(unknown)}")
    checkpoint_args = _distill_checkpoint_args(cfg.preflop_buckets)
    missing = [
        bucket_to_key[bucket]
        for bucket in required_buckets
        if checkpoint_args[bucket_to_key[bucket]] is None
    ]
    if missing:
        raise ValueError(
            "preflop_buckets.distill_checkpoints missing required fields: "
            f"{', '.join(missing)}"
        )
    preflop_bi.run_distill(execution, base_template=cfg)


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="config_rebel_preflop_buckets",
)
def main(dict_config: DictConfig) -> None:
    cfg = load_rebel_config(dict_config)
    train_rebel_preflop_buckets(cfg)


if __name__ == "__main__":
    main()
