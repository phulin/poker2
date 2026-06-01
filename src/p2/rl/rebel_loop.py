from __future__ import annotations

import glob
import os
import time
from typing import Any

import torch

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.utils.training_utils import print_preflop_range_grid


def cleanup_old_checkpoints(checkpoint_dir: str, current_path: str) -> None:
    """Clean up old checkpoints, keeping only best_model.pt and latest checkpoint."""
    actual_current_path = os.path.realpath(current_path)

    if not os.path.exists(checkpoint_dir):
        return

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "rebel_step_*.pt"))

    files_to_keep = {
        os.path.join(checkpoint_dir, "latest_model.pt"),
        os.path.join(checkpoint_dir, "best_model.pt"),
        actual_current_path,
    }

    if checkpoint_files:
        latest_step = -1
        latest_checkpoint = None
        for file_path in checkpoint_files:
            filename = os.path.basename(file_path)
            if filename.startswith("rebel_step_") and filename.endswith(".pt"):
                try:
                    step_num = int(filename[11:-3])
                    if step_num > latest_step:
                        latest_step = step_num
                        latest_checkpoint = file_path
                except ValueError:
                    continue

        if latest_checkpoint:
            files_to_keep.add(latest_checkpoint)

    deleted_count = 0
    for file_path in checkpoint_files:
        if file_path not in files_to_keep:
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as exc:
                print(f"Warning: Could not delete {file_path}: {exc}")

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old checkpoint(s)")


def print_rebel_training_stats(
    metrics: dict,
    step: int,
    total_steps: int,
    step_elapsed: float,
    total_elapsed: float,
) -> None:
    loss_str = f"{metrics['loss']:.4f}" if metrics["loss"] is not None else "N/A"
    policy_str = (
        f"{metrics['policy_loss']:.5f}" if metrics["policy_loss"] is not None else "N/A"
    )
    value_str = (
        f"{metrics['value_loss']:.5f}" if metrics["value_loss"] is not None else "N/A"
    )
    exploitability_str = (
        f"{metrics['local_exploitability']:.5f}"
        if metrics["local_exploitability"] is not None
        else "N/A"
    )
    exploitability_mbbg = metrics.get("local_exploitability_mbbg")
    exploitability_mbbg_str = (
        f"{exploitability_mbbg:.2f}" if exploitability_mbbg is not None else "N/A"
    )
    street_str = (
        f"{metrics['evaluator_street']:.4f}"
        if metrics["evaluator_street"] is not None
        else "N/A"
    )

    print(
        f"[Step {step:05d}/{total_steps}] "
        f"loss={loss_str} "
        f"policy={policy_str} "
        f"value={value_str} "
        f"exploit={exploitability_str} "
        f"exploit_mbbg={exploitability_mbbg_str} "
        f"street={street_str} "
        f"time={step_elapsed:.2f}s total={total_elapsed / 60:.1f}m"
    )


def run_training_loop(
    trainer: RebelCFRTrainer,
    cfg: Config,
    run: Any,
    *,
    start_step: int,
    stop_step: int,
    stage_tag: str | None = None,
) -> int:
    del stage_tag

    print(
        f"Starting ReBeL CFR training for {stop_step - start_step} steps "
        f"(total target: {stop_step})"
    )
    if cfg.use_wandb:
        print(
            f"📊 Wandb logging to project '{cfg.wandb_project}'"
            + (f" as run '{cfg.wandb_name}'" if cfg.wandb_name else "")
        )

    training_start = time.time()
    last_completed_step = start_step - 1

    for step in range(start_step, stop_step):
        step_start = time.time()
        metrics = trainer.train_step(step)
        step_elapsed = time.time() - step_start
        total_elapsed = time.time() - training_start

        print_rebel_training_stats(
            metrics, step, stop_step, step_elapsed, total_elapsed
        )

        if cfg.use_wandb:
            metrics["step_time_s"] = step_elapsed
            run.log(metrics, step=metrics["step"])

        if (step + 1) % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(cfg.checkpoint_dir, f"rebel_step_{step + 1}.pt")
            wandb_run_id = run.id if run else None
            trainer.save_checkpoint(
                ckpt_path,
                step,
                wandb_run_id=wandb_run_id,
                save_optimizer=False,
                save_dtype=torch.bfloat16,
            )
            trainer.save_checkpoint(
                os.path.join(cfg.checkpoint_dir, "rebel_latest.pt"),
                step,
                wandb_run_id=wandb_run_id,
                save_optimizer=True,
                save_dtype=None,
            )

            if cfg.economize_checkpoints:
                cleanup_old_checkpoints(cfg.checkpoint_dir, ckpt_path)

            print(f"Checkpoint saved at step {step + 1} -> {ckpt_path}")
            print_preflop_range_grid(trainer, step, rebel=True)

        if (
            trainer.trueskill_tracker is not None
            and trainer.trueskill_tracker.should_snapshot(step + 1)
        ):
            weights = trainer.trueskill_snapshot_weights()
            trainer.trueskill_tracker.snapshot_and_evaluate(
                step=step + 1,
                candidate_weights=weights,
                wandb_run=run if cfg.use_wandb else None,
            )

        last_completed_step = step

    final_path = os.path.join(cfg.checkpoint_dir, "rebel_final.pt")
    trainer.save_checkpoint(
        final_path, stop_step, save_optimizer=False, save_dtype=None
    )
    total_elapsed = time.time() - training_start
    print(
        f"Training complete in {total_elapsed / 3600:.2f} hours. "
        f"Final checkpoint: {final_path}"
    )
    print_preflop_range_grid(
        trainer, stop_step, title="Final Preflop Range Grid", rebel=True
    )
    return last_completed_step


__all__ = [
    "cleanup_old_checkpoints",
    "print_rebel_training_stats",
    "run_training_loop",
]
