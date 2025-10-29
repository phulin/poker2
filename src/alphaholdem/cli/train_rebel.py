#!/usr/bin/env python3
"""
Training script for ReBeL-style CFR with the feed-forward model.

Mirrors the structure of train_kbest.py but drives the RebelCFRTrainer and logs
search-driven supervision metrics to Weights & Biases.
"""

from __future__ import annotations

import glob
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict

import hydra
import torch
import wandb
from omegaconf import DictConfig

from alphaholdem.core.structured_config import Config
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.utils.training_utils import print_preflop_range_grid


def _cleanup_old_checkpoints(checkpoint_dir: str, current_path: str) -> None:
    """Clean up old checkpoints, keeping only best_model.pt and latest checkpoint."""
    # Resolve symlinks to get the actual checkpoint file
    actual_current_path = os.path.realpath(current_path)

    if not os.path.exists(checkpoint_dir):
        return

    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "rebel_step_*.pt"))

    # Keep best_model.pt and the current checkpoint
    files_to_keep = {
        os.path.join(checkpoint_dir, "latest_model.pt"),
        os.path.join(checkpoint_dir, "best_model.pt"),
        actual_current_path,  # Keep the current checkpoint (resolved path)
    }

    # Also keep the checkpoint with the latest step number
    if checkpoint_files:
        # Extract step numbers and find the latest
        latest_step = -1
        latest_checkpoint = None
        for file_path in checkpoint_files:
            filename = os.path.basename(file_path)
            # Extract step number from "rebel_step_XXX.pt"
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

    # Remove old checkpoint files
    deleted_count = 0
    for file_path in checkpoint_files:
        if file_path not in files_to_keep:
            try:
                os.remove(file_path)
                deleted_count += 1
            except OSError as e:
                print(f"Warning: Could not delete {file_path}: {e}")

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old checkpoint(s)")


def _device_from_config(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _init_wandb(cfg: Config, device: torch.device) -> Any:
    if not cfg.use_wandb:
        return nullcontext()

    # Handle wandb resumption from checkpoint
    wandb_run_id_from_checkpoint = None
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"Loading checkpoint to extract wandb run ID: {cfg.resume_from}")

        # Extract wandb run ID from checkpoint
        checkpoint = torch.load(
            cfg.resume_from, weights_only=False, map_location=device
        )
        wandb_run_id_from_checkpoint = checkpoint.get("wandb_run_id")
        if wandb_run_id_from_checkpoint:
            print(f"Found wandb run ID in checkpoint: {wandb_run_id_from_checkpoint}")
        else:
            print("No wandb run ID found in checkpoint")

    init_kwargs: Dict[str, Any] = {
        "project": cfg.wandb_project,
        "name": cfg.wandb_name,
        "tags": cfg.wandb_tags or [],
        "config": asdict(cfg),
    }
    if wandb_run_id_from_checkpoint:
        init_kwargs["id"] = cfg.wandb_run_id or wandb_run_id_from_checkpoint
        init_kwargs["resume"] = "must"

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"Wandb init failed ({exc}); continuing without logging.")
        cfg.use_wandb = False
        return nullcontext()


def train_rebel(cfg: Config) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = _device_from_config(cfg)
    print(f"Using device: {device}")

    torch.set_float32_matmul_precision("high")

    torch.manual_seed(cfg.seed)

    run_cm = _init_wandb(cfg, device)

    with run_cm as run:
        trainer = RebelCFRTrainer(cfg=cfg, device=device)

        start_step = 0
        if cfg.resume_from and os.path.exists(cfg.resume_from):
            print(f"Resuming from checkpoint: {cfg.resume_from}")
            start_step = trainer.load_checkpoint(cfg.resume_from)
            print(f"Resumed at global step {start_step}")

        print(
            f"Starting ReBeL CFR training for {cfg.num_steps - start_step} steps "
            f"(total target: {cfg.num_steps})"
        )
        if cfg.use_wandb:
            print(
                f"📊 Wandb logging to project '{cfg.wandb_project}'"
                + (f" as run '{cfg.wandb_name}'" if cfg.wandb_name else "")
            )

        training_start = time.time()

        for step in range(start_step, cfg.num_steps):
            step_start = time.time()
            metrics = trainer.train_step(step)
            step_elapsed = time.time() - step_start
            total_elapsed = time.time() - training_start

            loss_str = (
                f"{metrics['loss']:.4f}" if metrics["loss"] is not None else "N/A"
            )
            policy_str = (
                f"{metrics['policy_loss']:.4f}"
                if metrics["policy_loss"] is not None
                else "N/A"
            )
            value_str = (
                f"{metrics['value_loss']:.4f}"
                if metrics["value_loss"] is not None
                else "N/A"
            )
            exploitability_str = (
                f"{metrics['local_exploitability']:.2f}"
                if metrics["local_exploitability"] is not None
                else "N/A"
            )

            print(
                f"[Step {metrics['step']:05d}] "
                f"loss={loss_str} "
                f"policy={policy_str} "
                f"value={value_str} "
                f"exploit={exploitability_str} "
                f"time={step_elapsed:.2f}s total={total_elapsed/60:.1f}m"
            )

            if cfg.use_wandb:
                metrics["step_time_s"] = step_elapsed
                wandb.log(metrics, step=metrics["step"])

            if (step + 1) % cfg.checkpoint_interval == 0:
                ckpt_path = os.path.join(
                    cfg.checkpoint_dir, f"rebel_step_{step + 1}.pt"
                )
                wandb_run_id = run.id if run else None
                # Save compressed checkpoint without optimizer state
                trainer.save_checkpoint(
                    ckpt_path,
                    step,
                    wandb_run_id=wandb_run_id,
                    save_optimizer=False,
                    save_dtype=torch.bfloat16,
                )
                # Save latest checkpoint with full state
                trainer.save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, "rebel_latest.pt"),
                    step,
                    wandb_run_id=wandb_run_id,
                    save_optimizer=True,
                    save_dtype=None,  # Keep host dtype
                )

                # Clean up old checkpoints if economize_checkpoints is enabled
                if cfg.economize_checkpoints:
                    _cleanup_old_checkpoints(cfg.checkpoint_dir, ckpt_path)

                print(f"Checkpoint saved at step {step + 1} -> {ckpt_path}")
                print_preflop_range_grid(trainer, step + 1, rebel=True)

        final_path = os.path.join(cfg.checkpoint_dir, "rebel_final.pt")
        trainer.save_checkpoint(
            final_path, cfg.num_steps, save_optimizer=True, save_dtype=None
        )
        total_elapsed = time.time() - training_start
        print(
            f"Training complete in {total_elapsed/3600:.2f} hours. "
            f"Final checkpoint: {final_path}"
        )
        print_preflop_range_grid(
            trainer, cfg.num_steps, title="Final Preflop Range Grid", rebel=True
        )


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="config_rebel_cfr"
)
def main(dict_config: DictConfig) -> None:
    config = Config.from_dict_config(dict_config)
    train_rebel(config)


if __name__ == "__main__":
    main()
