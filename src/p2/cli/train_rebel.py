#!/usr/bin/env python3
"""
Training script for ReBeL-style CFR with the feed-forward model.

Mirrors the structure of train_kbest.py but drives the RebelCFRTrainer and logs
search-driven supervision metrics to Weights & Biases.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import asdict
from typing import Any, Dict

import hydra
import torch
import wandb
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_loop import run_training_loop
from p2.utils.model_utils import count_model_parameters
from p2.utils.profiling import install_triton_compile_logger_from_env


def _device_from_config(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _init_wandb(
    cfg: Config,
    device: torch.device,
    *,
    group: str | None = None,
    name: str | None = None,
) -> Any:
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
        "name": cfg.wandb_name if name is None else name,
        "tags": cfg.wandb_tags or [],
        "config": asdict(cfg),
    }
    if group is not None:
        init_kwargs["group"] = group
    if wandb_run_id_from_checkpoint:
        init_kwargs["id"] = cfg.wandb_run_id or wandb_run_id_from_checkpoint
        init_kwargs["resume"] = "must"

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:
        print(f"Wandb init failed ({exc}); continuing without logging.")
        cfg.use_wandb = False
        return nullcontext()


def _log_model_parameter_summary(model: torch.nn.Module, run: Any) -> None:
    metrics = count_model_parameters(model)
    print(
        "Model parameters: "
        f"total={metrics['total_parameters']:,}; "
        f"trainable={metrics['trainable_parameters']:,}"
    )
    if isinstance(run, wandb.Run):
        run.summary.update(metrics)


def train_rebel(cfg: Config) -> None:
    if install_triton_compile_logger_from_env():
        print("Triton compile logging enabled via P2_TRITON_COMPILE_LOG=1")

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    device = _device_from_config(cfg)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    torch._dynamo.config.recompile_limit = 16

    torch.manual_seed(cfg.seed)

    run_cm = _init_wandb(cfg, device)

    with run_cm as run:
        trainer = RebelCFRTrainer(cfg=cfg, device=device)
        _log_model_parameter_summary(trainer.model, run)
        if isinstance(run, wandb.Run):
            run.watch(trainer.model, log_freq=100)

        print(
            f"Initialized trainer. Value buffer capacity: {trainer.value_buffer.capacity}; policy buffer capacity: {trainer.policy_buffer.capacity}."
        )
        print(
            f"Data generation rate: {trainer.K_value} value samples per training step."
        )

        start_step = 0
        if cfg.resume_from and os.path.exists(cfg.resume_from):
            print(f"Resuming from checkpoint: {cfg.resume_from}")
            start_step = trainer.load_checkpoint(cfg.resume_from) + 1
            print(f"Resumed at global step {start_step}")

        run_training_loop(
            trainer,
            cfg,
            run,
            start_step=start_step,
            stop_step=cfg.num_steps,
        )


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="config_rebel_cfr"
)
def main(dict_config: DictConfig) -> None:
    config = Config.from_dict_config(dict_config)
    train_rebel(config)


if __name__ == "__main__":
    main()
