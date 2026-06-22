#!/usr/bin/env python3
"""
Training script for ReBeL-style CFR with the feed-forward model.

Mirrors the structure of train_kbest.py but drives the RebelCFRTrainer and logs
search-driven supervision metrics to Weights & Biases.
"""

from __future__ import annotations

import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_loop import run_training_loop
from p2.runtime.training_run import (
    device_from_config,
    log_model_parameter_summary,
    training_run,
    wandb_run,
)
from p2.utils.profiling import install_triton_compile_logger_from_env


def _device_from_config(cfg: Config) -> torch.device:
    return device_from_config(cfg)


def _init_wandb(
    cfg: Config,
    device: torch.device,
    *,
    group: str | None = None,
    name: str | None = None,
) -> Any:
    del device
    return wandb_run(cfg, group=group, name=name)


def _log_model_parameter_summary(model: torch.nn.Module, run: Any) -> None:
    log_model_parameter_summary(model, run)


def train_rebel(cfg: Config) -> None:
    if install_triton_compile_logger_from_env():
        print("Triton compile logging enabled via P2_TRITON_COMPILE_LOG=1")

    with training_run(cfg) as runtime:
        device = runtime.device
        run = runtime.run
        trainer = RebelCFRTrainer(cfg=cfg, device=device)
        runtime.log_model_parameter_summary(trainer.model)
        runtime.watch_model(trainer.model, log_freq=100)

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
