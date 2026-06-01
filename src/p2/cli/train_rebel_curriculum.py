#!/usr/bin/env python3
"""Curriculum orchestrator for staged postflop ReBeL training."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict
from typing import Any

import hydra
import torch
import wandb
from omegaconf import DictConfig

from p2.cli.train_rebel import (
    _device_from_config,
    _init_wandb,
    _log_model_parameter_summary,
)
from p2.core.structured_config import Config, CurriculumSubstepConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_loop import run_training_loop
from p2.utils.profiling import install_triton_compile_logger_from_env


def _stage_checkpoint_dir(cfg: Config, substep_name: str) -> str:
    return os.path.join(cfg.checkpoint_dir, substep_name)


def _stage_wandb_name(cfg: Config, substep_name: str) -> str | None:
    if cfg.wandb_name:
        return f"{cfg.wandb_name}-{substep_name}"
    return substep_name


def _checkpoint_metadata(
    substep_name: str, substep: CurriculumSubstepConfig
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "curriculum_substep": substep_name,
        "curriculum_kind": substep.kind,
        "curriculum_net": substep.net,
    }
    if substep.from_net is not None:
        metadata["curriculum_from_net"] = substep.from_net
    if substep.closing_net is not None:
        metadata["curriculum_closing_net"] = substep.closing_net
    if substep.chance is not None:
        metadata["curriculum_chance"] = substep.chance
    return metadata


def _read_checkpoint_metadata(path: str, device: torch.device) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    metadata = checkpoint.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


def _stage_names(cfg: Config) -> list[str]:
    names = list(cfg.curriculum.stages)
    if not names:
        names = list(cfg.curriculum.substeps.keys())
    if not names:
        raise ValueError("curriculum.stages must list at least one sub-step")
    missing = [name for name in names if name not in cfg.curriculum.substeps]
    if missing:
        raise ValueError(f"curriculum stages missing substep configs: {missing}")
    return names


def _resume_start_index(stage_names: list[str], metadata: dict[str, Any]) -> int:
    substep = metadata.get("curriculum_substep")
    if isinstance(substep, str) and substep in stage_names:
        return stage_names.index(substep)
    return 0


def _stage_config(
    cfg: Config,
    substep_name: str,
    substep: CurriculumSubstepConfig,
    *,
    resume_from: str | None,
    promoted: dict[str, str] | None = None,
) -> Config:
    stage_cfg = copy.deepcopy(cfg)
    stage_cfg.num_steps = int(substep.num_steps)
    stage_cfg.checkpoint_dir = substep.output_dir or _stage_checkpoint_dir(
        cfg, substep_name
    )
    stage_cfg.resume_from = resume_from
    stage_cfg.wandb_name = _stage_wandb_name(cfg, substep_name)
    closing_checkpoint = substep.closing_checkpoint
    if closing_checkpoint is None and promoted is not None and substep.closing_net:
        closing_checkpoint = promoted.get(substep.closing_net)
    stage_cfg.search.closing_leaf_checkpoint = closing_checkpoint
    return stage_cfg


def _run_train_substep(
    cfg: Config,
    substep_name: str,
    substep: CurriculumSubstepConfig,
    *,
    device: torch.device,
    resume_from: str | None,
    promoted: dict[str, str] | None = None,
) -> str:
    if cfg.data.mode != "live":
        raise NotImplementedError(
            "Curriculum train substeps currently support data.mode=live only; "
            "pregenerated mode depends on the RebelDataSource refactor."
        )

    stage_cfg = _stage_config(
        cfg,
        substep_name,
        substep,
        resume_from=resume_from,
        promoted=promoted,
    )
    os.makedirs(stage_cfg.checkpoint_dir, exist_ok=True)

    run_cm = _init_wandb(
        stage_cfg,
        device,
        group=cfg.curriculum.wandb_group,
        name=stage_cfg.wandb_name,
    )
    with run_cm as run:
        trainer = RebelCFRTrainer(cfg=stage_cfg, device=device)
        _log_model_parameter_summary(trainer.model, run)
        if isinstance(run, wandb.Run):
            run.watch(trainer.model, log_freq=100)

        start_step = 0
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming curriculum substep {substep_name}: {resume_from}")
            start_step = trainer.load_checkpoint(resume_from) + 1
            print(f"Resumed {substep_name} at step {start_step}")

        run_training_loop(
            trainer,
            stage_cfg,
            run,
            start_step=start_step,
            stop_step=stage_cfg.num_steps,
            stage_tag=substep_name,
            checkpoint_metadata=_checkpoint_metadata(substep_name, substep),
        )

    final_path = os.path.join(stage_cfg.checkpoint_dir, "rebel_final.pt")
    print(f"Promoted train substep {substep_name}: {final_path}")
    return final_path


def train_rebel_curriculum(cfg: Config) -> None:
    if install_triton_compile_logger_from_env():
        print("Triton compile logging enabled via P2_TRITON_COMPILE_LOG=1")

    device = _device_from_config(cfg)
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    stage_names = _stage_names(cfg)
    resume_metadata = _read_checkpoint_metadata(cfg.resume_from or "", device)
    start_index = _resume_start_index(stage_names, resume_metadata)
    promoted: dict[str, str] = {}

    for index, substep_name in enumerate(stage_names[start_index:], start=start_index):
        substep = cfg.curriculum.substeps[substep_name]
        resume_from = cfg.resume_from if index == start_index else None
        if substep.kind == "train":
            promoted[substep.net] = _run_train_substep(
                cfg,
                substep_name,
                substep,
                device=device,
                resume_from=resume_from,
                promoted=promoted,
            )
        elif substep.kind == "distill":
            raise NotImplementedError(
                "Curriculum distill substeps require the E_X distiller and "
                "chance-target dataset path, which are not implemented yet. "
                f"Blocked at substep {substep_name}: {asdict(substep)}"
            )
        else:
            raise ValueError(
                f"Unsupported curriculum substep kind for {substep_name}: "
                f"{substep.kind!r}"
            )

    print(f"Curriculum complete for implemented substeps: {promoted}")


@hydra.main(
    version_base=None, config_path="../../../conf", config_name="config_rebel_cfr"
)
def main(dict_config: DictConfig) -> None:
    config = Config.from_dict_config(dict_config)
    train_rebel_curriculum(config)


if __name__ == "__main__":
    main()
