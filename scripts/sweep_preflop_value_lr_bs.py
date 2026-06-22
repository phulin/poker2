#!/usr/bin/env python3
"""Sweep LR and batch size on a pregenerated preflop value dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.rebel_solved_dataset import RebelSolvedDataset
from p2.stages.preflop_backward_induction import (
    _evaluate_validation_set,
    _jsonable,
    _load_model_weights,
    _save_trainer_checkpoint,
    _train_value_minibatches,
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)


def _config_container_from_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_steps": resolved["run"]["num_steps"],
        "checkpoint_interval": resolved["checkpoint"]["checkpoint_interval"],
        "checkpoint_dir": resolved["checkpoint"]["checkpoint_dir"],
        "device": resolved["run"]["device"],
        "use_tensor_env": resolved["run"]["use_tensor_env"],
        "num_envs": resolved["run"]["num_envs"],
        "use_wandb": False,
        "wandb_project": resolved["logging"].get("wandb_project"),
        "wandb_name": None,
        "wandb_tags": resolved["logging"].get("wandb_tags") or ["rebel", "cfr"],
        "wandb_run_id": None,
        "resume_from": None,
        "seed": resolved["run"]["seed"],
        "config": resolved["run"].get("config"),
        "economize_checkpoints": resolved["checkpoint"].get(
            "economize_checkpoints", True
        ),
        "strict_model_loading": resolved["checkpoint"].get(
            "strict_model_loading", False
        ),
        "train": resolved["train"],
        "model": resolved["model"],
        "env": resolved["env"],
        "search": resolved["search"],
        "trueskill": resolved.get("trueskill", {}),
        "data": resolved.get("data", {}),
        "curriculum": resolved.get("curriculum", {}),
        "rebel_pregenerate": resolved.get("rebel_pregenerate", {}),
        "validation_set": resolved.get("validation_set", {}),
        "preflop_validation": resolved.get("preflop_validation", {}),
        "preflop_buckets": resolved.get("preflop_buckets", {}),
    }


def _load_base_config(path: Path) -> Config:
    resolved = json.loads(path.read_text())
    return Config.from_dict(_config_container_from_resolved(resolved))


def _trial_config(
    base_cfg: Config,
    *,
    lr: float,
    batch_size: int,
    total_updates: int,
    trial_dir: Path,
    device: str,
) -> Config:
    cfg = copy.deepcopy(base_cfg)
    cfg.device = device
    cfg.num_steps = max(1, int(total_updates))
    cfg.checkpoint_dir = str(trial_dir / "checkpoints")
    cfg.use_wandb = False
    cfg.wandb_name = None
    cfg.train.batch_size = int(batch_size)
    cfg.train.learning_rate = float(lr)
    cfg.train.learning_rate_final = float(lr)
    cfg.train.warmup_steps = 0
    cfg.train.policy_head_muon_learning_rate = float(lr)
    if cfg.train.adamw_learning_rate is not None:
        cfg.train.adamw_learning_rate = float(lr)
    return cfg


def _shuffle_order(length: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(int(length), generator=generator)


def _train_one_epoch(
    trainer: RebelCFRTrainer,
    dataset: RebelSolvedDataset,
    *,
    batch_size: int,
    seed: int,
) -> tuple[int, dict[str, float], float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shard_order = _shuffle_order(len(dataset.shards["value"]), generator)
    step = 0
    examples = 0
    weighted_stats: dict[str, float] = {}
    started = time.time()
    for shard_pos, shard_idx_tensor in enumerate(shard_order):
        shard_idx = int(shard_idx_tensor.item())
        shard = dataset.shards["value"][shard_idx]
        start = int(shard["start"])
        count = int(shard["end"]) - start
        batch = dataset.get_batch("value", start, count, float_dtype=torch.float32)
        row_order = _shuffle_order(len(batch), generator)
        batch = batch[row_order]
        step, stats, updates = _train_value_minibatches(
            trainer,
            batch,
            step=step,
            batch_size=batch_size,
        )
        examples += len(batch)
        for key, value in stats.items():
            if value is None:
                continue
            weighted_stats[key] = weighted_stats.get(key, 0.0) + float(value) * len(batch)
        if shard_pos == 0 or (shard_pos + 1) % 10 == 0 or shard_pos + 1 == len(shard_order):
            elapsed = time.time() - started
            print(
                f"  shard {shard_pos + 1}/{len(shard_order)} "
                f"examples={examples:,} updates={step} "
                f"last_updates={updates} elapsed={elapsed:.1f}s",
                flush=True,
            )
    averaged = {
        key: value / max(1, examples)
        for key, value in weighted_stats.items()
    }
    return step, averaged, time.time() - started


def _validation_value_loss(metrics: dict[str, float]) -> float:
    for key in ("validation_value_loss", "validation_value_value_loss"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"validation metrics missing value loss: {sorted(metrics)}")


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    solved_dir = Path(args.solved_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = Path(args.validation_cache)
    validation = torch.load(validation_path, map_location="cpu", weights_only=False)
    validation["policy_batch"] = None
    base_cfg = _load_base_config(Path(args.resolved_config))
    base_cfg.device = str(device)

    dataset = RebelSolvedDataset(
        solved_dir,
        num_players=base_cfg.env.num_players,
        num_actions=base_cfg.model.num_actions,
        context_length=int(validation["metadata"].get("context_length", 93))
        if "context_length" in validation.get("metadata", {})
        else int(json.loads((solved_dir / "manifest.json").read_text()).get("context_length", 93)),
        street_support=[0],
        async_shard_prefetch=True,
    )
    total_examples = dataset.stream_len("value")
    if total_examples <= 0:
        raise ValueError(f"{solved_dir} has no value examples")

    initial_metrics: dict[str, float] | None = None
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    trial_index = 0
    for lr in args.learning_rates:
        for batch_size in args.batch_sizes:
            trial_index += 1
            total_updates = math.ceil(total_examples / int(batch_size))
            trial_dir = output_dir / f"lr{lr:g}_bs{int(batch_size)}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            cfg = _trial_config(
                base_cfg,
                lr=float(lr),
                batch_size=int(batch_size),
                total_updates=total_updates,
                trial_dir=trial_dir,
                device=str(device),
            )
            print(
                f"trial {trial_index}: lr={lr:g} batch_size={batch_size} "
                f"examples={total_examples:,} updates={total_updates}",
                flush=True,
            )
            trainer = RebelCFRTrainer(cfg=cfg, device=device)
            _load_model_weights(trainer, args.base_checkpoint)
            if initial_metrics is None:
                initial_metrics = _evaluate_validation_set(
                    trainer,
                    validation,
                    eval_batch_size=args.validation_eval_batch_size,
                )
                print(
                    "initial validation: "
                    + json.dumps(_jsonable(initial_metrics), sort_keys=True),
                    flush=True,
                )
            step, train_stats, train_elapsed = _train_one_epoch(
                trainer,
                dataset,
                batch_size=int(batch_size),
                seed=int(args.seed) + trial_index * 1000,
            )
            metrics = _evaluate_validation_set(
                trainer,
                validation,
                eval_batch_size=args.validation_eval_batch_size,
            )
            value_loss = _validation_value_loss(metrics)
            result = {
                "trial": trial_index,
                "learning_rate": float(lr),
                "batch_size": int(batch_size),
                "updates": int(step),
                "examples": int(total_examples),
                "train_elapsed_s": train_elapsed,
                "train_stats": train_stats,
                "validation": metrics,
                "validation_value_loss": value_loss,
                "trial_dir": str(trial_dir),
            }
            results.append(result)
            (trial_dir / "result.json").write_text(
                json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
            )
            print(
                f"trial {trial_index} done: lr={lr:g} bs={batch_size} "
                f"validation_value_loss={value_loss:.8g} "
                f"elapsed={train_elapsed:.1f}s",
                flush=True,
            )
            if best is None or value_loss < float(best["validation_value_loss"]):
                best = result
                _save_trainer_checkpoint(
                    trainer,
                    output_dir / "best_rebel_latest.pt",
                    step=step,
                    run_id=None,
                    metadata={
                        "kind": "preflop_value_lr_bs_sweep_best",
                        "sweep_result": result,
                        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
                        "solved_dir": str(solved_dir.resolve()),
                        "validation_cache": str(validation_path.resolve()),
                    },
                )
            del trainer
            if device.type == "cuda":
                torch.cuda.empty_cache()

            summary = {
                "solved_dir": str(solved_dir),
                "base_checkpoint": str(args.base_checkpoint),
                "validation_cache": str(validation_path),
                "initial_validation": initial_metrics,
                "results": results,
                "best": best,
            }
            (output_dir / "summary.json").write_text(
                json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
            )

    assert best is not None
    print(
        "best: "
        f"lr={best['learning_rate']:g} bs={best['batch_size']} "
        f"validation_value_loss={best['validation_value_loss']:.8g}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solved-dir", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--validation-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.08],
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
    )
    parser.add_argument("--validation-eval-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
