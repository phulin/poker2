#!/usr/bin/env python3
"""Sweep LR and batch size on a pregenerated preflop value dataset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config, LrSchedule, ValueLossType
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.rebel_solved_dataset import RebelSolvedDataset
from p2.stages.preflop_backward_induction import (
    _aggregate_minibatch_stats,
    _evaluate_validation_set,
    _jsonable,
    _load_model_weights,
    _save_trainer_checkpoint,
)


@dataclass(frozen=True)
class TrialSpec:
    trial: int
    learning_rate: float
    batch_size: int
    lr_schedule: str
    learning_rate_final: float
    lr_final_ratio: float
    lr_wsd_decay_fraction: float
    adamw_learning_rate: float
    weight_decay: float
    grad_clip: float
    value_loss_type: str

    @property
    def name(self) -> str:
        return "_".join(
            [
                f"lr{_float_slug(self.learning_rate)}",
                f"bs{self.batch_size}",
                self.lr_schedule,
                f"final{_float_slug(self.lr_final_ratio)}",
                f"wsd{_float_slug(self.lr_wsd_decay_fraction)}",
                f"adamw{_float_slug(self.adamw_learning_rate)}",
                f"wd{_float_slug(self.weight_decay)}",
                f"gc{_float_slug(self.grad_clip)}",
                self.value_loss_type,
            ]
        )


def _float_slug(value: float) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace("+", "").replace(".", "p")


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
    trial: TrialSpec,
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
    cfg.train.batch_size = int(trial.batch_size)
    cfg.train.learning_rate = float(trial.learning_rate)
    cfg.train.learning_rate_final = float(trial.learning_rate_final)
    schedule = "cosine" if trial.lr_schedule == "constant" else trial.lr_schedule
    cfg.train.lr_schedule = LrSchedule(schedule)
    cfg.train.lr_wsd_decay_fraction = float(trial.lr_wsd_decay_fraction)
    cfg.train.warmup_steps = 0
    cfg.train.adamw_learning_rate = float(trial.adamw_learning_rate)
    cfg.train.weight_decay = float(trial.weight_decay)
    cfg.train.grad_clip = float(trial.grad_clip)
    cfg.train.value_loss_type = ValueLossType(trial.value_loss_type)
    return cfg


def _shuffle_order(length: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randperm(int(length), generator=generator)


def _train_one_epoch(
    trainer: RebelCFRTrainer,
    dataset: RebelSolvedDataset,
    *,
    batch_size: int,
    seed: int,
    tail_steps: int,
) -> tuple[int, dict[str, float], dict[str, float], float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shard_order = _shuffle_order(len(dataset.shards["value"]), generator)
    step = 0
    examples = 0
    weighted_stats: dict[str, float] = {}
    tail: deque[tuple[int, dict[str, float]]] = deque(maxlen=max(1, int(tail_steps)))
    started = time.time()
    for shard_pos, shard_idx_tensor in enumerate(shard_order):
        shard_idx = int(shard_idx_tensor.item())
        shard = dataset.shards["value"][shard_idx]
        start = int(shard["start"])
        count = int(shard["end"]) - start
        batch = dataset.get_batch("value", start, count, float_dtype=torch.float32)
        row_order = _shuffle_order(len(batch), generator)
        batch = batch[row_order]
        updates = 0
        for part_start in range(0, len(batch), batch_size):
            part = batch[part_start : min(part_start + batch_size, len(batch))]
            stats = trainer.train_value_batch(
                part,
                step,
                sync_inference_model=True,
            )
            stats = {
                key: float(value)
                for key, value in stats.items()
                if value is not None
            }
            updates += 1
            step += 1
            for key, value in stats.items():
                weighted_stats[key] = weighted_stats.get(key, 0.0) + value * len(part)
            tail.append((len(part), stats))
        examples += len(batch)
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
    tail_stats = _aggregate_minibatch_stats(list(tail))
    return step, averaged, tail_stats, time.time() - started


def _validation_value_loss(metrics: dict[str, float]) -> float:
    for key in ("validation_value_loss", "validation_value_value_loss"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"validation metrics missing value loss: {sorted(metrics)}")


def _tail_selection_loss(result: dict[str, Any]) -> float:
    stats = result.get("tail_train_stats") or {}
    for key in ("value_loss", "loss", "total_loss"):
        if key in stats:
            return float(stats[key])
    raise KeyError(f"tail_train_stats missing loss: {sorted(stats)}")


def _values_or_default(values: list[float] | None, default: float) -> list[float]:
    return [float(default)] if values is None else [float(value) for value in values]


def _trial_specs(args: argparse.Namespace, base_cfg: Config) -> list[TrialSpec]:
    learning_rates = [float(value) for value in args.learning_rates]
    batch_sizes = [int(value) for value in args.batch_sizes]
    lr_schedules = [str(value) for value in args.lr_schedules]
    lr_final_ratios = [float(value) for value in args.lr_final_ratios]
    wsd_decay_fractions = [float(value) for value in args.wsd_decay_fractions]
    weight_decays = _values_or_default(args.weight_decays, base_cfg.train.weight_decay)
    grad_clips = _values_or_default(args.grad_clips, base_cfg.train.grad_clip)
    value_loss_types = [str(value) for value in args.value_loss_types]

    specs: list[TrialSpec] = []
    for lr in learning_rates:
        adamw_lrs = _values_or_default(args.adamw_learning_rates, lr)
        for batch_size in batch_sizes:
            for schedule in lr_schedules:
                schedule_final_ratios = [1.0] if schedule == "constant" else lr_final_ratios
                schedule_wsd_fractions = (
                    wsd_decay_fractions
                    if schedule == "wsd"
                    else [wsd_decay_fractions[0]]
                )
                for final_ratio in schedule_final_ratios:
                    for wsd_fraction in schedule_wsd_fractions:
                        for adamw_lr in adamw_lrs:
                            for weight_decay in weight_decays:
                                for grad_clip in grad_clips:
                                    for value_loss_type in value_loss_types:
                                        specs.append(
                                            TrialSpec(
                                                trial=len(specs) + 1,
                                                learning_rate=lr,
                                                batch_size=batch_size,
                                                lr_schedule=schedule,
                                                learning_rate_final=lr
                                                * final_ratio,
                                                lr_final_ratio=final_ratio,
                                                lr_wsd_decay_fraction=wsd_fraction,
                                                adamw_learning_rate=adamw_lr,
                                                weight_decay=weight_decay,
                                                grad_clip=grad_clip,
                                                value_loss_type=value_loss_type,
                                            )
                                        )
    return specs


def _print_result_table(results: Iterable[dict[str, Any]]) -> None:
    for result in results:
        print(
            "  "
            f"trial={result['trial']} "
            f"lr={result['learning_rate']:.8g} "
            f"bs={result['batch_size']} "
            f"schedule={result['lr_schedule']} "
            f"final_ratio={result['lr_final_ratio']:.8g} "
            f"wsd={result['lr_wsd_decay_fraction']:.8g} "
            f"adamw={result['adamw_learning_rate']:.8g} "
            f"tail_loss={_tail_selection_loss(result):.8g}",
            flush=True,
        )


def run(args: argparse.Namespace) -> None:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    solved_dir = Path(args.solved_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = Path(args.validation_cache) if args.validation_cache else None
    validation = (
        torch.load(validation_path, map_location="cpu", weights_only=False)
        if validation_path is not None
        else None
    )
    if validation is not None:
        validation["policy_batch"] = None
    base_cfg = _load_base_config(Path(args.resolved_config))
    base_cfg.device = str(device)

    dataset = RebelSolvedDataset(
        solved_dir,
        num_players=base_cfg.env.num_players,
        num_actions=base_cfg.model.num_actions,
        context_length=(
            int(validation["metadata"].get("context_length", 93))
            if validation is not None and "context_length" in validation.get("metadata", {})
            else int(json.loads((solved_dir / "manifest.json").read_text()).get("context_length", 93))
        ),
        street_support=[0],
        async_shard_prefetch=True,
    )
    total_examples = dataset.stream_len("value")
    if total_examples <= 0:
        raise ValueError(f"{solved_dir} has no value examples")

    trial_specs = _trial_specs(args, base_cfg)
    if not trial_specs:
        raise ValueError("empty sweep")

    initial_metrics: dict[str, float] | None = None
    results: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for trial in trial_specs:
        total_updates = math.ceil(total_examples / int(trial.batch_size))
        trial_dir = output_dir / trial.name
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg = _trial_config(
            base_cfg,
            trial=trial,
            total_updates=total_updates,
            trial_dir=trial_dir,
            device=str(device),
        )
        print(
            f"trial {trial.trial}/{len(trial_specs)}: "
            f"lr={trial.learning_rate:g} batch_size={trial.batch_size} "
            f"schedule={trial.lr_schedule} "
            f"final_ratio={trial.lr_final_ratio:g} "
            f"wsd={trial.lr_wsd_decay_fraction:g} "
            f"adamw={trial.adamw_learning_rate:g} "
            f"examples={total_examples:,} updates={total_updates}",
            flush=True,
        )
        torch.manual_seed(int(args.model_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(args.model_seed))
        trainer = RebelCFRTrainer(cfg=cfg, device=device)
        if args.base_checkpoint:
            _load_model_weights(trainer, args.base_checkpoint)
        if validation is not None and initial_metrics is None:
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
        step, train_stats, tail_train_stats, train_elapsed = _train_one_epoch(
            trainer,
            dataset,
            batch_size=int(trial.batch_size),
            seed=int(args.seed) + trial.trial * int(args.trial_seed_stride),
            tail_steps=int(args.tail_steps),
        )
        metrics = (
            _evaluate_validation_set(
                trainer,
                validation,
                eval_batch_size=args.validation_eval_batch_size,
            )
            if validation is not None
            else {}
        )
        value_loss = _validation_value_loss(metrics) if metrics else None
        result = {
            "trial": trial.trial,
            "learning_rate": trial.learning_rate,
            "batch_size": trial.batch_size,
            "lr_schedule": trial.lr_schedule,
            "learning_rate_final": trial.learning_rate_final,
            "lr_final_ratio": trial.lr_final_ratio,
            "lr_wsd_decay_fraction": trial.lr_wsd_decay_fraction,
            "adamw_learning_rate": trial.adamw_learning_rate,
            "weight_decay": trial.weight_decay,
            "grad_clip": trial.grad_clip,
            "value_loss_type": trial.value_loss_type,
            "updates": int(step),
            "examples": int(total_examples),
            "train_elapsed_s": train_elapsed,
            "train_stats": train_stats,
            "tail_steps": int(args.tail_steps),
            "tail_train_stats": tail_train_stats,
            "validation": metrics,
            "validation_value_loss": value_loss,
            "trial_dir": str(trial_dir),
        }
        results.append(result)
        (trial_dir / "result.json").write_text(
            json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
        )
        done_msg = (
            f"trial {trial.trial} done: lr={trial.learning_rate:g} "
            f"bs={trial.batch_size} schedule={trial.lr_schedule} "
            f"adamw={trial.adamw_learning_rate:g} "
            f"tail_value_loss={_tail_selection_loss(result):.8g} "
        )
        if value_loss is not None:
            done_msg += f"validation_value_loss={value_loss:.8g} "
        done_msg += f"elapsed={train_elapsed:.1f}s"
        print(done_msg, flush=True)
        result_loss = _tail_selection_loss(result)
        best_loss = _tail_selection_loss(best) if best is not None else math.inf
        if result_loss < best_loss:
            best = result
            _save_trainer_checkpoint(
                trainer,
                output_dir / "best_rebel_latest.pt",
                step=step,
                run_id=None,
                metadata={
                    "kind": "preflop_value_lr_bs_sweep_best",
                    "sweep_result": result,
                    "base_checkpoint": (
                        str(Path(args.base_checkpoint).resolve())
                        if args.base_checkpoint
                        else None
                    ),
                    "solved_dir": str(solved_dir.resolve()),
                    "validation_cache": (
                        str(validation_path.resolve())
                        if validation_path is not None
                        else None
                    ),
                },
            )
        del trainer
        if device.type == "cuda":
            torch.cuda.empty_cache()

        summary = {
            "solved_dir": str(solved_dir),
            "base_checkpoint": str(args.base_checkpoint),
            "validation_cache": str(validation_path) if validation_path is not None else None,
            "initial_validation": initial_metrics,
            "trial_count": len(trial_specs),
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
        f"schedule={best['lr_schedule']} "
        f"adamw={best['adamw_learning_rate']:g} "
        f"tail_value_loss={_tail_selection_loss(best):.8g}",
        flush=True,
    )
    _print_result_table(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solved-dir", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--validation-cache", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.08],
    )
    parser.add_argument(
        "--adamw-learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="AdamW LR values. Defaults to matching each main Muon LR.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
    )
    parser.add_argument(
        "--lr-schedules",
        choices=["constant", "cosine", "linear", "wsd"],
        nargs="+",
        default=["constant"],
    )
    parser.add_argument("--lr-final-ratios", type=float, nargs="+", default=[0.1])
    parser.add_argument("--wsd-decay-fractions", type=float, nargs="+", default=[0.2])
    parser.add_argument("--weight-decays", type=float, nargs="+", default=None)
    parser.add_argument("--grad-clips", type=float, nargs="+", default=None)
    parser.add_argument(
        "--value-loss-types",
        choices=["huber", "mse", "quantile"],
        nargs="+",
        default=["huber"],
    )
    parser.add_argument("--validation-eval-batch-size", type=int, default=1024)
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=10,
        help="Number of final optimizer updates to average into tail_train_stats.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=42,
        help="Seed applied before constructing each trial model.",
    )
    parser.add_argument(
        "--trial-seed-stride",
        type=int,
        default=1000,
        help="Added to --seed per trial; set 0 to use identical shard/row shuffles.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
