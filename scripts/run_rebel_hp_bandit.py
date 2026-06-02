#!/usr/bin/env python3
"""Bounded MAB-style HP search for pregenerated ReBeL river training."""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from p2.core.structured_config import Config, PregeneratedDatasetConfig
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.rebel_solved_dataset import MANIFEST_NAME


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Arm:
    name: str
    overrides: dict[str, Any]


@dataclass
class ArmState:
    trials: int = 0
    reward_sum: float = 0.0
    best_objective: float = float("inf")

    @property
    def mean_reward(self) -> float:
        return self.reward_sum / max(1, self.trials)


def _device_from_name(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_manifest(dataset: Path) -> dict[str, Any]:
    manifest_path = dataset / MANIFEST_NAME if dataset.is_dir() else dataset
    return json.loads(manifest_path.read_text())


def _steps_from_epochs(
    manifest: dict[str, Any],
    *,
    batch_size: int,
    epochs: float | None,
    min_steps: int,
    explicit_steps: int | None,
) -> int:
    if explicit_steps is not None:
        return max(min_steps, int(explicit_steps))
    if epochs is None:
        return min_steps
    value_examples = int(manifest.get("value_examples", 0))
    if value_examples <= 0:
        raise ValueError("dataset manifest has no value_examples")
    epoch_steps = math.ceil(float(epochs) * value_examples / float(batch_size))
    return max(min_steps, epoch_steps)


def _load_base_config(config_path: Path) -> Config:
    container = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"expected mapping config at {config_path}")
    return Config.from_dict(container)


def _set_nested(obj: Any, dotted_key: str, value: Any) -> None:
    target = obj
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _make_trial_config(
    *,
    base_config_path: Path,
    dataset: Path,
    output_dir: Path,
    arm: Arm,
    trial_index: int,
    steps: int,
    batch_size: int,
    device_name: str,
    compile_mode: str,
    seed: int,
    keep_checkpoints: bool,
) -> Config:
    cfg = _load_base_config(base_config_path)
    cfg.num_steps = int(steps)
    cfg.seed = int(seed)
    cfg.device = device_name
    if device_name != "cuda":
        cfg.search.sparse_fused = False
    cfg.use_wandb = False
    cfg.wandb_name = None
    cfg.trueskill.enabled = False
    cfg.model.compile = compile_mode
    cfg.data.mode = "pregenerated"
    cfg.data.pregenerated.datasets = [
        PregeneratedDatasetConfig(path=str(dataset), value_weight=1.0, policy_weight=1.0)
    ]
    # Random direct sampling touches most shards for each large batch. Keep a
    # one-batch staging buffer and stream sequentially instead.
    cfg.data.pregenerated.direct_sample = False
    cfg.data.pregenerated.shuffle = False
    cfg.data.pregenerated.value_batch_size = batch_size
    cfg.data.pregenerated.policy_batch_size = batch_size
    cfg.train.batch_size = batch_size
    cfg.train.value_reuse_goal = 1.0
    cfg.train.replay_buffer_batches = 1
    cfg.train.episodes_per_step = 1
    cfg.checkpoint_interval = max(steps + 1, 1_000_000)
    cfg.economize_checkpoints = True
    cfg.checkpoint_dir = str(output_dir / "checkpoints" / f"trial_{trial_index:04d}")

    for key, value in arm.overrides.items():
        _set_nested(cfg, key, value)

    lr = float(cfg.train.learning_rate)
    if "train.learning_rate_final" not in arm.overrides:
        cfg.train.learning_rate_final = lr * 0.1
    if "train.adamw_learning_rate" not in arm.overrides:
        cfg.train.adamw_learning_rate = lr * 0.2
    if "train.policy_head_muon_learning_rate" not in arm.overrides:
        cfg.train.policy_head_muon_learning_rate = lr * 2.0
    if not keep_checkpoints:
        cfg.checkpoint_dir = str(output_dir / "scratch_checkpoints" / f"trial_{trial_index:04d}")

    return cfg


def _candidate_arms(
    rng: random.Random,
    max_arms: int,
    *,
    arch_set: str = "all",
) -> list[Arm]:
    standard_architectures = [
        (
            "compact",
            {
                "model.hidden_dim": 384,
                "model.range_hidden_dim": 192,
                "model.ffn_dim": 768,
                "model.num_hidden_layers": 2,
                "model.num_value_layers": 2,
                "model.num_policy_layers": 4,
                "model.policy_rank": 96,
                "model.policy_hand_bias_rank": 24,
            },
        ),
        (
            "base",
            {
                "model.hidden_dim": 512,
                "model.range_hidden_dim": 256,
                "model.ffn_dim": 1024,
                "model.num_hidden_layers": 3,
                "model.num_value_layers": 3,
                "model.num_policy_layers": 5,
                "model.policy_rank": 128,
                "model.policy_hand_bias_rank": 32,
            },
        ),
        (
            "wide",
            {
                "model.hidden_dim": 512,
                "model.range_hidden_dim": 256,
                "model.ffn_dim": 1536,
                "model.num_hidden_layers": 3,
                "model.num_value_layers": 3,
                "model.num_policy_layers": 5,
                "model.policy_rank": 128,
                "model.policy_hand_bias_rank": 32,
            },
        ),
        (
            "range512",
            {
                "model.hidden_dim": 512,
                "model.range_hidden_dim": 384,
                "model.ffn_dim": 1024,
                "model.num_hidden_layers": 3,
                "model.num_value_layers": 3,
                "model.num_policy_layers": 5,
                "model.policy_rank": 128,
                "model.policy_hand_bias_rank": 32,
            },
        ),
        (
            "deep_value",
            {
                "model.hidden_dim": 512,
                "model.range_hidden_dim": 256,
                "model.ffn_dim": 1024,
                "model.num_hidden_layers": 3,
                "model.num_value_layers": 5,
                "model.num_policy_layers": 4,
                "model.policy_rank": 128,
                "model.policy_hand_bias_rank": 32,
            },
        ),
    ]
    small_architectures = [
        (
            "tiny",
            {
                "model.hidden_dim": 256,
                "model.range_hidden_dim": 128,
                "model.ffn_dim": 512,
                "model.num_hidden_layers": 2,
                "model.num_value_layers": 2,
                "model.num_policy_layers": 3,
                "model.policy_rank": 64,
                "model.policy_hand_bias_rank": 16,
            },
        ),
        (
            "small",
            {
                "model.hidden_dim": 320,
                "model.range_hidden_dim": 160,
                "model.ffn_dim": 640,
                "model.num_hidden_layers": 2,
                "model.num_value_layers": 3,
                "model.num_policy_layers": 4,
                "model.policy_rank": 80,
                "model.policy_hand_bias_rank": 20,
            },
        ),
        (
            "compact",
            {
                "model.hidden_dim": 384,
                "model.range_hidden_dim": 192,
                "model.ffn_dim": 768,
                "model.num_hidden_layers": 2,
                "model.num_value_layers": 2,
                "model.num_policy_layers": 4,
                "model.policy_rank": 96,
                "model.policy_hand_bias_rank": 24,
            },
        ),
        (
            "compact_deep_value",
            {
                "model.hidden_dim": 384,
                "model.range_hidden_dim": 192,
                "model.ffn_dim": 768,
                "model.num_hidden_layers": 2,
                "model.num_value_layers": 5,
                "model.num_policy_layers": 4,
                "model.policy_rank": 96,
                "model.policy_hand_bias_rank": 24,
            },
        ),
    ]
    if arch_set == "all":
        architectures = standard_architectures
    elif arch_set == "small":
        architectures = small_architectures
    elif arch_set == "all_plus_small":
        architectures = small_architectures + standard_architectures
    else:
        raise ValueError(f"unsupported arch_set: {arch_set}")
    lr_starts = [0.006, 0.01, 0.016]
    schedules = ["cosine", "linear", "wsd"]

    arms: list[Arm] = []
    for (arch_name, arch), lr, schedule in itertools.product(
        architectures, lr_starts, schedules
    ):
        overrides = dict(arch)
        overrides.update(
            {
                "train.learning_rate": lr,
                "train.lr_schedule": schedule,
                "train.lr_wsd_decay_fraction": 0.2,
            }
        )
        arms.append(Arm(f"{arch_name}_lr{lr:g}_{schedule}", overrides))
    rng.shuffle(arms)
    return arms[:max_arms]


def _select_arm(
    arms: list[Arm], states: dict[str, ArmState], *, total_trials: int, ucb_c: float
) -> Arm:
    for arm in arms:
        if states[arm.name].trials == 0:
            return arm
    log_total = math.log(max(2, total_trials + 1))
    return max(
        arms,
        key=lambda arm: states[arm.name].mean_reward
        + ucb_c * math.sqrt(log_total / states[arm.name].trials),
    )


def _tail_average(metrics: list[dict[str, float]], tail_fraction: float) -> dict[str, float]:
    if not metrics:
        raise ValueError("no metrics collected")
    tail_count = max(1, int(math.ceil(len(metrics) * tail_fraction)))
    tail = metrics[-tail_count:]
    keys = sorted(set().union(*(m.keys() for m in tail)))
    out: dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in tail if key in m and m[key] is not None]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def _run_trial(
    cfg: Config,
    *,
    device: torch.device,
    steps: int,
    log_every: int,
    keep_checkpoint: bool,
) -> tuple[dict[str, float], float]:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(cfg.seed))
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    history: list[dict[str, float]] = []
    start = time.time()
    for step in range(steps):
        metrics = trainer.train_step(step)
        keep = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value is not None
        }
        history.append(keep)
        if log_every > 0 and ((step + 1) % log_every == 0 or step == 0):
            print(
                f"  step {step + 1:04d}/{steps} "
                f"loss={keep.get('loss', float('nan')):.6g} "
                f"value={keep.get('value_loss', float('nan')):.6g} "
                f"policy={keep.get('policy_loss', float('nan')):.6g}",
                flush=True,
            )
    if keep_checkpoint:
        trainer.save_checkpoint(
            os.path.join(cfg.checkpoint_dir, "rebel_final.pt"),
            steps - 1,
            save_optimizer=False,
            save_dtype=torch.bfloat16 if device.type == "cuda" else None,
        )
    elapsed = time.time() - start
    del trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _tail_average(history, tail_fraction=0.2), elapsed


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "conf" / "config_rebel_cfr.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "rebel_hp_bandit")
    parser.add_argument("--max-arms", type=int, default=12)
    parser.add_argument(
        "--arch-set",
        default="all",
        choices=["all", "small", "all_plus_small"],
        help="Architecture candidate family to sample.",
    )
    parser.add_argument("--trials", type=int, default=18)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--steps-per-trial", type=int, default=None)
    parser.add_argument("--min-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", default="off", choices=["off", "default", "max-autotune"])
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--ucb-c", type=float, default=0.01)
    parser.add_argument("--objective", default="value_loss", choices=["value_loss", "policy_loss", "loss"])
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = _load_manifest(args.dataset)
    steps = _steps_from_epochs(
        manifest,
        batch_size=args.batch_size,
        epochs=args.epochs,
        min_steps=args.min_steps,
        explicit_steps=args.steps_per_trial,
    )
    rng = random.Random(args.seed)
    arms = _candidate_arms(rng, args.max_arms, arch_set=args.arch_set)
    states = {arm.name: ArmState() for arm in arms}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"dataset value={manifest.get('value_examples')} policy={manifest.get('policy_examples')} "
        f"steps_per_trial={steps} arms={len(arms)} trials={args.trials}",
        flush=True,
    )
    for arm in arms:
        print(f"arm {arm.name}: {arm.overrides}", flush=True)
    if args.dry_run:
        return

    device = _device_from_name(args.device)
    results_jsonl = args.output_dir / "results.jsonl"
    results_csv = args.output_dir / "results.csv"

    for trial_index in range(args.trials):
        arm = _select_arm(
            arms, states, total_trials=trial_index, ucb_c=float(args.ucb_c)
        )
        trial_seed = int(args.seed + trial_index * 1009)
        print(
            f"\ntrial {trial_index + 1}/{args.trials}: {arm.name} seed={trial_seed}",
            flush=True,
        )
        cfg = _make_trial_config(
            base_config_path=args.config,
            dataset=args.dataset,
            output_dir=args.output_dir,
            arm=arm,
            trial_index=trial_index,
            steps=steps,
            batch_size=args.batch_size,
            device_name=args.device,
            compile_mode=args.compile,
            seed=trial_seed,
            keep_checkpoints=args.keep_checkpoints,
        )
        metrics, elapsed = _run_trial(
            cfg,
            device=device,
            steps=steps,
            log_every=args.log_every,
            keep_checkpoint=args.keep_checkpoints,
        )
        objective = float(metrics[args.objective])
        reward = -objective
        state = states[arm.name]
        state.trials += 1
        state.reward_sum += reward
        state.best_objective = min(state.best_objective, objective)
        row: dict[str, Any] = {
            "trial": trial_index,
            "arm": arm.name,
            "seed": trial_seed,
            "steps": steps,
            "elapsed_s": elapsed,
            "objective": args.objective,
            "objective_value": objective,
            "arm_trials": state.trials,
            "arm_mean_reward": state.mean_reward,
            **{f"metric_{k}": v for k, v in metrics.items()},
            "overrides": json.dumps(arm.overrides, sort_keys=True),
        }
        with results_jsonl.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        _append_csv(results_csv, row)
        print(
            f"trial done: objective={objective:.6g} elapsed={elapsed / 60:.1f}m "
            f"best_for_arm={state.best_objective:.6g}",
            flush=True,
        )

    ranking = sorted(states.items(), key=lambda item: -item[1].mean_reward)
    print("\nranking by mean reward:", flush=True)
    for name, state in ranking:
        print(
            f"{name}: trials={state.trials} "
            f"mean_objective={-state.mean_reward:.6g} best={state.best_objective:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
