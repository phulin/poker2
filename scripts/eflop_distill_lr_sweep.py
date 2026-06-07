#!/usr/bin/env python3
"""Sequential fixed-LR sweep for E_flop distillation runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"\[Step\s+(?P<step>\d+)/(?P<total>\d+)\].*?value=(?P<value>[0-9.eE+-]+).*?time=(?P<time>[0-9.]+)s"
)
WANDB_RE = re.compile(r"https://wandb\.ai/\S+/runs/(?P<run_id>[A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class TrialConfig:
    learning_rate: float
    batch_size: int
    num_steps: int
    lr_schedule: str = "linear"


@dataclass
class TrialResult:
    id: int
    config: TrialConfig
    wandb_name: str
    checkpoint_dir: str
    log_path: str
    command: list[str] = field(default_factory=list)
    status: str = "planned"
    started_at: str | None = None
    completed_at: str | None = None
    wandb_run_id: str | None = None
    exit_code: int | None = None
    final_value_loss: float | None = None
    best_value_loss: float | None = None
    best_step: int | None = None
    avg_step_time_s: float | None = None


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_lrs(text: str) -> list[float]:
    values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one LR")
    return values


def _fmt_lr(lr: float) -> str:
    return f"{lr:g}".replace(".", "p")


def _write_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _write_spec(path: Path, metadata: dict[str, Any], trials: list[TrialResult]) -> None:
    payload = {
        "metadata": metadata,
        "trials": [
            {
                **asdict(trial),
                "config": asdict(trial.config),
            }
            for trial in trials
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_command(args: argparse.Namespace, trial: TrialResult) -> list[str]:
    cfg = trial.config
    lr_final = cfg.learning_rate * args.final_lr_ratio
    return [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "p2.cli.train_rebel_curriculum",
        "--config-name",
        "config_rebel_curriculum_flop",
        f"num_steps={cfg.num_steps}",
        "curriculum.stages=[distill_E_flop]",
        f"checkpoint_dir={trial.checkpoint_dir}",
        f"curriculum.promote_dir={trial.checkpoint_dir}/promoted",
        f"+curriculum.distill_E_flop.checkpoint={args.source_checkpoint}",
        "use_wandb=true",
        f"+wandb_name={trial.wandb_name}",
        f"curriculum.distill_E_flop.train_overrides.learning_rate={cfg.learning_rate}",
        f"curriculum.distill_E_flop.train_overrides.learning_rate_final={lr_final}",
        f"curriculum.distill_E_flop.train_overrides.adamw_learning_rate={cfg.learning_rate}",
        f"curriculum.distill_E_flop.train_overrides.lr_schedule={cfg.lr_schedule}",
        f"curriculum.distill_E_flop.train_overrides.batch_size={cfg.batch_size}",
        "trueskill.enabled=false",
    ]


def _run_trial(trial: TrialResult, env: dict[str, str]) -> dict[str, Any]:
    values: list[tuple[int, float]] = []
    times: list[float] = []
    wandb_run_id = None
    log_path = Path(trial.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            trial.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
            log_fh.flush()
            m_run = WANDB_RE.search(line)
            if m_run:
                wandb_run_id = m_run.group("run_id")
            m_step = STEP_RE.search(line)
            if m_step:
                values.append((int(m_step.group("step")), float(m_step.group("value"))))
                times.append(float(m_step.group("time")))
        exit_code = proc.wait()
    best_step = None
    best_value = None
    final_value = None
    if values:
        best_step, best_value = min(values, key=lambda item: item[1])
        final_value = values[-1][1]
    return {
        "exit_code": exit_code,
        "wandb_run_id": wandb_run_id,
        "final_value_loss": final_value,
        "best_value_loss": best_value,
        "best_step": best_step,
        "avg_step_time_s": sum(times) / len(times) if times else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint",
        default=(
            "checkpoints-rebel-curriculum-turn-2000-live-from-sriver2000-eturn25000-"
            "fused-grouped-allin-wandb-full/promoted/S_turn.pt"
        ),
    )
    parser.add_argument("--learning-rates", type=_parse_lrs, default=[0.01, 0.02, 0.04, 0.08, 0.16])
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", default="linear")
    parser.add_argument("--checkpoint-root", default="checkpoints-eflop-distill-lr-sweep")
    parser.add_argument("--log-dir", default="outputs/experiments/eflop_distill_lr_logs")
    parser.add_argument("--spec", default="outputs/experiments/eflop_distill_lr_sweep.json")
    parser.add_argument("--results", default="outputs/experiments/eflop_distill_lr_results.jsonl")
    parser.add_argument("--wandb-prefix", default="eflop_lr_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.source_checkpoint):
        raise FileNotFoundError(args.source_checkpoint)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.spec).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at": _iso_now(),
        "source_checkpoint": args.source_checkpoint,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "lr_schedule": args.lr_schedule,
        "final_lr_ratio": args.final_lr_ratio,
        "override_target": "curriculum.distill_E_flop.train_overrides",
    }
    trials: list[TrialResult] = []
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for idx, lr in enumerate(args.learning_rates, start=1):
        suffix = f"t{idx:03d}_lr{_fmt_lr(lr)}_{args.num_steps}st_b{args.batch_size}"
        trial = TrialResult(
            id=idx,
            config=TrialConfig(
                learning_rate=lr,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                lr_schedule=args.lr_schedule,
            ),
            wandb_name=f"{args.wandb_prefix}_{suffix}",
            checkpoint_dir=f"{args.checkpoint_root}/{suffix}",
            log_path=f"{args.log_dir}/{suffix}.log",
        )
        trial.command = _build_command(args, trial)
        trial.status = "running"
        trial.started_at = _iso_now()
        trials.append(trial)
        metadata["updated_at"] = _iso_now()
        _write_spec(Path(args.spec), metadata, trials)

        outcome = _run_trial(trial, env)
        trial.completed_at = _iso_now()
        trial.status = "completed" if outcome["exit_code"] == 0 else "failed"
        trial.exit_code = outcome["exit_code"]
        trial.wandb_run_id = outcome["wandb_run_id"]
        trial.final_value_loss = outcome["final_value_loss"]
        trial.best_value_loss = outcome["best_value_loss"]
        trial.best_step = outcome["best_step"]
        trial.avg_step_time_s = outcome["avg_step_time_s"]
        metadata["updated_at"] = _iso_now()
        record = {
            **asdict(trial),
            "config": asdict(trial.config),
        }
        _write_jsonl(Path(args.results), record)
        _write_spec(Path(args.spec), metadata, trials)
        if trial.exit_code != 0:
            break

    metadata["finished_at"] = _iso_now()
    _write_spec(Path(args.spec), metadata, trials)


if __name__ == "__main__":
    main()
