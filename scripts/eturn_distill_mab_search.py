#!/usr/bin/env python3
"""Sequential MAB-style search for E_turn distillation hyperparameters."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_RE = re.compile(
    r"\[Step\s+(?P<step>\d+)/(?P<total>\d+)\].*?value=(?P<value>[0-9.]+).*?time=(?P<time>[0-9.]+)s"
)
WANDB_RE = re.compile(r"https://wandb\.ai/\S+/runs/(?P<run_id>[A-Za-z0-9_-]+)")


@dataclass(frozen=True)
class Arm:
    name: str
    learning_rate: float
    learning_rate_final: float
    adamw_learning_rate: float
    batch_size: int
    lr_schedule: str
    weight_decay: float = 0.01
    optimizer: str = "muon"
    num_steps_override: int | None = None


@dataclass
class Trial:
    id: int
    arm: Arm
    num_steps: int
    examples: int
    wandb_name: str
    checkpoint_dir: str
    log_path: str
    status: str = "planned"
    started_at: str | None = None
    completed_at: str | None = None
    wandb_run_id: str | None = None
    final_value_loss: float | None = None
    best_value_loss: float | None = None
    best_step: int | None = None
    return_score: float | None = None
    exit_code: int | None = None
    command: list[str] = field(default_factory=list)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def _write_trials_yaml(path: Path, *, metadata: dict[str, Any], trials: list[Trial]) -> None:
    lines = ["metadata:"]
    for key, value in metadata.items():
        lines.append(f"  {key}: {_format_scalar(value)}")
    lines.append("trials:")
    for trial in trials:
        lines.extend(
            [
                f"  - id: {trial.id}",
                f"    status: {_format_scalar(trial.status)}",
                f"    started_at: {_format_scalar(trial.started_at)}",
                f"    completed_at: {_format_scalar(trial.completed_at)}",
                f"    wandb_name: {_format_scalar(trial.wandb_name)}",
                f"    wandb_run_id: {_format_scalar(trial.wandb_run_id)}",
                f"    checkpoint_dir: {_format_scalar(trial.checkpoint_dir)}",
                f"    log_path: {_format_scalar(trial.log_path)}",
                f"    num_steps: {trial.num_steps}",
                f"    examples: {trial.examples}",
                "    arm:",
            ]
        )
        for key, value in trial.arm.__dict__.items():
            lines.append(f"      {key}: {_format_scalar(value)}")
        lines.extend(
            [
                "    metrics:",
                f"      final_value_loss: {_format_scalar(trial.final_value_loss)}",
                f"      best_value_loss: {_format_scalar(trial.best_value_loss)}",
                f"      best_step: {_format_scalar(trial.best_step)}",
                f"      return_score: {_format_scalar(trial.return_score)}",
                f"      exit_code: {_format_scalar(trial.exit_code)}",
                "    command:",
            ]
        )
        for part in trial.command:
            lines.append(f"      - {_format_scalar(part)}")
    path.write_text("\n".join(lines) + "\n")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _trials_from_results(results: list[dict[str, Any]]) -> list[Trial]:
    trials = []
    arm_fields = set(Arm.__dataclass_fields__)
    for result in results:
        arm = Arm(
            **{
                key: value
                for key, value in result["arm"].items()
                if key in arm_fields
            }
        )
        trial = Trial(
            id=int(result["id"]),
            arm=arm,
            num_steps=int(result["num_steps"]),
            examples=int(result["examples"]),
            wandb_name=str(result["wandb_name"]),
            checkpoint_dir=str(result["checkpoint_dir"]),
            log_path=str(result["log_path"]),
            status=str(result["status"]),
            started_at=result.get("started_at"),
            completed_at=result.get("completed_at"),
            wandb_run_id=result.get("wandb_run_id"),
            final_value_loss=result.get("final_value_loss"),
            best_value_loss=result.get("best_value_loss"),
            best_step=result.get("best_step"),
            return_score=result.get("return_score"),
            exit_code=result.get("exit_code"),
        )
        trials.append(trial)
    return trials


def _arms(preset: str) -> list[Arm]:
    if preset == "next":
        return [
            Arm("lr0p06_cosine_b1024", 0.06, 0.006, 0.06, 1024, "cosine"),
            Arm("lr0p08_cosine_b1024_repeat", 0.08, 0.008, 0.08, 1024, "cosine"),
            Arm("lr0p12_cosine_b1024", 0.12, 0.012, 0.12, 1024, "cosine"),
            Arm("lr0p16_cosine_b1024", 0.16, 0.016, 0.16, 1024, "cosine"),
            Arm("lr0p08_cosine_b512", 0.08, 0.008, 0.08, 512, "cosine"),
            Arm("lr0p06_cosine_b512", 0.06, 0.006, 0.06, 512, "cosine"),
            Arm("lr0p08_linear_b1024", 0.08, 0.008, 0.08, 1024, "linear"),
            Arm(
                "lr0p08_cosine_b1024_1000step",
                0.08,
                0.008,
                0.08,
                1024,
                "cosine",
                num_steps_override=1000,
            ),
        ]
    return [
        Arm("base_cosine_b2048", 0.04, 0.004, 0.04, 2048, "cosine"),
        Arm("lr2x_cosine_b2048", 0.08, 0.008, 0.08, 2048, "cosine"),
        Arm("lr0p5_cosine_b2048", 0.02, 0.002, 0.02, 2048, "cosine"),
        Arm("base_linear_b2048", 0.04, 0.004, 0.04, 2048, "linear"),
        Arm("lr2x_linear_b2048", 0.08, 0.008, 0.08, 2048, "linear"),
        Arm("base_cosine_b1024", 0.04, 0.004, 0.04, 1024, "cosine"),
        Arm("lr2x_cosine_b1024", 0.08, 0.008, 0.08, 1024, "cosine"),
        Arm("base_linear_b1024", 0.04, 0.004, 0.04, 1024, "linear"),
        Arm("base_cosine_b4096", 0.04, 0.004, 0.04, 4096, "cosine"),
        Arm("lr2x_cosine_b4096", 0.08, 0.008, 0.08, 4096, "cosine"),
        Arm("lr4x_cosine_b2048", 0.16, 0.016, 0.16, 2048, "cosine"),
        Arm("lr0p25_linear_b4096", 0.01, 0.001, 0.01, 4096, "linear"),
    ]


def _seed_order(preset: str) -> list[str]:
    if preset == "next":
        return [
            "lr0p06_cosine_b1024",
            "lr0p08_cosine_b1024_repeat",
            "lr0p12_cosine_b1024",
            "lr0p16_cosine_b1024",
            "lr0p08_cosine_b512",
            "lr0p06_cosine_b512",
            "lr0p08_linear_b1024",
            "lr0p08_cosine_b1024_1000step",
        ]
    return [
        "base_cosine_b2048",
        "lr2x_cosine_b2048",
        "lr0p5_cosine_b2048",
        "base_linear_b2048",
        "base_cosine_b1024",
        "base_cosine_b4096",
    ]


def _choose_arm(arms: list[Arm], results: list[dict[str, Any]], rng: random.Random, *, seed_order: list[str]) -> Arm:
    by_arm: dict[str, list[float]] = {arm.name: [] for arm in arms}
    attempted = set()
    for result in results:
        attempted.add(str(result["arm"]["name"]))
        if result.get("status") == "completed" and result.get("return_score") is not None:
            by_arm.setdefault(str(result["arm"]["name"]), []).append(float(result["return_score"]))
    for seeded_name in seed_order:
        seeded = next((arm for arm in arms if arm.name == seeded_name), None)
        if seeded is not None and seeded.name not in attempted:
            return seeded
    untried = [arm for arm in arms if arm.name not in attempted]
    if untried:
        return rng.choice(untried)
    total = sum(len(values) for values in by_arm.values())
    if rng.random() < 0.15:
        return rng.choice(arms)
    best_arm = arms[0]
    best_score = -float("inf")
    for arm in arms:
        values = by_arm[arm.name]
        if not values:
            continue
        mean = sum(values) / len(values)
        bonus = 0.35 * math.sqrt(math.log(max(total, 2)) / len(values))
        score = mean + bonus
        if score > best_score:
            best_score = score
            best_arm = arm
    if best_score == -float("inf"):
        return rng.choice(arms)
    return best_arm


def _build_command(args: argparse.Namespace, trial: Trial) -> list[str]:
    arm = trial.arm
    return [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "p2.cli.train_rebel_curriculum",
        "--config-name",
        "config_rebel_curriculum_turn",
        f"num_steps={trial.num_steps}",
        "curriculum.stages=[distill_E_turn]",
        f"checkpoint_dir={trial.checkpoint_dir}",
        f"curriculum.promote_dir={trial.checkpoint_dir}/promoted",
        f"+curriculum.distill_E_turn.checkpoint={args.source_checkpoint}",
        "use_wandb=true",
        f"+wandb_name={trial.wandb_name}",
        f"train.batch_size={arm.batch_size}",
        f"train.learning_rate={arm.learning_rate}",
        f"train.learning_rate_final={arm.learning_rate_final}",
        f"train.adamw_learning_rate={arm.adamw_learning_rate}",
        f"train.lr_schedule={arm.lr_schedule}",
        f"train.weight_decay={arm.weight_decay}",
        f"train.optimizer={arm.optimizer}",
        "trueskill.enabled=false",
    ]


def _run_trial(trial: Trial, *, env: dict[str, str]) -> dict[str, Any]:
    values: list[tuple[int, float]] = []
    times: list[float] = []
    wandb_run_id = None
    log_path = Path(trial.log_path)
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
    return_score = -math.log10(max(best_value, 1e-12)) if best_value is not None else None
    return {
        "exit_code": exit_code,
        "wandb_run_id": wandb_run_id,
        "final_value_loss": final_value,
        "best_value_loss": best_value,
        "best_step": best_step,
        "return_score": return_score,
        "avg_step_time_s": sum(times) / len(times) if times else None,
    }


def _parse_until_utc(text: str) -> float:
    now = datetime.now(timezone.utc)
    hour, minute = [int(part) for part in text.split(":", 1)]
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target = target.replace(day=now.day + 1)
    return target.timestamp()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="outputs/experiments/eturn_distill_mab.yaml")
    parser.add_argument("--results", default="outputs/experiments/eturn_distill_mab_results.jsonl")
    parser.add_argument("--log-dir", default="outputs/experiments/logs")
    parser.add_argument("--checkpoint-root", default="checkpoints-eturn-distill-mab")
    parser.add_argument("--source-checkpoint", default="checkpoints-rebel-curriculum-sapdcfr-200-80-500it-2000/promoted/S_river.pt")
    parser.add_argument("--until-utc", default="22:00")
    parser.add_argument("--base-batch-size", type=int, default=2048)
    parser.add_argument("--base-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--preset", choices=["initial", "next"], default="initial")
    parser.add_argument("--max-trials", type=int, default=None)
    args = parser.parse_args()

    Path(args.spec).parent.mkdir(parents=True, exist_ok=True)
    Path(args.results).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_root).mkdir(parents=True, exist_ok=True)
    target_examples = args.base_batch_size * args.base_steps
    until_ts = _parse_until_utc(args.until_utc)
    rng = random.Random(args.seed)
    arms = _arms(args.preset)
    seed_order = _seed_order(args.preset)
    results = _read_results(Path(args.results))
    trials = _trials_from_results(results)
    metadata = {
        "strategy": "ucb1_epsilon_greedy",
        "epsilon": 0.15,
        "preset": args.preset,
        "objective": "minimize best printed value_loss over equal example budget unless arm overrides num_steps",
        "source_checkpoint": args.source_checkpoint,
        "target_examples_per_trial": target_examples,
        "base_batch_size": args.base_batch_size,
        "base_steps": args.base_steps,
        "until_utc": args.until_utc,
        "max_trials": args.max_trials,
        "created_at": _iso_now(),
        "updated_at": _iso_now(),
    }

    trial_id = len(results)
    while time.time() < until_ts and (args.max_trials is None or len(results) < args.max_trials):
        arm = _choose_arm(arms, results, rng, seed_order=seed_order)
        num_steps = arm.num_steps_override or math.ceil(target_examples / arm.batch_size)
        examples = num_steps * arm.batch_size
        trial_id += 1
        suffix = f"t{trial_id:03d}_{arm.name}_{num_steps}st_{arm.batch_size}b"
        trial = Trial(
            id=trial_id,
            arm=arm,
            num_steps=num_steps,
            examples=examples,
            wandb_name=f"eturn_mab_{suffix}",
            checkpoint_dir=f"{args.checkpoint_root}/{suffix}",
            log_path=f"{args.log_dir}/{suffix}.log",
        )
        trial.command = _build_command(args, trial)
        trial.status = "running"
        trial.started_at = _iso_now()
        trials.append(trial)
        metadata["updated_at"] = _iso_now()
        _write_trials_yaml(Path(args.spec), metadata=metadata, trials=trials)

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["UV_CACHE_DIR"] = "/tmp/uv-cache"
        outcome = _run_trial(trial, env=env)
        trial.completed_at = _iso_now()
        trial.status = "completed" if outcome["exit_code"] == 0 else "failed"
        trial.exit_code = outcome["exit_code"]
        trial.wandb_run_id = outcome["wandb_run_id"]
        trial.final_value_loss = outcome["final_value_loss"]
        trial.best_value_loss = outcome["best_value_loss"]
        trial.best_step = outcome["best_step"]
        trial.return_score = outcome["return_score"]
        metadata["updated_at"] = _iso_now()
        record = {
            "id": trial.id,
            "status": trial.status,
            "started_at": trial.started_at,
            "completed_at": trial.completed_at,
            "arm": trial.arm.__dict__,
            "num_steps": trial.num_steps,
            "examples": trial.examples,
            "wandb_name": trial.wandb_name,
            "wandb_run_id": trial.wandb_run_id,
            "checkpoint_dir": trial.checkpoint_dir,
            "log_path": trial.log_path,
            **outcome,
        }
        _append_jsonl(Path(args.results), record)
        results.append(record)
        _write_trials_yaml(Path(args.spec), metadata=metadata, trials=trials)

    metadata["finished_at"] = _iso_now()
    _write_trials_yaml(Path(args.spec), metadata=metadata, trials=trials)


if __name__ == "__main__":
    main()
