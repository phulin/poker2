#!/usr/bin/env python3
"""Pregenerate actions_8_11 solved data and sweep supervised LRs on it."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2.cli.train_rebel_preflop_buckets import _execution_config_from_config  # noqa: E402
from p2.config.rebel_load import load_rebel_config  # noqa: E402
from p2.rl.cfr_trainer import RebelCFRTrainer, _compile_kwargs  # noqa: E402
from p2.runtime.training_run import setup_torch_runtime  # noqa: E402
from p2.search.rebel_solved_dataset import (  # noqa: E402
    RebelSolvedDataset,
    RebelSolvedDatasetWriter,
)
from p2.stages.preflop_backward_induction import (  # noqa: E402
    PublicStateBucketReader,
    _build_validation_cache,
    _bucket_cfr_batch_size,
    _bucket_spec,
    _checkpoint_model_config,
    _copy_public_states_to_env,
    _evaluate_validation_set,
    _filter_batch_by_action_bucket,
    _load_model_weights,
    _make_env_from_manifest,
    _maybe_load_student_weights,
    _policy_only,
    _policy_train_batch_size,
    _random_beliefs,
    _solve_public_state_batch,
    _train_policy_minibatches,
    _train_value_minibatches,
    _validation_to_device,
    _value_only,
)
from p2.stages.preflop_buckets import build_run_config  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/preflop_backward_induction/"
    "gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5"
)
DEFAULT_STATE_DATASET = (
    REPO_ROOT
    / "outputs/preflop_policy_states/"
    "eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622"
)
DEFAULT_SOLVER_CHECKPOINT = DEFAULT_RUN_DIR / "actions_12_15/checkpoints/specialist_final.pt"
DEFAULT_TRAINER_CHECKPOINT = (
    DEFAULT_RUN_DIR / "actions_8_11/checkpoints/specialist_inprogress.pt"
)
DEFAULT_ROOT = REPO_ROOT / "outputs/preflop_lr_sweep/actions_8_11"


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _wandb_context(args: argparse.Namespace, *, name: str, stage: str, config: dict[str, Any]):
    if not bool(args.wandb):
        return nullcontext(None)
    import wandb

    return wandb.init(
        project="poker-rebel-preflop-lr-sweep",
        name=name,
        group="actions_8_11_pregen",
        tags=["preflop", "actions_8_11", "lr_sweep", stage],
        config=config,
    )


def _load_cfg(args: argparse.Namespace, *, output_dir: Path):
    overrides = [
        "device=cuda",
        f"use_wandb={str(bool(args.wandb)).lower()}",
        "preflop_buckets.command=train_specialists",
        f"preflop_buckets.state_dataset={args.state_dataset}",
        f"preflop_buckets.base_checkpoint={args.solver_checkpoint}",
        f"preflop_buckets.output_dir={output_dir}",
        "preflop_buckets.train_bucket=actions_8_11",
        "preflop_buckets.validation_eval_batch_size=1024",
        "preflop_buckets.compile=default",
    ]
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_preflop_buckets",
            overrides=overrides,
        )
    cfg = load_rebel_config(dc)
    execution = _execution_config_from_config(cfg)
    return cfg, execution


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _compile_solver_models(solver: RebelCFRTrainer, cfg: Any, device: torch.device) -> None:
    if device.type != "cuda":
        return
    compile_kwargs = _compile_kwargs(cfg)
    models = [
        getattr(solver.cfr_evaluator, "model", None),
        getattr(solver.cfr_evaluator, "closing_leaf_value_model", None),
    ]
    for model in models:
        if model is not None and hasattr(model, "compile_forward_modes"):
            model.compile_forward_modes(**compile_kwargs)


def _dataset_batch_by_shard(
    dataset: RebelSolvedDataset,
    stream: str,
    shard_idx: int,
    *,
    device: torch.device,
) -> Any:
    shard = dataset.shards[stream][shard_idx]
    start = int(shard["start"])
    count = int(shard["end"]) - start
    return dataset.get_batch(stream, start, count, device=device)


def pregenerate(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir or DEFAULT_ROOT / f"pregen_200steps_{_now_slug()}"
    solved_dir = output_dir / "solved"
    if solved_dir.exists() and any(solved_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{solved_dir} exists; pass --overwrite")
    solved_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, execution = _load_cfg(args, output_dir=output_dir)
    execution = replace(
        execution,
        states_per_bucket=int(args.steps) * _bucket_cfr_batch_size(
            execution, "actions_8_11"
        ),
        write_solved_shards=True,
        overwrite=bool(args.overwrite),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    cfr_batch_size = _bucket_cfr_batch_size(execution, "actions_8_11")
    cfg = build_run_config(
        base_cfg,
        execution,
        checkpoint_dir=output_dir / "checkpoints",
        num_steps=max(1, int(args.steps)),
        num_envs=cfr_batch_size,
        bucket_label="actions_8_11",
    )
    setup_torch_runtime(cfg, device, recompile_limit=16)
    cutoff_cfg = _checkpoint_model_config(cfg, str(args.solver_checkpoint))
    reader = PublicStateBucketReader(
        str(args.state_dataset),
        "actions_8_11",
        allow_partial=execution.allow_partial,
        seed=execution.seed + 10_000,
    )
    env = _make_env_from_manifest(
        reader.manifest,
        num_envs=cfr_batch_size,
        device=device,
        seed=execution.seed + 101,
    )
    solver = RebelCFRTrainer(
        cfg=copy.deepcopy(cutoff_cfg), device=device, pregeneration_only=True
    )
    _load_model_weights(solver, str(args.solver_checkpoint))
    if args.compile_cfr_models:
        _compile_solver_models(solver, cutoff_cfg, device)

    with _wandb_context(
        args,
        name=f"pregen-8-11-{args.steps}steps-{_now_slug()}",
        stage="pregenerate",
        config={
            "steps": int(args.steps),
            "cfr_batch_size": cfr_batch_size,
            "solver_checkpoint": str(args.solver_checkpoint),
            "closing_leaf_checkpoint": str(cfg.search.closing_leaf_checkpoint),
            "state_dataset": str(args.state_dataset),
        },
    ) as run:
        writer = RebelSolvedDatasetWriter(
            solved_dir,
            storage_float_dtype=execution.storage_dtype,
        )
        rng = torch.Generator(device=device)
        rng.manual_seed(int(execution.seed))
        spec = _bucket_spec("actions_8_11")
        roots = 0
        value_examples = 0
        policy_examples = 0
        started = time.time()
        for step, states in enumerate(
            reader.iter_state_batches(
                batch_size=cfr_batch_size,
                max_rows=int(args.steps) * cfr_batch_size,
                seed=execution.seed + 10_000,
            )
        ):
            rows = _copy_public_states_to_env(env, states)
            beliefs = _random_beliefs(
                rows,
                solver.num_players,
                device=device,
                rng=rng,
                mode=execution.belief_mode,
                profile=getattr(execution, "belief_profile", "actions_8_11"),
                hand_dim=getattr(execution, "belief_hand_dim", 169),
            )
            value_batch, policy_batch, tree_stats = _solve_public_state_batch(
                solver,
                env,
                beliefs,
                include_policy=True,
            )
            value_batch = _filter_batch_by_action_bucket(
                value_batch, low=spec.low, high=spec.high
            )
            policy_batch = _filter_batch_by_action_bucket(
                policy_batch, low=spec.low, high=spec.high
            )
            value_stream = _value_only(value_batch)
            policy_stream = _policy_only(policy_batch)
            writer.append("value", value_stream)
            writer.append("policy", policy_stream)
            roots += rows
            value_examples += len(value_stream)
            policy_examples += len(policy_stream)
            if (step + 1) % max(1, int(args.progress_steps)) == 0:
                elapsed = time.time() - started
                roots_s = roots / max(elapsed, 1.0e-9)
                print(
                    f"pregen step={step + 1}/{args.steps} roots={roots:,} "
                    f"value={value_examples:,} policy={policy_examples:,} "
                    f"nodes={int(tree_stats.get('evaluator_total_nodes', 0)):,} "
                    f"roots/s={roots_s:.2f}",
                    flush=True,
                )
                if run is not None:
                    run.log(
                        {
                            "pregen/step": step + 1,
                            "pregen/roots": roots,
                            "pregen/value_examples": value_examples,
                            "pregen/policy_examples": policy_examples,
                            "pregen/roots_per_s": roots_s,
                            "pregen/evaluator_total_nodes": int(
                                tree_stats.get("evaluator_total_nodes", 0)
                            ),
                        },
                        step=step + 1,
                    )
            if step + 1 >= int(args.steps):
                break

        manifest = writer.finalize(
            {
                "format_note": "fixed actions_8_11 LR-sweep solved data",
                "bucket_label": "actions_8_11",
                "root_states_solved": roots,
                "generation_steps": int(args.steps),
                "cfr_batch_size": cfr_batch_size,
                "solver_checkpoint": str(args.solver_checkpoint.resolve()),
                "closing_leaf_checkpoint": str(
                    Path(cfg.search.closing_leaf_checkpoint).resolve()
                )
                if cfg.search.closing_leaf_checkpoint is not None
                else None,
                "state_dataset": str(Path(args.state_dataset).resolve()),
                "belief_mode": execution.belief_mode,
                "belief_profile": getattr(execution, "belief_profile", "actions_8_11"),
                "belief_hand_dim": getattr(execution, "belief_hand_dim", 169),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if run is not None:
            run.summary["solved_manifest"] = str(solved_dir / "manifest.json")
            run.summary["root_states_solved"] = roots
            run.summary["value_examples"] = value_examples
            run.summary["policy_examples"] = policy_examples
    (output_dir / "pregen_summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote solved dataset {solved_dir / 'manifest.json'}", flush=True)
    return solved_dir


def _lr_trials(args: argparse.Namespace) -> list[dict[str, float | str]]:
    if args.adamw_lrs:
        muon_lr = float(args.base_muon_lr)
        return [
            {
                "name": (
                    f"lr{muon_lr:g}_adamw{adamw_lr:g}"
                    .replace(".", "p")
                    .replace("-", "m")
                ),
                "muon_lr": muon_lr,
                "adamw_lr": float(adamw_lr),
            }
            for adamw_lr in args.adamw_lrs
        ]
    if args.headline_lrs:
        base_muon = float(args.base_muon_lr)
        base_adamw = float(args.base_adamw_lr)
        return [
            {
                "name": f"lr{lr:g}".replace(".", "p").replace("-", "m"),
                "muon_lr": float(lr),
                "adamw_lr": base_adamw * float(lr) / base_muon,
            }
            for lr in args.headline_lrs
        ]
    if args.lr_scales:
        base_muon = float(args.base_muon_lr)
        base_adamw = float(args.base_adamw_lr)
        return [
            {
                "name": f"scale{scale:g}".replace(".", "p"),
                "muon_lr": base_muon * float(scale),
                "adamw_lr": base_adamw * float(scale),
            }
            for scale in args.lr_scales
        ]
    return [
        {"name": "lr00105_adamw000875", "muon_lr": 0.00105, "adamw_lr": 0.000875},
        {"name": "lr0015_adamw00125", "muon_lr": 0.0015, "adamw_lr": 0.00125},
        {"name": "lr0021_adamw00175", "muon_lr": 0.0021, "adamw_lr": 0.00175},
    ]


def _set_lrs(cfg: Any, *, muon_lr: float, adamw_lr: float) -> None:
    final_ratio = 0.15
    cfg.train.learning_rate = float(muon_lr)
    cfg.train.learning_rate_final = float(muon_lr) * final_ratio
    cfg.train.adamw_learning_rate = float(adamw_lr)


def sweep(args: argparse.Namespace) -> Path:
    if args.solved_dataset is None:
        raise ValueError("--solved-dataset is required for sweep")
    output_dir = args.output_dir or DEFAULT_ROOT / f"sweep_{_now_slug()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg, execution = _load_cfg(args, output_dir=output_dir)
    execution = replace(
        execution,
        states_per_bucket=max(args.step_counts) * _bucket_cfr_batch_size(
            execution, "actions_8_11"
        ),
        write_solved_shards=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    cfr_batch_size = _bucket_cfr_batch_size(execution, "actions_8_11")
    full_schedule_steps = max(1, max(int(x) for x in args.step_counts))
    cfg = build_run_config(
        base_cfg,
        execution,
        checkpoint_dir=output_dir / "template_checkpoints",
        num_steps=full_schedule_steps,
        num_envs=cfr_batch_size,
        bucket_label="actions_8_11",
    )
    cfg.train.warmup_steps = max(0, int(args.warmup_steps))
    if args.lr_schedule is not None:
        cfg.train.lr_schedule = str(args.lr_schedule)
    if args.wsd_decay_fraction is not None:
        cfg.train.lr_wsd_decay_fraction = float(args.wsd_decay_fraction)
    setup_torch_runtime(cfg, device, recompile_limit=16)

    validation = None
    if not bool(args.skip_validation):
        spec = _bucket_spec("actions_8_11")
        cutoff_cfg = _checkpoint_model_config(cfg, str(args.solver_checkpoint))
        reader = PublicStateBucketReader(
            str(args.state_dataset),
            "actions_8_11",
            allow_partial=execution.allow_partial,
            seed=execution.seed + 10_000,
        )
        validation = _build_validation_cache(
            args=execution,
            bucket_label="actions_8_11",
            spec=spec,
            bucket_dir=DEFAULT_RUN_DIR / "actions_8_11",
            cutoff_checkpoint=str(args.solver_checkpoint),
            cfg=cutoff_cfg,
            reader=reader,
            device=device,
            bucket_index=1,
        )
        validation = _validation_to_device(validation, device)
    dataset = RebelSolvedDataset(
        args.solved_dataset,
        pin_memory=False,
        async_shard_prefetch=True,
    )
    value_shards = len(dataset.shards["value"])
    policy_shards = len(dataset.shards["policy"])
    max_available_steps = value_shards if args.value_only else min(value_shards, policy_shards)
    if max(args.step_counts) > max_available_steps:
        raise ValueError(
            f"requested {max(args.step_counts)} steps but dataset has only "
            f"{max_available_steps} paired shards"
        )

    results: list[dict[str, Any]] = []
    with _wandb_context(
        args,
        name=f"sweep-8-11-{_now_slug()}",
        stage="sweep",
        config={
            "solved_dataset": str(args.solved_dataset),
            "step_counts": list(args.step_counts),
            "fresh_optimizer": bool(args.fresh_optimizer),
            "fresh_bucket_init": bool(args.fresh_bucket_init),
            "value_only": bool(args.value_only),
            "schedule_reset": bool(args.schedule_reset),
            "lr_schedule": str(cfg.train.lr_schedule),
            "warmup_steps": int(args.warmup_steps),
            "wsd_decay_fraction": cfg.train.lr_wsd_decay_fraction,
            "metric_last_n": int(args.metric_last_n),
            "value_train_batch_size": int(execution.train_batch_size),
            "policy_train_batch_size": _policy_train_batch_size(execution),
        },
    ) as run:
        for trial in _lr_trials(args):
            for step_count in args.step_counts:
                trial_cfg = copy.deepcopy(cfg)
                _set_lrs(
                    trial_cfg,
                    muon_lr=float(trial["muon_lr"]),
                    adamw_lr=float(trial["adamw_lr"]),
                )
                trial_name = f"{trial['name']}_{step_count}steps"
                trial_dir = output_dir / trial_name
                trial_cfg.checkpoint_dir = str(trial_dir / "checkpoints")
                trainer = RebelCFRTrainer(cfg=trial_cfg, device=device)
                if args.fresh_bucket_init:
                    _maybe_load_student_weights(
                        trainer,
                        str(args.solver_checkpoint),
                        required=True,
                    )
                    loaded_step = 0
                    schedule_start = 0
                else:
                    loaded_step = trainer.load_checkpoint(
                        str(args.trainer_checkpoint),
                        load_optimizer=not args.fresh_optimizer,
                    )
                    schedule_start = 0 if args.schedule_reset else int(loaded_step)
                before = (
                    _evaluate_validation_set(
                        trainer,
                        validation,
                        eval_batch_size=execution.validation_eval_batch_size,
                    )
                    if validation is not None
                    else {}
                )
                started = time.time()
                value_losses: list[float] = []
                for local_step in range(int(step_count)):
                    schedule_step = schedule_start + local_step
                    value_batch = _dataset_batch_by_shard(
                        dataset, "value", local_step, device=device
                    )
                    value_stats, value_minibatches = _train_value_minibatches(
                        trainer,
                        value_batch,
                        step=schedule_step,
                        batch_size=max(1, int(execution.train_batch_size)),
                    )
                    value_loss = float(value_stats["value_loss"])
                    value_losses.append(value_loss)
                    if not args.value_only:
                        policy_batch = _dataset_batch_by_shard(
                            dataset, "policy", local_step, device=device
                        )
                        _train_policy_minibatches(
                            trainer,
                            policy_batch,
                            step=schedule_step,
                            batch_size=_policy_train_batch_size(execution),
                        )
                    if run is not None:
                        run.log(
                            {
                                "train/trial": trial_name,
                                "train/local_step": local_step + 1,
                                "train/value_loss": value_loss,
                                "train/value_minibatches": value_minibatches,
                                "trial/muon_lr": float(trial["muon_lr"]),
                                "trial/adamw_lr": float(trial["adamw_lr"]),
                            }
                        )
                    if (local_step + 1) % max(1, int(args.progress_steps)) == 0:
                        print(
                            f"{trial_name}: step={local_step + 1}/{step_count} "
                            f"value_loss={value_loss:.6f}",
                            flush=True,
                        )
                after = (
                    _evaluate_validation_set(
                        trainer,
                        validation,
                        eval_batch_size=execution.validation_eval_batch_size,
                    )
                    if validation is not None
                    else {}
                )
                elapsed = time.time() - started
                metric_n = min(int(args.metric_last_n), len(value_losses))
                mean_last_value_loss = (
                    sum(value_losses[-metric_n:]) / max(1, metric_n)
                )
                row = {
                    "trial": trial_name,
                    "steps": int(step_count),
                    "muon_lr": float(trial["muon_lr"]),
                    "muon_lr_final": float(trial["muon_lr"]) * 0.15,
                    "adamw_lr": float(trial["adamw_lr"]),
                    "fresh_optimizer": bool(args.fresh_optimizer),
                    "fresh_bucket_init": bool(args.fresh_bucket_init),
                    "value_only": bool(args.value_only),
                    "lr_schedule": str(trial_cfg.train.lr_schedule),
                    "warmup_steps": int(args.warmup_steps),
                    "wsd_decay_fraction": float(trial_cfg.train.lr_wsd_decay_fraction),
                    "loaded_step": int(loaded_step),
                    "schedule_start": schedule_start,
                    "elapsed_s": elapsed,
                    "value_losses": value_losses,
                    "mean_last_value_loss": mean_last_value_loss,
                    "metric_last_n": metric_n,
                    "validation_before": before,
                    "validation_after": after,
                }
                results.append(row)
                trial_dir.mkdir(parents=True, exist_ok=True)
                (trial_dir / "result.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True) + "\n"
                )
                if run is not None:
                    payload = {
                        "trial/steps": int(step_count),
                        "trial/muon_lr": float(trial["muon_lr"]),
                        "trial/adamw_lr": float(trial["adamw_lr"]),
                        "trial/elapsed_s": elapsed,
                        "trial/mean_last_value_loss": mean_last_value_loss,
                    }
                    payload.update({f"before/{k}": v for k, v in before.items()})
                    payload.update({f"after/{k}": v for k, v in after.items()})
                    run.log(payload)
                print(
                    f"{trial_name}: mean_last{metric_n}_value_loss="
                    f"{mean_last_value_loss:.6f}",
                    flush=True,
                )

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "solved_dataset": str(Path(args.solved_dataset).resolve()),
        "trainer_checkpoint": str(Path(args.trainer_checkpoint).resolve()),
        "solver_checkpoint": str(Path(args.solver_checkpoint).resolve()),
        "value_train_batch_size": int(execution.train_batch_size),
        "policy_train_batch_size": _policy_train_batch_size(execution),
        "results": results,
    }
    ranked = sorted(results, key=lambda row: float(row["mean_last_value_loss"]))
    out["ranked_by_mean_last_value_loss"] = [
        {
            "trial": row["trial"],
            "muon_lr": row["muon_lr"],
            "adamw_lr": row["adamw_lr"],
            "steps": row["steps"],
            "mean_last_value_loss": row["mean_last_value_loss"],
        }
        for row in ranked
    ]
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {results_path}", flush=True)
    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
        p.add_argument("--state-dataset", type=Path, default=DEFAULT_STATE_DATASET)
        p.add_argument("--solver-checkpoint", type=Path, default=DEFAULT_SOLVER_CHECKPOINT)
        p.add_argument("--trainer-checkpoint", type=Path, default=DEFAULT_TRAINER_CHECKPOINT)
        p.add_argument("--output-dir", type=Path, default=None)
        p.add_argument("--progress-steps", type=int, default=10)
        p.set_defaults(wandb=True)
        p.add_argument("--no-wandb", action="store_false", dest="wandb")

    p_gen = sub.add_parser("pregenerate")
    add_common(p_gen)
    p_gen.add_argument("--steps", type=int, default=200)
    p_gen.add_argument("--overwrite", action="store_true")
    p_gen.add_argument("--compile-cfr-models", action="store_true")

    p_sweep = sub.add_parser("sweep")
    add_common(p_sweep)
    p_sweep.add_argument("--solved-dataset", type=Path, required=True)
    p_sweep.add_argument("--step-counts", type=int, nargs="+", default=[100, 200])
    p_sweep.add_argument("--headline-lrs", type=float, nargs="*", default=None)
    p_sweep.add_argument("--adamw-lrs", type=float, nargs="*", default=None)
    p_sweep.add_argument("--lr-scales", type=float, nargs="*", default=None)
    p_sweep.add_argument("--base-muon-lr", type=float, default=0.00105)
    p_sweep.add_argument("--base-adamw-lr", type=float, default=0.000875)
    p_sweep.add_argument("--fresh-optimizer", action="store_true")
    p_sweep.add_argument(
        "--fresh-bucket-init",
        action="store_true",
        help=(
            "Start each trial like a new actions_8_11 bucket: fresh optimizer, "
            "fresh schedule, and student weights warm-started from --solver-checkpoint."
        ),
    )
    p_sweep.add_argument("--value-only", action="store_true")
    p_sweep.add_argument("--skip-validation", action="store_true")
    p_sweep.add_argument("--schedule-reset", action="store_true")
    p_sweep.add_argument(
        "--lr-schedule",
        choices=("cosine", "linear", "wsd"),
        default=None,
    )
    p_sweep.add_argument("--warmup-steps", type=int, default=0)
    p_sweep.add_argument("--wsd-decay-fraction", type=float, default=None)
    p_sweep.add_argument("--metric-last-n", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "pregenerate":
        pregenerate(args)
    elif args.command == "sweep":
        sweep(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
