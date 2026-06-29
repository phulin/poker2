#!/usr/bin/env python3
"""Benchmark the production compact preflop ``evaluate_cfr`` solve loop.

This complements ``bench_preflop_full_loop_profile.py``, which profiles direct
``cfr_iteration`` calls. The training/data-generation path calls
``evaluate_cfr`` and usually replays CUDA graphs, so this script measures the
actual per-solve hot loop for the current actions_4_7 shape.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2.cli.train_rebel_preflop_buckets import (  # noqa: E402
    _execution_config_from_config,
)
from p2.config.rebel_load import load_rebel_config  # noqa: E402
from p2.rl.cfr_trainer import RebelCFRTrainer  # noqa: E402
from p2.stages.preflop_backward_induction import (  # noqa: E402
    PublicStateBucketReader,
    _copy_public_states_to_env,
    _load_model_weights,
    _make_env_from_manifest,
    _random_beliefs,
    _seed_for_label,
)
from p2.stages.preflop_buckets import build_run_config  # noqa: E402


DEFAULT_DATASET = (
    REPO_ROOT
    / "outputs/preflop_policy_states/"
    "eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622"
)
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/preflop_backward_induction/"
    "gated_chain_from_sched7_distill_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260625"
)
DEFAULT_BASE_CHECKPOINT = (
    DEFAULT_RUN_DIR / "actions_4_7/checkpoints/specialist_inprogress.pt"
)
DEFAULT_CLOSING_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints-epreflop-distill-100k-lr2e4-sample256-from-sflop2000-169/"
    "promoted/E_preflop.pt"
)
DEFAULT_OUT = (
    REPO_ROOT / "outputs/preflop_full_loop_profile/evaluate_cfr_loop.json"
)


def _ancestor_pids(pid: int) -> set[int]:
    pids: set[int] = set()
    proc_dir = Path("/proc")
    current = pid
    while current > 0 and current not in pids:
        pids.add(current)
        try:
            for line in (proc_dir / str(current) / "status").read_text().splitlines():
                if line.startswith("PPid:"):
                    current = int(line.split()[1])
                    break
            else:
                break
        except OSError:
            break
    return pids


def _iter_processes() -> list[tuple[int, int, str]]:
    proc_dir = Path("/proc")
    own = _ancestor_pids(os.getpid())
    own_pgrp = os.getpgrp()
    out: list[tuple[int, int, str]] = []
    if not proc_dir.exists():
        return out
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in own:
            continue
        try:
            pgid = os.getpgid(pid)
            if pgid == own_pgrp:
                continue
            cmd = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        if cmd:
            out.append((pid, pgid, cmd))
    return out


@contextmanager
def _pause_processes(enabled: bool, pattern: str):
    paused: list[int] = []
    if enabled:
        seen_groups: set[int] = set()
        for _pid, pgid, cmd in _iter_processes():
            if pattern not in cmd:
                continue
            if pgid in seen_groups:
                continue
            seen_groups.add(pgid)
            try:
                os.killpg(pgid, signal.SIGSTOP)
                paused.append(pgid)
            except ProcessLookupError:
                continue
        if paused:
            print(f"Paused train process groups: {paused}", flush=True)
            time.sleep(0.5)
        else:
            print(f"No process matched pause pattern {pattern!r}.", flush=True)
    try:
        yield
    finally:
        for pgid in paused:
            try:
                os.killpg(pgid, signal.SIGCONT)
            except ProcessLookupError:
                continue
        if paused:
            print(f"Resumed train process groups: {paused}", flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _load_cfg(args: argparse.Namespace):
    overrides = [
        "device=cuda",
        "env.num_players=6",
        "preflop_buckets.command=train_specialists",
        f"preflop_buckets.state_dataset={args.state_dataset}",
        f"preflop_buckets.base_checkpoint={args.base_checkpoint}",
        f"preflop_buckets.output_dir={args.run_output_dir}",
        f"preflop_buckets.train_bucket={args.bucket}",
        "preflop_buckets.states_per_bucket=5000000",
        f"preflop_buckets.cfr_iterations={args.cfr_iterations}",
        "preflop_buckets.warm_start_iterations=0",
        "preflop_buckets.validation_items=4096",
        "preflop_buckets.validation_cfr_iterations=10000",
        "preflop_buckets.validation_interval_steps=10",
        "preflop_buckets.validation_eval_batch_size=1024",
        "preflop_buckets.train_batch_size=256",
        f"preflop_buckets.cfr_batch_size={args.cfr_batch_size}",
        "preflop_buckets.actions_8_11_cfr_batch_size=2048",
        "preflop_buckets.write_solved_shards=false",
        "preflop_buckets.allow_partial=false",
        "preflop_buckets.overwrite=false",
        "preflop_buckets.compile=default",
        "use_wandb=false",
        f"model.preflop_model_type={args.model_type}",
        "++model.preflop_hand_dim=169",
        f"model.hidden_dim={args.hidden_dim}",
        f"model.range_hidden_dim={args.range_hidden_dim}",
        f"model.ffn_dim={args.ffn_dim}",
        "model.board_interaction_dim=0",
        f"model.preflop_transformer_heads={args.transformer_heads}",
        f"model.num_hidden_layers={args.num_hidden_layers}",
        f"model.num_value_layers={args.num_value_layers}",
        f"model.num_policy_layers={args.num_policy_layers}",
        "model.enforce_zero_sum=false",
        "model.street_value_heads=both",
        f"model.compile={args.compile}",
        "search.model_scope=mixed_street",
    ]
    if args.no_closing_checkpoint:
        overrides.append("search.closing_leaf_checkpoint=null")
    else:
        overrides.append(f"search.closing_leaf_checkpoint={args.closing_checkpoint}")
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_preflop_buckets",
            overrides=overrides,
        )
    base_cfg = load_rebel_config(dc)
    execution = _execution_config_from_config(base_cfg)
    run_cfg = build_run_config(
        base_cfg,
        execution,
        checkpoint_dir=Path(args.run_output_dir) / args.bucket / "checkpoints",
        num_steps=1,
        num_envs=args.cfr_batch_size,
    )
    return execution, run_cfg


def _make_evaluator(args: argparse.Namespace) -> tuple[Any, int]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    execution, run_cfg = _load_cfg(args)
    reader = PublicStateBucketReader(
        args.state_dataset,
        args.bucket,
        allow_partial=execution.allow_partial,
        seed=execution.seed,
    )
    states = next(
        reader.iter_state_batches(
            batch_size=args.cfr_batch_size,
            max_rows=args.cfr_batch_size,
            seed=execution.seed,
        )
    )
    trainer = RebelCFRTrainer(cfg=run_cfg, device=device, pregeneration_only=True)
    if not args.skip_load_weights:
        _load_model_weights(trainer, str(args.base_checkpoint))
    env = _make_env_from_manifest(
        reader.manifest,
        num_envs=args.cfr_batch_size,
        device=device,
        seed=execution.seed + 100,
    )
    rows = _copy_public_states_to_env(env, states)
    rng = torch.Generator(device=device)
    rng.manual_seed(_seed_for_label(execution.seed, args.bucket, salt=500_000))
    beliefs = _random_beliefs(
        rows,
        env.num_players,
        device=device,
        rng=rng,
        mode=execution.belief_mode,
    )
    roots = torch.arange(rows, device=device)
    ev = trainer.cfr_evaluator
    ev.initialize_subgame(env, roots, beliefs)
    return ev, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument(
        "--closing-checkpoint", type=Path, default=DEFAULT_CLOSING_CHECKPOINT
    )
    parser.add_argument("--run-output-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--bucket", default="actions_4_7")
    parser.add_argument("--cfr-batch-size", type=int, default=512)
    parser.add_argument("--cfr-iterations", type=int, default=300)
    parser.add_argument(
        "--model-type",
        choices=("transformer", "ffn", "gated_token_mixer"),
        default="gated_token_mixer",
    )
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--range-hidden-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=0)
    parser.add_argument("--num-value-layers", type=int, default=5)
    parser.add_argument("--num-policy-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument(
        "--compile",
        choices=("off", "default", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--warmup-solves",
        type=int,
        default=1,
        help="Run untimed solves first so compile cost is excluded from timing.",
    )
    parser.add_argument("--no-closing-checkpoint", action="store_true")
    parser.add_argument("--skip-load-weights", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="preflop_backward_induction")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for _ in range(max(0, args.warmup_solves)):
        ev, _ = _make_evaluator(args)
        with torch.no_grad():
            ev.evaluate_cfr(training_mode=True, sample_continuation=False)
        _sync(ev.device)
    ev, rows = _make_evaluator(args)
    with _pause_processes(not args.no_pause, args.pause_pattern):
        _sync(ev.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            ev.evaluate_cfr(training_mode=True, sample_continuation=False)
        _sync(ev.device)
        wall_s = time.perf_counter() - t0
    result = {
        "wall_s": wall_s,
        "wall_ms_per_iter": 1e3 * wall_s / max(1, args.cfr_iterations),
        "rows": rows,
        "root_nodes": int(ev.root_nodes),
        "total_nodes": int(ev.total_nodes),
        "model_indices": int(ev.model_indices.numel()),
        "cfr_iterations": int(args.cfr_iterations),
    }
    print(
        f"wall={result['wall_s']:.3f}s "
        f"ms/iter={result['wall_ms_per_iter']:.3f}",
        flush=True,
    )
    output = {
        "config": {
            "bucket": args.bucket,
            "cfr_batch_size": args.cfr_batch_size,
            "cfr_iterations": args.cfr_iterations,
            "model_type": args.model_type,
            "hidden_dim": args.hidden_dim,
            "range_hidden_dim": args.range_hidden_dim,
            "ffn_dim": args.ffn_dim,
            "num_value_layers": args.num_value_layers,
            "num_policy_layers": args.num_policy_layers,
            "compile": args.compile,
        },
        "result": result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
