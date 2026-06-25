#!/usr/bin/env python3
"""Profile eager 6-player preflop fused CFR iterations on bucket states.

The defaults match the ongoing actions_4_7 backward-induction run:
6 players, depth 4, cfr batch 512, 300 CFR iterations, BetterFFN h192,
transformer-169, and the packed preflop state dataset.
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
from torch.profiler import ProfilerActivity, profile, record_function

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
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/preflop_full_loop_profile"
DEFAULT_BASE_CHECKPOINT = (
    REPO_ROOT
    / "outputs/preflop_backward_induction/"
    "rest_buckets_from12_mixed_300cfr_d4cutoff_20260622/"
    "actions_8_11/checkpoints/rebel_latest.pt"
)
DEFAULT_CLOSING_CHECKPOINT = (
    REPO_ROOT
    / "checkpoints-epreflop-distill-100k-lr2e4-sample256-from-sflop2000-169/"
    "promoted/E_preflop.pt"
)

COMPONENT_TAGS = (
    "cfr_iteration",
    "update_policy",
    "set_leaf_values",
    "compute_expected_values",
    "update_average_values",
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


def _wrap_method(obj: Any, attr: str, tag: str) -> None:
    orig = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        with record_function(tag):
            return orig(*args, **kwargs)

    setattr(obj, attr, wrapped)


def _patch_component_tags(ev: Any) -> None:
    for attr, tag in (
        ("cfr_iteration", "cfr_iteration"),
        ("update_policy", "update_policy"),
        ("set_leaf_values", "set_leaf_values"),
        ("compute_expected_values", "compute_expected_values"),
        ("update_average_values", "update_average_values"),
    ):
        if hasattr(ev, attr):
            _wrap_method(ev, attr, tag)


def _event_device_us(evt: Any) -> float:
    value = getattr(evt, "device_time_total", None)
    if value is not None:
        return float(value or 0.0)
    return float(getattr(evt, "cuda_time_total", 0.0) or 0.0)


def _component_rows(prof: Any, iters: int, wall_s: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tag in COMPONENT_TAGS:
        cuda_us = 0.0
        cpu_us = 0.0
        calls = 0
        for evt in prof.events():
            if evt.name != tag:
                continue
            cuda_us += _event_device_us(evt)
            cpu_us += float(evt.cpu_time_total or 0.0)
            calls += 1
        if calls == 0:
            continue
        rows.append(
            {
                "component": tag,
                "calls": calls,
                "calls_per_iter": calls / max(1, iters),
                "cuda_ms_per_iter": cuda_us / 1e3 / max(1, iters),
                "cpu_ms_per_iter": cpu_us / 1e3 / max(1, iters),
                "cuda_pct_of_wall": 100.0 * (cuda_us / 1e6) / max(wall_s, 1e-12),
            }
        )
    rows.sort(key=lambda row: float(row["cuda_ms_per_iter"]), reverse=True)
    return rows


def _kernel_rows(prof: Any, iters: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evt in prof.key_averages():
        self_cuda_us = float(
            getattr(evt, "self_device_time_total", None)
            or getattr(evt, "self_cuda_time_total", 0.0)
            or 0.0
        )
        if self_cuda_us <= 0.0:
            continue
        count = int(getattr(evt, "count", 1) or 1)
        rows.append(
            {
                "name": evt.key,
                "count_per_iter": count / max(1, iters),
                "self_cuda_ms_per_iter": self_cuda_us / 1e3 / max(1, iters),
            }
        )
    rows.sort(key=lambda row: float(row["self_cuda_ms_per_iter"]), reverse=True)
    return rows


def _jsonable_shapes(value: Any) -> Any:
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable_shapes(item) for item in value]
    return value


def _kernel_shape_rows(prof: Any, iters: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        events = prof.key_averages(group_by_input_shape=True)
    except TypeError:
        return rows
    for evt in events:
        self_cuda_us = float(
            getattr(evt, "self_device_time_total", None)
            or getattr(evt, "self_cuda_time_total", 0.0)
            or 0.0
        )
        if self_cuda_us <= 0.0:
            continue
        count = int(getattr(evt, "count", 1) or 1)
        rows.append(
            {
                "name": evt.key,
                "input_shapes": _jsonable_shapes(getattr(evt, "input_shapes", [])),
                "count_per_iter": count / max(1, iters),
                "self_cuda_ms_per_iter": self_cuda_us / 1e3 / max(1, iters),
            }
        )
    rows.sort(key=lambda row: float(row["self_cuda_ms_per_iter"]), reverse=True)
    return rows


def _load_cfg(args: argparse.Namespace):
    overrides = [
        "device=cuda",
        "+env.num_players=6",
        "preflop_buckets.command=train_specialists",
        f"preflop_buckets.state_dataset={args.state_dataset}",
        f"preflop_buckets.base_checkpoint={args.base_checkpoint}",
        f"preflop_buckets.output_dir={args.run_output_dir}",
        "preflop_buckets.train_bucket=actions_4_7",
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
        "+search.model_scope=mixed_street",
    ]
    if args.no_closing_checkpoint:
        overrides.append("+search.closing_leaf_checkpoint=null")
    else:
        overrides.append(f"+search.closing_leaf_checkpoint={args.closing_checkpoint}")
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
    return base_cfg, execution, run_cfg


def _prepare_evaluator(trainer: RebelCFRTrainer, env: Any, beliefs: torch.Tensor) -> Any:
    ev = trainer.cfr_evaluator
    roots = torch.arange(beliefs.shape[0], device=trainer.device)
    with torch.no_grad():
        ev.initialize_subgame(env, roots, beliefs)
        ev.initialize_policy_and_beliefs()
        if ev.warm_start_iterations > 0:
            ev.warm_start()
        ev.set_leaf_values(0)
        ev.compute_expected_values()
        ev.values_avg[:] = ev.latest_values
        ev.t_sample = ev._get_sampling_schedule()
        if hasattr(ev, "_prepare_sample_update_table"):
            ev._prepare_sample_update_table()
        if hasattr(ev, "_skip_record_stats"):
            ev._skip_record_stats = True
    _sync(trainer.device)
    return ev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument(
        "--closing-checkpoint", type=Path, default=DEFAULT_CLOSING_CHECKPOINT
    )
    parser.add_argument(
        "--run-output-dir",
        type=Path,
        default=REPO_ROOT
        / "outputs/preflop_backward_induction/"
        "rest_buckets_from12_mixed_300cfr_d4cutoff_20260622",
    )
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
    parser.add_argument("--no-closing-checkpoint", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--active-iters", type=int, default=10)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "profile.json")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="preflop_backward_induction")
    parser.add_argument("--skip-load-weights", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    base_cfg, execution, run_cfg = _load_cfg(args)
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

    with _pause_processes(not args.no_pause, args.pause_pattern):
        trainer = RebelCFRTrainer(
            cfg=run_cfg,
            device=device,
            pregeneration_only=True,
        )
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
        ev = _prepare_evaluator(trainer, env, beliefs)
        _patch_component_tags(ev)

        print(
            json.dumps(
                {
                    "bucket": args.bucket,
                    "rows": rows,
                    "total_nodes": int(ev.total_nodes),
                    "root_nodes": int(ev.root_nodes),
                    "tree_depth": int(ev.tree_depth),
                    "model_indices": int(ev.model_indices.numel()),
                    "showdown_indices": int(ev.showdown_indices.numel()),
                    "rank_stats_src_weights": os.environ.get(
                        "P2_PREFLOP_RANK_STATS_SRC_WEIGHTS", "1"
                    ),
                    "rank_stats_ev": os.environ.get("P2_PREFLOP_RANK_STATS_EV", "1"),
                    "rank_stats_parent_ev": os.environ.get(
                        "P2_PREFLOP_RANK_STATS_PARENT_EV", "1"
                    ),
                    "record_shapes": args.record_shapes,
                },
                indent=2,
            ),
            flush=True,
        )

        with torch.no_grad():
            for i in range(args.warmup_iters):
                ev.cfr_iteration(i)
        _sync(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=args.record_shapes,
                with_stack=False,
            ) as prof:
                with record_function("profiled_cfr_loop"):
                    for i in range(args.active_iters):
                        ev.cfr_iteration(args.warmup_iters + i)
                    _sync(device)
        wall_s = time.perf_counter() - t0

    output = {
        "config": {
            "bucket": args.bucket,
            "rows": rows,
            "cfr_batch_size": args.cfr_batch_size,
            "active_iters": args.active_iters,
            "warmup_iters": args.warmup_iters,
            "cfr_iterations": args.cfr_iterations,
            "model": {
                "preflop_model_type": run_cfg.model.preflop_model_type.value,
                "hidden_dim": run_cfg.model.hidden_dim,
                "range_hidden_dim": run_cfg.model.range_hidden_dim,
                "ffn_dim": run_cfg.model.ffn_dim,
                "num_hidden_layers": run_cfg.model.num_hidden_layers,
                "num_value_layers": run_cfg.model.num_value_layers,
                "num_policy_layers": run_cfg.model.num_policy_layers,
                "preflop_transformer_heads": run_cfg.model.preflop_transformer_heads,
                "compile": run_cfg.model.compile,
            },
            "closing_leaf_checkpoint": None
            if args.no_closing_checkpoint
            else str(args.closing_checkpoint),
            "rank_stats_src_weights": os.environ.get(
                "P2_PREFLOP_RANK_STATS_SRC_WEIGHTS", "1"
            ),
            "rank_stats_ev": os.environ.get("P2_PREFLOP_RANK_STATS_EV", "1"),
            "rank_stats_parent_ev": os.environ.get(
                "P2_PREFLOP_RANK_STATS_PARENT_EV", "1"
            ),
            "record_shapes": args.record_shapes,
        },
        "subgame": {
            "total_nodes": int(ev.total_nodes),
            "root_nodes": int(ev.root_nodes),
            "nodes_per_root": float(ev.total_nodes) / max(1, int(ev.root_nodes)),
            "tree_depth": int(ev.tree_depth),
            "model_indices": int(ev.model_indices.numel()),
            "showdown_indices": int(ev.showdown_indices.numel()),
        },
        "wall_s": wall_s,
        "wall_ms_per_iter": 1e3 * wall_s / max(1, args.active_iters),
        "components": _component_rows(prof, args.active_iters, wall_s),
        "top_kernels": _kernel_rows(prof, args.active_iters)[:60],
    }
    if args.record_shapes:
        output["top_kernel_shapes"] = _kernel_shape_rows(prof, args.active_iters)[:200]

    print(
        f"wall={output['wall_ms_per_iter']:.3f} ms/iter "
        f"nodes/root={output['subgame']['nodes_per_root']:.1f}",
        flush=True,
    )
    for row in output["components"]:
        print(
            f"  {row['component']:<24}"
            f" cuda={row['cuda_ms_per_iter']:>8.3f} ms/iter"
            f" cpu={row['cpu_ms_per_iter']:>8.3f}",
            flush=True,
        )
    print("Top kernels:", flush=True)
    for row in output["top_kernels"][:20]:
        print(
            f"  {row['self_cuda_ms_per_iter']:>8.3f} ms/iter "
            f"{row['count_per_iter']:>7.1f}x {row['name']}",
            flush=True,
        )
    if args.record_shapes:
        print("Top aten::mm/addmm/bmm shapes:", flush=True)
        for row in output.get("top_kernel_shapes", []):
            if row["name"] not in {"aten::mm", "aten::addmm", "aten::bmm"}:
                continue
            print(
                f"  {row['self_cuda_ms_per_iter']:>8.3f} ms/iter "
                f"{row['count_per_iter']:>7.1f}x {row['name']} "
                f"{row['input_shapes']}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
