#!/usr/bin/env python3
"""Microbenchmark turn all-in payoff value writeback kernels.

This isolates the S_turn all-in-call leaf path from the rest of CFR:

* build one realistic random-turn fused sparse subgame;
* propagate model policy once to get representative leaf beliefs;
* time the card-stat pass, the legacy per-node turn payoff dot, and the
  grouped turn-board dot kernel across block-size candidates.
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
from typing import Any, Callable

import hydra
import torch
from omegaconf import DictConfig

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.env.card_utils import NUM_HANDS
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.allin_payoff import (
    I16_SCALE,
    write_allin_belief_card_stats_split_triton_,
    write_allin_grouped_table_values_card_denom_dot_values_triton_,
    write_allin_table_values_card_denom_dot_values_triton_,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "turn_allin_values_microbench.json"


def _csv_ints(value: str) -> list[int]:
    out = [int(part) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _iter_processes() -> list[tuple[int, str]]:
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return []
    own = {os.getpid(), os.getppid()}
    out: list[tuple[int, str]] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in own:
            continue
        try:
            cmd = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        out.append((pid, cmd))
    return out


@contextmanager
def pause_train_rebel(enabled: bool, pattern: str):
    paused: list[int] = []
    if enabled:
        for pid, cmd in _iter_processes():
            if pattern not in cmd:
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
            except ProcessLookupError:
                continue
        if paused:
            print(f"Paused train_rebel processes: {paused}", flush=True)
            time.sleep(0.5)
        else:
            print(f"No process matched pause pattern {pattern!r}.", flush=True)
    try:
        yield
    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                continue
        if paused:
            print(f"Resumed train_rebel processes: {paused}", flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_cuda_ms(
    device: torch.device,
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    if device.type != "cuda":
        samples = []
        for _ in range(warmup):
            fn()
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append(1e3 * (time.perf_counter() - t0))
        return _summarize_samples(samples, prefix="wall")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    cuda_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        cuda_samples.append(float(start.elapsed_time(end)))
        wall_samples.append(1e3 * (time.perf_counter() - t0))
    out = _summarize_samples(cuda_samples, prefix="cuda")
    out.update(_summarize_samples(wall_samples, prefix="wall"))
    return out


def _summarize_samples(samples: list[float], *, prefix: str) -> dict[str, float]:
    if not samples:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_min_ms": 0.0,
            f"{prefix}_max_ms": 0.0,
        }
    return {
        f"{prefix}_mean_ms": sum(samples) / len(samples),
        f"{prefix}_min_ms": min(samples),
        f"{prefix}_max_ms": max(samples),
    }


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_steps = 1
    cfg.num_envs = args.num_envs
    cfg.data.mode = "live"
    cfg.data.live_root_source = "random_turn"
    cfg.data.include_pre_chance_value_batches = False
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    cfg.search.iterations = 1
    cfg.search.iterations_final = None
    cfg.search.warm_start_iterations = 0
    if args.depth is not None:
        cfg.search.depth = args.depth
    cfg.model.compile = "off"


def _load_cfg(args: argparse.Namespace) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_cfr",
            overrides=args.hydra_override,
        )
    cfg = load_rebel_config(dc)
    _apply_overrides(cfg, args)
    return cfg


def _make_trainer(cfg: Config) -> RebelCFRTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    try:
        torch._dynamo.config.recompile_limit = 32
    except Exception:
        pass
    torch.manual_seed(cfg.seed)
    return RebelCFRTrainer(cfg=cfg, device=device, pregeneration_only=True)


def _prepare_evaluator(trainer: RebelCFRTrainer, args: argparse.Namespace):
    if trainer.data_generator is None:
        raise RuntimeError("benchmark requires live data_generator")
    generator = trainer.data_generator
    evaluator = trainer.cfr_evaluator
    pbs = generator._sample_roots(args.num_envs)
    root_indices = torch.arange(pbs.env.N, device=trainer.device)
    evaluator.initialize_subgame(pbs.env, root_indices, pbs.beliefs)
    evaluator.initialize_policy_and_beliefs()
    _sync(trainer.device)
    return evaluator


def _shape_summary(ev) -> dict[str, Any]:
    turn_nodes = ev.allin_call_indices_by_street[2]
    table_ids = ev.allin_turn_table_ids
    group_positions = getattr(ev, "allin_turn_group_positions", None)
    if table_ids.numel() > 0:
        counts = torch.bincount(table_ids, minlength=ev.allin_turn_tables_i16.shape[0])
        counts_cpu = counts.detach().cpu()
        count_mean = float(counts_cpu.float().mean().item())
        count_max = int(counts_cpu.max().item())
        count_min = int(counts_cpu.min().item())
    else:
        count_mean = 0.0
        count_max = 0
        count_min = 0
    return {
        "total_nodes": int(ev.total_nodes),
        "root_nodes": int(ev.root_nodes),
        "tree_depth": int(ev.tree_depth),
        "allin_nodes_total": int(ev.allin_call_indices.numel()),
        "turn_allin_nodes": int(turn_nodes.numel()),
        "turn_unique_boards": int(ev.allin_turn_tables_i16.shape[0]),
        "turn_nodes_per_board_min": count_min,
        "turn_nodes_per_board_mean": count_mean,
        "turn_nodes_per_board_max": count_max,
        "group_positions_shape": (
            list(group_positions.shape) if group_positions is not None else None
        ),
        "model_indices": int(ev.model_indices.numel()),
        "showdown_indices": int(ev.showdown_indices.numel()),
    }


def _run_correctness(ev, args: argparse.Namespace) -> dict[str, float]:
    device = ev.device
    beliefs = ev.beliefs
    node_idx = ev.allin_call_indices_by_street[2]
    turn_tables = ev.allin_turn_tables_i16
    turn_ids = ev.allin_turn_table_ids
    turn_stats = ev.allin_turn_stats_buffer
    group_positions = ev.allin_turn_group_positions
    empty_nodes = torch.empty(0, dtype=torch.long, device=device)
    empty_stats = torch.empty(0, 2, 53, dtype=torch.float32, device=device)

    write_allin_belief_card_stats_split_triton_(
        beliefs=beliefs,
        node_indices0=empty_nodes,
        stats_buffer0=empty_stats,
        node_indices1=node_idx,
        stats_buffer1=turn_stats,
    )
    old_values = ev.latest_values.clone()
    grouped_values = ev.latest_values.clone()
    write_allin_table_values_card_denom_dot_values_triton_(
        table=turn_tables,
        beliefs=beliefs,
        node_indices=node_idx,
        latest_values=old_values,
        stacks=ev.env.stacks,
        pot=ev.env.pot,
        starting_stacks=ev.env.starting_stacks,
        env_scale=ev.env.scale,
        table_scale=I16_SCALE,
        canon_ids=turn_ids,
        stats_buffer=turn_stats,
        block_h=args.correctness_block_h,
        block_k=args.correctness_block_k,
        block_p=args.correctness_block_p,
    )
    write_allin_grouped_table_values_card_denom_dot_values_triton_(
        table=turn_tables,
        beliefs=beliefs,
        node_indices=node_idx,
        group_positions=group_positions,
        latest_values=grouped_values,
        stacks=ev.env.stacks,
        pot=ev.env.pot,
        starting_stacks=ev.env.starting_stacks,
        env_scale=ev.env.scale,
        table_scale=I16_SCALE,
        stats_buffer=turn_stats,
        block_h=args.correctness_block_h,
        block_k=args.correctness_block_k,
        block_nodes=args.correctness_block_nodes,
    )
    _sync(device)
    diff = (old_values[node_idx] - grouped_values[node_idx]).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rms": float(torch.sqrt((diff * diff).mean()).item()),
    }


def _benchmark(ev, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = ev.device
    beliefs = ev.beliefs
    node_idx = ev.allin_call_indices_by_street[2]
    turn_tables = ev.allin_turn_tables_i16
    turn_ids = ev.allin_turn_table_ids
    turn_stats = ev.allin_turn_stats_buffer
    group_positions = ev.allin_turn_group_positions
    latest = ev.latest_values.clone()
    empty_nodes = torch.empty(0, dtype=torch.long, device=device)
    empty_stats = torch.empty(0, 2, 53, dtype=torch.float32, device=device)

    def stats_pass() -> None:
        write_allin_belief_card_stats_split_triton_(
            beliefs=beliefs,
            node_indices0=empty_nodes,
            stats_buffer0=empty_stats,
            node_indices1=node_idx,
            stats_buffer1=turn_stats,
        )

    stats_timing = _time_cuda_ms(
        device,
        stats_pass,
        warmup=args.warmup,
        iters=args.iters,
    )
    stats_row: dict[str, Any] = {"kind": "stats_split", **stats_timing}
    rows: list[dict[str, Any]] = [stats_row]

    # Ensure dot kernels see initialized stats even when stats timing is skipped.
    stats_pass()

    for block_h in args.block_hs:
        for block_k in args.block_ks:
            for block_p in args.legacy_block_ps:
                row: dict[str, Any] = {
                    "kind": "legacy_dot",
                    "block_h": block_h,
                    "block_k": block_k,
                    "block_p": block_p,
                }

                def run_legacy() -> None:
                    write_allin_table_values_card_denom_dot_values_triton_(
                        table=turn_tables,
                        beliefs=beliefs,
                        node_indices=node_idx,
                        latest_values=latest,
                        stacks=ev.env.stacks,
                        pot=ev.env.pot,
                        starting_stacks=ev.env.starting_stacks,
                        env_scale=ev.env.scale,
                        table_scale=I16_SCALE,
                        canon_ids=turn_ids,
                        stats_buffer=turn_stats,
                        block_h=block_h,
                        block_k=block_k,
                        block_p=block_p,
                    )

                try:
                    row.update(
                        _time_cuda_ms(
                            device,
                            run_legacy,
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                    )
                except Exception as exc:
                    row["error"] = repr(exc)
                rows.append(row)
                print(_format_row(row), flush=True)

    for block_h in args.block_hs:
        for block_k in args.block_ks:
            for block_nodes in args.group_block_nodes:
                for num_warps in args.num_warps:
                    row = {
                        "kind": "grouped_dot",
                        "block_h": block_h,
                        "block_k": block_k,
                        "block_nodes": block_nodes,
                        "num_warps": num_warps,
                    }

                    def run_grouped() -> None:
                        write_allin_grouped_table_values_card_denom_dot_values_triton_(
                            table=turn_tables,
                            beliefs=beliefs,
                            node_indices=node_idx,
                            group_positions=group_positions,
                            latest_values=latest,
                            stacks=ev.env.stacks,
                            pot=ev.env.pot,
                            starting_stacks=ev.env.starting_stacks,
                            env_scale=ev.env.scale,
                            table_scale=I16_SCALE,
                            stats_buffer=turn_stats,
                            block_h=block_h,
                            block_k=block_k,
                            block_nodes=block_nodes,
                            num_warps=num_warps,
                        )

                    try:
                        row.update(
                            _time_cuda_ms(
                                device,
                                run_grouped,
                                warmup=args.warmup,
                                iters=args.iters,
                            )
                        )
                    except Exception as exc:
                        row["error"] = repr(exc)
                    rows.append(row)
                    print(_format_row(row), flush=True)

    best = {
        "stats_split": stats_row,
        "legacy_dot": _best_row(rows, "legacy_dot"),
        "grouped_dot": _best_row(rows, "grouped_dot"),
    }
    stats_ms = float(stats_row.get("cuda_mean_ms", stats_row.get("wall_mean_ms", 0.0)))
    for key in ("legacy_dot", "grouped_dot"):
        row = best[key]
        if row is not None:
            dot_ms = float(row.get("cuda_mean_ms", row.get("wall_mean_ms", 0.0)))
            row["with_stats_cuda_mean_ms"] = stats_ms + dot_ms
    return rows, best


def _best_row(rows: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("kind") == kind
        and "error" not in row
        and ("cuda_mean_ms" in row or "wall_mean_ms" in row)
    ]
    if not candidates:
        return None
    key = "cuda_mean_ms" if "cuda_mean_ms" in candidates[0] else "wall_mean_ms"
    return min(candidates, key=lambda row: float(row[key]))


def _format_row(row: dict[str, Any]) -> str:
    if "error" in row:
        return f"{row['kind']}: error {row['error']}"
    parts = [str(row["kind"])]
    for key in ("block_h", "block_k", "block_p", "block_nodes", "num_warps"):
        if key in row:
            parts.append(f"{key}={row[key]}")
    if "cuda_mean_ms" in row:
        parts.append(
            f"cuda_mean={row['cuda_mean_ms']:.3f}ms min={row['cuda_min_ms']:.3f}ms"
        )
    elif "wall_mean_ms" in row:
        parts.append(
            f"wall_mean={row['wall_mean_ms']:.3f}ms min={row['wall_min_ms']:.3f}ms"
        )
    return " ".join(parts)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--block-hs", type=_csv_ints, default=[32, 64, 128])
    parser.add_argument("--block-ks", type=_csv_ints, default=[32, 64, 128])
    parser.add_argument("--legacy-block-ps", type=_csv_ints, default=[8, 16])
    parser.add_argument("--group-block-nodes", type=_csv_ints, default=[16, 32])
    parser.add_argument("--num-warps", type=_csv_ints, default=[4, 8])
    parser.add_argument("--correctness-block-h", type=int, default=64)
    parser.add_argument("--correctness-block-k", type=int, default=128)
    parser.add_argument("--correctness-block-p", type=int, default=8)
    parser.add_argument("--correctness-block-nodes", type=int, default=32)
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument("hydra_override", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    cfg = _load_cfg(args)
    with pause_train_rebel(not args.no_pause, args.pause_pattern):
        trainer = _make_trainer(cfg)
        device = trainer.device
        if device.type != "cuda":
            raise RuntimeError("turn all-in Triton benchmark requires CUDA")
        ev = _prepare_evaluator(trainer, args)
        if ev.allin_call_indices_by_street[2].numel() == 0:
            raise RuntimeError("constructed subgame has no turn all-in leaves")
        summary = _shape_summary(ev)
        print(json.dumps(summary, indent=2), flush=True)
        correctness = _run_correctness(ev, args)
        print(f"correctness: {json.dumps(correctness)}", flush=True)
        rows, best = _benchmark(ev, args)

    payload = {
        "argv": argv,
        "config": {
            "num_envs": cfg.num_envs,
            "depth": cfg.search.depth,
            "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
            "allin_by_depth": cfg.search.allin_by_depth,
        },
        "shape_summary": summary,
        "correctness": correctness,
        "rows": rows,
        "best": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print("Best:", flush=True)
    for key, row in best.items():
        if row is None:
            print(f"  {key}: none", flush=True)
        else:
            print(f"  {key}: {_format_row(row)}", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
