#!/usr/bin/env python3
"""Microbenchmark fused sparse CFR subgame initialization from saved spots."""

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

from p2.cli.sample_spots import build_pbs_from_spots, load_spots
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "cfr_init_spots_micro.json"
STREET_TO_ID = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


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
    device: torch.device, fn: Callable[[], Any]
) -> tuple[float, float, Any]:
    if device.type != "cuda":
        t0 = time.perf_counter()
        result = fn()
        return 0.0, 1e3 * (time.perf_counter() - t0), result
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), 1e3 * (time.perf_counter() - t0), result


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = sum(vals) / len(vals)
        out[f"{key}_min"] = min(vals)
        out[f"{key}_max"] = max(vals)
    return out


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_envs = args.per_street * 4
    cfg.model.hidden_dim = 256
    cfg.model.range_hidden_dim = 128
    cfg.model.ffn_dim = 512
    cfg.model.num_hidden_layers = 3
    cfg.model.num_value_layers = 1
    cfg.model.num_policy_layers = 1
    cfg.model.compile = "off"
    cfg.search.depth = args.depth
    cfg.search.iterations = args.iterations
    cfg.search.iterations_final = None
    cfg.search.sparse = True
    cfg.search.sparse_fused = True


def _load_cfg(args: argparse.Namespace) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_cfr",
            overrides=args.hydra_override,
        )
    cfg = Config.from_dict_config(dc)
    _apply_overrides(cfg, args)
    return cfg


def _select_even_spots(
    payload: dict[str, Any], per_street: int, randomize: bool, seed: int
) -> torch.Tensor:
    chunks = []
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    streets = payload["spots"]["street"]
    for street_id in STREET_TO_ID.values():
        candidates = torch.where(streets == street_id)[0].cpu()
        if candidates.numel() < per_street:
            raise ValueError(
                f"Need {per_street} spots for street {street_id}, "
                f"but only found {candidates.numel()}."
            )
        if randomize:
            candidates = candidates[torch.randperm(candidates.numel(), generator=g)]
        chunks.append(candidates[:per_street])
    return torch.cat(chunks, dim=0)


def _run_initialize_once(trainer: RebelCFRTrainer, pbs) -> dict[str, float]:
    ev = trainer.cfr_evaluator
    src_indices = torch.arange(pbs.env.N, device=trainer.device)

    def run() -> None:
        ev.initialize_subgame(pbs.env, src_indices, pbs.beliefs)

    cuda_ms, wall_ms, _ = _time_cuda_ms(trainer.device, run)
    return {
        "cuda_ms": cuda_ms,
        "wall_ms": wall_ms,
        "total_nodes": float(ev.total_nodes),
        "model_indices": float(ev.model_indices.numel()),
        "showdown_indices": float(ev.showdown_indices.numel()),
    }


def _run_uniform_policy_micro(trainer: RebelCFRTrainer) -> dict[str, float]:
    ev = trainer.cfr_evaluator

    def old_way():
        return ev._fan_out(ev.child_count)

    def new_way():
        return ev.child_count[ev.parent_index[ev.root_nodes :]]

    old_cuda, old_wall, old = _time_cuda_ms(trainer.device, old_way)
    new_cuda, new_wall, new = _time_cuda_ms(trainer.device, new_way)
    torch.testing.assert_close(old, new)
    return {
        "old_cuda_ms": old_cuda,
        "old_wall_ms": old_wall,
        "new_cuda_ms": new_cuda,
        "new_wall_ms": new_wall,
        "speedup_cuda": old_cuda / new_cuda if new_cuda > 0 else 0.0,
        "speedup_wall": old_wall / new_wall if new_wall > 0 else 0.0,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots", default="outputs/spots.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-street", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random-spots", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument("hydra_override", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    payload = load_spots(args.spots, map_location="cpu")
    indices = _select_even_spots(
        payload,
        per_street=args.per_street,
        randomize=args.random_spots,
        seed=args.seed,
    )
    selected_streets = payload["spots"]["street"][indices]
    street_counts = {
        name: int((selected_streets == sid).sum().item())
        for name, sid in STREET_TO_ID.items()
    }

    cfg = _load_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    pbs = build_pbs_from_spots(payload, device=device, indices=indices)

    print(json.dumps({"street_counts": street_counts, "device": str(device)}, indent=2))
    init_rows = []
    uniform_rows = []
    with pause_train_rebel(not args.no_pause, args.pause_pattern):
        for i in range(args.repeats):
            init_row = _run_initialize_once(trainer, pbs)
            uniform_row = _run_uniform_policy_micro(trainer)
            init_rows.append(init_row)
            uniform_rows.append(uniform_row)
            print(
                f"[repeat {i}] init={init_row['cuda_ms']:.3f} ms cuda, "
                f"uniform old/new={uniform_row['old_cuda_ms']:.4f}/"
                f"{uniform_row['new_cuda_ms']:.4f} ms cuda",
                flush=True,
            )

    output = {
        "config": {
            "spots": str(args.spots),
            "num_envs": cfg.num_envs,
            "per_street": args.per_street,
            "street_counts": street_counts,
            "depth": cfg.search.depth,
            "iterations": cfg.search.iterations,
            "repeats": args.repeats,
        },
        "initialize_subgame": {
            "rows": init_rows,
            "summary": _summarize(init_rows),
        },
        "uniform_policy_denominator": {
            "rows": uniform_rows,
            "summary": _summarize(uniform_rows),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print("\nSummary:")
    print(json.dumps(output["initialize_subgame"]["summary"], indent=2))
    print(json.dumps(output["uniform_policy_denominator"]["summary"], indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
