#!/usr/bin/env python3
"""Interleaved CUDA benchmark for batched showdown tier evaluators."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch

from p2.env.card_utils import board_allowed_hands, hand_combos_tensor
from p2.env.rules import rank_hands
from p2.showdown.approximate import (
    tier1_hero_removal_by_hand,
    tier2_first_order_opp_collision_by_hand,
    tier3_second_order_opp_collision_by_hand,
)
from p2.showdown.multiway_showdown_estimators import PreparedShowdown
from p2.showdown.results import PerHandEquityResult


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "showdown_tiers_benchmark.json"


def _iter_processes() -> list[tuple[int, str]]:
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return []
    own = {os.getpid(), os.getppid()}
    rows: list[tuple[int, str]] = []
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
                .strip()
            )
        except OSError:
            continue
        rows.append((pid, cmd))
    return rows


@contextmanager
def pause_train(enabled: bool, pattern: str):
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
            print(f"Paused matching processes: {paused}", flush=True)
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
            print(f"Resumed matching processes: {paused}", flush=True)


def _make_prepared(
    *,
    batch_size: int,
    players: int,
    device: torch.device,
    seed: int,
) -> PreparedShowdown:
    generator = torch.Generator(device=device).manual_seed(seed)
    boards = torch.topk(
        torch.rand(batch_size, 52, device=device, generator=generator),
        5,
        dim=1,
    ).indices
    weights = torch.empty(
        batch_size,
        players,
        1326,
        dtype=torch.float32,
        device=device,
    ).exponential_(generator=generator)
    weights.masked_fill_(~board_allowed_hands(boards)[:, None, :], 0.0)
    beliefs = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-30)
    ranks, _ = rank_hands(boards)
    combos = hand_combos_tensor(device=device).long()
    masks = ((1 << combos[:, 0]) | (1 << combos[:, 1])).long().contiguous()
    return PreparedShowdown(
        board=boards,
        beliefs=beliefs,
        hand_ranks=ranks,
        combos=combos,
        hand_masks=masks,
        setup_seconds=0.0,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(
    fn: Callable[[PreparedShowdown], PerHandEquityResult],
    prepared: PreparedShowdown,
) -> tuple[float, float, PerHandEquityResult]:
    device = prepared.beliefs.device
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        _sync(device)
        t0 = time.perf_counter()
        start_event.record()
        result = fn(prepared)
        end_event.record()
        _sync(device)
        wall_ms = 1.0e3 * (time.perf_counter() - t0)
        return float(start_event.elapsed_time(end_event)), wall_ms, result
    t0 = time.perf_counter()
    result = fn(prepared)
    wall_ms = 1.0e3 * (time.perf_counter() - t0)
    return wall_ms, wall_ms, result


def _summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[mid]
    else:
        median = 0.5 * (ordered[mid - 1] + ordered[mid])
    p90_index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    trimmed = ordered[1:-1] if len(ordered) >= 5 else ordered
    return {
        "mean": sum(values) / len(values),
        "median": median,
        "trimmed_mean": sum(trimmed) / len(trimmed),
        "p90": ordered[p90_index],
        "min": min(values),
        "max": max(values),
    }


def _check_small_correctness(device: torch.device, players: int, seed: int) -> dict[str, float]:
    prepared = _make_prepared(batch_size=2, players=players, device=device, seed=seed)
    results = {
        "tier1": tier1_hero_removal_by_hand(prepared),
        "tier2": tier2_first_order_opp_collision_by_hand(prepared),
        "tier3": tier3_second_order_opp_collision_by_hand(prepared),
    }
    out: dict[str, float] = {}
    for name, result in results.items():
        for board_idx in range(prepared.beliefs.shape[0]):
            sliced = PreparedShowdown(
                board=prepared.board[board_idx : board_idx + 1],
                beliefs=prepared.beliefs[board_idx : board_idx + 1],
                hand_ranks=prepared.hand_ranks[board_idx : board_idx + 1],
                combos=prepared.combos,
                hand_masks=prepared.hand_masks,
                setup_seconds=0.0,
            )
            sliced_result = {
                "tier1": tier1_hero_removal_by_hand,
                "tier2": tier2_first_order_opp_collision_by_hand,
                "tier3": tier3_second_order_opp_collision_by_hand,
            }[name](sliced)
            out[f"{name}_equity_by_hand_max_abs"] = max(
                out.get(f"{name}_equity_by_hand_max_abs", 0.0),
                float(
                    (
                        result.equity_by_hand[board_idx]
                        - sliced_result.equity_by_hand[0]
                    )
                    .abs()
                    .max()
                    .item(),
                ),
            )
            out[f"{name}_aggregate_max_abs"] = max(
                out.get(f"{name}_aggregate_max_abs", 0.0),
                float(
                    (
                        result.aggregate_equity[board_idx]
                        - sliced_result.aggregate_equity[0]
                    )
                    .abs()
                    .max()
                    .item(),
                ),
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--tiers", default="tier2,tier3")
    parser.add_argument("--include-tier1", action="store_true")
    parser.add_argument("--pause-train", action="store_true")
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument("--check-small", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")
    prepared = _make_prepared(
        batch_size=args.batch_size,
        players=args.players,
        device=device,
        seed=args.seed,
    )
    functions: dict[str, Callable[[PreparedShowdown], PerHandEquityResult]] = {
        "tier1": tier1_hero_removal_by_hand,
        "tier2": tier2_first_order_opp_collision_by_hand,
        "tier3": tier3_second_order_opp_collision_by_hand,
    }
    requested = [name.strip() for name in args.tiers.split(",") if name.strip()]
    if args.include_tier1 and "tier1" not in requested:
        requested = ["tier1", *requested]
    for name in requested:
        if name not in functions:
            raise ValueError(f"Unknown tier {name!r}; choose from {sorted(functions)}")

    print(
        f"Benchmarking {requested} B={args.batch_size} players={args.players} "
        f"device={device}",
        flush=True,
    )
    rows: list[dict[str, float | int | str]] = []
    rng = random.Random(args.seed)
    with pause_train(args.pause_train, args.pause_pattern):
        for name in requested:
            for _ in range(args.warmup):
                _measure(functions[name], prepared)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for iteration in range(args.iters):
            order = requested[:]
            rng.shuffle(order)
            for name in order:
                cuda_ms, wall_ms, result = _measure(functions[name], prepared)
                row: dict[str, float | int | str] = {
                    "iteration": iteration,
                    "tier": name,
                    "cuda_ms": cuda_ms,
                    "wall_ms": wall_ms,
                    "result_seconds": result.seconds,
                    "ms_per_board": cuda_ms / args.batch_size,
                }
                rows.append(row)
                print(
                    f"{name} iter={iteration} cuda={cuda_ms:.3f}ms "
                    f"wall={wall_ms:.3f}ms per_board={cuda_ms / args.batch_size:.6f}ms",
                    flush=True,
                )
                del result

    grouped: dict[str, list[float]] = defaultdict(list)
    grouped_wall: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["tier"])].append(float(row["cuda_ms"]))
        grouped_wall[str(row["tier"])].append(float(row["wall_ms"]))
    summary = {
        name: {
            "cuda_ms": _summarize(grouped[name]),
            "steady_cuda_ms": _summarize(grouped[name][1:]),
            "wall_ms": _summarize(grouped_wall[name]),
            "steady_wall_ms": _summarize(grouped_wall[name][1:]),
            "cuda_ms_per_board_mean": _summarize(grouped[name]).get("mean", 0.0)
            / args.batch_size,
        }
        for name in sorted(grouped)
    }
    checks = _check_small_correctness(device, args.players, args.seed + 999) if args.check_small else {}
    output = {
        "batch_size": args.batch_size,
        "players": args.players,
        "device": str(device),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "rows": rows,
        "summary": summary,
        "checks": checks,
        "cuda_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "checks": checks}, indent=2, sort_keys=True))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
