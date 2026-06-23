#!/usr/bin/env python3
"""Microbenchmark compact preflop fused CFR hot-loop candidates.

The benchmark pauses live training processes with SIGSTOP before timing and
resumes them with SIGCONT afterward so concurrent GPU work does not distort
measurements.

Default sizes are anchored to the ongoing `actions_4_7` preflop training
bucket: 6 players, compact-169 hands, CFR batch 512, and median reported
solve-batch node count around 126k.

Examples:
    uv run python scripts/bench_preflop_fused_hotloop.py
    uv run python scripts/bench_preflop_fused_hotloop.py --iters 100
    uv run python scripts/bench_preflop_fused_hotloop.py --rows 8192 --top 4096
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
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2.env.card_utils import PREFLOP_HANDS  # noqa: E402
from p2.search.fused_cfr_triton import (  # noqa: E402
    fused_preflop169_parent_sum_opp_,
    fused_preflop169_parent_sum_opp_rank_stats_,
    fused_preflop169_parent_sum_opp_stats_,
    fused_preflop169_project_rows_,
    fused_preflop169_src_weights_from_unblocked_multiway_,
    fused_preflop169_src_weights_multiway_,
    fused_preflop169_src_weights_rank_stats_multiway_,
    fused_preflop169_src_weights_stats_multiway_,
    preflop169_unblocked_rank_mass_triton_out_,
    preflop169_unblocked_mass_triton_out_,
    preflop169_unblocked_rank_stats_out_,
    preflop169_unblocked_stats_out_,
    triton_is_available,
)

REALISTIC_SHAPE_BASIS = (
    "actions_4_7 live run: players=6, compact169, cfr_batch=512, "
    "median logged nodes=126606, logged node range=116514..135955"
)


def _find_training_processes(pattern: str) -> list[int]:
    self_pid = os.getpid()
    parent_pid = os.getppid()
    out: list[int] = []
    proc = Path("/proc")
    if not proc.exists():
        return out
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (self_pid, parent_pid):
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if pattern.encode() in cmdline:
            out.append(pid)
    return out


@contextmanager
def _paused_training(pattern: str, enabled: bool):
    pids = _find_training_processes(pattern) if enabled else []
    stopped: list[int] = []
    for pid in pids:
        try:
            os.kill(pid, signal.SIGSTOP)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    if stopped:
        time.sleep(0.5)
    try:
        yield stopped
    finally:
        for pid in stopped:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                continue


def _cuda_time(fn: Callable[[], None], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _make_projection(device: torch.device) -> torch.Tensor:
    from p2.env.card_utils import (
        preflop_class_compatibility_counts_tensor,
        preflop_class_multiplicity_tensor,
    )

    multiplicity = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    compatibility = preflop_class_compatibility_counts_tensor(device=device).to(
        torch.float32
    )
    return (compatibility.T * multiplicity.reciprocal()[:, None]).contiguous()


def _benchmark_projection(
    *,
    rows: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    source = torch.rand(rows, PREFLOP_HANDS, device=device)
    projection = _make_projection(device)
    out_mm = torch.empty_like(source)
    out_fused = torch.empty_like(source)
    out_stats = torch.empty_like(source)
    out_rank_stats = torch.empty_like(source)
    row_sum = torch.empty(rows, device=device)
    stats = torch.empty(rows, 53, device=device)
    rank_stats = torch.empty(rows, 14, device=device)

    def legacy() -> None:
        torch.mm(source, projection, out=out_mm)

    def fused() -> None:
        fused_preflop169_project_rows_(
            source,
            projection,
            out_fused,
            row_sum=row_sum,
        )

    def compact_stats() -> None:
        preflop169_unblocked_mass_triton_out_(
            source,
            out_stats,
            stats_out=stats,
            row_sum=row_sum,
        )

    def compact_rank_stats() -> None:
        preflop169_unblocked_rank_mass_triton_out_(
            source,
            out_rank_stats,
            stats_out=rank_stats,
            row_sum=row_sum,
        )

    legacy_ms = _cuda_time(legacy, warmup=warmup, iters=iters)
    fused_ms = _cuda_time(fused, warmup=warmup, iters=iters)
    stats_ms = _cuda_time(compact_stats, warmup=warmup, iters=iters)
    rank_stats_ms = _cuda_time(compact_rank_stats, warmup=warmup, iters=iters)
    legacy()
    fused()
    compact_stats()
    compact_rank_stats()
    torch.cuda.synchronize()
    max_diff = float((out_mm - out_fused).abs().max().item())
    stats_diff = float((out_mm - out_stats).abs().max().item())
    rank_stats_diff = float((out_mm - out_rank_stats).abs().max().item())
    sum_diff = float((source.sum(dim=-1) - row_sum).abs().max().item())
    return {
        "legacy_mm_ms": legacy_ms,
        "fused_project_ms": fused_ms,
        "compact_stats_project_ms": stats_ms,
        "compact_rank_stats_project_ms": rank_stats_ms,
        "speedup": legacy_ms / fused_ms if fused_ms > 0 else float("inf"),
        "compact_stats_speedup": legacy_ms / stats_ms if stats_ms > 0 else float("inf"),
        "compact_rank_stats_speedup": (
            legacy_ms / rank_stats_ms if rank_stats_ms > 0 else float("inf")
        ),
        "max_abs_diff": max_diff,
        "compact_stats_max_abs_diff": stats_diff,
        "compact_rank_stats_max_abs_diff": rank_stats_diff,
        "row_sum_max_abs_diff": sum_diff,
    }


def _benchmark_src_weights(
    *,
    top: int,
    players: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    class_mass = torch.rand(top, players, PREFLOP_HANDS, device=device)
    class_mass /= class_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = _make_projection(device)
    to_act = torch.randint(0, players, (top,), device=device)
    has_folded = torch.rand(top, players, device=device) < 0.15
    rows = torch.arange(top, device=device)
    has_folded[rows, to_act] = False
    allowed = torch.ones(top, PREFLOP_HANDS, dtype=torch.bool, device=device)
    out_legacy = torch.empty(top, PREFLOP_HANDS, device=device)
    out_optimized = torch.empty_like(out_legacy)
    out_hybrid = torch.empty_like(out_legacy)
    out_fused = torch.empty_like(out_legacy)
    out_stats = torch.empty_like(out_legacy)
    out_rank_stats = torch.empty_like(out_legacy)
    allowed_float = allowed.to(dtype=torch.float32).contiguous()
    stats = torch.empty(top * players, 53, device=device)
    rank_stats = torch.empty(top * players, 14, device=device)

    player_ids = torch.arange(players, device=device)

    def legacy() -> None:
        unblocked = class_mass @ projection
        other_live = player_ids[None, :, None] != to_act[:, None, None]
        other_live &= ~has_folded[:, :, None]
        weights = torch.where(
            other_live,
            unblocked.clamp_min(1e-12),
            torch.ones_like(unblocked),
        ).prod(dim=1)
        torch.mul(weights, allowed.to(dtype=weights.dtype), out=out_legacy)

    def optimized_fallback() -> None:
        unblocked = class_mass @ projection
        unblocked.clamp_min_(1e-12)
        other_live = player_ids[None, :, None] != to_act[:, None, None]
        other_live &= ~has_folded[:, :, None]
        weights = torch.where(other_live, unblocked, 1.0).prod(dim=1)
        torch.mul(weights, allowed_float, out=out_optimized)

    def hybrid() -> None:
        unblocked = (class_mass @ projection).contiguous()
        fused_preflop169_src_weights_from_unblocked_multiway_(
            unblocked=unblocked,
            to_act=to_act,
            allowed_weight=allowed_float,
            out=out_hybrid,
            has_folded=has_folded,
        )

    def fused() -> None:
        fused_preflop169_src_weights_multiway_(
            class_mass=class_mass,
            projection=projection,
            to_act=to_act,
            allowed_mask=allowed,
            out=out_fused,
            has_folded=has_folded,
        )

    def compact_stats() -> None:
        fused_preflop169_src_weights_stats_multiway_(
            class_mass=class_mass,
            to_act=to_act,
            allowed_weight=allowed_float,
            out=out_stats,
            has_folded=has_folded,
            stats_out=stats,
        )

    def compact_rank_stats() -> None:
        fused_preflop169_src_weights_rank_stats_multiway_(
            class_mass=class_mass,
            to_act=to_act,
            allowed_weight=allowed_float,
            out=out_rank_stats,
            has_folded=has_folded,
            stats_out=rank_stats,
        )

    legacy_ms = _cuda_time(legacy, warmup=warmup, iters=iters)
    optimized_ms = _cuda_time(optimized_fallback, warmup=warmup, iters=iters)
    hybrid_ms = _cuda_time(hybrid, warmup=warmup, iters=iters)
    fused_ms = _cuda_time(fused, warmup=warmup, iters=iters)
    stats_ms = _cuda_time(compact_stats, warmup=warmup, iters=iters)
    rank_stats_ms = _cuda_time(compact_rank_stats, warmup=warmup, iters=iters)
    legacy()
    optimized_fallback()
    hybrid()
    fused()
    compact_stats()
    compact_rank_stats()
    torch.cuda.synchronize()
    opt_diff = float((out_legacy - out_optimized).abs().max().item())
    hybrid_diff = float((out_legacy - out_hybrid).abs().max().item())
    fused_diff = float((out_legacy - out_fused).abs().max().item())
    stats_diff = float((out_legacy - out_stats).abs().max().item())
    rank_stats_diff = float((out_legacy - out_rank_stats).abs().max().item())
    return {
        "legacy_original_src_weights_ms": legacy_ms,
        "optimized_fallback_src_weights_ms": optimized_ms,
        "hybrid_src_weights_ms": hybrid_ms,
        "fused_src_weights_ms": fused_ms,
        "compact_stats_src_weights_ms": stats_ms,
        "compact_rank_stats_src_weights_ms": rank_stats_ms,
        "optimized_fallback_speedup": (
            legacy_ms / optimized_ms if optimized_ms > 0 else float("inf")
        ),
        "hybrid_speedup": legacy_ms / hybrid_ms if hybrid_ms > 0 else float("inf"),
        "fused_speedup": legacy_ms / fused_ms if fused_ms > 0 else float("inf"),
        "compact_stats_speedup": (
            legacy_ms / stats_ms if stats_ms > 0 else float("inf")
        ),
        "compact_rank_stats_speedup": (
            legacy_ms / rank_stats_ms if rank_stats_ms > 0 else float("inf")
        ),
        "optimized_fallback_max_abs_diff": opt_diff,
        "hybrid_max_abs_diff": hybrid_diff,
        "fused_max_abs_diff": fused_diff,
        "compact_stats_max_abs_diff": stats_diff,
        "compact_rank_stats_max_abs_diff": rank_stats_diff,
    }


def _benchmark_ev_backup(
    *,
    parents: int,
    fanout: int,
    players: int,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    child_base = parents
    num_children = parents * fanout
    total = child_base + num_children
    fanout_pow2 = 1
    while fanout_pow2 < fanout:
        fanout_pow2 *= 2
    values_legacy = torch.rand(total, players, PREFLOP_HANDS, device=device)
    values_stats = values_legacy.clone()
    values_rank = values_legacy.clone()
    values_rank_parent = values_legacy.clone()
    policy = torch.rand(total, PREFLOP_HANDS, device=device)
    policy /= policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    actor_beliefs = torch.rand(parents, PREFLOP_HANDS, device=device)
    actor_beliefs /= actor_beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    marginal_policy = torch.rand(num_children, PREFLOP_HANDS, device=device)
    marginal_policy /= marginal_policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    projection = _make_projection(device)
    denom_unblocked = torch.empty_like(actor_beliefs)
    numer_unblocked = torch.empty_like(marginal_policy)
    denom_rank_unblocked = torch.empty_like(actor_beliefs)
    numer_rank_unblocked = torch.empty_like(marginal_policy)
    marginal_action = torch.empty(num_children, device=device)
    marginal_rank_action = torch.empty(num_children, device=device)
    actor_stats = torch.empty(parents, 53, device=device)
    marginal_stats = torch.empty(num_children, 53, device=device)
    actor_rank_stats = torch.empty(parents, 14, device=device)
    marginal_rank_stats = torch.empty(num_children, 14, device=device)
    child_offsets = child_base + torch.arange(parents, device=device) * fanout
    child_count = torch.full((parents,), fanout, dtype=torch.long, device=device)
    prev_actor = torch.randint(0, players, (total,), device=device)
    has_folded = torch.rand(total, players, device=device) < 0.15
    has_folded[torch.arange(total, device=device), prev_actor] = False

    def legacy() -> None:
        torch.mm(actor_beliefs, projection, out=denom_unblocked)
        torch.mm(marginal_policy, projection, out=numer_unblocked)
        torch.sum(marginal_policy, dim=-1, out=marginal_action)
        fused_preflop169_parent_sum_opp_(
            values=values_legacy,
            prev_actor=prev_actor,
            policy=policy,
            marginal_action_policy=marginal_action,
            numer_unblocked=numer_unblocked,
            denom_unblocked=denom_unblocked,
            child_offsets=child_offsets,
            child_count=child_count,
            parent_base=0,
            child_base=child_base,
            max_children=fanout,
            max_children_pow2=fanout_pow2,
            has_folded=has_folded,
        )

    def compact_stats() -> None:
        preflop169_unblocked_stats_out_(actor_beliefs, actor_stats)
        preflop169_unblocked_stats_out_(marginal_policy, marginal_stats)
        fused_preflop169_parent_sum_opp_stats_(
            values=values_stats,
            prev_actor=prev_actor,
            policy=policy,
            actor_beliefs=actor_beliefs,
            marginal_policy=marginal_policy,
            actor_stats=actor_stats,
            marginal_stats=marginal_stats,
            child_offsets=child_offsets,
            child_count=child_count,
            parent_base=0,
            child_base=child_base,
            max_children=fanout,
            max_children_pow2=fanout_pow2,
            has_folded=has_folded,
        )

    def rank_project() -> None:
        preflop169_unblocked_rank_mass_triton_out_(
            actor_beliefs,
            denom_rank_unblocked,
            stats_out=actor_rank_stats,
        )
        preflop169_unblocked_rank_mass_triton_out_(
            marginal_policy,
            numer_rank_unblocked,
            stats_out=marginal_rank_stats,
            row_sum=marginal_rank_action,
        )
        fused_preflop169_parent_sum_opp_(
            values=values_rank,
            prev_actor=prev_actor,
            policy=policy,
            marginal_action_policy=marginal_rank_action,
            numer_unblocked=numer_rank_unblocked,
            denom_unblocked=denom_rank_unblocked,
            child_offsets=child_offsets,
            child_count=child_count,
            parent_base=0,
            child_base=child_base,
            max_children=fanout,
            max_children_pow2=fanout_pow2,
            has_folded=has_folded,
        )

    def rank_parent() -> None:
        preflop169_unblocked_rank_stats_out_(actor_beliefs, actor_rank_stats)
        preflop169_unblocked_rank_stats_out_(marginal_policy, marginal_rank_stats)
        fused_preflop169_parent_sum_opp_rank_stats_(
            values=values_rank_parent,
            prev_actor=prev_actor,
            policy=policy,
            actor_beliefs=actor_beliefs,
            marginal_policy=marginal_policy,
            actor_stats=actor_rank_stats,
            marginal_stats=marginal_rank_stats,
            child_offsets=child_offsets,
            child_count=child_count,
            parent_base=0,
            child_base=child_base,
            max_children=fanout,
            max_children_pow2=fanout_pow2,
            has_folded=has_folded,
        )

    legacy_ms = _cuda_time(legacy, warmup=warmup, iters=iters)
    stats_ms = _cuda_time(compact_stats, warmup=warmup, iters=iters)
    rank_project_ms = _cuda_time(rank_project, warmup=warmup, iters=iters)
    rank_parent_ms = _cuda_time(rank_parent, warmup=warmup, iters=iters)
    legacy()
    compact_stats()
    rank_project()
    rank_parent()
    torch.cuda.synchronize()
    stats_diff = float(
        (values_legacy[:parents] - values_stats[:parents]).abs().max().item()
    )
    rank_diff = float(
        (values_legacy[:parents] - values_rank[:parents]).abs().max().item()
    )
    rank_parent_diff = float(
        (values_legacy[:parents] - values_rank_parent[:parents]).abs().max().item()
    )
    return {
        "parents": float(parents),
        "fanout": float(fanout),
        "legacy_ev_backup_ms": legacy_ms,
        "compact_stats_ev_backup_ms": stats_ms,
        "rank_project_ev_backup_ms": rank_project_ms,
        "rank_parent_ev_backup_ms": rank_parent_ms,
        "compact_stats_speedup": legacy_ms / stats_ms if stats_ms > 0 else float("inf"),
        "rank_project_speedup": (
            legacy_ms / rank_project_ms if rank_project_ms > 0 else float("inf")
        ),
        "rank_parent_speedup": (
            legacy_ms / rank_parent_ms if rank_parent_ms > 0 else float("inf")
        ),
        "compact_stats_max_abs_diff": stats_diff,
        "rank_project_max_abs_diff": rank_diff,
        "rank_parent_max_abs_diff": rank_parent_diff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--top", type=int, default=65536)
    parser.add_argument("--ev-parents", type=int, default=8192)
    parser.add_argument("--ev-fanout", type=int, default=4)
    parser.add_argument("--players", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if not triton_is_available():
        raise RuntimeError("Triton is required for this benchmark.")

    device = torch.device("cuda")
    with _paused_training(args.pause_pattern, enabled=not args.no_pause) as stopped:
        result = {
            "config": {
                "rows": args.rows,
                "top": args.top,
                "ev_parents": args.ev_parents,
                "ev_fanout": args.ev_fanout,
                "players": args.players,
                "warmup": args.warmup,
                "iters": args.iters,
                "pause_pattern": args.pause_pattern,
                "paused_pids": stopped,
                "shape_basis": REALISTIC_SHAPE_BASIS,
            },
            "projection": _benchmark_projection(
                rows=args.rows,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            ),
            "src_weights": _benchmark_src_weights(
                top=args.top,
                players=args.players,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            ),
            "ev_backup": _benchmark_ev_backup(
                parents=args.ev_parents,
                fanout=args.ev_fanout,
                players=args.players,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            ),
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
