#!/usr/bin/env python3
"""Microbenchmark fused subgame same-street child construction.

The benchmark builds one real fused sparse subgame with the legacy child writer,
then replays the exact per-depth writer inputs against both the legacy and
optimized Triton kernels. CUDA events time only the writer kernel launch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_cfr_main_path import (  # noqa: E402
    _load_config,
    _make_trainer,
    _sync,
    pause_train_rebel,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv  # noqa: E402
from p2.search.subgame_constructor_triton import (  # noqa: E402
    legal_counts_triton_,
    write_children_same_street_triton_legacy_,
    write_children_same_street_triton_optimized_,
)


DEFAULT_OUT = REPO_ROOT / "outputs" / "write_children_kernel_bench.json"


@contextmanager
def _temporary_env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _clone_env_prefix(src: HUNLTensorEnv, rows: int) -> HUNLTensorEnv:
    dst = HUNLTensorEnv.from_proto(src, num_envs=rows, init_state=False)
    select = slice(0, rows, 1)
    dst.copy_state_from(src, select, select, copy_deck=True)
    return dst


def _clone_outputs(parent_index, action_from_parent, rewards, allin_leaf, rows: int):
    return (
        parent_index[:rows].clone(),
        action_from_parent[:rows].clone(),
        rewards[:rows].clone(),
        allin_leaf[:rows].clone(),
    )


def _writer_kwargs(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_start": case["parent_start"],
        "parent_count": case["parent_count"],
        "dst_start": case["dst_start"],
        "has_preflop_table": case["has_preflop_table"],
        "allin_abstraction": case["allin_abstraction"],
    }


def _run_writer(
    fn: Callable[..., None],
    env: HUNLTensorEnv,
    case: dict[str, Any],
    parent_index: torch.Tensor,
    action_from_parent: torch.Tensor,
    rewards: torch.Tensor,
    allin_leaf: torch.Tensor,
) -> None:
    fn(
        env,
        case["legal_mask"],
        case["child_offsets"],
        parent_index,
        action_from_parent,
        rewards,
        allin_leaf,
        case["bet_bins"],
        **_writer_kwargs(case),
    )


def _time_writer(
    device: torch.device,
    fn: Callable[..., None],
    base_env: HUNLTensorEnv,
    base_parent_index: torch.Tensor,
    base_action_from_parent: torch.Tensor,
    base_rewards: torch.Tensor,
    base_allin_leaf: torch.Tensor,
    case: dict[str, Any],
    *,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    env = _clone_env_prefix(base_env, case["total_nodes"])
    parent_index, action_from_parent, rewards, allin_leaf = _clone_outputs(
        base_parent_index,
        base_action_from_parent,
        base_rewards,
        base_allin_leaf,
        case["total_nodes"],
    )

    for _ in range(warmup):
        _run_writer(fn, env, case, parent_index, action_from_parent, rewards, allin_leaf)
    _sync(device)

    wall_ms: list[float] = []
    cuda_ms: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()
        _run_writer(fn, env, case, parent_index, action_from_parent, rewards, allin_leaf)
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            cuda_ms.append(float(start.elapsed_time(end)))
        else:
            _sync(device)
        wall_ms.append(1e3 * (time.perf_counter() - t0))

    cuda_sorted = sorted(cuda_ms)
    wall_sorted = sorted(wall_ms)
    mid = len(wall_sorted) // 2
    return {
        "iters": iters,
        "wall_ms_mean": sum(wall_ms) / max(1, len(wall_ms)),
        "wall_ms_p50": wall_sorted[mid] if wall_sorted else 0.0,
        "cuda_ms_mean": sum(cuda_ms) / max(1, len(cuda_ms)) if cuda_ms else 0.0,
        "cuda_ms_p50": cuda_sorted[mid] if cuda_sorted else 0.0,
    }


def _compare_outputs(
    legacy_env: HUNLTensorEnv,
    opt_env: HUNLTensorEnv,
    legacy_outputs,
    opt_outputs,
    case: dict[str, Any],
) -> dict[str, Any]:
    start = case["dst_start"]
    end = case["dst_end"]
    mismatches: dict[str, Any] = {}

    one_d_fields = [
        "button",
        "street",
        "to_act",
        "last_to_act",
        "pot",
        "min_raise",
        "actions_this_round",
        "actions_last_round",
        "acted_since_reset",
        "done",
        "winner",
        "scale",
    ]
    two_d_fields = [
        "stacks",
        "committed",
        "chips_placed",
        "starting_stacks",
        "is_allin",
    ]

    for name in one_d_fields + two_d_fields:
        a = getattr(legacy_env, name)[start:end]
        b = getattr(opt_env, name)[start:end]
        if a.dtype.is_floating_point:
            ok = torch.allclose(a, b, atol=0.0, rtol=0.0)
            if not bool(ok):
                mismatches[name] = {
                    "max_abs": float((a - b).abs().max().detach().cpu().item())
                }
        else:
            diff = int((a != b).sum().detach().cpu().item())
            if diff:
                mismatches[name] = {"diff_count": diff}

    for name, a, b in (
        ("parent_index", legacy_outputs[0][start:end], opt_outputs[0][start:end]),
        (
            "action_from_parent",
            legacy_outputs[1][start:end],
            opt_outputs[1][start:end],
        ),
        ("rewards", legacy_outputs[2][start:end], opt_outputs[2][start:end]),
        ("allin_leaf", legacy_outputs[3][start:end], opt_outputs[3][start:end]),
    ):
        if a.dtype.is_floating_point:
            ok = torch.allclose(a, b, atol=0.0, rtol=0.0)
            if not bool(ok):
                mismatches[name] = {
                    "max_abs": float((a - b).abs().max().detach().cpu().item())
                }
        else:
            diff = int((a != b).sum().detach().cpu().item())
            if diff:
                mismatches[name] = {"diff_count": diff}

    return {
        "ok": not mismatches,
        "mismatches": mismatches,
        "excluded_fields": [
            "has_folded",
            "deck_pos",
            "deck",
            "board_indices",
            "last_board_indices",
            "hole_indices",
        ],
    }


def _correctness_check(
    device: torch.device,
    base_env: HUNLTensorEnv,
    base_parent_index: torch.Tensor,
    base_action_from_parent: torch.Tensor,
    base_rewards: torch.Tensor,
    base_allin_leaf: torch.Tensor,
    case: dict[str, Any],
) -> dict[str, Any]:
    legacy_env = _clone_env_prefix(base_env, case["total_nodes"])
    opt_env = _clone_env_prefix(base_env, case["total_nodes"])
    legacy_outputs = _clone_outputs(
        base_parent_index,
        base_action_from_parent,
        base_rewards,
        base_allin_leaf,
        case["total_nodes"],
    )
    opt_outputs = _clone_outputs(
        base_parent_index,
        base_action_from_parent,
        base_rewards,
        base_allin_leaf,
        case["total_nodes"],
    )
    _run_writer(
        write_children_same_street_triton_legacy_,
        legacy_env,
        case,
        *legacy_outputs,
    )
    _run_writer(
        write_children_same_street_triton_optimized_,
        opt_env,
        case,
        *opt_outputs,
    )
    _sync(device)
    return _compare_outputs(legacy_env, opt_env, legacy_outputs, opt_outputs, case)


def _prepare_cases(trainer, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ev = trainer.cfr_evaluator
    device = trainer.device
    if not hasattr(ev, "_construct_subgame"):
        raise RuntimeError("benchmark requires a fused sparse evaluator")
    pbs = trainer.data_generator.current_pbs
    if pbs is None:
        pbs = trainer.data_generator._new_pbs(ev.root_nodes)
        trainer.data_generator.current_pbs = pbs

    root_indices = torch.arange(ev.root_nodes, device=device)
    with _temporary_env("P2_WRITE_CHILDREN_KERNEL", "legacy"):
        ev._construct_subgame(pbs.env, root_indices)
    _sync(device)

    if ev._subgame_env is None:
        raise RuntimeError("subgame construction did not create an env")
    total_nodes = int(ev.total_nodes)
    base_env = _clone_env_prefix(ev._subgame_env, total_nodes)
    base_parent_index = ev._subgame_parent_index[:total_nodes].clone()
    base_action_from_parent = ev._subgame_action_from_parent[:total_nodes].clone()
    base_rewards = ev._subgame_rewards[:total_nodes].clone()
    base_allin_leaf = ev._subgame_allin_leaf[:total_nodes].clone()

    search_cfg = getattr(getattr(ev, "cfg", None), "search", None)
    has_preflop_table = (
        getattr(
            search_cfg,
            "preflop_allin_table_path",
            getattr(ev, "preflop_allin_table_path", None),
        )
        is not None
    )
    allin_abstraction = ev._allin_abstraction_enabled()

    cases: list[dict[str, Any]] = []
    for depth in range(ev.tree_depth):
        parent_start = int(ev.depth_offsets[depth])
        parent_end = int(ev.depth_offsets[depth + 1])
        dst_start = int(ev.depth_offsets[depth + 1])
        dst_end = int(ev.depth_offsets[depth + 2])
        parent_count = parent_end - parent_start
        child_count = dst_end - dst_start
        if parent_count <= 0 or child_count <= 0:
            continue

        legal_mask = torch.empty(
            total_nodes, ev.num_actions, dtype=torch.bool, device=device
        )
        child_counts = torch.empty(parent_count, dtype=torch.long, device=device)
        child_offsets = torch.empty(parent_count, dtype=torch.long, device=device)
        legal_counts_triton_(
            base_env,
            legal_mask,
            child_counts,
            base_allin_leaf,
            ev._subgame_bet_bins_t,
            parent_start=parent_start,
            parent_count=parent_count,
            stop_new_street=depth > 0,
            num_actions=ev.num_actions,
        )
        torch.cumsum(child_counts, dim=0, out=child_offsets)
        child_offsets -= child_counts
        recomputed_child_count = int(child_counts.sum().detach().cpu().item())
        if recomputed_child_count != child_count:
            raise RuntimeError(
                "recomputed child count did not match constructed depth: "
                f"depth={depth} recomputed={recomputed_child_count} actual={child_count}"
            )

        cases.append(
            {
                "depth": depth,
                "parent_start": parent_start,
                "parent_count": parent_count,
                "dst_start": dst_start,
                "dst_end": dst_end,
                "child_count": child_count,
                "total_nodes": total_nodes,
                "legal_mask": legal_mask,
                "child_offsets": child_offsets,
                "bet_bins": ev._subgame_bet_bins_t,
                "has_preflop_table": has_preflop_table,
                "allin_abstraction": allin_abstraction,
            }
        )

    base = {
        "env": base_env,
        "parent_index": base_parent_index,
        "action_from_parent": base_action_from_parent,
        "rewards": base_rewards,
        "allin_leaf": base_allin_leaf,
        "total_nodes": total_nodes,
        "root_nodes": int(ev.root_nodes),
        "tree_depth": int(ev.tree_depth),
    }
    return base, cases


def run_benchmark(trainer, args: argparse.Namespace) -> dict[str, Any]:
    device = trainer.device
    base, cases = _prepare_cases(trainer, args)

    rows: list[dict[str, Any]] = []
    all_ok = True
    total_legacy = 0.0
    total_optimized = 0.0
    for case in cases:
        correctness = _correctness_check(
            device,
            base["env"],
            base["parent_index"],
            base["action_from_parent"],
            base["rewards"],
            base["allin_leaf"],
            case,
        )
        all_ok = all_ok and bool(correctness["ok"])
        legacy = _time_writer(
            device,
            write_children_same_street_triton_legacy_,
            base["env"],
            base["parent_index"],
            base["action_from_parent"],
            base["rewards"],
            base["allin_leaf"],
            case,
            warmup=args.warmup,
            iters=args.iters,
        )
        optimized = _time_writer(
            device,
            write_children_same_street_triton_optimized_,
            base["env"],
            base["parent_index"],
            base["action_from_parent"],
            base["rewards"],
            base["allin_leaf"],
            case,
            warmup=args.warmup,
            iters=args.iters,
        )
        total_legacy += legacy["cuda_ms_mean"]
        total_optimized += optimized["cuda_ms_mean"]
        rows.append(
            {
                "depth": case["depth"],
                "parent_count": case["parent_count"],
                "child_count": case["child_count"],
                "legacy": legacy,
                "optimized": optimized,
                "speedup": (
                    legacy["cuda_ms_mean"] / optimized["cuda_ms_mean"]
                    if optimized["cuda_ms_mean"]
                    else 0.0
                ),
                "correctness": correctness,
            }
        )

    return {
        "total_nodes": base["total_nodes"],
        "root_nodes": base["root_nodes"],
        "tree_depth": base["tree_depth"],
        "all_correct": all_ok,
        "depths": rows,
        "total_cuda_ms_legacy": total_legacy,
        "total_cuda_ms_optimized": total_optimized,
        "total_speedup": total_legacy / total_optimized if total_optimized else 0.0,
        "notes": {
            "timed_region": "write_children_same_street_* only",
            "optimized_writer_skips": [
                "has_folded",
                "deck_pos",
                "deck",
                "board_indices",
                "last_board_indices",
                "hole_indices",
            ],
        },
    }


def _print_summary(result: dict[str, Any]) -> None:
    print("\nwrite_children_same_street_triton_ microbenchmark:")
    print(
        f"  nodes={result['total_nodes']} roots={result['root_nodes']} "
        f"tree_depth={result['tree_depth']} correctness={result['all_correct']}"
    )
    print(
        f"  total cuda: legacy={result['total_cuda_ms_legacy']:.3f} ms "
        f"optimized={result['total_cuda_ms_optimized']:.3f} ms "
        f"speedup={result['total_speedup']:.2f}x"
    )
    for row in result["depths"]:
        print(
            f"  depth {row['depth']}: parents={row['parent_count']} "
            f"children={row['child_count']} "
            f"legacy={row['legacy']['cuda_ms_mean']:.3f} ms "
            f"optimized={row['optimized']['cuda_ms_mean']:.3f} ms "
            f"speedup={row['speedup']:.2f}x "
            f"ok={row['correctness']['ok']}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--keep-hydra-model-config", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("hydra_override", nargs="*")
    args = parser.parse_args(argv)
    # Compatibility with scripts.bench_cfr_main_path._load_config.
    args.warmup_steps = 0
    args.active_steps = 1
    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    cfg = _load_config(args)
    trainer = _make_trainer(cfg)
    config_summary = {
        "num_envs": cfg.num_envs,
        "depth": cfg.search.depth,
        "sparse_fused": cfg.search.sparse_fused,
        "num_actions": trainer.cfr_evaluator.num_actions,
        "model": {
            "hidden_dim": cfg.model.hidden_dim,
            "num_hidden_layers": cfg.model.num_hidden_layers,
            "num_value_layers": cfg.model.num_value_layers,
            "num_policy_layers": cfg.model.num_policy_layers,
            "compile": cfg.model.compile,
        },
        "benchmark": {"warmup": args.warmup, "iters": args.iters},
    }
    print(json.dumps({"config": config_summary}, indent=2), flush=True)
    with pause_train_rebel(not args.no_pause, args.pause_pattern):
        result = run_benchmark(trainer, args)
    output = {"config": config_summary, "result": result}
    _print_summary(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
