#!/usr/bin/env python3
"""
Coordinate-descent CFR-param search driver.

For each `iterations` budget we want to compare, greedily locks in the
best value for each search-time parameter (single pass through the
parameter list, in fixed order). The trainer + evaluator (FusedSparse
by default — comes from the checkpoint config) is built once.

Inputs:
  --spots, --checkpoint, --batch-size, --device   (same as tune_cfr.py)
  --out                JSON results path
  --iterations         comma-sep list, e.g. "500,1000,2000"
  --n-spots            cap on spots used per trial (default 1024)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict
from typing import Any

import torch

from p2.cli.sample_spots import build_pbs_from_spots, load_spots
from p2.cli.tune_cfr import (
    EVAL_ATTR_MAP,
    STREETS,
    _aggregate_trial,
    _apply_eval_overrides,
    _device_from_str,
    _restore_eval_params,
    _snapshot_eval_params,
)
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer

def param_grid(iters: int) -> list[tuple[str, list[Any]]]:
    """Iter-dependent candidate set. Sweep order matters — alpha first since
    it had the largest single-param effect in earlier runs.

    warm_start_iterations is scaled around iters/5 per user guidance.
    dcfr_plus_delay is scaled relative to iters since the previous boundary
    winner was 320 at iter=500/1000.
    dcfr_gamma is dropped — it had zero impact in earlier sweeps.
    """
    ws = [iters // 20, iters // 10, iters // 5, iters // 3, iters // 2]
    # Keep unique, sorted, drop zeros.
    ws = sorted({max(1, v) for v in ws})

    delay = sorted({iters // 10, iters // 5, iters // 4, iters // 3,
                    iters // 2, (iters * 3) // 4, 80, 320})

    return [
        ("dcfr_alpha",            [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]),
        ("dcfr_beta",             [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0]),
        ("dcfr_plus_delay",       delay),
        ("warm_start_iterations", ws),
        ("warm_start_multiplier", [1, 3, 5, 10, 20, 30, 50]),
    ]


def run_one(
    trainer: RebelCFRTrainer,
    base_snap: dict[str, Any],
    overrides: dict[str, Any],
    batch_pbs: list[Any],
    streets_cpu: torch.Tensor,
    src_indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    _restore_eval_params(trainer.cfr_evaluator, base_snap)
    _apply_eval_overrides(trainer.cfr_evaluator, overrides)
    per_batch_stats: list[dict[str, Any]] = []
    per_batch_n: list[int] = []
    t0 = time.time()
    try:
        for bi, pbs in enumerate(batch_pbs):
            trainer.cfr_evaluator.initialize_subgame(pbs.env, src_indices, pbs.beliefs)
            trainer.cfr_evaluator.evaluate_cfr(training_mode=False)
            stats = {k: v for k, v in trainer.cfr_evaluator.stats.items()}
            bs_streets = streets_cpu[bi * batch_size : (bi + 1) * batch_size]
            stats["_per_street_counts"] = {
                STREETS[i]: int((bs_streets == i).sum().item())
                for i in range(5) if (bs_streets == i).any()
            }
            per_batch_stats.append(stats)
            per_batch_n.append(batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as e:
        return {"error": repr(e), "wall_time_s_total": time.time() - t0,
                "overrides": overrides}
    agg = _aggregate_trial(per_batch_stats, per_batch_n)
    agg["wall_time_s_total"] = time.time() - t0
    agg["overrides"] = overrides
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--iterations", default="500,1000,2000")
    ap.add_argument("--n-spots", type=int, default=1024)
    ap.add_argument(
        "--initial-overrides",
        default=None,
        help="JSON file mapping iter (str) -> {param: value} to seed the locked overrides.",
    )
    ap.add_argument(
        "--params",
        default=None,
        help="Comma-separated subset of params to sweep (default: all). "
             "Example: warm_start_iterations,warm_start_multiplier",
    )
    args = ap.parse_args()

    iter_list = [int(x) for x in args.iterations.split(",")]
    device = _device_from_str(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}; iter budgets: {iter_list}")

    payload = load_spots(args.spots, map_location="cpu")
    n_total = payload["spots"]["street"].shape[0]
    n_used = min(args.n_spots, n_total)
    n_used = (n_used // args.batch_size) * args.batch_size
    if n_used == 0:
        raise ValueError(f"batch_size {args.batch_size} > n_spots cap {args.n_spots}.")
    print(f"Spots: {n_total} available, using first {n_used} (batch_size={args.batch_size}).")

    checkpoint = args.checkpoint or payload.get("source_checkpoint")
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    base_cfg = Config.from_dict(copy.deepcopy(ckpt["config"]))
    print(f"Checkpoint search config (from ckpt): sparse={base_cfg.search.sparse}, "
          f"sparse_fused={base_cfg.search.sparse_fused}, "
          f"cfr_type={base_cfg.search.cfr_type}, depth={base_cfg.search.depth}")

    base_cfg.num_envs = args.batch_size
    base_cfg.resume_from = checkpoint
    print("Building trainer (one-time compile)…")
    t0 = time.time()
    trainer = RebelCFRTrainer(cfg=base_cfg, device=device)
    trainer.load_checkpoint(checkpoint)
    trainer.model.eval()
    print(f"  trainer ready in {time.time() - t0:.1f}s")

    # Pre-build per-batch PBSes once.
    streets_cpu = payload["spots"]["street"][:n_used]
    batch_pbs: list[Any] = []
    for start in range(0, n_used, args.batch_size):
        idx = torch.arange(start, start + args.batch_size)
        batch_pbs.append(build_pbs_from_spots(payload, device, idx))
    print(f"Built {len(batch_pbs)} batches.")

    src_indices = torch.arange(args.batch_size, device=device)
    base_snap = _snapshot_eval_params(trainer.cfr_evaluator)
    print(f"Baseline evaluator params (from checkpoint): {base_snap}")

    history: list[dict[str, Any]] = []
    per_iter_best: dict[int, dict[str, Any]] = {}

    with torch.no_grad():
        with trainer._eval_swap():
            initial_seeds: dict[str, dict[str, Any]] = {}
            if args.initial_overrides:
                with open(args.initial_overrides) as f:
                    initial_seeds = json.load(f)
                print(f"Initial-overrides seeds loaded: {initial_seeds}")

            for iters in iter_list:
                print(f"\n========= Iter budget: {iters} =========")
                seed = {k: v for k, v in initial_seeds.get(str(iters), {}).items()
                        if k != "iterations"}
                # Start each iter sweep from baseline + iter override + seed.
                current_overrides: dict[str, Any] = {"iterations": iters, **seed}
                print(f"Initial baseline @ iter={iters}")
                base_result = run_one(
                    trainer, base_snap, current_overrides,
                    batch_pbs, streets_cpu, src_indices, args.batch_size, device,
                )
                base_result["name"] = f"iter{iters}_baseline"
                base_result["iterations"] = iters
                base_result["stage"] = "baseline"
                history.append(base_result)
                best = base_result
                if "error" in base_result:
                    print(f"  baseline FAILED: {base_result['error']}")
                    continue
                print(f"  baseline expl={best['local_exploitability']:.5f} "
                      f"time={best['wall_time_s_total']:.1f}s")
                locked: dict[str, Any] = {"iterations": iters, **seed}

                grid = param_grid(iters)
                if args.params:
                    keep = set(args.params.split(","))
                    grid = [(p, c) for p, c in grid if p in keep]
                for param, candidates in grid:
                    locked_view = {k: v for k, v in locked.items() if k != "iterations"}
                    print(f"\n  -- sweeping {param} (current best so far "
                          f"expl={best['local_exploitability']:.5f}, locked={locked_view}) --")
                    # Effective current value: locked override if set, else evaluator baseline.
                    current_v = locked.get(param, base_snap.get(_canonical_attr(param)))
                    for v in candidates:
                        if v == current_v:
                            print(f"    {param}={v}: skipped (matches current best)")
                            continue
                        cand_overrides = {**locked, param: v}
                        result = run_one(
                            trainer, base_snap, cand_overrides,
                            batch_pbs, streets_cpu, src_indices, args.batch_size, device,
                        )
                        result["name"] = f"iter{iters}_{param}={v}"
                        result["iterations"] = iters
                        result["stage"] = f"sweep_{param}"
                        history.append(result)
                        if "error" in result:
                            print(f"    {param}={v}: ERROR {result['error']}")
                            continue
                        improved = result["local_exploitability"] < best["local_exploitability"]
                        marker = " *new best*" if improved else ""
                        print(f"    {param}={v}: expl={result['local_exploitability']:.5f}  "
                              f"time={result['wall_time_s_total']:.1f}s{marker}")
                        if improved:
                            best = result
                            locked = {**locked, param: v}

                print(f"\n  >>> Best @ iter={iters}: expl={best['local_exploitability']:.5f} "
                      f"overrides={best['overrides']}")
                per_iter_best[iters] = best

    out = {
        "spots_path": os.path.realpath(args.spots),
        "checkpoint_path": os.path.realpath(checkpoint),
        "batch_size": args.batch_size,
        "n_spots_used": n_used,
        "iter_budgets": iter_list,
        "param_grid": {str(it): [{p: vs} for p, vs in param_grid(it)] for it in iter_list},
        "base_search_config": asdict(base_cfg.search),
        "baseline_evaluator_params": {k: str(v) for k, v in base_snap.items()},
        "per_iter_best": {str(k): v for k, v in per_iter_best.items()},
        "history": history,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")

    print("\n--- Final best per iter ---")
    for iters in iter_list:
        b = per_iter_best.get(iters)
        if not b:
            print(f"  iter={iters}: no result")
            continue
        print(f"  iter={iters}: expl={b['local_exploitability']:.5f}  "
              f"overrides={b['overrides']}")


def _canonical_attr(param: str) -> str:
    """Return one evaluator attr to read for the given trial-spec key."""
    return EVAL_ATTR_MAP[param][0]


if __name__ == "__main__":
    main()
