#!/usr/bin/env python3
"""
Evaluate different CFR search parameters on a fixed set of saved spots.

The model + spots are deterministic, so we build the trainer (and thus
the evaluator + compiled model) ONCE and just mutate the evaluator's
search-time parameters between trials.

Inputs:
  --spots          path to a spots.pt produced by sample_spots.py
  --trials         JSON file describing the trials
  --out            JSON path for results
  --checkpoint     model checkpoint (default: payload["source_checkpoint"])
  --batch-size     spots per evaluator pass (default: 256). Spots are
                   truncated to a multiple of batch-size.
  --device         cuda | mps | cpu

Trial JSON schema:
  {
    "batch_size": 256,
    "trials": [
      {"name": "baseline"},
      {"name": "iter400", "iterations": 400},
      {"name": "alpha2", "dcfr_alpha": 2.0}
    ]
  }

Each trial overrides fields on the evaluator. Supported keys:
  iterations, warm_start_iterations, warm_start_type,
  warm_start_multiplier, cfr_type, cfr_plus, cfr_avg,
  dcfr_alpha, dcfr_beta, dcfr_gamma,
  dcfr_alpha_final, dcfr_beta_final, dcfr_gamma_final,
  dcfr_plus_delay (a.k.a. dcfr_delay),
  value_targets_from_final_policy

Tree-shape parameters (depth, sparse, sparse_fused) are NOT supported
here — they'd require rebuilding the trainer. Run separately for those.
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
from p2.core.structured_config import CFRType, Config, WarmStartType
from p2.rl.cfr_trainer import RebelCFRTrainer

STREETS = ["preflop", "flop", "turn", "river", "showdown"]

# Map trial-spec keys -> list of evaluator attrs to set with the same value.
# (Some attrs come in pairs because the evaluator stores both "current"
# and "initial" copies of dcfr_* and resets the current to initial each
# evaluate_cfr.)
EVAL_ATTR_MAP: dict[str, list[str]] = {
    "iterations": ["cfr_iterations"],
    "warm_start_iterations": ["warm_start_iterations"],
    "warm_start_type": ["warm_start_type"],
    "warm_start_multiplier": ["warm_start_multiplier"],
    "cfr_type": ["cfr_type"],
    "cfr_plus": ["cfr_plus"],
    "cfr_avg": ["cfr_avg"],
    "dcfr_alpha": ["dcfr_alpha", "dcfr_alpha_initial"],
    "dcfr_beta":  ["dcfr_beta",  "dcfr_beta_initial"],
    "dcfr_gamma": ["dcfr_gamma", "dcfr_gamma_initial"],
    "dcfr_alpha_final": ["dcfr_alpha_final"],
    "dcfr_beta_final":  ["dcfr_beta_final"],
    "dcfr_gamma_final": ["dcfr_gamma_final"],
    "dcfr_plus_delay": ["dcfr_delay"],
    "dcfr_delay": ["dcfr_delay"],
    "value_targets_from_final_policy": ["use_final_policy_values"],
}

# trial-spec keys that require rebuild (rejected here)
UNSUPPORTED_KEYS = {"depth", "sparse", "sparse_fused", "iterations_final"}

# Values that need string -> enum coercion when set.
ENUM_COERCE = {
    "cfr_type": CFRType,
    "warm_start_type": WarmStartType,
}


def _device_from_str(s: str | None) -> torch.device:
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if s == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device(s or "cpu")


def _snapshot_eval_params(evaluator) -> dict[str, Any]:
    keys: set[str] = set()
    for attrs in EVAL_ATTR_MAP.values():
        keys.update(attrs)
    return {k: getattr(evaluator, k) for k in keys}


def _restore_eval_params(evaluator, snap: dict[str, Any]) -> None:
    for k, v in snap.items():
        setattr(evaluator, k, v)


def _apply_eval_overrides(evaluator, overrides: dict[str, Any]) -> None:
    for k, raw_v in overrides.items():
        if k == "name":
            continue
        if k in UNSUPPORTED_KEYS:
            raise ValueError(
                f"Override {k!r} requires rebuild; not supported in this script. "
                f"Run a separate sweep with that fixed."
            )
        if k not in EVAL_ATTR_MAP:
            raise ValueError(
                f"Unknown override key: {k!r}. Supported: {sorted(EVAL_ATTR_MAP)}"
            )
        v = ENUM_COERCE[k](raw_v) if k in ENUM_COERCE and isinstance(raw_v, str) else raw_v
        for attr in EVAL_ATTR_MAP[k]:
            setattr(evaluator, attr, v)


def _aggregate_trial(
    per_batch_stats: list[dict[str, Any]], per_batch_n: list[int]
) -> dict[str, Any]:
    total_n = sum(per_batch_n)

    def wmean(key: str) -> float | None:
        nums, dens = 0.0, 0
        for stats, n in zip(per_batch_stats, per_batch_n):
            v = stats.get(key)
            if v is None:
                continue
            nums += float(v) * n
            dens += n
        return nums / dens if dens else None

    out: dict[str, Any] = {
        "local_exploitability": wmean("local_exploitability"),
        "local_exploitability_init": wmean("local_exploitability_init"),
        "local_exploitability_max": max(
            (s.get("local_exploitability_max", float("-inf")) for s in per_batch_stats),
            default=None,
        ),
        "cfr_entropy": wmean("cfr_entropy"),
        "mean_positive_regret": wmean("mean_positive_regret"),
        "n_spots": total_n,
    }

    street_sums: dict[str, float] = {s: 0.0 for s in STREETS}
    street_counts: dict[str, int] = {s: 0 for s in STREETS}
    for stats in per_batch_stats:
        per_street = stats.get("local_exploitability_street", {}) or {}
        per_street_counts = stats.get("_per_street_counts", {}) or {}
        for s, v in per_street.items():
            cnt = per_street_counts.get(s, 0)
            if cnt > 0:
                street_sums[s] += float(v) * cnt
                street_counts[s] += cnt
    out["local_exploitability_street"] = {
        s: (street_sums[s] / street_counts[s]) if street_counts[s] else None
        for s in STREETS
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", required=True)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = _device_from_str(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    print(f"Loading spots: {args.spots}")
    payload = load_spots(args.spots, map_location="cpu")
    n_spots = payload["spots"]["street"].shape[0]
    print(f"  {n_spots} spots, per-street: {payload.get('per_street_counts')}")

    checkpoint = args.checkpoint or payload.get("source_checkpoint")
    if not checkpoint or not os.path.exists(checkpoint):
        raise ValueError(f"Checkpoint not found: {checkpoint!r}. Pass --checkpoint.")
    print(f"Loading checkpoint config: {checkpoint}")
    ckpt = torch.load(checkpoint, weights_only=False, map_location="cpu")
    base_cfg = Config.from_dict(copy.deepcopy(ckpt["config"]))

    with open(args.trials) as f:
        trial_spec = json.load(f)
    batch_size = int(trial_spec.get("batch_size", args.batch_size))
    trials = trial_spec["trials"]
    print(f"Running {len(trials)} trial(s) with batch_size={batch_size}")

    # Build trainer ONCE.
    base_cfg.num_envs = batch_size
    base_cfg.resume_from = checkpoint
    print("Building trainer (one-time compile)…")
    t0 = time.time()
    trainer = RebelCFRTrainer(cfg=base_cfg, device=device)
    trainer.load_checkpoint(checkpoint)
    trainer.model.eval()
    print(f"  trainer ready in {time.time() - t0:.1f}s")

    n_used = (n_spots // batch_size) * batch_size
    if n_used == 0:
        raise ValueError(f"batch_size {batch_size} > n_spots {n_spots}.")
    if n_used < n_spots:
        print(f"Truncating {n_spots} -> {n_used} spots to fit batch_size.")
    spots = payload["spots"]
    streets_cpu = spots["street"][:n_used]

    # Pre-build per-batch PBS once — these are the same across trials.
    print("Pre-building per-batch PBSes…")
    batch_pbs: list[Any] = []
    for start in range(0, n_used, batch_size):
        idx = torch.arange(start, start + batch_size)
        batch_pbs.append(build_pbs_from_spots(payload, device, idx))
    print(f"  {len(batch_pbs)} batches.")

    src_indices = torch.arange(batch_size, device=device)
    base_eval_snap = _snapshot_eval_params(trainer.cfr_evaluator)
    print(f"Baseline evaluator params: {base_eval_snap}")

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        with trainer._eval_swap():
            for trial in trials:
                name = trial.get("name", f"trial_{len(results)}")
                print(f"\n=== Trial: {name} ===")
                non_name_overrides = {k: v for k, v in trial.items() if k != "name"}
                print(f"  overrides: {non_name_overrides}")

                _restore_eval_params(trainer.cfr_evaluator, base_eval_snap)
                try:
                    _apply_eval_overrides(trainer.cfr_evaluator, trial)
                except Exception as e:
                    print(f"  FAILED to apply overrides: {e!r}")
                    results.append({"name": name, "error": repr(e), "overrides": non_name_overrides})
                    continue

                per_batch_stats: list[dict[str, Any]] = []
                per_batch_n: list[int] = []
                t_trial = time.time()
                trial_error: str | None = None
                try:
                    for bi, pbs in enumerate(batch_pbs):
                        trainer.cfr_evaluator.initialize_subgame(pbs.env, src_indices, pbs.beliefs)
                        trainer.cfr_evaluator.evaluate_cfr(training_mode=False)
                        stats = {k: v for k, v in trainer.cfr_evaluator.stats.items()}
                        bs_streets = streets_cpu[bi * batch_size : (bi + 1) * batch_size]
                        stats["_per_street_counts"] = {
                            STREETS[i]: int((bs_streets == i).sum().item())
                            for i in range(5)
                            if (bs_streets == i).any()
                        }
                        per_batch_stats.append(stats)
                        per_batch_n.append(batch_size)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                except Exception as e:
                    trial_error = repr(e)
                    print(f"  RUNTIME ERROR: {trial_error}")
                trial_time = time.time() - t_trial
                if trial_error is not None:
                    results.append({
                        "name": name, "error": trial_error,
                        "overrides": non_name_overrides,
                        "wall_time_s_total": trial_time,
                    })
                    continue

                agg = _aggregate_trial(per_batch_stats, per_batch_n)
                agg["wall_time_s_total"] = trial_time
                agg["wall_time_s_per_batch_mean"] = trial_time / max(1, len(batch_pbs))
                agg["batch_size"] = batch_size
                agg["overrides"] = non_name_overrides
                agg["name"] = name
                results.append(agg)

                expl = agg.get("local_exploitability") or float("nan")
                print(
                    f"  local_exploitability={expl:.5f} "
                    f"per_street={agg.get('local_exploitability_street')} "
                    f"time={trial_time:.1f}s"
                )

    out = {
        "spots_path": os.path.realpath(args.spots),
        "checkpoint_path": os.path.realpath(checkpoint),
        "batch_size": batch_size,
        "n_spots_total": n_spots,
        "n_spots_used": n_used,
        "base_search_config": asdict(base_cfg.search),
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")

    print("\n--- Summary (sorted by local_exploitability) ---")
    rows = [r for r in results if "local_exploitability" in r and r["local_exploitability"] is not None]
    rows.sort(key=lambda r: r["local_exploitability"])
    for r in rows:
        print(
            f"  {r['name']:<28s} expl={r['local_exploitability']:.5f}  "
            f"time={r['wall_time_s_total']:.1f}s  "
            f"{r['overrides']}"
        )


if __name__ == "__main__":
    main()
