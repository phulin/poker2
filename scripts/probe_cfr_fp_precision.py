#!/usr/bin/env python3
"""Probe fp32 precision in fused sparse CFR average-policy updates.

This script starts from a saved ``spots.pt`` payload, loads a trained ReBeL
checkpoint, runs the CUDA/Triton ``FusedSparseCFREvaluator`` eagerly, and
samples the weighted additions used by the average-policy update.  The probe
answers: "is the new contribution below fp32 resolution relative to the
existing average accumulator?"

Example:
  uv run python scripts/probe_cfr_fp_precision.py

Defaults are set for the local workspace:
  --spots outputs/spots.pt
  --checkpoint checkpoints-rebel/rebel_step_2500.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import asdict
from types import MethodType
from typing import Any

import torch

from p2.cli.sample_spots import build_pbs_from_spots, load_spots
from p2.config.rebel_load import load_rebel_config_file
from p2.core.structured_config import CFRType, Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

STREET_TO_ID = {
    "preflop": 0,
    "flop": 1,
    "turn": 2,
    "river": 3,
    "showdown": 4,
}


def _cuda_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type != "cuda":
        raise SystemExit("FusedSparseCFREvaluator requires a CUDA device.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment.")
    return device


def _float_or_none(x: torch.Tensor | None) -> float | None:
    if x is None or x.numel() == 0:
        return None
    return float(x.detach().cpu().item())


def _quantile_or_none(x: torch.Tensor, q: float) -> float | None:
    if x.numel() == 0:
        return None
    return _float_or_none(torch.quantile(x.float(), q))


def _mean_bool(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.float().mean().detach().cpu().item())


def _sample_flat_indices(
    total: int,
    sample_elements: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if sample_elements <= 0 or sample_elements >= total:
        return torch.arange(total, device=device)
    return torch.randint(total, (sample_elements,), device=device, generator=generator)


class PrecisionProbe:
    """Monkey-patches selected evaluator update methods for diagnostics."""

    def __init__(
        self,
        evaluator: FusedSparseCFREvaluator,
        sample_elements: int,
        sample_every: int,
        seed: int,
    ) -> None:
        self.evaluator = evaluator
        self.sample_elements = sample_elements
        self.sample_every = max(1, sample_every)
        self.generator = torch.Generator(device=evaluator.device)
        self.generator.manual_seed(seed)
        self.policy_records: list[dict[str, Any]] = []
        self.value_records: list[dict[str, Any]] = []
        self._orig_update_average_policy = evaluator.update_average_policy
        self._orig_update_average_values = evaluator.update_average_values

    def attach(self) -> None:
        probe = self

        def update_average_policy_wrapper(
            ev: FusedSparseCFREvaluator,
            t: int,
            update_reach: bool = False,
        ) -> None:
            if probe._should_sample(t):
                record = probe.policy_average_stats(t)
                if record is not None:
                    probe.policy_records.append(record)
            probe._orig_update_average_policy(t, update_reach=update_reach)

        def update_average_values_wrapper(
            ev: FusedSparseCFREvaluator,
            t: int,
        ) -> None:
            if probe._should_sample(t):
                record = probe.value_average_stats(t)
                if record is not None:
                    probe.value_records.append(record)
            probe._orig_update_average_values(t)

        self.evaluator.update_average_policy = MethodType(
            update_average_policy_wrapper, self.evaluator
        )
        self.evaluator.update_average_values = MethodType(
            update_average_values_wrapper, self.evaluator
        )

    def _should_sample(self, t: int) -> bool:
        return t == 0 or t % self.sample_every == 0 or t == self.evaluator.cfr_iterations - 1

    def policy_average_stats(self, t: int) -> dict[str, Any] | None:
        ev = self.evaluator
        if (
            ev.cfr_type == CFRType.discounted
            and t <= ev.dcfr_delay
        ):
            return {
                "kind": "policy_avg",
                "t": int(t),
                "mode": "copy_before_dcfr_delay",
                "dcfr_delay": int(ev.dcfr_delay),
            }
        bottom = ev.root_nodes
        total, hands = ev.policy_probs_avg.shape
        child_rows = total - bottom
        if child_rows <= 0:
            return None

        weight = ev._get_average_policy_weight(t)
        n_elements = child_rows * hands
        flat = _sample_flat_indices(
            n_elements, self.sample_elements, ev.device, self.generator
        )
        row = flat.div(hands, rounding_mode="floor") + bottom
        hand = flat.remainder(hands)
        parent = ev.parent_index[row]
        actor = ev.env.to_act[parent]

        avg = ev.policy_probs_avg[row, hand]
        cur = ev.policy_probs[row, hand]
        reach_cur = ev.self_reach[parent, actor, hand]
        avg_num, avg_den = ev._ensure_average_policy_accumulators()
        prev_num = avg_num[row, hand]
        prev_den = avg_den[row, hand]

        weight_t = torch.as_tensor(float(weight), dtype=avg.dtype, device=ev.device)
        new_num = reach_cur * weight_t * cur
        num32 = prev_num + new_num

        new_den = reach_cur * weight_t
        den32 = prev_den + new_den

        avg64 = avg.double()
        cur64 = cur.double()
        prev_num64 = prev_num.double()
        new_num64 = reach_cur.double() * float(weight) * cur64
        num64 = prev_num64 + new_num64
        den64 = prev_den.double() + reach_cur.double() * float(weight)

        out32 = torch.where(den32 > 1e-5, num32 / den32.clamp_min(1e-30), cur)
        out64 = torch.where(den64 > 1e-5, num64 / den64.clamp_min(1e-300), cur64)

        eps = torch.finfo(avg.dtype).eps
        nonzero_new_num = new_num != 0
        nonzero_new_den = new_den != 0
        numerator_lost = (num32 == prev_num) & nonzero_new_num
        denominator_lost = (den32 == prev_den) & nonzero_new_den
        true_change = (out64 - avg64).abs() > 0
        output_noop = (out32 == avg) & true_change

        ratio = new_num.abs() / (eps * prev_num.abs().clamp_min(torch.finfo(avg.dtype).tiny))
        ratio = ratio[nonzero_new_num & torch.isfinite(ratio)]
        abs_diff = (out32.double() - out64).abs()

        return {
            "kind": "policy_avg",
            "t": int(t),
            "mode": "weighted_mix",
            "new_weight": float(weight),
            "sampled_elements": int(flat.numel()),
            "new_numerator_nonzero_frac": _mean_bool(nonzero_new_num),
            "new_denominator_nonzero_frac": _mean_bool(nonzero_new_den),
            "numerator_lost_frac": _mean_bool(numerator_lost),
            "denominator_lost_frac": _mean_bool(denominator_lost),
            "output_noop_frac": _mean_bool(output_noop),
            "new_num_over_ulp_prev_min": _float_or_none(ratio.min() if ratio.numel() else None),
            "new_num_over_ulp_prev_p01": _quantile_or_none(ratio, 0.01),
            "new_num_over_ulp_prev_median": _quantile_or_none(ratio, 0.50),
            "max_abs_fp32_vs_fp64": _float_or_none(abs_diff.max()),
            "mean_abs_fp32_vs_fp64": _float_or_none(abs_diff.mean()),
        }

    def value_average_stats(self, t: int) -> dict[str, Any] | None:
        ev = self.evaluator
        if t == 0:
            return {"kind": "value_avg", "t": 0, "mode": "initial"}

        values_avg = ev.values_avg.reshape(-1)
        latest = ev.latest_values.reshape(-1)
        if values_avg.numel() == 0:
            return None

        old, new = ev._get_mixing_weights(t)
        flat = _sample_flat_indices(
            values_avg.numel(), self.sample_elements, ev.device, self.generator
        )
        avg = values_avg[flat]
        cur = latest[flat]

        old_t = torch.as_tensor(float(old), dtype=avg.dtype, device=ev.device)
        new_t = torch.as_tensor(float(new), dtype=avg.dtype, device=ev.device)
        prev_num = old_t * avg
        new_num = new_t * cur
        num32 = prev_num + new_num
        out32 = num32 / (old_t + new_t)

        out64 = (float(old) * avg.double() + float(new) * cur.double()) / (
            float(old) + float(new)
        )

        eps = torch.finfo(avg.dtype).eps
        nonzero_new_num = new_num != 0
        numerator_lost = (num32 == prev_num) & nonzero_new_num
        true_change = (out64 - avg.double()).abs() > 0
        output_noop = (out32 == avg) & true_change
        ratio = new_num.abs() / (eps * prev_num.abs().clamp_min(torch.finfo(avg.dtype).tiny))
        ratio = ratio[nonzero_new_num & torch.isfinite(ratio)]
        abs_diff = (out32.double() - out64).abs()

        return {
            "kind": "value_avg",
            "t": int(t),
            "mode": "weighted_mix",
            "old_weight": float(old),
            "new_weight": float(new),
            "sampled_elements": int(flat.numel()),
            "new_numerator_nonzero_frac": _mean_bool(nonzero_new_num),
            "numerator_lost_frac": _mean_bool(numerator_lost),
            "output_noop_frac": _mean_bool(output_noop),
            "new_num_over_ulp_prev_min": _float_or_none(ratio.min() if ratio.numel() else None),
            "new_num_over_ulp_prev_p01": _quantile_or_none(ratio, 0.01),
            "new_num_over_ulp_prev_median": _quantile_or_none(ratio, 0.50),
            "max_abs_fp32_vs_fp64": _float_or_none(abs_diff.max()),
            "mean_abs_fp32_vs_fp64": _float_or_none(abs_diff.mean()),
        }


def _load_config(config_path: str, checkpoint_path: str) -> Config:
    cfg = load_rebel_config_file(config_path)
    cfg.resume_from = checkpoint_path
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    return cfg


def _build_trainer_and_evaluator(
    cfg: Config,
    checkpoint_path: str,
    device: torch.device,
    batch_size: int,
    compile_model: bool,
) -> tuple[RebelCFRTrainer, FusedSparseCFREvaluator]:
    trainer_cfg = copy.deepcopy(cfg)
    trainer_cfg.num_envs = batch_size
    trainer_cfg.model.compile = "off"
    trainer_cfg.trueskill.enabled = False
    trainer_cfg.search.sparse = True
    trainer_cfg.search.sparse_fused = False

    trainer = RebelCFRTrainer(cfg=trainer_cfg, device=device)
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.num_envs = batch_size
    eval_cfg.model.compile = "off"
    eval_cfg.trueskill.enabled = False
    eval_cfg.search.sparse = True
    eval_cfg.search.sparse_fused = True

    evaluator = FusedSparseCFREvaluator(
        model=trainer.model,
        device=device,
        cfg=eval_cfg,
        generator=trainer.rng,
        compile_model=compile_model,
    )
    return trainer, evaluator


def _select_spot_indices(
    payload: dict[str, Any],
    street: str,
    offset: int,
    batch_size: int,
    random_spots: bool,
    seed: int,
) -> torch.Tensor:
    streets = payload["spots"]["street"]
    if street == "any":
        candidates = torch.arange(streets.shape[0])
    else:
        candidates = torch.where(streets == STREET_TO_ID[street])[0]
    if random_spots:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        candidates = candidates[torch.randperm(candidates.numel(), generator=generator)]
    end = offset + batch_size
    if end > candidates.numel():
        raise ValueError(
            f"Need {end} candidate spots for street={street!r}, "
            f"but spots.pt only has {candidates.numel()}."
        )
    return candidates[offset:end].cpu()


def _run_cfr_eager(evaluator: FusedSparseCFREvaluator, training_mode: bool = False):
    evaluator.model.eval()
    evaluator.initialize_policy_and_beliefs()
    if evaluator.warm_start_iterations > 0:
        evaluator.warm_start()

    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values

    evaluator.t_sample = evaluator._get_sampling_schedule()
    if hasattr(evaluator, "_prepare_sample_update_table"):
        evaluator._prepare_sample_update_table()

    for t in range(evaluator.warm_start_iterations, evaluator.cfr_iterations):
        evaluator.profiler_step()
        evaluator.cfr_iteration(t)

    if not evaluator.cfr_avg and evaluator.use_final_policy_values:
        evaluator._refresh_average_beliefs()

    if evaluator.use_final_policy_values:
        evaluator.update_average_values_final()

    evaluator._record_action_mix()
    evaluator._record_cfr_entropy()
    evaluator._record_cumulative_regret()
    return evaluator.sample_leaves(training_mode)


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    weighted = [r for r in records if r.get("mode") == "weighted_mix"]
    if not weighted:
        return {"weighted_samples": 0}

    def max_key(key: str) -> float | None:
        vals = [r.get(key) for r in weighted if r.get(key) is not None]
        return max(vals) if vals else None

    worst = max(weighted, key=lambda r: r.get("numerator_lost_frac", 0.0))
    last = weighted[-1]
    return {
        "weighted_samples": len(weighted),
        "worst_t": worst["t"],
        "worst_numerator_lost_frac": worst.get("numerator_lost_frac"),
        "worst_output_noop_frac": max_key("output_noop_frac"),
        "worst_denominator_lost_frac": max_key("denominator_lost_frac"),
        "lowest_new_num_over_ulp_prev_p01": min(
            (
                r["new_num_over_ulp_prev_p01"]
                for r in weighted
                if r.get("new_num_over_ulp_prev_p01") is not None
            ),
            default=None,
        ),
        "last_t": last["t"],
        "last_numerator_lost_frac": last.get("numerator_lost_frac"),
        "last_output_noop_frac": last.get("output_noop_frac"),
        "last_new_num_over_ulp_prev_p01": last.get("new_num_over_ulp_prev_p01"),
    }


def _print_summary(kind: str, summary: dict[str, Any]) -> None:
    if summary.get("weighted_samples", 0) == 0:
        print(f"{kind}: no weighted average samples were recorded.")
        return
    print(
        f"{kind}: worst lost={summary['worst_numerator_lost_frac']:.6f} "
        f"at t={summary['worst_t']}, "
        f"worst noop={summary.get('worst_output_noop_frac'):.6f}, "
        f"last lost={summary['last_numerator_lost_frac']:.6f} "
        f"at t={summary['last_t']}, "
        f"last p01(new/ulp(prev))={summary.get('last_new_num_over_ulp_prev_p01')}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", default="outputs/spots.pt")
    ap.add_argument("--config", default="conf/config_rebel_cfr.yaml")
    ap.add_argument("--checkpoint", default="checkpoints-rebel/rebel_step_2500.pt")
    ap.add_argument("--out", default="outputs/cfr_fp_precision_probe.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--spot-offset", type=int, default=0)
    ap.add_argument(
        "--random-spots",
        action="store_true",
        help="Randomly sample spots from the selected street set using --seed.",
    )
    ap.add_argument(
        "--street",
        choices=["any", *STREET_TO_ID.keys()],
        default="any",
        help="Select the first batch from a specific street in spots.pt.",
    )
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--dcfr-delay", type=int, default=None)
    ap.add_argument("--sample-elements", type=int, default=65536)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--stress-t",
        type=int,
        nargs="*",
        default=[100_000, 1_000_000, 10_000_000, 100_000_000],
        help="Extra t values to probe on the final evaluator state without mutating it.",
    )
    ap.add_argument(
        "--compile-model",
        action="store_true",
        help="Compile the model inside FusedSparseCFREvaluator.",
    )
    args = ap.parse_args()

    device = _cuda_device(args.device)
    torch.set_float32_matmul_precision("high")

    checkpoint = os.path.abspath(args.checkpoint)
    spots_path = os.path.abspath(args.spots)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)
    if not os.path.exists(spots_path):
        raise FileNotFoundError(spots_path)

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Spots: {spots_path}")

    cfg = _load_config(args.config, checkpoint)
    if args.iterations is not None:
        cfg.search.iterations = args.iterations
        cfg.search.iterations_final = None
    if args.dcfr_delay is not None:
        cfg.search.dcfr_plus_delay = args.dcfr_delay

    payload = load_spots(spots_path, map_location="cpu")
    spot_indices = _select_spot_indices(
        payload,
        street=args.street,
        offset=args.spot_offset,
        batch_size=args.batch_size,
        random_spots=args.random_spots,
        seed=args.seed,
    )
    selected_streets = payload["spots"]["street"][spot_indices]
    selected_counts = {
        name: int((selected_streets == street_id).sum().item())
        for name, street_id in STREET_TO_ID.items()
    }
    print(f"Selected spots: {spot_indices.tolist()}")
    print(f"Selected street counts: {selected_counts}")

    trainer, evaluator = _build_trainer_and_evaluator(
        cfg=cfg,
        checkpoint_path=checkpoint,
        device=device,
        batch_size=args.batch_size,
        compile_model=args.compile_model,
    )
    probe = PrecisionProbe(
        evaluator=evaluator,
        sample_elements=args.sample_elements,
        sample_every=args.sample_every,
        seed=args.seed,
    )
    probe.attach()

    pbs = build_pbs_from_spots(payload, device, spot_indices)
    src_indices = torch.arange(args.batch_size, device=device)

    print(
        "Evaluator config: "
        f"depth={evaluator.max_depth}, iterations={evaluator.cfr_iterations}, "
        f"warm_start={evaluator.warm_start_iterations}, "
        f"cfr_type={evaluator.cfr_type}, "
        f"dcfr_delay={evaluator.dcfr_delay}, "
        f"cfr_avg={evaluator.cfr_avg}, "
        f"use_final_policy_values={evaluator.use_final_policy_values}"
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        with trainer._eval_swap():
            evaluator.initialize_subgame(pbs.env, src_indices, pbs.beliefs)
            _run_cfr_eager(evaluator, training_mode=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            stress_records = [
                probe.policy_average_stats(t)
                for t in args.stress_t
                if t > evaluator.dcfr_delay
            ]
    elapsed = time.perf_counter() - t0

    policy_summary = _summarize(probe.policy_records)
    value_summary = _summarize(probe.value_records)
    stress_records = [r for r in stress_records if r is not None]
    stress_summary = _summarize(stress_records)

    print(f"Ran eager fused CFR in {elapsed:.2f}s over {evaluator.total_nodes:,} nodes.")
    _print_summary("policy_avg", policy_summary)
    _print_summary("value_avg", value_summary)
    _print_summary("stress_policy_avg", stress_summary)
    print(f"Evaluator stats: {evaluator.stats}")

    report = {
        "checkpoint": checkpoint,
        "spots": spots_path,
        "selected_spot_indices": spot_indices.tolist(),
        "selected_street_counts": selected_counts,
        "batch_size": args.batch_size,
        "random_spots": bool(args.random_spots),
        "seed": int(args.seed),
        "average_policy": "true_cfr_reach_weighted",
        "sample_elements": args.sample_elements,
        "sample_every": args.sample_every,
        "elapsed_s": elapsed,
        "total_nodes": int(evaluator.total_nodes),
        "tree_depth": int(evaluator.tree_depth),
        "search_config": asdict(cfg.search),
        "policy_records": probe.policy_records,
        "value_records": probe.value_records,
        "stress_policy_records": stress_records,
        "summary": {
            "policy_avg": policy_summary,
            "value_avg": value_summary,
            "stress_policy_avg": stress_summary,
        },
        "evaluator_stats": evaluator.stats,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
