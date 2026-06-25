#!/usr/bin/env python3
"""Benchmark deferred preflop average-policy materialization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.bench_preflop_evaluate_cfr_loop import (  # noqa: E402
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_CLOSING_CHECKPOINT,
    DEFAULT_DATASET,
    DEFAULT_RUN_DIR,
    _make_evaluator,
    _pause_processes,
    _sync,
)


def _event_time_ms(fn, *, iters: int, device: torch.device) -> float:
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA event timing.")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    _sync(device)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync(device)
    return float(start.elapsed_time(end)) / max(1, iters)


def _single_event_time_ms(fn, *, device: torch.device) -> float:
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires CUDA event timing.")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    _sync(device)
    start.record()
    fn()
    end.record()
    _sync(device)
    return float(start.elapsed_time(end))


def _make_loop_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        state_dataset=args.state_dataset,
        base_checkpoint=args.base_checkpoint,
        closing_checkpoint=args.closing_checkpoint,
        run_output_dir=args.run_output_dir,
        bucket=args.bucket,
        cfr_batch_size=args.cfr_batch_size,
        cfr_iterations=args.cfr_iterations,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        range_hidden_dim=args.range_hidden_dim,
        ffn_dim=args.ffn_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_value_layers=args.num_value_layers,
        num_policy_layers=args.num_policy_layers,
        transformer_heads=args.transformer_heads,
        compile=args.compile,
        graph="off",
        warmup_solves=0,
        no_closing_checkpoint=args.no_closing_checkpoint,
        skip_load_weights=args.skip_load_weights,
        no_pause=args.no_pause,
        pause_pattern=args.pause_pattern,
        out=args.out,
    )


def _make_evaluator_for_mode(args: argparse.Namespace, *, defer_avg_policy: bool):
    os.environ["P2_PREFLOP_DEFER_AVG_POLICY"] = "1" if defer_avg_policy else "0"
    os.environ.setdefault("P2_PREFLOP_DEFER_AVG_BELIEFS", "1")
    ev, rows = _make_evaluator(_make_loop_args(args))
    if ev.device.type != "cuda":
        raise RuntimeError("CUDA is required")
    with torch.no_grad():
        ev.initialize_policy_and_beliefs()
        ev._regret_match_current_policy(args.start_t)
    return ev, rows


def _set_defer_avg_policy(defer_avg_policy: bool) -> None:
    os.environ["P2_PREFLOP_DEFER_AVG_POLICY"] = "1" if defer_avg_policy else "0"


def _average_policy_step(ev, t_box: list[int], *, defer_avg_policy: bool) -> None:
    _set_defer_avg_policy(defer_avg_policy)
    with torch.no_grad():
        t = t_box[0]
        ev._refresh_fused_t_scalars(t)
        ev.update_average_policy(t, update_reach=False, update_beliefs=False)
        t_box[0] = t + 1


def _bench_mode(
    args: argparse.Namespace, *, defer_avg_policy: bool
) -> tuple[dict[str, float | int], object]:
    ev, rows = _make_evaluator_for_mode(args, defer_avg_policy=defer_avg_policy)
    t_box = [args.start_t]
    for _ in range(args.warmup_iters):
        _average_policy_step(ev, t_box, defer_avg_policy=defer_avg_policy)
    update_ms = _event_time_ms(
        lambda: _average_policy_step(ev, t_box, defer_avg_policy=defer_avg_policy),
        iters=args.iters,
        device=ev.device,
    )
    finalize_ms = 0.0
    if defer_avg_policy:
        finalize_ms = _single_event_time_ms(
            lambda: (
                _set_defer_avg_policy(defer_avg_policy),
                ev._finalize_deferred_average_policy(),
            ),
            device=ev.device,
        )
    total_ms_per_iter = update_ms + finalize_ms / max(1, args.iters)
    return (
        {
            "defer_avg_policy": int(defer_avg_policy),
            "rows": int(rows),
            "total_nodes": int(ev.total_nodes),
            "root_nodes": int(ev.root_nodes),
            "model_indices": int(ev.model_indices.numel()),
            "iters": int(args.iters),
            "update_ms": update_ms,
            "finalize_ms": finalize_ms,
            "total_ms_per_iter": total_ms_per_iter,
        },
        ev,
    )


def _correctness(args: argparse.Namespace) -> dict[str, float]:
    old, _rows = _make_evaluator_for_mode(args, defer_avg_policy=False)
    new, _rows = _make_evaluator_for_mode(args, defer_avg_policy=True)
    old_t = [args.start_t]
    new_t = [args.start_t]
    with torch.no_grad():
        for _ in range(args.correctness_iters):
            _average_policy_step(old, old_t, defer_avg_policy=False)
            _average_policy_step(new, new_t, defer_avg_policy=True)
        _set_defer_avg_policy(True)
        new._finalize_deferred_average_policy()
    _sync(old.device)
    return {
        "policy_probs_avg": float(
            (old.policy_probs_avg - new.policy_probs_avg).abs().max().item()
        ),
        "average_policy_numerator": float(
            (
                old.average_policy_numerator - new.average_policy_numerator
            )
            .abs()
            .max()
            .item()
        ),
        "average_policy_denominator": float(
            (
                old.average_policy_denominator - new.average_policy_denominator
            )
            .abs()
            .max()
            .item()
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument(
        "--closing-checkpoint",
        type=Path,
        default=DEFAULT_CLOSING_CHECKPOINT,
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
    parser.add_argument("--no-closing-checkpoint", action="store_true")
    parser.add_argument("--skip-load-weights", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="preflop_backward_induction")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--correctness-iters", type=int, default=12)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--start-t", type=int, default=64)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/preflop_defer_avg_policy.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with _pause_processes(not args.no_pause, args.pause_pattern):
        max_diff = _correctness(args)
        off, _ = _bench_mode(args, defer_avg_policy=False)
        on, _ = _bench_mode(args, defer_avg_policy=True)
        on_repeat, _ = _bench_mode(args, defer_avg_policy=True)
        off_repeat, _ = _bench_mode(args, defer_avg_policy=False)
    results = [off, on, on_repeat, off_repeat]
    output = {"max_diff": max_diff, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    for row in results:
        print(
            "defer_avg_policy={defer_avg_policy} update_ms={update_ms:.6f} "
            "finalize_ms={finalize_ms:.6f} total_ms_per_iter={total_ms_per_iter:.6f}".format(
                **row
            ),
            flush=True,
        )
    print(f"max_diff={max_diff}", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
