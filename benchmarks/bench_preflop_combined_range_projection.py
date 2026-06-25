#!/usr/bin/env python3
"""Benchmark combined compact-preflop range moment projections."""

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


def _prepare(args: argparse.Namespace, mode: str):
    os.environ["P2_PREFLOP_COMBINED_RANGE_PROJECTION"] = mode
    ev, rows = _make_evaluator(_make_loop_args(args))
    if ev.device.type != "cuda":
        raise RuntimeError("CUDA is required")
    with torch.no_grad():
        ev._regret_match_current_policy(1)
        beliefs = ev._model_beliefs_for_values(ev.beliefs)
        features = ev._model_features_for_beliefs(beliefs)
    return ev, rows, beliefs, features


def _compute_values(args: argparse.Namespace, mode: str) -> torch.Tensor:
    ev, _rows, beliefs, features = _prepare(args, mode)
    with torch.no_grad():
        ev._set_model_values(1, beliefs, features)
    _sync(ev.device)
    return ev.latest_values.detach().clone()


def _bench_mode(args: argparse.Namespace, mode: str) -> dict[str, float | int | str]:
    ev, rows, beliefs, features = _prepare(args, mode)

    def call() -> None:
        with torch.no_grad():
            ev._set_model_values(1, beliefs, features)

    for _ in range(args.warmup_iters):
        call()
    ms = _event_time_ms(call, iters=args.iters, device=ev.device)
    return {
        "mode": mode,
        "rows": int(rows),
        "total_nodes": int(ev.total_nodes),
        "model_indices": int(ev.model_indices.numel()),
        "cutoff_model_positions": int(ev.cutoff_model_positions.numel()),
        "new_street_hu_model_positions": int(
            ev.new_street_hu_model_positions.numel()
        ),
        "ms": ms,
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
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/preflop_combined_range_projection.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ("off", "range")
    with _pause_processes(not args.no_pause, args.pause_pattern):
        reference = _compute_values(args, "off")
        max_diffs = {}
        for mode in modes[1:]:
            actual = _compute_values(args, mode)
            max_diffs[mode] = float((actual - reference).abs().max().item())
        results = [
            _bench_mode(args, "off"),
            _bench_mode(args, "range"),
            _bench_mode(args, "range"),
            _bench_mode(args, "off"),
        ]
    output = {
        "max_diffs": max_diffs,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    for row in results:
        print(f"mode={row['mode']} ms={row['ms']:.6f}", flush=True)
    print(f"max_diffs={max_diffs}", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
