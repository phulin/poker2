#!/usr/bin/env python3
"""Benchmark live-entry routing/writeback for exact preflop all-in values."""

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


def _make_evaluator_for_mode(
    args: argparse.Namespace,
    *,
    live2_entries: bool,
    sparse_writeback: bool,
):
    os.environ["P2_PREFLOP_ALLIN_LIVE2_ENTRIES"] = "1" if live2_entries else "0"
    os.environ["P2_PREFLOP_ALLIN_SPARSE_WRITEBACK"] = (
        "1" if sparse_writeback else "0"
    )
    ev, rows = _make_evaluator(_make_loop_args(args))
    if ev.device.type != "cuda":
        raise RuntimeError("CUDA is required")
    with torch.no_grad():
        ev._regret_match_current_policy(1)
        ev._cache_preflop_allin_live_partitions()
        ev._ensure_preflop_allin_169_oracle()
    return ev, rows


def _bench_mode(
    args: argparse.Namespace,
    *,
    live2_entries: bool,
    sparse_writeback: bool,
) -> dict[str, float | int]:
    ev, rows = _make_evaluator_for_mode(
        args,
        live2_entries=live2_entries,
        sparse_writeback=sparse_writeback,
    )

    def call() -> None:
        with torch.no_grad():
            ev._set_allin_call_values(ev.beliefs)

    for _ in range(args.warmup_iters):
        call()
    ms = _event_time_ms(call, iters=args.iters, device=ev.device)
    return {
        "live2_entries": int(live2_entries),
        "sparse_writeback": int(sparse_writeback),
        "rows": int(rows),
        "total_nodes": int(ev.total_nodes),
        "allin_call_indices": int(ev.allin_call_indices.numel()),
        "live2_nodes": int(ev.preflop_allin_indices_by_live_count[2].numel()),
        "live2_entries_count": int(ev.preflop_allin_live2_entry_rows.numel()),
        "live3_nodes": int(ev.preflop_allin_indices_by_live_count[3].numel()),
        "live3_entries_count": int(ev.preflop_allin_live3_entry_rows.numel()),
        "ms": ms,
    }


def _correctness(args: argparse.Namespace) -> float:
    ev_old, _rows = _make_evaluator_for_mode(
        args,
        live2_entries=True,
        sparse_writeback=False,
    )
    ev_new, _rows = _make_evaluator_for_mode(
        args,
        live2_entries=True,
        sparse_writeback=True,
    )
    with torch.no_grad():
        ev_old._set_allin_call_values(ev_old.beliefs)
        ev_new._set_allin_call_values(ev_new.beliefs)
    _sync(ev_old.device)
    return float(
        (
            ev_new.latest_values[ev_new.allin_call_indices]
            - ev_old.latest_values[ev_old.allin_call_indices]
        )
        .abs()
        .max()
        .item()
    )


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
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/preflop_allin_live2_entries.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with _pause_processes(not args.no_pause, args.pause_pattern):
        max_diff = _correctness(args)
        results = [
            _bench_mode(args, live2_entries=True, sparse_writeback=False),
            _bench_mode(args, live2_entries=True, sparse_writeback=True),
            _bench_mode(args, live2_entries=True, sparse_writeback=True),
            _bench_mode(args, live2_entries=True, sparse_writeback=False),
        ]
    output = {
        "max_diff": max_diff,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    for row in results:
        print(
            "live2_entries={live2_entries} sparse_writeback={sparse_writeback} "
            "live2_nodes={live2_nodes} live3_nodes={live3_nodes} ms={ms:.6f}".format(
                **row
            ),
            flush=True,
        )
    print(f"max_diff={max_diff:.6g}", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
