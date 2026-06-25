#!/usr/bin/env python3
"""Microbenchmark preflop model-leaf belief scatter versus model-index gather."""

from __future__ import annotations

import argparse
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


def _bench_mode(args: argparse.Namespace, *, scatter: bool) -> dict[str, float | int]:
    os.environ["P2_FUSED_MODEL_LEAF_SCATTER"] = "1" if scatter else "0"
    loop_args = _make_loop_args(args)
    ev, rows = _make_evaluator(loop_args)
    device = ev.device
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    with torch.no_grad():
        ev._regret_match_current_policy(1)
    direct = torch.index_select(ev.beliefs, 0, ev.model_indices.contiguous())
    model_beliefs = ev._model_beliefs_for_values(ev.beliefs)
    _sync(device)
    max_diff = float((model_beliefs - direct).abs().max().item())

    gather_ms = _event_time_ms(
        lambda: ev._model_beliefs_for_values(ev.beliefs),
        iters=args.gather_iters,
        device=device,
    )

    t = 2

    def policy_plus_model_beliefs() -> None:
        nonlocal t
        with torch.no_grad():
            ev._regret_match_current_policy(t)
            ev._model_beliefs_for_values(ev.beliefs)
        t += 1

    for _ in range(args.warmup_iters):
        policy_plus_model_beliefs()
    policy_plus_gather_ms = _event_time_ms(
        policy_plus_model_beliefs,
        iters=args.policy_iters,
        device=device,
    )

    return {
        "scatter": int(scatter),
        "rows": int(rows),
        "total_nodes": int(ev.total_nodes),
        "tree_depth": int(ev.tree_depth),
        "model_indices": int(ev.model_indices.numel()),
        "max_diff": max_diff,
        "model_beliefs_ms": gather_ms,
        "policy_plus_model_beliefs_ms": policy_plus_gather_ms,
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
    parser.add_argument(
        "--range-hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num-value-layers",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num-policy-layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--compile",
        choices=("off", "default", "max-autotune"),
        default="default",
    )
    parser.add_argument("--no-closing-checkpoint", action="store_true")
    parser.add_argument("--skip-load-weights", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="preflop_backward_induction")
    parser.add_argument("--gather-iters", type=int, default=1000)
    parser.add_argument("--policy-iters", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("/tmp/preflop_leaf_scatter.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    with _pause_processes(not args.no_pause, args.pause_pattern):
        results.append(_bench_mode(args, scatter=False))
        results.append(_bench_mode(args, scatter=True))
    for row in results:
        print(
            "scatter={scatter} rows={rows} nodes={total_nodes} model={model_indices} "
            "diff={max_diff:.3g} model_beliefs={model_beliefs_ms:.6f}ms "
            "policy_plus_model_beliefs={policy_plus_model_beliefs_ms:.6f}ms".format(
                **row
            ),
            flush=True,
        )
    args.out.write_text(__import__("json").dumps({"results": results}, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
