#!/usr/bin/env python3
"""Benchmark skipping full reach/belief stores for compact preflop model leaves."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

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


def _make_mode_evaluator(
    args: argparse.Namespace, *, skip_store: bool
) -> tuple[Any, int]:
    os.environ["P2_FUSED_MODEL_LEAF_SCATTER"] = "1"
    os.environ["P2_PREFLOP_SKIP_MODEL_LEAF_BELIEF_STORE"] = (
        "1" if skip_store else "0"
    )
    os.environ["P2_PREFLOP_SKIP_MODEL_LEAF_REACH_STORE"] = (
        "1" if skip_store else "0"
    )
    return _make_evaluator(_make_loop_args(args))


def _make_short_solve_evaluator(
    args: argparse.Namespace, *, skip_store: bool
) -> tuple[Any, int]:
    loop_args = _make_loop_args(args)
    loop_args.cfr_iterations = 2
    os.environ["P2_FUSED_MODEL_LEAF_SCATTER"] = "1"
    os.environ["P2_PREFLOP_SKIP_MODEL_LEAF_BELIEF_STORE"] = (
        "1" if skip_store else "0"
    )
    os.environ["P2_PREFLOP_SKIP_MODEL_LEAF_REACH_STORE"] = (
        "1" if skip_store else "0"
    )
    return _make_evaluator(loop_args)


def _leaf_values_after_one_update(
    args: argparse.Namespace, *, skip_store: bool
) -> tuple[torch.Tensor, torch.Tensor, Any, int]:
    ev, rows = _make_mode_evaluator(args, skip_store=skip_store)
    with torch.no_grad():
        ev._regret_match_current_policy(1)
        model_beliefs = ev._model_beliefs_for_values(ev.beliefs).detach().clone()
        ev.set_leaf_values(1)
        leaf_values = ev.latest_values[ev.leaf_mask].detach().clone()
    _sync(ev.device)
    return leaf_values, model_beliefs, ev, rows


def _correctness(args: argparse.Namespace) -> dict[str, float | int]:
    leaf_off, beliefs_off, ev_off, rows = _leaf_values_after_one_update(
        args, skip_store=False
    )
    leaf_on, beliefs_on, _ev_on, _rows_on = _leaf_values_after_one_update(
        args, skip_store=True
    )
    old_graph = os.environ.get("P2_PREFLOP_CUDA_GRAPH_EVALUATE")
    os.environ["P2_PREFLOP_CUDA_GRAPH_EVALUATE"] = "0"
    try:
        ev_iter_off, _rows = _make_short_solve_evaluator(args, skip_store=False)
        ev_iter_on, _rows = _make_short_solve_evaluator(args, skip_store=True)
        with torch.no_grad():
            ev_iter_off.evaluate_cfr(training_mode=True, sample_continuation=False)
            ev_iter_on.evaluate_cfr(training_mode=True, sample_continuation=False)
        _sync(ev_iter_off.device)
    finally:
        if old_graph is None:
            os.environ.pop("P2_PREFLOP_CUDA_GRAPH_EVALUATE", None)
        else:
            os.environ["P2_PREFLOP_CUDA_GRAPH_EVALUATE"] = old_graph
    policy_diff = float(
        (ev_iter_on.policy_probs - ev_iter_off.policy_probs).abs().max().item()
    )
    latest_diff = float(
        (ev_iter_on.latest_values - ev_iter_off.latest_values).abs().max().item()
    )
    values_avg_diff = float(
        (ev_iter_on.values_avg - ev_iter_off.values_avg).abs().max().item()
    )
    cumulative_regret_diff = float(
        (ev_iter_on.cumulative_regrets - ev_iter_off.cumulative_regrets)
        .abs()
        .max()
        .item()
    )

    return {
        "rows": int(rows),
        "root_nodes": int(ev_off.root_nodes),
        "total_nodes": int(ev_off.total_nodes),
        "tree_depth": int(ev_off.tree_depth),
        "model_indices": int(ev_off.model_indices.numel()),
        "showdown_indices": int(ev_off.showdown_indices.numel()),
        "leaf_value_max_diff": float((leaf_on - leaf_off).abs().max().item()),
        "model_belief_max_diff": float((beliefs_on - beliefs_off).abs().max().item()),
        "cfr_iteration_policy_max_diff": policy_diff,
        "cfr_iteration_latest_values_max_diff": latest_diff,
        "cfr_iteration_values_avg_max_diff": values_avg_diff,
        "cfr_iteration_cumulative_regrets_max_diff": cumulative_regret_diff,
    }


def _bench_mode(args: argparse.Namespace, *, skip_store: bool) -> dict[str, float | int]:
    ev, rows = _make_mode_evaluator(args, skip_store=skip_store)
    if ev.device.type != "cuda":
        raise RuntimeError("CUDA is required")

    t_policy = 1

    def policy_update() -> None:
        nonlocal t_policy
        with torch.no_grad():
            ev._regret_match_current_policy(t_policy)
        t_policy += 1

    for _ in range(args.warmup_iters):
        policy_update()
    policy_ms = _event_time_ms(
        policy_update,
        iters=args.policy_iters,
        device=ev.device,
    )

    ev2, _rows2 = _make_mode_evaluator(args, skip_store=skip_store)
    t_values = 1

    def policy_plus_values() -> None:
        nonlocal t_values
        with torch.no_grad():
            ev2._regret_match_current_policy(t_values)
            ev2.set_leaf_values(t_values)
        t_values += 1

    for _ in range(args.value_warmup_iters):
        policy_plus_values()
    policy_plus_values_ms = _event_time_ms(
        policy_plus_values,
        iters=args.value_iters,
        device=ev2.device,
    )

    return {
        "skip_store": int(skip_store),
        "rows": int(rows),
        "root_nodes": int(ev.root_nodes),
        "total_nodes": int(ev.total_nodes),
        "tree_depth": int(ev.tree_depth),
        "model_indices": int(ev.model_indices.numel()),
        "policy_update_ms": policy_ms,
        "policy_plus_set_leaf_values_ms": policy_plus_values_ms,
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
    parser.add_argument("--policy-iters", type=int, default=200)
    parser.add_argument("--value-iters", type=int, default=30)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--value-warmup-iters", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/preflop_skip_model_leaf_belief_store.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with _pause_processes(not args.no_pause, args.pause_pattern):
        correctness = _correctness(args)
        results = [
            _bench_mode(args, skip_store=False),
            _bench_mode(args, skip_store=True),
        ]

    for row in results:
        print(
            "skip_store={skip_store} rows={rows} nodes={total_nodes} "
            "model={model_indices} policy={policy_update_ms:.6f}ms "
            "policy_plus_values={policy_plus_set_leaf_values_ms:.6f}ms".format(**row),
            flush=True,
        )
    print(
        "diff leaf_values={leaf_value_max_diff:.3g} "
        "model_beliefs={model_belief_max_diff:.3g}".format(**correctness),
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "correctness": correctness,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
