#!/usr/bin/env python3
"""Benchmark the public-preflop policy reach/belief propagation specialization."""

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
        raise RuntimeError("CUDA is required")
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


def _prepare_evaluator(args: argparse.Namespace):
    ev, rows = _make_evaluator(_make_loop_args(args))
    if ev.device.type != "cuda":
        raise RuntimeError("CUDA is required")
    with torch.no_grad():
        ev.initialize_policy_and_beliefs()
        ev._ensure_fused_attrs()
        ev._prepare_tree_slices()
    _sync(ev.device)
    return ev, rows


def _run_propagation(ev, *, public: bool) -> None:
    old = os.environ.get("P2_PREFLOP_PUBLIC_POLICY_REACH_BELIEFS")
    os.environ["P2_PREFLOP_PUBLIC_POLICY_REACH_BELIEFS"] = "1" if public else "0"
    try:
        ev._renormalize_policy_reach_beliefs(
            ev.policy_probs,
            ev.self_reach,
            ev.beliefs,
        )
    finally:
        if old is None:
            os.environ.pop("P2_PREFLOP_PUBLIC_POLICY_REACH_BELIEFS", None)
        else:
            os.environ["P2_PREFLOP_PUBLIC_POLICY_REACH_BELIEFS"] = old


def run(args: argparse.Namespace) -> dict[str, object]:
    ev_old, rows = _prepare_evaluator(args)
    ev_new, _ = _prepare_evaluator(args)
    ev_new.policy_probs.copy_(ev_old.policy_probs)
    ev_new.self_reach.copy_(ev_old.self_reach)
    ev_new.beliefs.copy_(ev_old.beliefs)

    with torch.no_grad():
        _run_propagation(ev_old, public=False)
        _run_propagation(ev_new, public=True)
    _sync(ev_old.device)
    policy_diff = float((ev_new.policy_probs - ev_old.policy_probs).abs().max().item())
    reach_diff = float((ev_new.self_reach - ev_old.self_reach).abs().max().item())
    belief_diff = float((ev_new.beliefs - ev_old.beliefs).abs().max().item())
    leaf_diff = 0.0
    if ev_old._model_leaf_beliefs_buf is not None and ev_new._model_leaf_beliefs_buf is not None:
        leaf_diff = float(
            (ev_new._model_leaf_beliefs_buf - ev_old._model_leaf_beliefs_buf)
            .abs()
            .max()
            .item()
        )

    ev_old_t, _ = _prepare_evaluator(args)
    ev_new_t, _ = _prepare_evaluator(args)
    for _ in range(args.warmup_iters):
        _run_propagation(ev_old_t, public=False)
        _run_propagation(ev_new_t, public=True)

    old_ms = _event_time_ms(
        lambda: _run_propagation(ev_old_t, public=False),
        iters=args.iters,
        device=ev_old_t.device,
    )
    new_ms = _event_time_ms(
        lambda: _run_propagation(ev_new_t, public=True),
        iters=args.iters,
        device=ev_new_t.device,
    )
    depth_offsets = [int(x) for x in ev_old.depth_offsets]
    return {
        "rows": int(rows),
        "root_nodes": int(ev_old.root_nodes),
        "total_nodes": int(ev_old.total_nodes),
        "tree_depth": int(ev_old.tree_depth),
        "depth_offsets": depth_offsets,
        "child_count_by_depth": [int(t.numel()) for t in ev_old._child_count_by_depth],
        "child_sum_by_depth": [int(t.sum().item()) for t in ev_old._child_count_by_depth],
        "child_count_pow2_by_depth": [int(x) for x in ev_old._child_count_pow2_by_depth],
        "allowed_all": bool(ev_old._preflop_all_hands_allowed),
        "policy_max_diff": policy_diff,
        "reach_max_diff": reach_diff,
        "belief_max_diff": belief_diff,
        "model_leaf_belief_max_diff": leaf_diff,
        "generic_ms": old_ms,
        "public_ms": new_ms,
        "speedup": old_ms / new_ms if new_ms > 0 else float("inf"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--closing-checkpoint", type=Path, default=DEFAULT_CLOSING_CHECKPOINT)
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
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/preflop_policy_reach_public.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with _pause_processes(not args.no_pause, args.pause_pattern):
        result = run(args)
    print(
        "nodes={total_nodes} depths={child_count_by_depth} "
        "generic={generic_ms:.6f}ms public={public_ms:.6f}ms "
        "speedup={speedup:.3f} diffs policy/reach/belief/leaf="
        "{policy_max_diff:.3g}/{reach_max_diff:.3g}/{belief_max_diff:.3g}/"
        "{model_leaf_belief_max_diff:.3g}".format(**result),
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
