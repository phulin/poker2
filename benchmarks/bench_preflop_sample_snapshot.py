#!/usr/bin/env python3
"""Benchmark preflop sample snapshot updates on the realistic actions_4_7 shape."""

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

from p2.search.fused_cfr_triton import (  # noqa: E402
    fused_preflop_sample_snapshot_multiway_,
)
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


def _prepare_evaluator(args: argparse.Namespace):
    ev, rows = _make_evaluator(_make_loop_args(args))
    if ev.device.type != "cuda":
        raise RuntimeError("CUDA is required")
    with torch.no_grad():
        ev.initialize_policy_and_beliefs()
        ev.t_sample = ev._get_sampling_schedule()
        ev._ensure_fused_attrs()
    root_schedule = ev.t_sample[: ev.root_nodes]
    unique, counts = root_schedule.unique(return_counts=True)
    t = int(unique[counts.argmax()].item())
    ev._t_scalars.t_tensor.fill_(t)
    return ev, rows, t, int(counts.max().item())


def _old_snapshot(ev) -> None:
    sample_mask = ev.t_sample == ev._t_scalars.t_tensor
    torch.where(
        sample_mask[:, None],
        ev.policy_probs,
        ev.policy_probs_sample,
        out=ev.policy_probs_sample,
    )
    torch.where(
        sample_mask[:, None, None],
        ev.beliefs,
        ev.beliefs_sample,
        out=ev.beliefs_sample,
    )


def _new_snapshot(ev) -> None:
    fused_preflop_sample_snapshot_multiway_(
        ev.policy_probs,
        ev.policy_probs_sample,
        ev.beliefs,
        ev.beliefs_sample,
        ev.t_sample.contiguous(),
        ev._t_scalars.t_tensor,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    ev_old, rows, t, sampled_roots = _prepare_evaluator(args)
    ev_new, _rows, _t, _sampled_roots = _prepare_evaluator(args)
    ev_new.t_sample.copy_(ev_old.t_sample)
    ev_new.policy_probs_sample.copy_(ev_old.policy_probs_sample)
    ev_new.beliefs_sample.copy_(ev_old.beliefs_sample)
    ev_new._t_scalars.t_tensor.fill_(t)

    with torch.no_grad():
        _old_snapshot(ev_old)
        _new_snapshot(ev_new)
    _sync(ev_old.device)
    policy_diff = float(
        (ev_new.policy_probs_sample - ev_old.policy_probs_sample).abs().max().item()
    )
    belief_diff = float(
        (ev_new.beliefs_sample - ev_old.beliefs_sample).abs().max().item()
    )

    ev_old_t, _rows, _t, _sampled_roots = _prepare_evaluator(args)
    ev_new_t, _rows, _t, _sampled_roots = _prepare_evaluator(args)
    ev_new_t.t_sample.copy_(ev_old_t.t_sample)
    ev_new_t._t_scalars.t_tensor.fill_(t)
    ev_old_t._t_scalars.t_tensor.fill_(t)

    for _ in range(args.warmup_iters):
        _old_snapshot(ev_old_t)
        _new_snapshot(ev_new_t)

    old_ms = _event_time_ms(
        lambda: _old_snapshot(ev_old_t),
        iters=args.iters,
        device=ev_old_t.device,
    )
    new_ms = _event_time_ms(
        lambda: _new_snapshot(ev_new_t),
        iters=args.iters,
        device=ev_new_t.device,
    )
    return {
        "rows": int(rows),
        "root_nodes": int(ev_old.root_nodes),
        "total_nodes": int(ev_old.total_nodes),
        "tree_depth": int(ev_old.tree_depth),
        "model_indices": int(ev_old.model_indices.numel()),
        "t": t,
        "sampled_roots_at_t": sampled_roots,
        "policy_max_diff": policy_diff,
        "belief_max_diff": belief_diff,
        "old_where_ms": old_ms,
        "fused_snapshot_ms": new_ms,
        "speedup": old_ms / new_ms if new_ms > 0 else float("inf"),
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
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/preflop_sample_snapshot.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with _pause_processes(not args.no_pause, args.pause_pattern):
        result = run(args)
    print(
        "rows={rows} nodes={total_nodes} sampled_roots={sampled_roots_at_t} "
        "old={old_where_ms:.6f}ms fused={fused_snapshot_ms:.6f}ms "
        "speedup={speedup:.3f} policy_diff={policy_max_diff:.3g} "
        "belief_diff={belief_max_diff:.3g}".format(**result),
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
