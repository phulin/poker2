#!/usr/bin/env python3
"""Benchmark exact 3-player preflop all-in denominator production path."""

from __future__ import annotations

import argparse
import json
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


def _prepare_inputs(args: argparse.Namespace):
    ev, rows = _make_evaluator(_make_loop_args(args))
    if ev.device.type != "cuda":
        raise RuntimeError("CUDA is required")
    with torch.no_grad():
        ev.initialize_policy_and_beliefs()
        ev._regret_match_current_policy(1)
        ev._cache_preflop_allin_live_partitions()
        oracle = ev._ensure_preflop_allin_169_oracle()
        entry_nodes = ev.preflop_allin_live3_entry_nodes
        if args.max_entries is not None:
            entry_nodes = entry_nodes[: args.max_entries]
            hero_players = ev.preflop_allin_live3_hero_players[: args.max_entries]
            opp0_players = ev.preflop_allin_live3_opp0_players[: args.max_entries]
            opp1_players = ev.preflop_allin_live3_opp1_players[: args.max_entries]
        else:
            hero_players = ev.preflop_allin_live3_hero_players
            opp0_players = ev.preflop_allin_live3_opp0_players
            opp1_players = ev.preflop_allin_live3_opp1_players
        hero = ev._preflop_allin_entry_beliefs(
            ev.beliefs,
            entry_nodes,
            hero_players,
            "_bench_share3_hero",
        )
        opp0 = ev._preflop_allin_entry_beliefs(
            ev.beliefs,
            entry_nodes,
            opp0_players,
            "_bench_share3_opp0",
        )
        opp1 = ev._preflop_allin_entry_beliefs(
            ev.beliefs,
            entry_nodes,
            opp1_players,
            "_bench_share3_opp1",
        )
    return ev, oracle, hero, opp0, opp1, rows


def _share3_dense_reference(oracle, hero, opp0, opp1) -> torch.Tensor:
    with torch.no_grad():
        oracle._share3_values(hero[:1], opp0[:1], opp1[:1])
        assert oracle._share3_weighted_flat is not None
        assert oracle._compat3_flat is not None
        opp0_per_combo = opp0.to(torch.float32) / oracle._multiplicity
        opp1_per_combo = opp1.to(torch.float32) / oracle._multiplicity
        outer = (opp0_per_combo[:, :, None] * opp1_per_combo[:, None, :]).flatten(1)
        numer = outer @ oracle._share3_weighted_flat
        denom = outer @ oracle._compat3_flat
        return numer / denom.clamp_min(1.0e-8)


def _call_share3(oracle, hero, opp0, opp1) -> torch.Tensor:
    with torch.no_grad():
        return oracle._share3_values(hero, opp0, opp1)


def run(args: argparse.Namespace) -> dict[str, object]:
    ev, oracle, hero, opp0, opp1, rows = _prepare_inputs(args)
    out = _call_share3(oracle, hero, opp0, opp1)
    _sync(ev.device)
    ref = _share3_dense_reference(oracle, hero, opp0, opp1)
    diff = float((out - ref).abs().max().item())
    for _ in range(args.warmup_iters):
        _call_share3(oracle, hero, opp0, opp1)
    timing = _event_time_ms(
        lambda: _call_share3(oracle, hero, opp0, opp1),
        iters=args.iters,
        device=ev.device,
    )
    return {
        "rows": int(rows),
        "total_nodes": int(ev.total_nodes),
        "live3_nodes": int(ev.preflop_allin_indices_by_live_count[3].numel()),
        "live3_entries": int(hero.shape[0]),
        "iters": int(args.iters),
        "warmup_iters": int(args.warmup_iters),
        "max_diff_vs_dense_reference": diff,
        "share3_ms": timing,
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
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/preflop_allin_share3_denom.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with _pause_processes(not args.no_pause, args.pause_pattern):
        result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
