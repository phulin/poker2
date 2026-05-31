"""Microbenchmark for all-in training data generation.

Measures the per-step cost of producing one training batch the way
``p2.allin.train`` does: draw a random preflop all-in batch, then estimate
its Monte Carlo terminal-value targets. Target estimation dominates, so the
batch-generation and target-estimation costs are timed separately.

The timed estimation loop uses ``compute_stats=False`` to mirror the common
(non-logging) training step, which skips the per-step host reductions and
device sync; the benchmark adds its own synchronize around the timed region.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from p2.allin.data import make_random_preflop_allin_batch
from p2.allin.kernels import triton_available
from p2.allin.sampler import (
    PreflopAllInEstimatorWorkspace,
    estimate_preflop_allin_values,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="64",
        help="Comma-separated training batch sizes (roots per step)",
    )
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--sample-count", type=int, default=32_768)
    parser.add_argument("--board-samples", type=int, default=2_048)
    parser.add_argument("--board-chunk", type=int, default=64)
    parser.add_argument("--tuple-tries", type=int, default=4)
    parser.add_argument("--hand-chunk", type=int, default=128)
    parser.add_argument("--bb", type=int, default=100)
    parser.add_argument(
        "--no-exact-two-player",
        action="store_true",
        help="Force the MC sampler for every row (skip the exact 2-player table)",
    )
    parser.add_argument(
        "--persistent-workspace",
        action="store_true",
        help="Reuse the target-estimator workspace across benchmark repeats.",
    )
    parser.add_argument(
        "--legacy-board-allowed",
        action="store_true",
        help="Use the legacy materialized [rows, 1326] board-allowed matrix path.",
    )
    parser.add_argument(
        "--no-skip-folded-heroes",
        action="store_true",
        help="Do not early-return folded heroes in the compact board-mask scorer.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment.")
    if not triton_available():
        raise SystemExit("Triton is not installed in this environment.")

    device = torch.device("cuda")
    tuple_samples = (args.sample_count + args.board_samples - 1) // args.board_samples
    use_exact = not args.no_exact_two_player
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(
        f"players={args.players} sample_count={args.sample_count:,} "
        f"board_samples={args.board_samples:,} tuple_samples={tuple_samples} "
        f"board_chunk={args.board_chunk} tuple_tries={args.tuple_tries} "
        f"exact_two_player={use_exact} "
        f"persistent_workspace={args.persistent_workspace} "
        f"board_masks={not args.legacy_board_allowed} "
        f"skip_folded_heroes={not args.no_skip_folded_heroes}"
    )
    print(f"Warmup: {args.warmup}, repeats: {args.repeats}")

    def estimate(batch, *, compute_stats):
        return estimate_preflop_allin_values(
            batch,
            sample_count=args.sample_count,
            board_samples=args.board_samples,
            tuple_samples=None,
            tuple_tries=args.tuple_tries,
            board_chunk=args.board_chunk,
            hand_chunk=args.hand_chunk,
            generator=generator,
            use_exact_two_player=use_exact,
            compute_stats=compute_stats,
            workspace=workspace,
            use_board_masks=not args.legacy_board_allowed,
            skip_folded_heroes=not args.no_skip_folded_heroes,
        )

    for batch_size in [int(x) for x in args.batch_sizes.split(",") if x.strip()]:
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_size)
        workspace = (
            PreflopAllInEstimatorWorkspace() if args.persistent_workspace else None
        )

        def make_batch():
            return make_random_preflop_allin_batch(
                batch_size,
                args.players,
                bb=args.bb,
                device=device,
                generator=generator,
            )

        # Time batch generation on its own.
        for _ in range(args.warmup):
            batch = make_batch()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.repeats):
            batch = make_batch()
        synchronize(device)
        batch_gen_time = (time.perf_counter() - start) / args.repeats

        # One stats-on call for sanity diagnostics (and to warm the kernels).
        _, diag = estimate(batch, compute_stats=True)

        for _ in range(args.warmup):
            estimate(batch, compute_stats=False)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.repeats):
            estimate(batch, compute_stats=False)
        synchronize(device)
        target_time = (time.perf_counter() - start) / args.repeats

        total_time = batch_gen_time + target_time
        boards_per_s = batch_size * args.board_samples / target_time
        samples_per_s = batch_size * args.sample_count / target_time
        exact_rows = diag.get("target_exact_two_player_rows", 0.0)
        mc_rows = diag.get("target_mc_rows", float(batch_size))
        print(
            f"batch={batch_size:>5,d} | "
            f"batch_gen={batch_gen_time * 1e3:>7.2f} ms | "
            f"target={target_time * 1e3:>9.2f} ms | "
            f"total={total_time * 1e3:>9.2f} ms | "
            f"{boards_per_s:>12,.0f} boards/s | "
            f"{samples_per_s / 1e6:>8.2f} M samples/s | "
            f"exact/mc rows={exact_rows:.0f}/{mc_rows:.0f} | "
            f"val_mean={diag.get('target_value_mean', float('nan')):+.4f}"
        )


if __name__ == "__main__":
    main()
