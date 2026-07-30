"""Round-robin evaluation ladder over checkpoints and CFR search budgets.

Why a ladder rather than adjacent-checkpoint matches: the surviving v3
checkpoints are all clustered at the end of training (10000, 12750, 15000), and
a 10000-vs-12750 match measured -3.81 +/- 24.17 bb/100 over 3072 hands -- an
interval spanning [-51, +44]. Gaps that small are unaffordable to resolve. So
the rungs here are deliberately spaced by two axes that *do* produce large,
resolvable differences:

* **Training time**: an at-init control (untrained leaf values, identical search
  machinery) and the 10000-step checkpoint.
* **Search budget**: the final 15000-step checkpoint at four CFR iteration
  counts.

Putting both axes on one ladder answers the question that decides whether
exploitability is a meaningful model metric at all: is a step of training worth
more or less than a CFR iteration? If search budget dominates, then the
exploitability of a depth-limited search agent is mostly a measurement of its
iteration count, not of the value net underneath it.

The default ladder is now the training-time axis over the full v3 lineage --
init, 10k, 15k, 22k, every one of them at 300 CFR iterations. The search-budget
rungs remain available by name via ``--rungs`` for the attribution question.

Every game is written to JSONL so ratings can be refit offline without
replaying poker. Matchups are resumable: a completed matchup is skipped on a
re-run, so this can be interrupted.

Measured throughput (A100, 10k@300 vs 15k@300, steady-state marginal rate
excluding agent load), which is what the defaults are set from:

    num_envs   256 -> 5.2 hands/s
    num_envs   512 -> 6.2 hands/s
    num_envs  1024 -> 6.3 hands/s

Returns are flat past 512 and the card is nowhere near full, so this is not
memory-bound. ``torch.compile`` was measured too: with ``dynamic=True`` it
recompiles only a handful of times and breaks no graphs, but it changes the
steady-state rate by nothing and costs ~160s of warmup *per matchup*. The leaf
model forward is simply not the bottleneck -- the fused Triton CFR kernels are.
Leave ``--compile off`` unless that changes.

Throughput is strongly matchup-dependent: matches involving the untrained
control run far faster because those games end sooner, so never compare rates
across different matchups.

Usage:
    uv run python scripts/eval_ladder.py --out-dir eval_runs/ladder_v1 --batches 8
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from p2.eval.checkpoints import SearchFidelity, load_search_agent
from p2.eval.duplicate_match import play_duplicate_match, pool_results
from p2.eval.records import RecordWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHORS = REPO_ROOT / "eval_anchors"
RESOLVED_CONFIG = ANCHORS / "v3_resolved_config.json"

CKPT_INIT = ANCHORS / "v3_init_seed0.pt"
CKPT_10K = ANCHORS / "checkpoints-rebel-hu-context-v3@rebel_step_10000.pt"
CKPT_15K = ANCHORS / "checkpoints-rebel-hu-context-v3-to15k@rebel_step_15000.pt"
CKPT_22K = ANCHORS / "checkpoints-rebel-hu-context-v3-to22k@rebel_step_22000.pt"

# The terminal training fidelity of this lineage; the reference iteration count.
TERMINAL_ITERS = 300


@dataclass(frozen=True)
class Rung:
    """One ladder entry: a checkpoint played at a pinned search budget."""

    name: str
    checkpoint: Path
    cfr_iterations: int

    def fidelity(self) -> SearchFidelity:
        # Keep the lineage's training values (warm start 10, DCFR+ delay 80)
        # whenever they fit under the iteration count -- that is what "fidelity"
        # means here, and a rung at the terminal budget must reproduce training
        # exactly. Only the deliberately-starved low-K rungs scale down, and
        # they must: a delay at or above K would disable DCFR+ entirely and a
        # warm start above K gets clamped by the evaluator, either of which
        # would silently change what those rungs measure.
        iters = self.cfr_iterations
        return SearchFidelity(
            cfr_iterations=iters,
            warm_start_iterations=10 if iters > 40 else max(1, iters // 4),
            dcfr_delay=80 if iters > 160 else max(1, iters // 4),
        )


def all_rungs() -> list[Rung]:
    """Every rung this script knows about, on both axes.

    The training-time rungs all sit at ``TERMINAL_ITERS`` so that a match
    between two of them varies only the value net. The search-budget rungs all
    share one checkpoint so that a match between two of *them* varies only the
    iteration count. Mixing the two axes in a single match is legitimate but
    confounded, and the reported number should be read accordingly.
    """
    return [
        # Training-time axis (fixed search budget).
        Rung("init@300", CKPT_INIT, TERMINAL_ITERS),
        Rung("10k@300", CKPT_10K, TERMINAL_ITERS),
        Rung("15k@300", CKPT_15K, TERMINAL_ITERS),
        Rung("22k@300", CKPT_22K, TERMINAL_ITERS),
        # Search-budget axis (fixed checkpoint).
        Rung("15k@10", CKPT_15K, 10),
        Rung("15k@30", CKPT_15K, 30),
        Rung("15k@100", CKPT_15K, 100),
    ]


def default_rungs() -> list[Rung]:
    """The training-time ladder: init -> 10k -> 15k -> 22k, all at 300 iters."""
    keep = {"init@300", "10k@300", "15k@300", "22k@300"}
    return [rung for rung in all_rungs() if rung.name in keep]


@contextlib.contextmanager
def loaded(rung: Rung, device: str, num_envs: int, compile_mode: str = "off"):
    """Load a rung's agent, yield it, and free the GPU memory afterwards.

    Agents are loaded per matchup rather than all at once: each one carries a
    trainer, a model and a CFR subgame arena sized for ``num_envs``, and holding
    six of those resident would dominate the card for no benefit.
    """
    handle = load_search_agent(
        rung.checkpoint,
        resolved_config=RESOLVED_CONFIG,
        device=device,
        fidelity=rung.fidelity(),
        name=rung.name,
        num_envs=num_envs,
        compile_mode=compile_mode,
    )
    evaluator_type = type(handle.trainer.cfr_evaluator).__name__
    if "Fused" not in evaluator_type:
        raise RuntimeError(
            f"expected the fused sparse CFR evaluator, got {evaluator_type}. "
            "The ladder must not mix evaluator implementations across rungs."
        )
    try:
        yield handle
    finally:
        del handle
        gc.collect()
        torch.cuda.empty_cache()


def run_matchup(
    rung_a: Rung,
    rung_b: Rung,
    *,
    device: str,
    pairs_per_batch: int,
    batches: int,
    num_envs: int,
    seed: int,
    out_dir: Path,
    compile_mode: str = "off",
) -> dict:
    """Play one matchup, streaming per-game records to JSONL."""
    tag = f"{rung_a.name}__vs__{rung_b.name}".replace("@", "at")
    records_path = out_dir / f"{tag}.jsonl"
    # The writer appends, and only *completed* matchups are skipped on resume,
    # so a matchup interrupted mid-way must start from an empty file rather than
    # leave half a run's games duplicated in the record set.
    records_path.unlink(missing_ok=True)
    started = time.time()

    with loaded(rung_a, device, num_envs, compile_mode) as handle_a:
        with loaded(rung_b, device, num_envs, compile_mode) as handle_b:
            writer = RecordWriter(records_path)
            parts = []
            for batch in range(batches):
                part = play_duplicate_match(
                    handle_a.agent,
                    handle_b.agent,
                    handle_a.env_proto,
                    num_pairs=pairs_per_batch,
                    seed=seed + batch,
                    device=torch.device(device),
                    recorder=writer,
                    eval_id=tag,
                    extra_manifest={
                        "rung_a": asdict(rung_a)
                        | {"checkpoint": str(rung_a.checkpoint)},
                        "rung_b": asdict(rung_b)
                        | {"checkpoint": str(rung_b.checkpoint)},
                    },
                )
                parts.append(part)
                pooled = pool_results(parts)
                elapsed = time.time() - started
                print(
                    f"  [{tag}] batch {batch + 1}/{batches}: {pooled.summary()} "
                    f"({pooled.num_games / elapsed:.1f} hands/s)",
                    flush=True,
                )
            pooled = pool_results(parts)

    elapsed = time.time() - started
    return {
        "matchup": tag,
        "a": rung_a.name,
        "b": rung_b.name,
        "mean_bb_per_100": pooled.mean_bb_per_100,
        "se_bb_per_100": pooled.se_bb_per_100,
        "num_games": pooled.num_games,
        "num_pairs": pooled.num_pairs,
        "seconds": elapsed,
        "hands_per_second": pooled.num_games / elapsed,
        "records": str(records_path),
    }


def resolved(row: dict) -> str:
    """Sign of the result if its 95% interval excludes zero, else 'unresolved'."""
    mean, se = row["mean_bb_per_100"], row["se_bb_per_100"]
    if not math.isfinite(se):
        return "unresolved"
    lo, hi = mean - 1.959964 * se, mean + 1.959964 * se
    if hi < 0:
        return "B wins"
    if lo > 0:
        return "A wins"
    return "unresolved"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument(
        "--pairs-per-batch",
        type=int,
        default=512,
        help="duplicate pairs per batch; the match env holds 2x this many envs",
    )
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--compile",
        default="off",
        choices=["off", "default", "static", "reduce-overhead", "max-autotune"],
        help=(
            "torch.compile mode for the leaf model. Anything but 'static' "
            "compiles with dynamic=True. Enables recompile logging."
        ),
    )
    parser.add_argument(
        "--rungs",
        nargs="+",
        default=None,
        help=(
            "rung names to play round-robin, overriding the default "
            "training-time ladder; see all_rungs() for the full set"
        ),
    )
    args = parser.parse_args(argv)

    if args.compile != "off":
        # The active set shrinks every decision round, so a new batch size shows
        # up on nearly every call. dynamic=True is supposed to keep that to one
        # graph; this logging is how we find out whether it actually does rather
        # than silently recompiling per shape and losing more than we gain.
        torch._logging.set_logs(recompiles=True, graph_breaks=True)
        # The fused evaluator swallows compile failures in a bare `except`, so
        # a compiled run that silently fell back would otherwise look like a
        # real "compile does not help" measurement.
        torch._dynamo.config.suppress_errors = False
        print(f"torch.compile={args.compile} (dynamic), recompile logging on")

    rungs = default_rungs()
    if args.rungs:
        by_name = {rung.name: rung for rung in all_rungs()}
        unknown = sorted(set(args.rungs) - by_name.keys())
        if unknown:
            raise SystemExit(f"unknown rungs {unknown}; have {sorted(by_name)}")
        rungs = [by_name[name] for name in args.rungs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "matchups.json"
    rows: list[dict] = []
    if summary_path.exists():
        rows = json.loads(summary_path.read_text())
        print(f"resuming: {len(rows)} matchups already complete")
    done = {row["matchup"] for row in rows}

    for rung_a, rung_b in itertools.combinations(rungs, 2):
        tag = f"{rung_a.name}__vs__{rung_b.name}".replace("@", "at")
        if tag in done:
            print(f"skip {tag} (already complete)")
            continue
        print(f"=== {rung_a.name} vs {rung_b.name} ===", flush=True)
        row = run_matchup(
            rung_a,
            rung_b,
            device=args.device,
            pairs_per_batch=args.pairs_per_batch,
            batches=args.batches,
            num_envs=args.num_envs,
            seed=args.seed,
            out_dir=args.out_dir,
            compile_mode=args.compile,
        )
        rows.append(row)
        # Written after every matchup so an interrupted run resumes cleanly.
        summary_path.write_text(json.dumps(rows, indent=2))

    print()
    header = f"{'matchup':<34}{'bb/100':>10}{'SE':>9}{'hands':>9}  verdict"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['a'] + ' vs ' + row['b']:<34}"
            f"{row['mean_bb_per_100']:>+10.2f}{row['se_bb_per_100']:>9.2f}"
            f"{row['num_games']:>9,}  {resolved(row)}"
        )
    print(f"\nwrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
