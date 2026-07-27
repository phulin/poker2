#!/usr/bin/env python3
"""Calibration gates for the real-hand duplicate evaluation harness.

The evaluation system this harness replaces was silently wrong for months
because nothing ever checked it against a known answer. These are the checks
that would have caught it. Every gate is a statement whose expected value is
known *before* the harness runs.

Gates
-----
``fold_exact``
    A fold-bot in the small blind folds preflop and therefore loses exactly
    0.5 bb, whoever it is playing. This is deterministic, so it is asserted
    exactly (float tolerance), not within a standard error. One number checks
    sign convention, chip scale, blind accounting, and seat handling end to
    end. Model-free. If this fails, nothing downstream means anything.

``self_play``
    Identical agents on both sides must score zero. Asserted as
    ``|mean| < 3 SE``. Model-free variants (call/random) plus a real
    checkpoint-vs-itself variant.

``coupling``
    Not a pass/fail assertion but a measurement: raw per-game SD, the SD of
    the per-pair duplicate statistic, and the ratio. Duplicate pairing spends
    2x the compute per paired observation, so it only pays for itself if it
    removes more than 2x the variance (i.e. SD ratio > sqrt(2)). Also reports
    the fraction of pairs whose two halves produced identical public action
    sequences -- CFR is stochastic and GPU-nondeterministic, so this is not
    100% and the real number matters.

``known_gap``
    Two checkpoints from one lineage, widely separated, must show a large
    correctly-signed gap. Reported honestly: if the confidence interval
    straddles zero the gate says "unresolved" and prints the sample size that
    would be needed.

Run everything that is possible in this environment::

    uv run python scripts/eval_calibration_gates.py

Model-free gates only (no GPU, no checkpoint needed)::

    uv run python scripts/eval_calibration_gates.py --model-free
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.eval.agents import CallAgent, FoldAgent, MatchAgent, RandomAgent
from p2.eval.duplicate_match import MatchResult, play_duplicate_match, pool_results

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHORS = REPO_ROOT / "eval_anchors"
RESOLVED_CONFIG_V3 = ANCHORS / "v3_resolved_config.json"
CKPT_EARLY = ANCHORS / "checkpoints-rebel-hu-context-v3@rebel_step_10000.pt"
CKPT_LATE = ANCHORS / "checkpoints-rebel-hu-context-v3-to15k@rebel_step_12750.pt"

# Exactness tolerance for the fold gate. Rewards travel through float32 as a
# fraction of the effective stack and are scaled back to bb, so the error floor
# is float32 epsilon times the stack-to-bb ratio, not zero.
FOLD_ATOL_BB = 1e-6

# A "large" gap the known-gap gate is calibrated to detect, in bb/100.
TARGET_RESOLVABLE_BB100 = 5.0


# ------------------------------------------------------------------- results


@dataclass
class GateResult:
    """One gate outcome. ``status`` is pass / FAIL / info / skip."""

    gate: str
    detail: str
    status: str
    measured: Optional[float] = None
    expected: Optional[str] = None
    se: Optional[float] = None
    hands: Optional[int] = None
    units: str = "bb/100"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _fmt(value: Optional[float], width: int = 12, digits: int = 4) -> str:
    return " " * width if value is None else f"{value:>{width}.{digits}f}"


def print_table(results: Sequence[GateResult]) -> None:
    header = (
        f"{'gate':<12} {'case':<34} {'status':<7} {'measured':>12} "
        f"{'expected':>14} {'SE':>10} {'hands':>8}  units"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        print(
            f"{r.gate:<12} {r.detail:<34} {r.status:<7} {_fmt(r.measured)} "
            f"{(r.expected or ''):>14} {_fmt(r.se, 10, 4)} "
            f"{(str(r.hands) if r.hands is not None else ''):>8}  {r.units}"
        )
        if r.notes:
            print(f"{'':<12} {'':<34}   {r.notes}")
    print("=" * len(header))
    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "skip"]
    print(
        f"{len(results)} rows: "
        f"{sum(r.status == 'pass' for r in results)} pass, {len(failed)} FAIL, "
        f"{sum(r.status == 'info' for r in results)} info, {len(skipped)} skip"
    )
    for r in failed:
        print(f"  FAIL  {r.gate}/{r.detail}: {r.notes}")


# ------------------------------------------------------------------ env protos


def fixed_stack_env(device: torch.device, sb: int = 50, bb: int = 100) -> HUNLTensorEnv:
    """Simplest possible env: fixed 100bb stacks, standard blinds."""
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    return HUNLTensorEnv(
        num_envs=1,
        starting_stack=100 * bb,
        sb=sb,
        bb=bb,
        device=device,
        rng=rng,
    )


def randomized_stack_env(device: torch.device) -> HUNLTensorEnv:
    """Stack-randomizing env, matching how the v3 lineage was trained."""
    rng = torch.Generator(device=device)
    rng.manual_seed(0)
    return HUNLTensorEnv(
        num_envs=1,
        starting_stack=10_000,
        sb=50,
        bb=100,
        default_bet_bins=[0.25, 0.5, 0.75, 1.0, 1.5],
        device=device,
        rng=rng,
        stack_mode="weighted_uniform_bb",
        min_stack_bb=10,
        mid_stack_bb=200,
        max_stack_bb=400,
        high_stack_mass_ratio=1.0 / 3.0,
    )


# ------------------------------------------------------- gate 1: fold-bot exact


def fold_bot_sb_rewards(result: MatchResult) -> list[float]:
    """Agent A's bb reward on the games where agent A held the small blind.

    Heads-up, the button *is* the small blind (``HUNLTensorEnv.reset`` sets
    ``p_sb = button``), and agent A always sits in seat 0, so agent A is the SB
    exactly when ``button == 0``.
    """
    return [r.reward_a_bb for r in result.records if r.button == 0]


def gate_fold_exact(
    device: torch.device, num_pairs: int, seed: int = 1
) -> list[GateResult]:
    """A fold-bot in the SB loses exactly 0.5 bb. Deterministic, so exact."""
    results: list[GateResult] = []
    envs = {
        "fixed-100bb": fixed_stack_env(device),
        "weighted-uniform-bb": randomized_stack_env(device),
    }
    opponents: list[Callable[[], MatchAgent]] = [
        lambda: FoldAgent("fold_opp"),
        CallAgent,
        RandomAgent,
    ]
    for env_name, env_proto in envs.items():
        for make_opponent in opponents:
            opponent = make_opponent()
            result = play_duplicate_match(
                FoldAgent(),
                opponent,
                env_proto,
                num_pairs=num_pairs,
                seed=seed,
                device=device,
            )
            sb = fold_bot_sb_rewards(result)
            worst = max(abs(v + 0.5) for v in sb) if sb else float("inf")
            ok = bool(sb) and worst <= FOLD_ATOL_BB
            mean_bb100 = 100.0 * sum(sb) / max(len(sb), 1)
            results.append(
                GateResult(
                    gate="fold_exact",
                    detail=f"{env_name} vs {opponent.identity.name}",
                    status="pass" if ok else "FAIL",
                    measured=mean_bb100,
                    expected="-50.0000",
                    se=0.0,
                    hands=len(sb),
                    notes=(
                        f"max |reward + 0.5bb| = {worst:.3e} bb (tol {FOLD_ATOL_BB:g})"
                        if not ok
                        else f"exact: max deviation {worst:.2e} bb"
                    ),
                    extra={"max_abs_deviation_bb": worst},
                )
            )
    return results


# -------------------------------------------------- gate 2: self-play symmetry


def gate_self_play(result: MatchResult, detail: str, sigma: float = 3.0) -> GateResult:
    """Identical agents must score within ``sigma`` SE of zero."""
    se = result.se_bb_per_100
    mean = result.mean_bb_per_100
    if not math.isfinite(se) or se == 0.0:
        # Deterministic identical agents cancel exactly; then the assertion is
        # exactness, not a z-test.
        ok = abs(mean) <= 1e-4
        return GateResult(
            gate="self_play",
            detail=detail,
            status="pass" if ok else "FAIL",
            measured=mean,
            expected="0.0000",
            se=se if math.isfinite(se) else None,
            hands=result.num_games,
            notes="SE is exactly zero (deterministic cancellation)",
        )
    z = mean / se
    ok = abs(z) < sigma
    return GateResult(
        gate="self_play",
        detail=detail,
        status="pass" if ok else "FAIL",
        measured=mean,
        expected="0.0000",
        se=se,
        hands=result.num_games,
        notes=f"z = {z:+.2f} (gate |z| < {sigma:g})",
    )


# --------------------------------------------------------------- match pooling


# `pool_results` lives in the package so non-gate runners share it; it is
# re-exported here because the gates are the harness's documented entry point.
__all__ = ["pool_results"]


def run_batched_match(
    agent_a: MatchAgent,
    agent_b: MatchAgent,
    env_proto: HUNLTensorEnv,
    *,
    pairs_per_batch: int,
    batches: int,
    seed: int,
    device: torch.device,
    tape: "ActionTape | None" = None,
    label: str = "",
) -> MatchResult:
    parts: list[MatchResult] = []
    for batch in range(batches):
        parts.append(
            play_duplicate_match(
                agent_a,
                agent_b,
                env_proto,
                num_pairs=pairs_per_batch,
                seed=seed + batch,
                device=device,
            )
        )
        if tape is not None:
            tape.harvest()
        if label:
            pooled = pool_results(parts)
            print(
                f"  [{label}] batch {batch + 1}/{batches}: {pooled.summary()}",
                flush=True,
            )
    return pool_results(parts)


# ------------------------------------------------------- gate 3: duplicate coupling


class ActionTape(MatchAgent):
    """Transparent wrapper that records the public bin taken in every game.

    Wraps one seat. ``commit`` is called on *both* agents with the same public
    actions, so wrapping a single seat captures the whole public sequence. Used
    only by the coupling diagnostic; it costs one device->host sync per
    decision round and is never in the path of a scored gate.
    """

    def __init__(self, inner: MatchAgent):
        self.inner = inner
        self.identity = inner.identity
        self.tape: list[list[int]] = []
        self._active: list[int] = []
        self._total_pairs = 0
        self._coupled_pairs = 0

    def search_fidelity(self) -> dict[str, Any]:
        return self.inner.search_fidelity()

    def begin_match(self, env: HUNLTensorEnv, seat: int) -> None:
        self.tape = [[] for _ in range(env.N)]
        self.inner.begin_match(env, seat)

    def observe(self, env: HUNLTensorEnv, active_indices: torch.Tensor) -> None:
        self._active = active_indices.tolist()
        self.inner.observe(env, active_indices)

    def action_probs(self, env, active_indices, hands, legal_mask):
        return self.inner.action_probs(env, active_indices, hands, legal_mask)

    def commit(self, actions: torch.Tensor, keep_mask: torch.Tensor) -> None:
        for game, action in zip(self._active, actions.tolist(), strict=True):
            self.tape[game].append(int(action))
        self.inner.commit(actions, keep_mask)

    def harvest(self) -> None:
        """Fold the finished match's tape into the running coupling counts.

        Call once per match; ``begin_match`` clears the per-match tape, so a
        tape reused across batches must be harvested between them.
        """
        pairs = len(self.tape) // 2
        self._total_pairs += pairs
        self._coupled_pairs += sum(
            self.tape[2 * i] == self.tape[2 * i + 1] for i in range(pairs)
        )
        self.tape = []

    def perfect_coupling_fraction(self) -> float:
        """Fraction of pairs whose two halves played identical bin sequences."""
        if self.tape:
            self.harvest()
        if self._total_pairs == 0:
            return float("nan")
        return self._coupled_pairs / self._total_pairs


def coupling_report(
    result: MatchResult, detail: str, tape: ActionTape | None = None
) -> list[GateResult]:
    """Measure what duplicate pairing actually buys, per-pair vs per-game."""
    raw_sd = float(result.reward_bb.std(unbiased=True).item()) * 100.0
    paired_sd = float(result.pair_diff_bb.std(unbiased=True).item()) * 100.0
    degenerate = paired_sd == 0.0
    ratio = raw_sd / paired_sd if paired_sd > 0 else float("inf")
    var_ratio = ratio * ratio
    # N games unpaired -> SE = raw_sd/sqrt(N). The same N games as N/2 pairs ->
    # SE = paired_sd/sqrt(N/2). Pairing wins iff var_ratio > 2.
    clears = var_ratio > 2.0
    # Scripted bots are deterministic, so common random numbers couple the two
    # halves perfectly and the pair statistic is identically zero. That is a
    # correctness result, not a variance measurement: only a stochastic agent
    # (CFR search) gives a meaningful coupling ratio.
    degenerate_note = (
        "degenerate: deterministic agents cancel exactly, so the pair "
        "statistic has no variance to measure"
    )
    rows = [
        GateResult(
            gate="coupling",
            detail=f"{detail}: raw per-game SD",
            status="info",
            measured=raw_sd,
            hands=result.num_games,
            units="bb/100",
        ),
        GateResult(
            gate="coupling",
            detail=f"{detail}: paired SD",
            status="info",
            measured=paired_sd,
            hands=result.num_games,
            units="bb/100",
            notes=degenerate_note if degenerate else "",
        ),
        GateResult(
            gate="coupling",
            detail=f"{detail}: SD ratio (raw/paired)",
            status="info",
            measured=ratio,
            expected=">1.4142",
            units="ratio",
            notes=(
                degenerate_note
                if degenerate
                else (
                    f"variance ratio {var_ratio:.2f}x -- "
                    + (
                        "clears the 2x compute cost of pairing"
                        if clears
                        else "does NOT clear the 2x compute cost of pairing"
                    )
                )
            ),
            extra={"variance_ratio": var_ratio, "clears_2x": clears},
        ),
    ]
    if tape is not None:
        frac = tape.perfect_coupling_fraction()
        rows.append(
            GateResult(
                gate="coupling",
                detail=f"{detail}: identical action seqs",
                status="info",
                measured=100.0 * frac,
                hands=result.num_games,
                units="% of pairs",
            )
        )
    rows.append(
        GateResult(
            gate="coupling",
            detail=f"{detail}: hands to resolve {TARGET_RESOLVABLE_BB100:g}bb/100",
            status="info",
            measured=float(hands_to_resolve(paired_sd, TARGET_RESOLVABLE_BB100)),
            units="hands",
            notes=(
                degenerate_note
                if degenerate
                else "80% power, 5% two-sided, paired scoring"
            ),
        )
    )
    return rows


def hands_to_resolve(
    paired_sd_bb100: float,
    difference_bb100: float = TARGET_RESOLVABLE_BB100,
    z_alpha: float = 1.959964,
    z_power: float = 0.8416212,
) -> int:
    """Hands (= 2 x pairs) needed to detect ``difference_bb100`` at 80% power."""
    if difference_bb100 <= 0 or not math.isfinite(paired_sd_bb100):
        return -1
    pairs = ((z_alpha + z_power) * paired_sd_bb100 / difference_bb100) ** 2
    return int(math.ceil(pairs)) * 2


# ------------------------------------------------------------ gate 4: known gap


def gate_known_gap(result: MatchResult, detail: str) -> list[GateResult]:
    """Later checkpoint (agent B) should beat the earlier one (agent A).

    ``result`` is scored from agent A's perspective, so a correctly-signed gap
    is *negative*. Nothing is tuned to make this come out positive: if the
    interval straddles zero the gate reports "unresolved" and the sample size
    that would settle it.
    """
    mean = result.mean_bb_per_100
    se = result.se_bb_per_100
    lo, hi = mean - 1.959964 * se, mean + 1.959964 * se
    resolved = hi < 0.0 or lo > 0.0
    if not resolved:
        needed = hands_to_resolve(
            se * math.sqrt(result.num_pairs), max(abs(mean), 1e-9)
        )
        status = "info"
        notes = (
            f"95% CI [{lo:+.2f}, {hi:+.2f}] straddles zero: cannot resolve at this "
            f"sample size. Detecting the observed {abs(mean):.2f} bb/100 at 80% "
            f"power would need ~{needed:,} hands."
        )
    else:
        status = "pass" if mean < 0 else "FAIL"
        direction = "later beats earlier" if mean < 0 else "WRONG SIGN"
        notes = f"95% CI [{lo:+.2f}, {hi:+.2f}] excludes zero: {direction}"
    return [
        GateResult(
            gate="known_gap",
            detail=detail,
            status=status,
            measured=mean,
            expected="< 0",
            se=se,
            hands=result.num_games,
            notes=notes,
        ),
        GateResult(
            gate="known_gap",
            detail=f"{detail}: hands for {TARGET_RESOLVABLE_BB100:g}bb/100",
            status="info",
            measured=float(
                hands_to_resolve(
                    se * math.sqrt(result.num_pairs), TARGET_RESOLVABLE_BB100
                )
            ),
            units="hands",
            notes="80% power, 5% two-sided, paired scoring",
        ),
    ]


# ------------------------------------------------------------- gate collections


def run_model_free_gates(
    device: torch.device,
    fold_pairs: int = 512,
    symmetry_pairs: int = 4096,
    seed: int = 1,
) -> list[GateResult]:
    """Every gate that needs no model, no GPU and no checkpoint."""
    results = gate_fold_exact(device, num_pairs=fold_pairs, seed=seed)

    env_proto = randomized_stack_env(device)
    call_result = play_duplicate_match(
        CallAgent("call_a"),
        CallAgent("call_b"),
        env_proto,
        num_pairs=symmetry_pairs,
        seed=seed + 1,
        device=device,
    )
    results.append(gate_self_play(call_result, "call_bot vs call_bot"))

    tape = ActionTape(RandomAgent("rand_a"))
    random_result = play_duplicate_match(
        tape,
        RandomAgent("rand_b"),
        env_proto,
        num_pairs=symmetry_pairs,
        seed=seed + 2,
        device=device,
    )
    results.append(gate_self_play(random_result, "random_bot vs random_bot"))
    results.extend(coupling_report(random_result, "random_bot self-play", tape))
    results.extend(coupling_report(call_result, "call_bot self-play"))
    return results


def run_model_gates(
    device: torch.device,
    *,
    pairs_per_batch: int,
    self_play_batches: int,
    known_gap_batches: int,
    cfr_iterations: int,
    warm_start_iterations: int,
    dcfr_delay: int,
    seed: int = 1,
) -> list[GateResult]:
    """Gates that need a real checkpoint and a GPU."""
    from p2.eval.checkpoints import SearchFidelity, load_search_agent

    fidelity = SearchFidelity(
        cfr_iterations=cfr_iterations,
        warm_start_iterations=warm_start_iterations,
        dcfr_delay=dcfr_delay,
    )
    results: list[GateResult] = []

    def load(path: Path, name: str):
        return load_search_agent(
            path,
            resolved_config=RESOLVED_CONFIG_V3,
            device=device,
            fidelity=fidelity,
            name=name,
            num_envs=8,
        )

    # -- 2b: a real checkpoint against itself ---------------------------------
    early_a = load(CKPT_EARLY, "v3_step10000_a")
    early_b = load(CKPT_EARLY, "v3_step10000_b")
    tape = ActionTape(early_a.agent)
    t0 = time.time()
    self_result = run_batched_match(
        tape,
        early_b.agent,
        early_a.env_proto,
        pairs_per_batch=pairs_per_batch,
        batches=self_play_batches,
        seed=seed + 3,
        device=device,
        tape=tape,
        label="self-play",
    )
    elapsed = time.time() - t0
    label = f"v3@10000 vs itself (cfr={cfr_iterations})"
    results.append(gate_self_play(self_result, label))
    results[-1].notes += f"; {elapsed:.0f}s for {self_result.num_games} hands"
    results.extend(coupling_report(self_result, "v3@10000 self-play", tape))

    del early_b
    torch.cuda.empty_cache()

    # -- 4: known gap ---------------------------------------------------------
    late = load(CKPT_LATE, "v3to15k_step12750")
    t0 = time.time()
    gap_result = run_batched_match(
        early_a.agent,
        late.agent,
        early_a.env_proto,
        pairs_per_batch=pairs_per_batch,
        batches=known_gap_batches,
        seed=seed + 4,
        device=device,
        label="known-gap",
    )
    elapsed = time.time() - t0
    results.extend(gate_known_gap(gap_result, "v3@10000 (A) vs v3to15k@12750 (B)"))
    results[-2].notes += f"; {elapsed:.0f}s"
    return results


# --------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-free",
        action="store_true",
        help="run only the gates that need no checkpoint and no GPU",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--fold-pairs", type=int, default=512)
    parser.add_argument("--symmetry-pairs", type=int, default=4096)
    parser.add_argument(
        "--pairs-per-batch",
        type=int,
        default=64,
        help="duplicate pairs per search match; bounds GPU memory",
    )
    parser.add_argument("--self-play-batches", type=int, default=8)
    parser.add_argument("--known-gap-batches", type=int, default=16)
    parser.add_argument("--cfr-iterations", type=int, default=300)
    parser.add_argument("--warm-start-iterations", type=int, default=10)
    parser.add_argument("--dcfr-delay", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    results: list[GateResult] = []
    # Gate 1 and the model-free symmetry gates run on the CPU: they are cheap,
    # deterministic, and must not depend on the GPU being free.
    results.extend(
        run_model_free_gates(
            torch.device("cpu"),
            fold_pairs=args.fold_pairs,
            symmetry_pairs=args.symmetry_pairs,
            seed=args.seed,
        )
    )
    fold_failures = [
        r for r in results if r.gate == "fold_exact" and r.status == "FAIL"
    ]
    if fold_failures:
        print_table(results)
        print(
            "\nSTOP: the fold-bot exactness gate failed. The harness or the env "
            "is wrong about sign, scale, blinds or seats; every downstream number "
            "is meaningless until this is fixed."
        )
        return 1

    if not args.model_free:
        missing = [
            p for p in (CKPT_EARLY, CKPT_LATE, RESOLVED_CONFIG_V3) if not p.exists()
        ]
        if missing:
            results.append(
                GateResult(
                    gate="model",
                    detail="checkpoint gates",
                    status="skip",
                    notes=f"missing anchors: {[str(p) for p in missing]}",
                )
            )
        elif not torch.cuda.is_available():
            results.append(
                GateResult(
                    gate="model",
                    detail="checkpoint gates",
                    status="skip",
                    notes="no CUDA device available",
                )
            )
        else:
            results.extend(
                run_model_gates(
                    torch.device(args.device),
                    pairs_per_batch=args.pairs_per_batch,
                    self_play_batches=args.self_play_batches,
                    known_gap_batches=args.known_gap_batches,
                    cfr_iterations=args.cfr_iterations,
                    warm_start_iterations=args.warm_start_iterations,
                    dcfr_delay=args.dcfr_delay,
                    seed=args.seed,
                )
            )

    print_table(results)
    return 1 if any(r.status == "FAIL" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
