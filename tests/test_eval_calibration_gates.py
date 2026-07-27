"""Tests for the evaluation calibration gates.

The cheap, model-free gates run here on every CI pass: they are the ones that
would have caught a silently broken scoring path. The model/GPU gates are
marked so they skip cleanly when no checkpoint or CUDA device is available --
they are exercised by ``scripts/eval_calibration_gates.py`` on a real box.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from p2.eval.agents import CallAgent, FoldAgent, RandomAgent
from p2.eval.duplicate_match import play_duplicate_match

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_gates_module():
    """Import ``scripts/eval_calibration_gates.py`` (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / "eval_calibration_gates.py"
    spec = importlib.util.spec_from_file_location("eval_calibration_gates", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gates = _load_gates_module()
DEVICE = torch.device("cpu")


# ---------------------------------------------------- gate 1: fold exactness


@pytest.mark.parametrize(
    "make_opponent", [lambda: FoldAgent("fold_opp"), CallAgent, RandomAgent]
)
@pytest.mark.parametrize("env_factory", ["fixed_stack_env", "randomized_stack_env"])
def test_fold_bot_in_the_small_blind_loses_exactly_half_a_blind(
    make_opponent, env_factory
):
    """The one gate that must be exact, not merely within a standard error.

    A fold-bot in the SB folds preflop for exactly 0.5 bb, whoever it plays. One
    number covering sign convention, chip scale, blind accounting and seats.
    """
    env_proto = getattr(gates, env_factory)(DEVICE)
    result = play_duplicate_match(
        FoldAgent(),
        make_opponent(),
        env_proto,
        num_pairs=64,
        seed=5,
        device=DEVICE,
    )
    sb_rewards = gates.fold_bot_sb_rewards(result)
    assert sb_rewards, "no games where agent A held the small blind"
    assert len(sb_rewards) == result.num_games // 2  # exactly one half per pair
    worst = max(abs(v + 0.5) for v in sb_rewards)
    assert worst <= gates.FOLD_ATOL_BB, f"max deviation {worst:.3e} bb"


def test_gate_fold_exact_reports_pass_rows():
    rows = gates.gate_fold_exact(DEVICE, num_pairs=16)
    assert len(rows) == 6  # 2 envs x 3 opponents
    assert all(r.status == "pass" for r in rows), [r.notes for r in rows]
    assert all(r.measured == pytest.approx(-50.0, abs=1e-4) for r in rows)


def test_button_zero_really_is_the_small_blind_seat():
    """The fold gate's SB subset relies on ``button == 0`` meaning seat 0 posts
    the small blind; assert that against the env directly."""
    env = gates.fixed_stack_env(DEVICE)
    env = type(env).from_proto(env, num_envs=8)
    env.reset(force_button=torch.arange(8, device=DEVICE) % 2)
    # Seat 0's committed chips preflop are sb when it has the button, bb when not.
    committed = env.committed[:, 0]
    button0 = env.button == 0
    assert torch.allclose(
        committed[button0], torch.full_like(committed[button0], float(env.sb))
    )
    assert torch.allclose(
        committed[~button0], torch.full_like(committed[~button0], float(env.bb))
    )


# ------------------------------------------------ gate 2: self-play symmetry


@pytest.mark.parametrize("agent_cls", [CallAgent, RandomAgent])
def test_model_free_self_play_scores_zero(agent_cls):
    result = play_duplicate_match(
        agent_cls("a"),
        agent_cls("b"),
        gates.randomized_stack_env(DEVICE),
        num_pairs=256,
        seed=7,
        device=DEVICE,
    )
    row = gates.gate_self_play(result, "self")
    assert row.status == "pass", row.notes
    # Deterministic agents under common random numbers cancel exactly.
    assert result.mean_bb_per_100 == pytest.approx(0.0, abs=1e-4)


def test_gate_self_play_flags_a_nonzero_mean():
    """The symmetry gate must actually fail when the mean is off."""
    result = play_duplicate_match(
        FoldAgent(),
        CallAgent(),
        gates.fixed_stack_env(DEVICE),
        num_pairs=64,
        seed=11,
        device=DEVICE,
    )
    row = gates.gate_self_play(result, "fold vs call (not symmetric)")
    assert row.status == "FAIL"
    assert row.measured < 0.0


# --------------------------------------------------- gate 3: coupling report


def test_coupling_report_measures_raw_and_paired_spread():
    result = play_duplicate_match(
        RandomAgent("a"),
        RandomAgent("b"),
        gates.randomized_stack_env(DEVICE),
        num_pairs=128,
        seed=13,
        device=DEVICE,
    )
    rows = gates.coupling_report(result, "random")
    by_detail = {r.detail.split(": ")[-1]: r for r in rows}
    assert by_detail["raw per-game SD"].measured > 0.0
    # Scripted agents couple perfectly, so the pair statistic is exactly zero.
    assert by_detail["paired SD"].measured == pytest.approx(0.0, abs=1e-6)
    assert "degenerate" in by_detail["paired SD"].notes


def test_action_tape_is_transparent_and_counts_coupling():
    inner = RandomAgent("taped")
    tape = gates.ActionTape(inner)
    env_proto = gates.randomized_stack_env(DEVICE)
    taped = play_duplicate_match(
        tape, RandomAgent("b"), env_proto, num_pairs=32, seed=17, device=DEVICE
    )
    plain = play_duplicate_match(
        RandomAgent("taped"),
        RandomAgent("b"),
        env_proto,
        num_pairs=32,
        seed=17,
        device=DEVICE,
    )
    # Wrapping must not change a single played hand.
    assert torch.equal(taped.reward_bb, plain.reward_bb)
    # Scripted agents under common random numbers are perfectly coupled.
    assert tape.perfect_coupling_fraction() == pytest.approx(1.0)


def test_action_tape_accumulates_across_batches():
    tape = gates.ActionTape(RandomAgent("a"))
    env_proto = gates.randomized_stack_env(DEVICE)
    for seed in (1, 2, 3):
        play_duplicate_match(
            tape, RandomAgent("b"), env_proto, num_pairs=8, seed=seed, device=DEVICE
        )
        tape.harvest()
    assert tape._total_pairs == 24
    assert tape.perfect_coupling_fraction() == pytest.approx(1.0)


def test_hands_to_resolve_scales_as_variance_over_squared_difference():
    # Doubling the SD quadruples the required sample.
    assert gates.hands_to_resolve(200.0, 5.0) == pytest.approx(
        4 * gates.hands_to_resolve(100.0, 5.0), rel=1e-3
    )
    # Halving the detectable difference quadruples it too.
    assert gates.hands_to_resolve(100.0, 2.5) == pytest.approx(
        4 * gates.hands_to_resolve(100.0, 5.0), rel=1e-3
    )
    # Sanity against the closed form: pairs = ((1.96+0.84) sd / d)^2, hands = 2 pairs.
    expected = 2 * math.ceil(((1.959964 + 0.8416212) * 700.0 / 5.0) ** 2)
    assert gates.hands_to_resolve(700.0, 5.0) == expected


# ------------------------------------------------------ pooling across batches


def test_pool_results_matches_a_single_large_match():
    """Pooled batches must give the same statistics as one big match would.

    Search matches are run in memory-bounded batches, so the pooling has to be
    arithmetically honest.
    """
    env_proto = gates.randomized_stack_env(DEVICE)
    parts = [
        play_duplicate_match(
            RandomAgent("a"),
            FoldAgent(),
            env_proto,
            num_pairs=16,
            seed=100 + i,
            device=DEVICE,
        )
        for i in range(4)
    ]
    pooled = gates.pool_results(parts)
    assert pooled.num_pairs == 64
    assert pooled.num_games == 128
    all_diffs = torch.cat([p.pair_diff_bb for p in parts])
    assert pooled.mean_bb_per_100 == pytest.approx(100.0 * all_diffs.mean().item())
    assert pooled.se_bb_per_100 == pytest.approx(
        100.0 * all_diffs.std(unbiased=True).item() / math.sqrt(64), rel=1e-5
    )
    assert len(pooled.records) == 128


def test_run_model_free_gates_all_pass():
    rows = gates.run_model_free_gates(DEVICE, fold_pairs=16, symmetry_pairs=64)
    assert not [r for r in rows if r.status == "FAIL"]
    assert {r.gate for r in rows} == {"fold_exact", "self_play", "coupling"}


def test_main_model_free_exits_zero(capsys):
    assert (
        gates.main(["--model-free", "--fold-pairs", "8", "--symmetry-pairs", "32"]) == 0
    )
    out = capsys.readouterr().out
    assert "fold_exact" in out and "FAIL" not in out.replace("FAIL,", "")


# ------------------------------------------------------- model / GPU-only gates

ANCHORS_PRESENT = gates.CKPT_EARLY.exists() and gates.RESOLVED_CONFIG_V3.exists()


def _triton_can_find_libcuda() -> bool:
    """Whether Triton will be able to load libcuda.

    `torch.cuda.is_available()` is not sufficient: the sparse CFR evaluator
    compiles Triton kernels, and Triton resolves libcuda through the ldconfig
    cache. On hosts where that cache lacks an entry (this one), the import
    raises deep inside the driver instead of degrading, which would turn a
    missing host setup into a spurious test failure. Skip with an actionable
    message instead.
    """
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return True
    try:
        listing = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(
            errors="ignore"
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "libcuda.so.1" in listing


requires_model = pytest.mark.skipif(
    not (ANCHORS_PRESENT and torch.cuda.is_available() and _triton_can_find_libcuda()),
    reason=(
        "needs the eval_anchors checkpoints, a CUDA device, and a Triton-visible "
        "libcuda (set TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu on hosts whose "
        "ldconfig cache lacks it)"
    ),
)


@requires_model
def test_load_search_agent_pins_fidelity():
    from p2.eval.checkpoints import SearchFidelity, load_search_agent

    fidelity = SearchFidelity(cfr_iterations=16, warm_start_iterations=4, dcfr_delay=8)
    loaded = load_search_agent(
        gates.CKPT_EARLY,
        resolved_config=gates.RESOLVED_CONFIG_V3,
        device="cuda",
        fidelity=fidelity,
        name="v3_early",
        num_envs=4,
    )
    assert loaded.step == 9999
    ev = loaded.agent.evaluator
    assert ev.cfr_iterations == 16
    assert ev.warm_start_iterations == 4
    # The pinned values reach the per-game provenance record.
    recorded = loaded.agent.search_fidelity()
    assert recorded["cfr_iterations"] == 16
    assert loaded.agent.identity.extra["fidelity"] == fidelity.to_dict()
    assert loaded.agent.identity.checkpoint == str(gates.CKPT_EARLY)


@requires_model
def test_search_agent_self_play_is_symmetric():
    from p2.eval.checkpoints import SearchFidelity, load_search_agent

    fidelity = SearchFidelity(cfr_iterations=16, warm_start_iterations=4, dcfr_delay=8)
    a = load_search_agent(
        gates.CKPT_EARLY,
        resolved_config=gates.RESOLVED_CONFIG_V3,
        device="cuda",
        fidelity=fidelity,
        name="a",
        num_envs=4,
    )
    b = load_search_agent(
        gates.CKPT_EARLY,
        resolved_config=gates.RESOLVED_CONFIG_V3,
        device="cuda",
        fidelity=fidelity,
        name="b",
        num_envs=4,
    )
    result = play_duplicate_match(
        a.agent, b.agent, a.env_proto, num_pairs=16, seed=3, device=torch.device("cuda")
    )
    row = gates.gate_self_play(result, "search self-play")
    assert row.status == "pass", row.notes
