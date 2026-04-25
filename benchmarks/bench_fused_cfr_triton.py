"""Benchmark fused Triton CFR helpers vs the unfused PyTorch baseline.

Measures:

1. ``fused_dcfr_update_`` vs the equivalent PyTorch op sequence at a realistic
   tensor shape.
2. One CUDA-graph-captured ``cfr_iteration`` replay vs one uncaptured
   ``cfr_iteration`` call.

Example::

    uv run --python 3.12 python benchmarks/bench_fused_cfr_triton.py

Hydra overrides are accepted for the full-iteration benchmark, e.g.::

    ... num_envs=192 search.depth=5
"""

from __future__ import annotations

import time

import hydra
import torch
from omegaconf import DictConfig

from p2.core.structured_config import CFRType, Config
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.rebel_ffn import RebelFFN
from p2.search.fused_cfr_triton import (
    GraphedCFRIteration,
    fused_dcfr_update_,
    triton_is_available,
)
from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


def _cuda_timed(fn, n_iters: int, n_warmup: int = 5) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iters  # ms


def bench_dcfr_kernel(total_nodes: int = 10_000, num_hands: int = 1326) -> None:
    if not triton_is_available():
        print("Triton unavailable; skipping dcfr kernel bench.")
        return

    device = torch.device("cuda")
    shape = (total_nodes, num_hands)
    torch.manual_seed(0)

    cumul = torch.randn(shape, device=device)
    weight = torch.rand(shape, device=device) * 5.0
    regrets = torch.randn(shape, device=device) * 0.1
    pos_out = torch.empty_like(cumul)

    alpha, beta, t = 1.5, 0.5, 50

    # Baseline: PyTorch op sequence.
    def baseline() -> None:
        c = cumul  # in-place
        w = weight
        numerator = torch.where(c > 0, t**alpha, t**beta)
        denominator = torch.where(c > 0, t**alpha + 1, t**beta + 1)
        c.mul_(numerator)
        c.div_(denominator)
        w.mul_(numerator)
        w.div_(denominator)
        w.add_(1)
        c.add_(regrets)
        torch.clamp(c, min=0, out=pos_out)

    # Fused kernel.
    def fused() -> None:
        fused_dcfr_update_(
            cumulative_regrets=cumul,
            regret_weight_sums=weight,
            regrets=regrets,
            t=t,
            cfr_type=CFRType.discounted_plus,
            dcfr_alpha=alpha,
            dcfr_beta=beta,
            cfr_plus=False,
            positive_regrets_out=pos_out,
        )

    # Reset state between timings — both mutate in place. Snapshot and restore
    # via clones to give each a fair start.
    cumul0 = cumul.clone()
    weight0 = weight.clone()

    def baseline_wrapped() -> None:
        cumul.copy_(cumul0)
        weight.copy_(weight0)
        baseline()

    def fused_wrapped() -> None:
        cumul.copy_(cumul0)
        weight.copy_(weight0)
        fused()

    # These wrapped versions include a 2-tensor copy per iter, same for both,
    # so the delta is still a fair comparison.
    t_baseline = _cuda_timed(baseline_wrapped, n_iters=200)
    t_fused = _cuda_timed(fused_wrapped, n_iters=200)

    print("\n=== fused_dcfr_update kernel ===")
    print(f"  shape: {shape} ({cumul.numel():,} elements, "
          f"{cumul.numel() * 4 / 2**20:.1f} MB per tensor)")
    print(f"  PyTorch sequence : {t_baseline:.4f} ms / call")
    print(f"  Triton fused     : {t_fused:.4f} ms / call")
    print(f"  Speedup          : {t_baseline / t_fused:.2f}x")


def bench_graphed_iteration(cfg: Config) -> None:
    device = torch.device(cfg.device)

    env = _build_mixed_street_env(cfg, device)
    root_indices = torch.arange(cfg.num_envs, dtype=torch.long, device=device)

    torch.manual_seed(cfg.seed)
    model = RebelFFN(
        input_dim=cfg.model.input_dim,
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
        detach_value_head=cfg.model.detach_value_head,
        num_players=2,
    )
    cpu_rng = torch.Generator(device="cpu").manual_seed(42)
    model.init_weights(cpu_rng)
    model.to(device).eval()

    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    evaluator = FusedSparseCFREvaluator(model=model, device=device, cfg=cfg)
    evaluator.initialize_subgame(env, root_indices)
    evaluator.initialize_policy_and_beliefs()
    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values
    evaluator.t_sample = evaluator._get_sampling_schedule()

    # Prime a bit so we're out of any t<=1 early-branch territory.
    for t in range(1, 5):
        evaluator.cfr_iteration(t)
    torch.cuda.synchronize()

    T = 10

    # Stub stats to match the graph path (fair comparison: the graph can't
    # replay the .item() sync in _record_stats).
    orig_stats = evaluator._record_stats
    evaluator._record_stats = lambda t, old: None

    replay_counter = {"t": T}

    def baseline() -> None:
        evaluator.cfr_iteration(replay_counter["t"])
        replay_counter["t"] += 1

    try:
        t_baseline = _cuda_timed(baseline, n_iters=25, n_warmup=3)

        runner = GraphedCFRIteration(evaluator)
        runner.capture(t_capture=replay_counter["t"], num_warmup=2)

        def replay() -> None:
            runner.replay(t=replay_counter["t"])
            replay_counter["t"] += 1

        t_graphed = _cuda_timed(replay, n_iters=25, n_warmup=3)
    finally:
        evaluator._record_stats = orig_stats

    print("\n=== full cfr_iteration: uncaptured vs CUDA-graph replay ===")
    print(f"  num_envs={cfg.num_envs}, depth={cfg.search.depth}")
    print(f"  uncaptured (stats stubbed) : {t_baseline:.3f} ms / iter")
    print(f"  CUDA-graph replay          : {t_graphed:.3f} ms / iter")
    print(f"  Speedup                    : {t_baseline / t_graphed:.2f}x")


def bench_sibling_sum(
    num_parents: int = 30_000, avg_children: int = 5, h: int = 1326
) -> None:
    """Compare sibling-sum (Triton 2D tile) vs native scatter_reduce_ + repeat_interleave."""
    if not triton_is_available():
        print("Triton unavailable; skipping sibling_sum bench.")
        return
    from p2.search.fused_cfr_triton import fused_sibling_sum

    device = torch.device("cuda")
    torch.manual_seed(0)

    # Synthetic sibling layout: most parents have avg_children children, a few have fewer/more.
    child_count = torch.full((num_parents,), avg_children, device=device, dtype=torch.long)
    # Randomize a bit to stress the mask path.
    jitter = torch.randint(-1, 2, (num_parents,), device=device, dtype=torch.long)
    child_count = (child_count + jitter).clamp(min=1, max=8)

    num_children = int(child_count.sum().item())
    bottom = num_parents  # roots then children
    total = bottom + num_children

    child_offsets = bottom + torch.cumsum(child_count, 0) - child_count
    values = torch.randn(total, h, device=device)
    parent_index = torch.arange(num_parents, device=device).repeat_interleave(child_count)
    parent_index_abs = parent_index  # maps child row (in [0, num_children)) → parent row

    # Native baseline: scatter_reduce to [num_parents, H] then fan_out.
    def baseline() -> None:
        parent_sum = torch.zeros(num_parents, h, device=device)
        parent_sum.scatter_reduce_(
            0,
            parent_index_abs[:, None].expand(-1, h),
            values[bottom:],
            reduce="sum",
            include_self=True,
        )
        parent_sum.repeat_interleave(child_count, dim=0, output_size=num_children)

    # Triton 2D-tile kernel.
    def fused() -> None:
        fused_sibling_sum(
            values=values.contiguous(),
            child_offsets=child_offsets.contiguous(),
            child_count=child_count.contiguous(),
            bottom=bottom,
            num_children=num_children,
            max_children=8,
        )

    t_base = _cuda_timed(baseline, n_iters=50, n_warmup=5)
    t_fused = _cuda_timed(fused, n_iters=50, n_warmup=5)
    print("\n=== sibling_sum: scatter_reduce+repeat_interleave vs Triton 2D tile ===")
    print(
        f"  num_parents={num_parents:,}, avg_children={avg_children}, "
        f"num_children={num_children:,}, H={h}"
    )
    print(f"  PyTorch (scatter+fan)  : {t_base:.3f} ms / call")
    print(f"  Triton 2D tile         : {t_fused:.3f} ms / call")
    print(f"  Speedup                : {t_base / t_fused:.2f}x")


def bench_approach_c(
    num_parents: int = 30_000, avg_children: int = 5, h: int = 1326
) -> None:
    """Approach A (sibling_sum → denom) vs Approach C (parent_sum + fused divide)."""
    if not triton_is_available():
        print("Triton unavailable; skipping approach_c bench.")
        return
    from p2.search.fused_cfr_triton import (
        fused_divide_by_parent_sum_,
        fused_parent_sum,
        fused_regret_matching_divide_,
        fused_sibling_sum,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)

    child_count = torch.full((num_parents,), avg_children, device=device, dtype=torch.long)
    jitter = torch.randint(-1, 2, (num_parents,), device=device, dtype=torch.long)
    child_count = (child_count + jitter).clamp(min=1, max=8)

    num_children = int(child_count.sum().item())
    bottom = num_parents
    total = bottom + num_children
    child_offsets = bottom + torch.cumsum(child_count, 0) - child_count
    parent_index_rel = torch.repeat_interleave(
        torch.arange(num_parents, device=device), child_count
    )

    values = torch.randn(total, h, device=device)
    pos = torch.rand(num_children, h, device=device)
    fallback = torch.rand(num_children, h, device=device)
    out = torch.empty_like(pos)

    def approach_a() -> None:
        denom = fused_sibling_sum(
            values=values.contiguous(),
            child_offsets=child_offsets.contiguous(),
            child_count=child_count.contiguous(),
            bottom=bottom,
            num_children=num_children,
            max_children=8,
        )
        fused_regret_matching_divide_(
            positive_regrets=pos.contiguous(),
            denom=denom,
            uniform_fallback=fallback.contiguous(),
            out=out,
        )

    def approach_c() -> None:
        parent_sum = fused_parent_sum(
            values=values.contiguous(),
            child_offsets=child_offsets.contiguous(),
            child_count=child_count.contiguous(),
            max_children=8,
        )
        fused_divide_by_parent_sum_(
            pos=pos.contiguous(),
            fallback=fallback.contiguous(),
            parent_sum=parent_sum,
            parent_index=parent_index_rel.contiguous(),
            out=out,
        )

    t_a = _cuda_timed(approach_a, n_iters=50, n_warmup=5)
    t_c = _cuda_timed(approach_c, n_iters=50, n_warmup=5)
    print("\n=== Approach A (sibling_sum + divide) vs Approach C (parent_sum + fused divide) ===")
    print(f"  num_parents={num_parents:,}, num_children={num_children:,}, H={h}")
    print(f"  A: sibling_sum + divide  : {t_a:.3f} ms")
    print(f"  C: parent_sum + indirect : {t_c:.3f} ms")
    print(f"  Speedup of C over A      : {t_a / t_c:.2f}x")


def bench_unblocked_mass(b: int = 400_000) -> None:
    if not triton_is_available():
        print("Triton unavailable; skipping unblocked_mass bench.")
        return
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import unblocked_mass_triton

    device = torch.device("cuda")
    target = torch.rand(b, 1326, device=device)

    def baseline() -> None:
        calculate_unblocked_mass(target)

    def fused() -> None:
        unblocked_mass_triton(target)

    t_base = _cuda_timed(baseline, n_iters=20, n_warmup=3)
    t_fused = _cuda_timed(fused, n_iters=20, n_warmup=3)
    print("\n=== calculate_unblocked_mass: fp64 GEMM vs O(N) Triton ===")
    print(f"  B={b:,}, H=1326")
    print(f"  fp64 GEMM (baseline) : {t_base:.3f} ms / call")
    print(f"  O(N) Triton          : {t_fused:.3f} ms / call")
    print(f"  Speedup              : {t_base / t_fused:.2f}x")


def _build_mixed_street_env(cfg: Config, device: torch.device) -> HUNLTensorEnv:
    """Construct one HUNLTensorEnv of cfg.num_envs roots split evenly across
    preflop / flop / turn / river.

    Each quarter of the root indices is advanced to its target street by
    stepping a dedicated sub-env through check/call action bins (bin=1) until
    the target ``street`` value is reached, then the sub-env's state is copied
    into the final combined env via ``copy_state_from``. This gives the sparse
    subgame a realistic mix of street depths so ``_showdown_value_both`` and
    ``new_street_mask`` paths actually get exercised, not left empty as they
    are with a pure-preflop root.
    """
    N = cfg.num_envs
    assert N % 4 == 0, f"num_envs={N} must be divisible by 4 for mixed-street bench"
    quarter = N // 4

    full = HUNLTensorEnv(
        num_envs=N,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    full.reset()

    # Target streets for the four quarters: preflop (0), flop (1), turn (2), river (3).
    targets = [0, 1, 2, 3]
    for q_idx, target_street in enumerate(targets):
        sub = HUNLTensorEnv(
            num_envs=quarter,
            starting_stack=cfg.env.stack,
            sb=cfg.env.sb,
            bb=cfg.env.bb,
            default_bet_bins=cfg.env.bet_bins,
            device=device,
            float_dtype=torch.float32,
            flop_showdown=cfg.env.flop_showdown,
        )
        sub.reset()
        # Advance by stepping check/call (bin=1) until all sub-envs reach target.
        # Each betting round closes after two actions (SB-checkcall, BB-check).
        safety_cap = 4 * (target_street + 1) + 2
        steps = 0
        while int(sub.street.min().item()) < target_street and steps < safety_cap:
            bins = torch.ones(quarter, dtype=torch.long, device=device)
            # Fold any env that finished the hand early (shouldn't happen on
            # pure check/call paths, but be safe).
            bins[sub.done] = -1
            sub.step_bins(bins)
            steps += 1
        assert int(sub.street.min().item()) >= target_street, (
            f"Failed to advance quarter {q_idx} to street {target_street} "
            f"(reached min={int(sub.street.min().item())} after {steps} steps)"
        )
        src_idx = torch.arange(quarter, dtype=torch.long, device=device)
        dst_idx = src_idx + q_idx * quarter
        full.copy_state_from(sub, src_idx, dst_idx)
    return full


def bench_fused_evaluator(cfg: Config) -> None:
    """End-to-end: SparseCFREvaluator vs FusedSparseCFREvaluator."""
    device = torch.device(cfg.device)

    def _build(klass):
        env = _build_mixed_street_env(cfg, device)
        root_indices = torch.arange(cfg.num_envs, dtype=torch.long, device=device)

        torch.manual_seed(cfg.seed)
        model = RebelFFN(
            input_dim=cfg.model.input_dim,
            num_actions=cfg.model.num_actions,
            hidden_dim=cfg.model.hidden_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            detach_value_head=cfg.model.detach_value_head,
            num_players=2,
        )
        cpu_rng = torch.Generator(device="cpu").manual_seed(42)
        model.init_weights(cpu_rng)
        model.to(device).eval()

        ev = klass(model=model, device=device, cfg=cfg)
        ev.initialize_subgame(env, root_indices)
        ev.initialize_policy_and_beliefs()
        ev.set_leaf_values(0)
        ev.compute_expected_values()
        ev.values_avg[:] = ev.latest_values
        ev.t_sample = ev._get_sampling_schedule()
        return ev

    ev_base = _build(SparseCFREvaluator)
    ev_fused = _build(FusedSparseCFREvaluator)

    # Prime both past warm-start / early-t branches.
    for t in range(1, 5):
        ev_base.cfr_iteration(t)
        ev_fused.cfr_iteration(t)
    torch.cuda.synchronize()

    T = 10

    def run_base() -> None:
        ev_base.cfr_iteration(T)

    def run_fused() -> None:
        ev_fused.cfr_iteration(T)

    t_base = _cuda_timed(run_base, n_iters=100, n_warmup=20)
    t_fused = _cuda_timed(run_fused, n_iters=100, n_warmup=20)

    print("\n=== SparseCFREvaluator vs FusedSparseCFREvaluator (full iter) ===")
    print(f"  num_envs={cfg.num_envs}, depth={cfg.search.depth}")
    print(f"  SparseCFREvaluator        : {t_base:.3f} ms / iter")
    print(f"  FusedSparseCFREvaluator   : {t_fused:.3f} ms / iter")
    print(f"  Speedup                   : {t_base / t_fused:.2f}x")


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    bench_dcfr_kernel()
    bench_sibling_sum()
    bench_unblocked_mass()
    cfg = Config.from_dict_config(dict_config)
    cfg.use_wandb = False
    bench_fused_evaluator(cfg)
    bench_graphed_iteration(cfg)


if __name__ == "__main__":
    main()
