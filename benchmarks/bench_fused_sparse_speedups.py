"""Benchmark candidate speedups for FusedSparseCFREvaluator.

The evaluator reads optimization flags from environment variables at
construction time, so this script builds a fresh evaluator per variant and
times a short run of uncaptured `cfr_iteration` calls.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch

from p2.core.structured_config import (
    CFRType,
    Config,
    ModelType,
    NonlinearityType,
    WarmStartType,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.rebel_ffn import RebelFFN
from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator


OPT_FLAGS = {
    "reuse_positive": "P2_FUSED_OPT_REUSE_POSITIVE",
    "parent_sum_divide": "P2_FUSED_OPT_PARENT_SUM_DIVIDE",
    "sparse_sample": "P2_FUSED_OPT_SPARSE_SAMPLE",
    "leaf_feature_cache": "P2_FUSED_OPT_LEAF_FEATURE_CACHE",
    "child_opp_policy": "P2_FUSED_OPT_CHILD_OPP_POLICY",
}


@dataclass
class BenchResult:
    name: str
    ms_per_iter: float
    nodes: int
    model_leaves: int
    showdown_leaves: int


def _cuda_timed(fn, n_iters: int, n_warmup: int = 3) -> float:
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
    return start.elapsed_time(end) / n_iters


def _set_flags(enabled: set[str]) -> None:
    for name, env_name in OPT_FLAGS.items():
        os.environ[env_name] = "1" if name in enabled else "0"


def _make_cfg(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.device = "cuda"
    cfg.seed = 42
    cfg.num_envs = args.num_envs
    cfg.env.bet_bins = [0.5, 1.0]
    cfg.env.flop_showdown = False
    cfg.model.name = (
        ModelType.better_ffn if args.model == "better" else ModelType.rebel_ffn
    )
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.range_hidden_dim = args.range_hidden_dim
    cfg.model.ffn_dim = args.ffn_dim
    cfg.model.num_hidden_layers = args.layers
    cfg.model.num_policy_layers = 2
    cfg.model.num_value_layers = 2
    cfg.model.detach_value_head = False
    cfg.model.compile = False
    cfg.model.enforce_zero_sum = True
    cfg.model.nonlinearity = (
        NonlinearityType.swiglu
        if cfg.model.name == ModelType.better_ffn
        else NonlinearityType.gelu
    )
    cfg.search.depth = args.depth
    cfg.search.iterations = args.iterations
    cfg.search.warm_start_iterations = args.warm_start
    cfg.search.warm_start_type = WarmStartType.model
    cfg.search.warm_start_multiplier = 10
    cfg.search.dcfr_plus_delay = args.dcfr_delay
    cfg.search.cfr_type = CFRType.discounted
    cfg.search.cfr_avg = False
    cfg.search.cfr_plus = False
    cfg.search.value_targets_from_final_policy = True
    cfg.__post_init__()
    return cfg


def _make_model(cfg: Config, device: torch.device) -> torch.nn.Module:
    if cfg.model.name == ModelType.better_ffn:
        model = BetterFFN(
            num_actions=cfg.model.num_actions,
            hidden_dim=cfg.model.hidden_dim,
            range_hidden_dim=cfg.model.range_hidden_dim,
            ffn_dim=cfg.model.ffn_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            num_policy_layers=cfg.model.num_policy_layers,
            num_value_layers=cfg.model.num_value_layers,
            num_players=2,
            shared_trunk=cfg.model.shared_trunk,
            enforce_zero_sum=cfg.model.enforce_zero_sum,
            nonlinearity=cfg.model.nonlinearity,
        )
    else:
        model = RebelFFN(
            input_dim=cfg.model.input_dim,
            num_actions=cfg.model.num_actions,
            hidden_dim=cfg.model.hidden_dim,
            num_hidden_layers=cfg.model.num_hidden_layers,
            detach_value_head=cfg.model.detach_value_head,
            num_players=2,
            nonlinearity=cfg.model.nonlinearity,
            enforce_zero_sum=cfg.model.enforce_zero_sum,
        )
    cpu_rng = torch.Generator(device="cpu").manual_seed(42)
    model.init_weights(cpu_rng)
    return model.to(device).eval()


def _build_mixed_street_env(cfg: Config, device: torch.device) -> HUNLTensorEnv:
    n = cfg.num_envs
    assert n % 4 == 0, f"num_envs={n} must be divisible by 4"
    quarter = n // 4
    full = HUNLTensorEnv(
        num_envs=n,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    full.reset()

    for q_idx, target_street in enumerate([0, 1, 2, 3]):
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
        steps = 0
        while int(sub.street.min().item()) < target_street and steps < 18:
            bins = torch.ones(quarter, dtype=torch.long, device=device)
            bins[sub.done] = -1
            sub.step_bins(bins)
            steps += 1
        src_idx = torch.arange(quarter, dtype=torch.long, device=device)
        full.copy_state_from(sub, src_idx, src_idx + q_idx * quarter)
    return full


def _build_evaluator(cfg: Config) -> FusedSparseCFREvaluator:
    device = torch.device(cfg.device)
    env = _build_mixed_street_env(cfg, device)
    roots = torch.arange(cfg.num_envs, dtype=torch.long, device=device)
    model = _make_model(cfg, device)
    evaluator = FusedSparseCFREvaluator(
        model=model,
        device=device,
        cfg=cfg,
        compile_model=False,
    )
    evaluator.initialize_subgame(env, roots)
    evaluator.initialize_policy_and_beliefs()
    if evaluator.warm_start_iterations > 0:
        evaluator.warm_start()
    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values
    evaluator.t_sample = evaluator._get_sampling_schedule()
    evaluator._prepare_sample_update_table()
    return evaluator


def bench_variant(name: str, enabled: set[str], cfg: Config, args: argparse.Namespace) -> BenchResult:
    _set_flags(enabled)
    evaluator = _build_evaluator(cfg)

    next_t = max(evaluator.warm_start_iterations, evaluator.dcfr_delay + 1, 2)

    def step() -> None:
        nonlocal next_t
        if next_t >= evaluator.cfr_iterations:
            next_t = max(evaluator.warm_start_iterations, evaluator.dcfr_delay + 1, 2)
        evaluator.cfr_iteration(next_t)
        next_t += 1

    ms = _cuda_timed(step, n_iters=args.repeats, n_warmup=args.warmup)
    return BenchResult(
        name=name,
        ms_per_iter=ms,
        nodes=evaluator.total_nodes,
        model_leaves=int(evaluator.model_indices.numel()),
        showdown_leaves=int(evaluator.showdown_indices.numel()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--warm-start", type=int, default=4)
    parser.add_argument("--dcfr-delay", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model", choices=["better", "rebel"], default="better")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--range-hidden-dim", type=int, default=128)
    parser.add_argument("--ffn-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fused sparse benchmarks.")

    cfg = _make_cfg(args)
    variants: list[tuple[str, set[str]]] = [("baseline", set())]
    variants.extend((name, {name}) for name in OPT_FLAGS)
    selected = {"reuse_positive", "parent_sum_divide", "leaf_feature_cache"}
    variants.append(("selected", selected))
    variants.append(("selected_plus_child", selected | {"child_opp_policy"}))
    variants.append(("selected_plus_sparse", selected | {"sparse_sample"}))
    variants.append(("all", set(OPT_FLAGS)))
    variants.extend(
        (f"all_except_{name}", set(OPT_FLAGS) - {name})
        for name in OPT_FLAGS
    )

    results = [bench_variant(name, enabled, cfg, args) for name, enabled in variants]
    base = results[0].ms_per_iter
    print(
        f"num_envs={cfg.num_envs} depth={cfg.search.depth} "
        f"iters={cfg.search.iterations} model={args.model}"
    )
    print(f"nodes={results[0].nodes:,} model_leaves={results[0].model_leaves:,} "
          f"showdown_leaves={results[0].showdown_leaves:,}")
    print(f"{'variant':<24} {'ms/iter':>10} {'speedup':>10}")
    for result in results:
        print(f"{result.name:<24} {result.ms_per_iter:>10.3f} {base / result.ms_per_iter:>10.3f}x")


if __name__ == "__main__":
    main()
