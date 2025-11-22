#!/usr/bin/env python3
"""
Benchmark comparing two approaches to _set_model_values in SparseCFREvaluator:
1. Current: Encode all features, then index with model_indices
2. Alternative: Index beliefs/new_street_mask first, then encode only those
"""

import argparse
import time
from typing import Tuple

import torch
from torch.testing import assert_close

from alphaholdem.core.structured_config import (
    CFRType,
    Config,
    EnvConfig,
    ModelConfig,
    ModelType,
    SearchConfig,
    TrainingConfig,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.search.sparse_cfr_evaluator import SparseCFREvaluator


def synchronize_device_if_needed(device: torch.device) -> None:
    """Synchronize device operations for accurate timing."""
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.synchronize()


def create_mock_model(cfg: Config, device: torch.device) -> BetterFFN:
    """Create a simple mock model for benchmarking."""
    model = BetterFFN(
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        range_hidden_dim=getattr(cfg.model, "range_hidden_dim", 128),
        ffn_dim=getattr(cfg.model, "ffn_dim", 1024),
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_policy_layers=getattr(cfg.model, "num_policy_layers", 3),
        num_value_layers=getattr(cfg.model, "num_value_layers", 3),
        num_players=2,
        shared_trunk=getattr(cfg.model, "shared_trunk", True),
        enforce_zero_sum=getattr(cfg.model, "enforce_zero_sum", True),
    )
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(42)
    model.init_weights(cpu_rng)
    model.to(device)
    model.eval()
    return model


def create_config() -> Config:
    """Create a default configuration for benchmarking."""
    cfg = Config()
    cfg.device = "mps"
    cfg.seed = 42

    cfg.train = TrainingConfig()
    cfg.train.batch_size = 1024

    cfg.model = ModelConfig()
    cfg.model.name = ModelType.better_ffn
    cfg.model.hidden_dim = 512
    cfg.model.range_hidden_dim = 128
    cfg.model.ffn_dim = 1024
    cfg.model.num_hidden_layers = 3
    cfg.model.num_policy_layers = 3
    cfg.model.num_value_layers = 3
    cfg.model.shared_trunk = True
    cfg.model.enforce_zero_sum = True

    cfg.env = EnvConfig()
    cfg.env.stack = 1000
    cfg.env.sb = 5
    cfg.env.bb = 10
    cfg.env.bet_bins = [0.5, 1.5]
    cfg.env.flop_showdown = False

    cfg.search = SearchConfig()
    cfg.search.iterations = 10
    cfg.search.warm_start_iterations = 5
    cfg.search.depth = 3
    cfg.search.branching = 4
    cfg.search.dcfr_alpha = 1.5
    cfg.search.dcfr_beta = 0.0
    cfg.search.dcfr_gamma = 2.0
    cfg.search.cfr_type = CFRType.linear
    cfg.search.cfr_avg = True

    cfg.model.num_actions = len(cfg.env.bet_bins) + 3

    return cfg


def setup_evaluator(
    cfg: Config,
    model: BetterFFN,
    device: torch.device,
    num_envs: int = 1,
) -> Tuple[SparseCFREvaluator, HUNLTensorEnv, torch.Tensor]:
    """Set up evaluator and return it along with env and root indices."""
    env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    env.reset()
    root_indices = torch.arange(num_envs, dtype=torch.long, device=device)

    evaluator = SparseCFREvaluator(
        model=model,
        device=device,
        cfg=cfg,
    )
    evaluator.initialize_subgame(env, root_indices)

    return evaluator, env, root_indices


@torch.no_grad()
def _set_model_values_current(
    evaluator: SparseCFREvaluator,
    t: int,
    beliefs: torch.Tensor | None = None,
) -> None:
    """Current approach: encode all features, then index with model_indices."""
    if beliefs is None:
        beliefs = evaluator.beliefs_avg if evaluator.cfr_avg else evaluator.beliefs

    features = evaluator.feature_encoder.encode(
        beliefs, pre_chance_node=evaluator.new_street_mask
    )
    model_output = evaluator.model(features[evaluator.model_indices])

    if not evaluator.cfr_avg or t <= 1 or evaluator.last_model_values is None:
        # Use torch.index_copy for out-of-place operation (torch.compile compatible)
        new_values = torch.index_copy(
            evaluator.latest_values,
            0,
            evaluator.model_indices,
            model_output.hand_values,
        )
        evaluator.latest_values = new_values
    else:
        old, new = evaluator._get_mixing_weights(t)
        unmixed = (
            old + new
        ) * model_output.hand_values - old * evaluator.last_model_values
        unmixed /= new
        # Use torch.index_copy for out-of-place operation (torch.compile compatible)
        new_values = torch.index_copy(
            evaluator.latest_values,
            0,
            evaluator.model_indices,
            unmixed,
        )
        evaluator.latest_values = evaluator._maybe_enforce_zero_sum(
            new_values,
            evaluator.beliefs,
            ignore_mask=evaluator.env.done,
        )
    evaluator.last_model_values = model_output.hand_values.clone()


@torch.no_grad()
def _set_model_values_alternative(
    evaluator: SparseCFREvaluator,
    t: int,
    beliefs: torch.Tensor | None = None,
    temp_encoder=None,
    indexed_env=None,
) -> None:
    """Alternative approach: index beliefs/new_street_mask first, then encode only those."""
    if beliefs is None:
        beliefs = evaluator.beliefs_avg if evaluator.cfr_avg else evaluator.beliefs

    # Index beliefs and new_street_mask first
    indexed_beliefs = beliefs[evaluator.model_indices]
    indexed_new_street_mask = evaluator.new_street_mask[evaluator.model_indices]

    # Encode only the indexed subset using the pre-created encoder
    # (indexed_env state was already copied outside the loop)
    features = temp_encoder.encode(
        indexed_beliefs, pre_chance_node=indexed_new_street_mask
    )
    model_output = evaluator.model(features)

    if not evaluator.cfr_avg or t <= 1 or evaluator.last_model_values is None:
        # Use torch.index_copy for out-of-place operation (torch.compile compatible)
        new_values = torch.index_copy(
            evaluator.latest_values,
            0,
            evaluator.model_indices,
            model_output.hand_values,
        )
        evaluator.latest_values = new_values
    else:
        old, new = evaluator._get_mixing_weights(t)
        unmixed = (
            old + new
        ) * model_output.hand_values - old * evaluator.last_model_values
        unmixed /= new
        # Use torch.index_copy for out-of-place operation (torch.compile compatible)
        new_values = torch.index_copy(
            evaluator.latest_values,
            0,
            evaluator.model_indices,
            unmixed,
        )
        evaluator.latest_values = evaluator._maybe_enforce_zero_sum(
            new_values,
            evaluator.beliefs,
            ignore_mask=evaluator.env.done,
        )
    evaluator.last_model_values = model_output.hand_values.clone()


@torch.no_grad()
def verify_correctness(
    cfg: Config,
    model: BetterFFN,
    device: torch.device,
    num_envs: int = 1,
) -> None:
    """Verify that both approaches produce identical results."""
    print("Verifying correctness...")

    # Create two evaluators with identical state
    evaluator1, env1, root_indices1 = setup_evaluator(cfg, model, device, num_envs)
    evaluator2, env2, root_indices2 = setup_evaluator(cfg, model, device, num_envs)

    # Ensure they have the same state
    evaluator2.beliefs.copy_(evaluator1.beliefs)
    evaluator2.beliefs_avg.copy_(evaluator1.beliefs_avg)
    evaluator2.last_model_values = (
        evaluator1.last_model_values.clone()
        if evaluator1.last_model_values is not None
        else None
    )

    # Create temporary encoder and environment for alternative approach
    from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
    from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder

    num_indexed = evaluator2.model_indices.numel()
    indexed_env = HUNLTensorEnv.from_proto(evaluator2.env, num_envs=num_indexed)
    # Copy state once (env state doesn't change)
    indexed_env.copy_state_from(
        evaluator2.env,
        evaluator2.model_indices,
        torch.arange(num_indexed, device=evaluator2.device),
    )

    if isinstance(evaluator2.feature_encoder, BetterFeatureEncoder):
        temp_encoder = BetterFeatureEncoder(
            env=indexed_env,
            device=evaluator2.device,
            dtype=evaluator2.float_dtype,
        )
    else:
        temp_encoder = RebelFeatureEncoder(
            env=indexed_env,
            device=evaluator2.device,
            dtype=evaluator2.float_dtype,
        )

    # Run both approaches
    t = 2  # Use t > 1 to test the mixing path
    _set_model_values_current(evaluator1, t)
    _set_model_values_alternative(
        evaluator2, t, temp_encoder=temp_encoder, indexed_env=indexed_env
    )

    # Compare results
    try:
        assert_close(
            evaluator1.latest_values,
            evaluator2.latest_values,
            msg="latest_values should match",
        )
        assert_close(
            evaluator1.last_model_values,
            evaluator2.last_model_values,
            msg="last_model_values should match",
        )
        print("✓ Correctness check passed: both approaches produce identical results")
    except AssertionError as e:
        print(f"✗ Correctness check failed: {e}")


@torch.no_grad()
def benchmark_approach(
    evaluator: SparseCFREvaluator,
    approach_fn,
    t: int,
    repeats: int,
    device: torch.device,
    **kwargs,
) -> float:
    """Benchmark a single approach."""
    # Warmup
    for _ in range(3):
        approach_fn(evaluator, t, **kwargs)

    synchronize_device_if_needed(device)

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(repeats):
        approach_fn(evaluator, t, **kwargs)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    return end - start


@torch.no_grad()
def run_benchmark(
    device: torch.device,
    depth: int,
    repeats: int,
    cfg: Config,
    model: BetterFFN,
    num_envs: int = 1,
) -> None:
    """Run benchmark for a specific depth."""
    print(f"\n{'='*60}")
    print(f"Benchmarking _set_model_values indexing approaches")
    print(f"Depth: {depth}, Repeats: {repeats}")
    print(f"{'='*60}")

    cfg.search.depth = depth

    # Verify correctness first
    try:
        verify_correctness(cfg, model, device, num_envs)
    except Exception as e:
        print(f"Correctness check failed: {e}")
        return

    # Setup evaluators for benchmarking
    evaluator_current, _, _ = setup_evaluator(cfg, model, device, num_envs)
    evaluator_alternative, _, _ = setup_evaluator(cfg, model, device, num_envs)

    # Ensure same state
    evaluator_alternative.beliefs.copy_(evaluator_current.beliefs)
    evaluator_alternative.beliefs_avg.copy_(evaluator_current.beliefs_avg)

    # Create temporary encoder and environment for alternative approach (once, outside the loop)
    from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
    from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder

    num_indexed = evaluator_alternative.model_indices.numel()
    indexed_env = HUNLTensorEnv.from_proto(
        evaluator_alternative.env, num_envs=num_indexed
    )
    # Copy state once outside the loop (env state doesn't change during benchmark)
    indexed_env.copy_state_from(
        evaluator_alternative.env,
        evaluator_alternative.model_indices,
        torch.arange(num_indexed, device=evaluator_alternative.device),
    )

    if isinstance(evaluator_alternative.feature_encoder, BetterFeatureEncoder):
        temp_encoder = BetterFeatureEncoder(
            env=indexed_env,
            device=evaluator_alternative.device,
            dtype=evaluator_alternative.float_dtype,
        )
    else:
        temp_encoder = RebelFeatureEncoder(
            env=indexed_env,
            device=evaluator_alternative.device,
            dtype=evaluator_alternative.float_dtype,
        )

    t = 2  # Use t > 1 to test the mixing path

    # Benchmark current approach
    print("\nBenchmarking current approach (encode all, then index)...")
    time_current = benchmark_approach(
        evaluator_current, _set_model_values_current, t, repeats, device
    )

    # Benchmark alternative approach
    print("Benchmarking alternative approach (index first, then encode)...")
    time_alternative = benchmark_approach(
        evaluator_alternative,
        _set_model_values_alternative,
        t,
        repeats,
        device,
        temp_encoder=temp_encoder,
        indexed_env=indexed_env,
    )

    # Report results
    num_model_nodes = evaluator_current.model_indices.numel()
    total_nodes = evaluator_current.total_nodes

    print(f"\nResults:")
    print(f"  Total nodes: {total_nodes:,}")
    print(
        f"  Model nodes: {num_model_nodes:,} ({100 * num_model_nodes / total_nodes:.1f}%)"
    )
    print(
        f"  Current approach:    {time_current:.6f} s total, {time_current / repeats:.6e} s/iter"
    )
    print(
        f"  Alternative approach: {time_alternative:.6f} s total, {time_alternative / repeats:.6e} s/iter"
    )

    if time_alternative > 0:
        speedup = time_current / time_alternative
        print(
            f"  Speedup: {speedup:.2f}x ({'Alternative' if speedup > 1 else 'Current'} is faster)"
        )
    else:
        print(f"  Speedup: N/A")


@torch.no_grad()
def run_benchmark_compile(
    device: torch.device,
    depth: int,
    repeats: int,
    cfg: Config,
    model: BetterFFN,
    num_envs: int = 1,
) -> None:
    """Run benchmark comparing current vs compiled _set_model_values with different modes."""
    print(f"\n{'='*60}")
    print(f"Benchmarking _set_model_values: Current vs Compiled (all modes)")
    print(f"Depth: {depth}, Repeats: {repeats}")
    print(f"{'='*60}")

    cfg.search.depth = depth

    # Setup evaluators for benchmarking
    evaluator_current, _, _ = setup_evaluator(cfg, model, device, num_envs)
    evaluators_compiled = {}
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        eval_compiled, _, _ = setup_evaluator(cfg, model, device, num_envs)
        eval_compiled.beliefs.copy_(evaluator_current.beliefs)
        eval_compiled.beliefs_avg.copy_(evaluator_current.beliefs_avg)
        evaluators_compiled[mode] = eval_compiled

    t = 2  # Use t > 1 to test the mixing path

    # Verify correctness first
    print("Verifying correctness...")
    try:
        # Run both approaches once to verify they produce the same results
        evaluator_test1, _, _ = setup_evaluator(cfg, model, device, num_envs)
        for mode in ["default", "reduce-overhead", "max-autotune"]:
            evaluator_test2, _, _ = setup_evaluator(cfg, model, device, num_envs)
            evaluator_test2.beliefs.copy_(evaluator_test1.beliefs)
            evaluator_test2.beliefs_avg.copy_(evaluator_test1.beliefs_avg)
            evaluator_test2.last_model_values = (
                evaluator_test1.last_model_values.clone()
                if evaluator_test1.last_model_values is not None
                else None
            )

            # Compile the function
            compiled_fn = torch.compile(_set_model_values_current, mode=mode)

            _set_model_values_current(evaluator_test1, t)
            compiled_fn(evaluator_test2, t)

            assert_close(
                evaluator_test1.latest_values,
                evaluator_test2.latest_values,
                msg=f"latest_values should match for mode {mode}",
            )
            assert_close(
                evaluator_test1.last_model_values,
                evaluator_test2.last_model_values,
                msg=f"last_model_values should match for mode {mode}",
            )
        print(
            "✓ Correctness check passed: all compiled modes produce identical results"
        )
    except Exception as e:
        print(f"✗ Correctness check failed: {e}")

    # Benchmark current approach
    print("\nBenchmarking current approach (uncompiled)...")
    synchronize_device_if_needed(device)
    start = time.perf_counter()
    for _ in range(repeats):
        _set_model_values_current(evaluator_current, t)
    synchronize_device_if_needed(device)
    time_current = time.perf_counter() - start

    # Benchmark each compilation mode
    results = {}
    for mode in ["default", "reduce-overhead", "max-autotune"]:
        print(f"\nBenchmarking compiled approach (mode: {mode})...")
        evaluator_compiled = evaluators_compiled[mode]

        synchronize_device_if_needed(device)
        start = time.perf_counter()
        compiled_fn = torch.compile(_set_model_values_current, mode=mode)
        # Warmup to trigger compilation
        for _ in range(3):
            compiled_fn(evaluator_compiled, t)
        synchronize_device_if_needed(device)
        compilation_time = time.perf_counter() - start

        # Actual benchmark with compiled function
        synchronize_device_if_needed(device)
        start = time.perf_counter()
        for _ in range(repeats):
            compiled_fn(evaluator_compiled, t)
        synchronize_device_if_needed(device)
        execution_time = time.perf_counter() - start

        time_compiled_total = compilation_time + execution_time
        results[mode] = {
            "compilation_time": compilation_time,
            "execution_time": execution_time,
            "total_time": time_compiled_total,
        }

    # Report results
    num_model_nodes = evaluator_current.model_indices.numel()
    total_nodes = evaluator_current.total_nodes

    print(f"\nResults:")
    print(f"  Total nodes: {total_nodes:,}")
    print(
        f"  Model nodes: {num_model_nodes:,} ({100 * num_model_nodes / total_nodes:.1f}%)"
    )
    print(
        f"  Current (uncompiled):     {time_current:.6f} s total, {time_current / repeats:.6e} s/iter"
    )
    print()

    for mode in ["default", "reduce-overhead", "max-autotune"]:
        r = results[mode]
        print(f"  {mode:20s}:")
        print(
            f"    Total:                 {r['total_time']:.6f} s total, {r['total_time'] / repeats:.6e} s/iter"
        )
        print(f"    Compilation time:     {r['compilation_time']:.6f} s")
        print(
            f"    Execution time:       {r['execution_time']:.6f} s total, {r['execution_time'] / repeats:.6e} s/iter"
        )
        speedup_total = time_current / r["total_time"] if r["total_time"] > 0 else 0
        speedup_execution = (
            time_current / r["execution_time"]
            if r["execution_time"] > 0
            else float("inf")
        )
        print(
            f"    Speedup (total):       {speedup_total:.2f}x ({'Compiled' if speedup_total > 1 else 'Uncompiled'} is faster)"
        )
        print(
            f"    Speedup (execution):   {speedup_execution:.2f}x ({'Compiled' if speedup_execution > 1 else 'Uncompiled'} is faster)"
        )
        print()

    # Compare modes
    print("Mode comparison (execution time only):")
    best_execution = min(results.items(), key=lambda x: x[1]["execution_time"])
    worst_execution = max(results.items(), key=lambda x: x[1]["execution_time"])
    print(
        f"  Fastest: {best_execution[0]} ({best_execution[1]['execution_time'] / repeats:.6e} s/iter)"
    )
    print(
        f"  Slowest: {worst_execution[0]} ({worst_execution[1]['execution_time'] / repeats:.6e} s/iter)"
    )
    if worst_execution[1]["execution_time"] > 0:
        print(
            f"  Ratio:   {worst_execution[1]['execution_time'] / best_execution[1]['execution_time']:.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark _set_model_values indexing approaches"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run benchmark on",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of times to repeat the benchmark",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="3,4",
        help="Comma-separated list of depths to benchmark",
    )
    parser.add_argument(
        "--benchmark-type",
        type=str,
        default="both",
        choices=["indexing", "compile", "both"],
        help="Type of benchmark to run: indexing comparison, compile comparison, or both",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments to use",
    )
    args = parser.parse_args()

    device_str = args.device
    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available; exiting.")
        return
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; exiting.")
        return

    device = torch.device(device_str)

    # Enable TF32 for CUDA (Ampere and later GPUs)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for CUDA operations")

    # Parse depths
    depths = [int(d.strip()) for d in args.depths.split(",")]

    print(f"\n{'='*60}")
    print(f"_set_model_values Indexing Benchmark")
    print(f"Device: {device}")
    print(f"Repeats: {args.repeats}")
    print(f"Depths: {depths}")
    print(f"Num envs: {args.num_envs}")
    print(f"{'='*60}")

    # Create config and model
    cfg = create_config()
    model = create_mock_model(cfg, device)

    # Run benchmarks for each depth
    for depth in depths:
        if args.benchmark_type in ("indexing", "both"):
            run_benchmark(device, depth, args.repeats, cfg, model, args.num_envs)
        if args.benchmark_type in ("compile", "both"):
            run_benchmark_compile(
                device, depth, args.repeats, cfg, model, args.num_envs
            )

    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
