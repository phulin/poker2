#!/usr/bin/env python3
"""
Microbenchmark comparing SparseCFREvaluator vs RebelCFREvaluator
on MPS for depth 3 and 4 trees with 150 iterations.
"""

import argparse
import time
from typing import Tuple

import torch

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
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
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


def create_mock_model(cfg: Config, device: torch.device) -> RebelFFN:
    """Create a simple mock model for benchmarking."""
    model = RebelFFN(
        input_dim=cfg.model.input_dim,
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
        detach_value_head=cfg.model.detach_value_head,
        num_players=2,
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
    cfg.model.name = ModelType.rebel_ffn
    cfg.model.input_dim = 2661
    cfg.model.hidden_dim = 512  # Smaller for benchmarking
    cfg.model.num_hidden_layers = 3  # Smaller for benchmarking
    cfg.model.value_head_type = "scalar"
    cfg.model.detach_value_head = True

    cfg.env = EnvConfig()
    cfg.env.stack = 1000
    cfg.env.sb = 5
    cfg.env.bb = 10
    cfg.env.bet_bins = [0.5, 1.5]
    cfg.env.flop_showdown = False

    cfg.search = SearchConfig()
    cfg.search.iterations = 150
    cfg.search.warm_start_iterations = 15
    cfg.search.branching = 4
    cfg.search.dcfr_alpha = 1.5
    cfg.search.dcfr_beta = 0.0
    cfg.search.dcfr_gamma = 2.0
    cfg.search.cfr_type = CFRType.linear
    cfg.search.cfr_avg = True

    # Ensure num_actions is set correctly (Config.__post_init__ does this, but let's be explicit)
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3

    return cfg


def setup_environment(
    cfg: Config, device: torch.device
) -> Tuple[HUNLTensorEnv, torch.Tensor]:
    """Set up environment and return root indices."""
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    env.reset()
    root_indices = torch.tensor([0], dtype=torch.long, device=device)
    return env, root_indices


def benchmark_rebel_cfr(
    cfg: Config,
    model: RebelFFN,
    env: HUNLTensorEnv,
    root_indices: torch.Tensor,
    depth: int,
    iterations: int,
    device: torch.device,
) -> Tuple[float, int]:
    """Benchmark RebelCFREvaluator."""
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    warm_start_iterations = min(
        cfg.search.warm_start_iterations, max(0, iterations - 1)
    )

    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=cfg.env.bet_bins,
        max_depth=depth,
        cfr_iterations=iterations,
        device=device,
        float_dtype=torch.float32,
        generator=generator,
        warm_start_iterations=warm_start_iterations,
        cfr_type=cfg.search.cfr_type,
        cfr_avg=cfg.search.cfr_avg,
        dcfr_alpha=cfg.search.dcfr_alpha,
        dcfr_beta=cfg.search.dcfr_beta,
        dcfr_gamma=cfg.search.dcfr_gamma,
        dcfr_delay=cfg.search.dcfr_plus_delay,
    )

    # Initialize search
    evaluator.initialize_subgame(env, root_indices)

    # Warmup
    evaluator.evaluate_cfr(training_mode=False)

    synchronize_device_if_needed(device)

    # Actual benchmark
    start = time.perf_counter()
    evaluator.evaluate_cfr(training_mode=False)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    elapsed_time = end - start
    num_nodes = evaluator.total_nodes

    return elapsed_time, num_nodes


def benchmark_sparse_cfr(
    cfg: Config,
    model: RebelFFN,
    env: HUNLTensorEnv,
    root_indices: torch.Tensor,
    depth: int,
    iterations: int,
    device: torch.device,
) -> Tuple[float, int]:
    """Benchmark SparseCFREvaluator.

    Note: Sparse evaluator builds tree dynamically based on legal actions,
    but is still limited by cfg.search.depth (max_depth).
    """
    # Set depth in config to match the requested depth for fair comparison
    cfg.search.depth = depth
    evaluator = SparseCFREvaluator(
        model=model,
        device=device,
        cfg=cfg,
    )

    # Initialize subgame (this builds the tree naturally based on legal actions)
    evaluator.initialize_subgame(env, root_indices)

    # Warmup
    evaluator.cfr_iterations = iterations
    evaluator.evaluate_cfr(training_mode=False)

    synchronize_device_if_needed(device)

    # Actual benchmark
    start = time.perf_counter()
    evaluator.evaluate_cfr(training_mode=False)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    elapsed_time = end - start
    num_nodes = evaluator.total_nodes

    return elapsed_time, num_nodes


def run_benchmark(
    device: torch.device,
    depth: int,
    iterations: int,
    cfg: Config,
    model: RebelFFN,
) -> None:
    """Run benchmark for a specific depth."""
    print(f"\n{'=' * 60}")
    print(f"Benchmarking Depth {depth}, {iterations} iterations")
    print(f"{'=' * 60}")

    env, root_indices = setup_environment(cfg, device)

    # Benchmark RebelCFREvaluator
    print("\nBenchmarking RebelCFREvaluator...")
    rebel_time, rebel_nodes = benchmark_rebel_cfr(
        cfg, model, env, root_indices, depth, iterations, device
    )
    print(f"RebelCFREvaluator: {rebel_time:.4f} s, {rebel_nodes:,} nodes")
    print(f"  Time per iteration: {rebel_time / iterations:.6f} s")
    print(f"  Nodes per second: {rebel_nodes / rebel_time:.2f}")

    # Benchmark SparseCFREvaluator
    print("\nBenchmarking SparseCFREvaluator...")
    sparse_time, sparse_nodes = benchmark_sparse_cfr(
        cfg, model, env, root_indices, depth, iterations, device
    )
    print(f"SparseCFREvaluator: {sparse_time:.4f} s, {sparse_nodes:,} nodes")
    print(f"  Time per iteration: {sparse_time / iterations:.6f} s")
    print(f"  Nodes per second: {sparse_nodes / sparse_time:.2f}")

    # Compare
    speedup = rebel_time / sparse_time if sparse_time > 0 else float("inf")
    print("\nComparison:")
    print(
        f"  Speedup: {speedup:.2f}x ({'Sparse' if speedup > 1 else 'Rebel'} is faster)"
    )
    print(
        f"  Node ratio: {sparse_nodes / rebel_nodes:.2f}x"
        if rebel_nodes > 0
        else "  Node ratio: N/A"
    )
    print("  Note: Sparse builds tree naturally (may differ from Rebel's fixed depth)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SparseCFREvaluator vs RebelCFREvaluator"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run benchmark on",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=150,
        help="Number of CFR iterations",
    )
    parser.add_argument(
        "--depths",
        type=str,
        default="3,4",
        help="Comma-separated list of depths to benchmark",
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

    # Parse depths
    depths = [int(d.strip()) for d in args.depths.split(",")]

    print(f"\n{'=' * 60}")
    print("CFR Evaluator Benchmark")
    print(f"Device: {device}")
    print(f"Iterations: {args.iterations}")
    print(f"Depths: {depths}")
    print(f"{'=' * 60}")

    # Create config and model
    cfg = create_config()
    model = create_mock_model(cfg, device)

    # Run benchmarks for each depth
    for depth in depths:
        run_benchmark(device, depth, args.iterations, cfg, model)

    print(f"\n{'=' * 60}")
    print("Benchmark completed!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
