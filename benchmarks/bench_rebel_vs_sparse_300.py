#!/usr/bin/env python3
"""
Benchmark comparing RebelCFREvaluator vs SparseCFREvaluator
with 300 CFR iterations.
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
from alphaholdem.models.mlp.better_ffn import BetterFFN
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


def create_mock_model(cfg: Config, device: torch.device) -> BetterFFN:
    """Create a simple mock model for benchmarking."""
    model = BetterFFN(
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        range_hidden_dim=cfg.model.range_hidden_dim,
        ffn_dim=cfg.model.ffn_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_policy_layers=cfg.model.num_policy_layers,
        num_value_layers=cfg.model.num_value_layers,
        num_players=2,
    )
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(42)
    model.init_weights(cpu_rng)
    model.to(device)
    model.eval()
    return model


def create_config(iterations: int = 300) -> Config:
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
    cfg.model.value_head_type = "scalar"

    cfg.env = EnvConfig()
    cfg.env.stack = 1000
    cfg.env.sb = 5
    cfg.env.bb = 10
    cfg.env.bet_bins = [0.5, 1.5]
    cfg.env.flop_showdown = False

    cfg.search = SearchConfig()
    cfg.search.iterations = iterations
    cfg.search.warm_start_iterations = 30
    cfg.search.depth = 4
    cfg.search.branching = 4
    cfg.search.dcfr_alpha = 1.5
    cfg.search.dcfr_beta = 0.0
    cfg.search.dcfr_gamma = 2.0
    cfg.search.cfr_type = CFRType.linear
    cfg.search.cfr_avg = True

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
    model: BetterFFN,
    env: HUNLTensorEnv,
    root_indices: torch.Tensor,
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
        max_depth=cfg.search.depth,
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
    evaluator.initialize_search(env, root_indices)

    synchronize_device_if_needed(device)

    # Run 300 iterations
    start = time.perf_counter()
    evaluator.evaluate_cfr(training_mode=False)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    elapsed_time = end - start
    num_nodes = evaluator.total_nodes

    return elapsed_time, num_nodes


def benchmark_sparse_cfr(
    cfg: Config,
    model: BetterFFN,
    env: HUNLTensorEnv,
    root_indices: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> Tuple[float, int]:
    """Benchmark SparseCFREvaluator."""
    evaluator = SparseCFREvaluator(
        model=model,
        device=device,
        cfg=cfg,
    )

    # Initialize subgame
    evaluator.initialize_subgame(env, root_indices)

    synchronize_device_if_needed(device)

    # Run 300 iterations
    start = time.perf_counter()
    evaluator.evaluate_cfr(num_iterations=iterations)
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    elapsed_time = end - start
    num_nodes = evaluator.total_nodes

    return elapsed_time, num_nodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark RebelCFREvaluator vs SparseCFREvaluator"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Compute device to use (default: cpu)",
    )
    args = parser.parse_args()

    iterations = 1000
    device_str = args.device

    if device_str == "mps" and not torch.backends.mps.is_available():
        print("MPS not available; falling back to CPU.")
        device_str = "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device_str = "cpu"

    device = torch.device(device_str)

    print(f"\n{'='*60}")
    print(f"CFR Evaluator Benchmark: RebelCFREvaluator vs SparseCFREvaluator")
    print(f"Device: {device}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}")

    # Create config and model
    cfg = create_config(iterations=iterations)
    model = create_mock_model(cfg, device)

    # Setup environment
    env, root_indices = setup_environment(cfg, device)

    # Benchmark SparseCFREvaluator
    print("\nInitializing and running SparseCFREvaluator...")
    sparse_time, sparse_nodes = benchmark_sparse_cfr(
        cfg, model, env, root_indices, iterations, device
    )
    print(f"SparseCFREvaluator:")
    print(f"  Total time: {sparse_time:.4f} s")
    print(f"  Total nodes: {sparse_nodes:,}")
    print(f"  Time per iteration: {sparse_time / iterations:.6f} s")
    print(f"  Nodes per second: {sparse_nodes / sparse_time:.2f}")

    # Benchmark RebelCFREvaluator
    print("\nInitializing and running RebelCFREvaluator...")
    rebel_time, rebel_nodes = benchmark_rebel_cfr(
        cfg, model, env, root_indices, iterations, device
    )
    print(f"RebelCFREvaluator:")
    print(f"  Total time: {rebel_time:.4f} s")
    print(f"  Total nodes: {rebel_nodes:,}")
    print(f"  Time per iteration: {rebel_time / iterations:.6f} s")
    print(f"  Nodes per second: {rebel_nodes / rebel_time:.2f}")

    # Compare
    speedup = rebel_time / sparse_time if sparse_time > 0 else float("inf")
    print(f"\n{'='*60}")
    print(f"Comparison:")
    print(
        f"  Speedup: {speedup:.2f}x ({'Sparse' if speedup > 1 else 'Rebel'} is faster)"
    )
    if rebel_nodes > 0:
        print(f"  Node ratio: {sparse_nodes / rebel_nodes:.2f}x")
    print(f"  Time difference: {abs(rebel_time - sparse_time):.4f} s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
