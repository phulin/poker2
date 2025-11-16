#!/usr/bin/env python3
"""
Profile cfr_iteration using PyTorch profiler with record_function around each call.

This script instruments the cfr_iteration method to add record_function calls
around each major operation for detailed profiling.

Usage:
    python debugging/profile_cfr_iteration.py
    python debugging/profile_cfr_iteration.py search.iterations=10 search.depth=3
    python debugging/profile_cfr_iteration.py search.sparse=true search.iterations=10
    python debugging/profile_cfr_iteration.py num_envs=4 search.iterations=10
"""

from __future__ import annotations

import types
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

from alphaholdem.core.structured_config import CFRType, Config
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator
from alphaholdem.search.sparse_cfr_evaluator import SparseCFREvaluator


def create_instrumented_cfr_iteration(original_method):
    """Create an instrumented version of cfr_iteration with record_function around each call."""

    def instrumented_cfr_iteration(self, t: int) -> None:
        """Instrumented version of cfr_iteration with profiling."""
        with record_function("cfr_iteration.update_policy_probs_sample"):
            torch.where(
                (self.t_sample == t)[:, None],
                self.policy_probs,
                self.policy_probs_sample,
                out=self.policy_probs_sample,
            )

        # Compute regrets
        with record_function("cfr_iteration.compute_instantaneous_regrets"):
            regrets = self.compute_instantaneous_regrets(self.latest_values)

        if self.cfr_type == CFRType.linear:  # Alternate updates.
            with record_function("cfr_iteration.masked_fill_linear"):
                regrets.masked_fill_(
                    self.prev_actor[:, None] == t % self.num_players, 0.0
                )
        elif self.cfr_type in [CFRType.discounted, CFRType.discounted_plus]:
            with record_function("cfr_iteration.dcfr_numerator"):
                numerator = torch.where(
                    self.cumulative_regrets > 0,
                    t**self.dcfr_alpha,
                    t**self.dcfr_beta,
                )
            with record_function("cfr_iteration.dcfr_denominator"):
                denominator = torch.where(
                    self.cumulative_regrets > 0,
                    (t + 1) ** self.dcfr_alpha,
                    (t + 1) ** self.dcfr_beta,
                )
            with record_function("cfr_iteration.dcfr_update_cumulative_regrets"):
                self.cumulative_regrets *= numerator
                self.cumulative_regrets /= denominator
            with record_function("cfr_iteration.dcfr_update_regret_weight_sums"):
                self.regret_weight_sums *= numerator
                self.regret_weight_sums /= denominator

        # Update cumulative regrets
        with record_function("cfr_iteration.update_regret_weight_sums"):
            self.regret_weight_sums += 1
        with record_function("cfr_iteration.update_cumulative_regrets"):
            self.cumulative_regrets += regrets

        # CFR+ trick: clamp regrets to non-negative
        with record_function("cfr_iteration.clamp_regrets"):
            self.cumulative_regrets.clamp_(min=0)

        # Update policy
        with record_function("cfr_iteration.clone_policy_probs"):
            old_policy_probs = self.policy_probs.clone()
        with record_function("cfr_iteration.update_policy"):
            self.update_policy(t)
        with record_function("cfr_iteration.record_stats"):
            self._record_stats(t, old_policy_probs)

        # Set leaf values and back up
        with record_function("cfr_iteration.set_leaf_values"):
            self.set_leaf_values(t)
        with record_function("cfr_iteration.compute_expected_values"):
            self.compute_expected_values()

        # Update average values
        with record_function("cfr_iteration.get_mixing_weights"):
            old, new = self._get_mixing_weights(t)
        with record_function("cfr_iteration.update_values_avg"):
            self.values_avg *= old
            self.values_avg += new * self.latest_values
            self.values_avg /= old + new

    return instrumented_cfr_iteration


def profile_cfr_iteration(cfg: Config) -> None:
    """Run CFR iteration profiling."""
    device = torch.device(cfg.device)

    print(f"Using device: {device}")
    print(f"Config: batch_size={cfg.train.batch_size}, num_envs={cfg.num_envs}")
    print(f"Search: depth={cfg.search.depth}, iterations={cfg.search.iterations}")
    print(f"Search: sparse={cfg.search.sparse}")
    evaluator_type = "SparseCFREvaluator" if cfg.search.sparse else "RebelCFREvaluator"
    print(f"Using evaluator: {evaluator_type}")
    print("\nInitializing evaluator...")

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)

    # Create environment
    env = HUNLTensorEnv(
        num_envs=cfg.num_envs,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    env.reset()
    root_indices = torch.arange(cfg.num_envs, dtype=torch.long, device=device)

    # Create model
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

    # Create evaluator based on config.search.sparse
    if cfg.search.sparse:
        evaluator = SparseCFREvaluator(
            model=model,
            device=device,
            cfg=cfg,
        )
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.seed)

        warm_start_iterations = min(
            cfg.search.warm_start_iterations,
            max(0, cfg.search.iterations - 1),
        )

        evaluator = RebelCFREvaluator(
            search_batch_size=cfg.num_envs,
            env_proto=env,
            model=model,
            bet_bins=cfg.env.bet_bins,
            max_depth=cfg.search.depth,
            cfr_iterations=cfg.search.iterations,
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

    # Initialize subgame
    evaluator.initialize_subgame(env, root_indices)
    evaluator.initialize_policy_and_beliefs()

    if evaluator.warm_start_iterations > 0:
        evaluator.warm_start()

    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values
    evaluator.t_sample = evaluator._get_sampling_schedule()

    # Instrument the cfr_iteration method
    original_cfr_iteration = evaluator.cfr_iteration
    instrumented_method = create_instrumented_cfr_iteration(original_cfr_iteration)
    evaluator.cfr_iteration = types.MethodType(instrumented_method, evaluator)

    print("Evaluator initialized. Starting profiling...")
    print("=" * 80)

    # Configure profiler activities
    if device.type == "cuda":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    elif device.type == "mps":
        # MPS doesn't support CUDA profiling, use CPU only
        activities = [ProfilerActivity.CPU]
    else:
        activities = [ProfilerActivity.CPU]

    # Use profiler with detailed options
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        # Run CFR iterations with profiling
        warm_start_iterations = evaluator.warm_start_iterations
        available_iterations = cfg.search.iterations - warm_start_iterations
        if available_iterations <= 0:
            print(
                f"Warning: No iterations available to profile (iterations={cfg.search.iterations}, warm_start={warm_start_iterations})"
            )
            return

        num_profile_iterations = available_iterations
        print(f"\nProfiling {num_profile_iterations} CFR iterations...")

        for t in range(
            warm_start_iterations,
            warm_start_iterations + num_profile_iterations,
        ):
            print(f"  Running iteration {t}...")
            with record_function(f"cfr_iteration_t_{t}"):
                evaluator.cfr_iteration(t)

    print("\n" + "=" * 80)
    print("Profiling complete. Processing results...")

    # Export profiling results
    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get summary statistics
    print("\nComputing summary statistics...")
    key_averages_summary = prof.key_averages(group_by_input_shape=False)

    # Print summary table
    print("\n" + "=" * 80)
    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    print(
        f"PROFILING SUMMARY - Top Operations by Self {('CUDA' if device.type == 'cuda' else 'CPU')} Time"
    )
    print("=" * 80)
    print(
        key_averages_summary.table(
            sort_by=sort_key,
            row_limit=50,
        )
    )

    # Print grouped by function name
    print("\n" + "=" * 80)
    print("PROFILING SUMMARY - Grouped by Function Name")
    print("=" * 80)
    key_averages_grouped = prof.key_averages(group_by_stack_n=0)
    print(
        key_averages_grouped.table(
            sort_by=sort_key,
            row_limit=50,
        )
    )

    # Export trace file
    trace_file = output_dir / f"cfr_iteration_trace_{timestamp}.json"
    print(f"\nExporting trace to {trace_file}...")
    prof.export_chrome_trace(str(trace_file))
    print(f"Trace exported to {trace_file}")

    # Export stack trace
    stack_file = output_dir / f"cfr_iteration_stacks_{timestamp}.txt"
    print(f"Exporting stack trace to {stack_file}...")
    with open(stack_file, "w") as f:
        f.write(prof.key_averages(group_by_stack_n=5).table(sort_by=sort_key))
    print(f"Stack trace exported to {stack_file}")

    print("\n" + "=" * 80)
    print("Profiling complete!")
    print("=" * 80)


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config_rebel_cfr",
)
def main(dict_config: DictConfig) -> None:
    """Main entry point with Hydra config."""
    # Convert DictConfig to Config
    config = Config.from_dict_config(dict_config)

    # Override some defaults for profiling
    config.use_wandb = False  # Disable wandb for profiling

    # Warn if iterations is very high (profiling can be slow)
    if config.search.iterations > 50:
        print(
            f"Warning: search.iterations={config.search.iterations} is high for profiling. This may take a while."
        )

    print("Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Search depth: {config.search.depth}")
    print(f"  Search iterations: {config.search.iterations}")
    print(f"  Warm start iterations: {config.search.warm_start_iterations}")
    print(f"  Search sparse: {config.search.sparse}")
    print()

    profile_cfr_iteration(config)


if __name__ == "__main__":
    main()
