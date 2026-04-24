#!/usr/bin/env python3
"""Profile SparseCFREvaluator methods with torch.profiler.

Wraps both the generic cfr_iteration sub-calls and the sparse-specific
methods (_fan_out, _push_down, _pull_back, _pull_back_sum,
_propagate_level_beliefs, _fan_out_deep, sample_leaves, _mask_invalid)
with record_function to attribute time.

Runs with vastly shortened defaults (few iterations, small num_envs, shallow
depth) so profiling completes quickly.
"""

from __future__ import annotations

import types
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

from p2.core.structured_config import CFRType, Config
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.rebel_ffn import RebelFFN
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


SPARSE_METHODS = [
    "_fan_out",
    "_fan_out_deep",
    "_push_down",
    "_pull_back",
    "_pull_back_sum",
    "_propagate_level_beliefs",
    "_mask_invalid",
    "sample_leaves",
    "_construct_subgame",
]

BASE_METHODS = [
    "set_leaf_values",
    "compute_expected_values",
    "compute_instantaneous_regrets",
    "update_policy",
    "_record_stats",
    "_get_mixing_weights",
    "_set_model_values",
    "_set_model_values_impl",
    "_get_model_policy_probs",
    "_calculate_reach_weights",
    "_propagate_all_beliefs",
    "_block_beliefs",
    "_normalize_beliefs",
    "_showdown_value",
    "_showdown_value_both",
]


def wrap_method(evaluator, name: str) -> None:
    """Wrap evaluator.<name> with a record_function tagged f'sparse.{name}'."""
    if not hasattr(evaluator, name):
        return
    original = getattr(evaluator, name)
    tag = f"sparse.{name}"

    def wrapped(*args, __orig=original, __tag=tag, **kwargs):
        with record_function(__tag):
            return __orig(*args, **kwargs)

    setattr(evaluator, name, wrapped)


def create_instrumented_cfr_iteration(original_method):
    def instrumented_cfr_iteration(self, t: int) -> None:
        with record_function("cfr_iteration.update_policy_probs_sample"):
            torch.where(
                (self.t_sample == t)[:, None],
                self.policy_probs,
                self.policy_probs_sample,
                out=self.policy_probs_sample,
            )

        with record_function("cfr_iteration.compute_instantaneous_regrets"):
            regrets = self.compute_instantaneous_regrets(self.latest_values)

        if self.cfr_type == CFRType.linear:
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

        with record_function("cfr_iteration.update_regret_weight_sums"):
            self.regret_weight_sums += 1
        with record_function("cfr_iteration.update_cumulative_regrets"):
            self.cumulative_regrets += regrets

        with record_function("cfr_iteration.clamp_regrets"):
            self.cumulative_regrets.clamp_(min=0)

        with record_function("cfr_iteration.clone_policy_probs"):
            old_policy_probs = self.policy_probs.clone()
        with record_function("cfr_iteration.update_policy"):
            self.update_policy(t)
        with record_function("cfr_iteration.record_stats"):
            self._record_stats(t, old_policy_probs)

        with record_function("cfr_iteration.set_leaf_values"):
            self.set_leaf_values(t)
        with record_function("cfr_iteration.compute_expected_values"):
            self.compute_expected_values()

        with record_function("cfr_iteration.get_mixing_weights"):
            old, new = self._get_mixing_weights(t)
        with record_function("cfr_iteration.update_values_avg"):
            self.values_avg *= old
            self.values_avg += new * self.latest_values
            self.values_avg /= old + new

    return instrumented_cfr_iteration


def profile_sparse(cfg: Config) -> None:
    device = torch.device(cfg.device)
    print(f"Device: {device}")
    print(
        f"num_envs={cfg.num_envs}, depth={cfg.search.depth}, "
        f"iterations={cfg.search.iterations}, warm_start={cfg.search.warm_start_iterations}"
    )

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)

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

    evaluator = SparseCFREvaluator(model=model, device=device, cfg=cfg)

    # Wrap sparse-specific methods BEFORE initialize_subgame so _construct_subgame is captured.
    for name in SPARSE_METHODS + BASE_METHODS:
        wrap_method(evaluator, name)

    evaluator.initialize_subgame(env, root_indices)
    evaluator.initialize_policy_and_beliefs()

    if evaluator.warm_start_iterations > 0:
        evaluator.warm_start()

    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values
    evaluator.t_sample = evaluator._get_sampling_schedule()

    original_cfr_iteration = evaluator.cfr_iteration
    instrumented_method = create_instrumented_cfr_iteration(original_cfr_iteration)
    evaluator.cfr_iteration = types.MethodType(instrumented_method, evaluator)

    if device.type == "cuda":
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        activities = [ProfilerActivity.CPU]

    warm_start_iterations = evaluator.warm_start_iterations
    num_profile_iterations = max(1, cfg.search.iterations - warm_start_iterations)
    print(f"Profiling {num_profile_iterations} iterations...")

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for t in range(
            warm_start_iterations,
            warm_start_iterations + num_profile_iterations,
        ):
            with record_function(f"cfr_iteration_t_{t}"):
                evaluator.cfr_iteration(t)
        if device.type == "cuda":
            torch.cuda.synchronize()

    sort_key = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )

    print("\n" + "=" * 80)
    print("ALL OPS — top by self time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=40))

    # Filter to just our tagged record_function entries.
    tagged_prefixes = ("cfr_iteration.", "sparse.", "cfr_iteration_t_")
    tagged = [
        e
        for e in prof.key_averages()
        if any(e.key.startswith(p) for p in tagged_prefixes)
    ]
    def _self_time(e):
        if device.type == "cuda":
            for attr in ("self_cuda_time_total", "self_device_time_total"):
                if hasattr(e, attr):
                    return getattr(e, attr)
        return e.self_cpu_time_total

    def _dev_total(e):
        for attr in ("cuda_time_total", "device_time_total"):
            if hasattr(e, attr):
                return getattr(e, attr)
        return 0

    tagged.sort(key=_self_time, reverse=True)
    print("\n" + "=" * 80)
    print("TAGGED RECORD_FUNCTION REGIONS (our wrappers)")
    print("=" * 80)
    header = f"{'name':<60s} {'count':>6s} {'cpu_tot_ms':>12s} {'cuda_tot_ms':>12s} {'cpu_avg_us':>12s} {'cuda_avg_us':>12s}"
    print(header)
    print("-" * len(header))
    for e in tagged:
        cpu_tot_ms = e.cpu_time_total / 1000.0
        cuda_tot_ms = _dev_total(e) / 1000.0
        cpu_avg_us = e.cpu_time_total / max(1, e.count)
        cuda_avg_us = _dev_total(e) / max(1, e.count)
        print(
            f"{e.key[:60]:<60s} {e.count:>6d} {cpu_tot_ms:>12.3f} {cuda_tot_ms:>12.3f} {cpu_avg_us:>12.1f} {cuda_avg_us:>12.1f}"
        )

    output_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_file = output_dir / f"sparse_methods_trace_{timestamp}.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\nTrace exported to {trace_file}")


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="config_rebel_cfr",
)
def main(dict_config: DictConfig) -> None:
    config = Config.from_dict_config(dict_config)
    config.use_wandb = False
    profile_sparse(config)


if __name__ == "__main__":
    main()
