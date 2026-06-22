#!/usr/bin/env python3
"""Breakdown benchmark: per-step GPU time across major training components.

Wraps the following regions with torch.profiler record_function tags so the
profiler can attribute their CUDA self-time:

    cfr_iter            — one cfr_iteration call (per-CFR-iter cost)
    cfr_model_fwd       — model forward inside CFR (set_leaf_values)
    cfr_set_leaf        — full set_leaf_values (model fwd + showdown gather)
    training_supervise  — one full _supervise call (fwd + bwd + opt step)
    training_model_fwd  — model forward inside _supervise
    suit_permutation    — with_permuted_targets + _compute_permutation_loss
    chance_node         — every method on the evaluator's chance_node_helper

Then prints per-step (averaged over the profile window) total CUDA time per
region. Use to decide where to focus optimization effort.
"""

from __future__ import annotations

import time

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


NUM_WARMUP = 3
NUM_ACTIVE = 3
STEP_TAG = "train_step"


def _wrap_class_call(cls: type, tag: str, _patched: set) -> None:
    """Patch cls.__call__ to be wrapped in record_function(tag)."""
    if cls in _patched:
        return
    _patched.add(cls)
    orig_call = cls.__call__

    def wrapped(self, *args, **kwargs):
        with record_function(tag):
            return orig_call(self, *args, **kwargs)

    cls.__call__ = wrapped


def _wrap_method(obj, attr: str, tag: str) -> None:
    """Patch a bound method on obj with record_function(tag)."""
    orig = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        with record_function(tag):
            return orig(*args, **kwargs)

    setattr(obj, attr, wrapped)


def _wrap_chance_node(helper) -> None:
    """Tag every method *defined on the chance_node_helper class itself* with
    one tag. Skips inherited attrs (model, generator, etc.) to avoid e.g.
    shadowing self.model.eval()."""
    cls = helper.__class__
    for name in vars(cls):
        if name.startswith("_"):
            continue
        attr = getattr(cls, name, None)
        if not callable(attr) or not hasattr(helper, name):
            continue
        bound = getattr(helper, name)
        if not callable(bound):
            continue
        try:
            _wrap_method(helper, name, "chance_node")
        except (AttributeError, TypeError):
            continue


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    cfg = load_rebel_config(dict_config)
    cfg.use_wandb = False
    cfg.num_steps = max(cfg.num_steps, NUM_WARMUP + NUM_ACTIVE + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device)

    # Patch the two distinct model classes so we can tell CFR-side vs
    # training-side forwards apart. trainer.model is the eager class
    # (BetterFFN), evaluator.model is the torch.compile OptimizedModule.
    patched: set = set()
    _wrap_class_call(
        trainer.model.__class__, "training_model_fwd", patched
    )
    _wrap_class_call(
        trainer.cfr_evaluator.model.__class__, "cfr_model_fwd", patched
    )

    # Per-CFR-iter and per-supervise tags via bound methods on the live objects.
    _wrap_method(trainer.cfr_evaluator, "cfr_iteration", "cfr_iter")
    _wrap_method(trainer.cfr_evaluator, "set_leaf_values", "cfr_set_leaf")
    _wrap_method(trainer.cfr_evaluator, "evaluate_cfr", "cfr_evaluate")
    _wrap_method(trainer.cfr_evaluator, "initialize_subgame", "cfr_init_subgame")
    _wrap_method(trainer.cfr_evaluator, "warm_start", "cfr_warm_start")
    _wrap_method(trainer.cfr_evaluator, "sample_leaves", "cfr_sample_leaves")
    _wrap_method(trainer.cfr_evaluator, "training_data", "cfr_training_data")
    # Sub-calls inside cfr_iteration:
    _wrap_method(
        trainer.cfr_evaluator,
        "compute_instantaneous_regrets",
        "cfr_iter_regrets",
    )
    _wrap_method(trainer.cfr_evaluator, "update_policy", "cfr_iter_update_policy")
    _wrap_method(
        trainer.cfr_evaluator, "compute_expected_values", "cfr_iter_expected_values"
    )
    _wrap_method(
        trainer.cfr_evaluator, "update_average_values", "cfr_iter_avg_values"
    )
    _wrap_method(
        trainer.cfr_evaluator, "update_average_policy", "cfr_iter_avg_policy"
    )
    # Sub-calls inside set_leaf_values:
    _wrap_method(
        trainer.cfr_evaluator, "_showdown_value_both", "cfr_leaf_showdown"
    )
    _wrap_method(
        trainer.cfr_evaluator, "_set_model_values", "cfr_leaf_set_model_values"
    )
    for stat_name in (
        "_record_action_mix",
        "_record_cfr_entropy",
        "_record_cumulative_regret",
        "_record_initial_exploitability",
    ):
        if hasattr(trainer.cfr_evaluator, stat_name):
            _wrap_method(trainer.cfr_evaluator, stat_name, "cfr_record_stats")
    _wrap_method(trainer, "_supervise", "training_supervise")
    _wrap_method(trainer, "_compute_permutation_loss", "suit_permutation")

    # Chance-node helper: tag every callable method.
    if hasattr(trainer.cfr_evaluator, "chance_helper"):
        _wrap_chance_node(trainer.cfr_evaluator.chance_helper)
    elif hasattr(trainer.cfr_evaluator, "chance_node_helper"):
        _wrap_chance_node(trainer.cfr_evaluator.chance_node_helper)

    print(f"Using device: {device}; warmup={NUM_WARMUP} active={NUM_ACTIVE}")

    for s in range(NUM_WARMUP):
        t0 = time.time()
        trainer.train_step(s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[warmup {s}] {time.time() - t0:.2f}s")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("Profiling...")
    with profile(activities=activities, record_shapes=False) as prof:
        for s in range(NUM_WARMUP, NUM_WARMUP + NUM_ACTIVE):
            with record_function(STEP_TAG):
                trainer.train_step(s)
                if device.type == "cuda":
                    torch.cuda.synchronize()

    tags = [
        STEP_TAG,
        "cfr_evaluate",
        "cfr_init_subgame",
        "cfr_warm_start",
        "cfr_iter",
        "cfr_iter_regrets",
        "cfr_iter_update_policy",
        "cfr_iter_expected_values",
        "cfr_iter_avg_values",
        "cfr_iter_avg_policy",
        "cfr_set_leaf",
        "cfr_leaf_showdown",
        "cfr_leaf_set_model_values",
        "cfr_model_fwd",
        "cfr_sample_leaves",
        "cfr_training_data",
        "cfr_record_stats",
        "training_supervise",
        "training_model_fwd",
        "suit_permutation",
        "chance_node",
    ]
    cuda_us: dict[str, float] = {t: 0.0 for t in tags}
    cpu_us: dict[str, float] = {t: 0.0 for t in tags}
    counts: dict[str, int] = {t: 0 for t in tags}
    for evt in prof.events():
        if evt.name in cuda_us:
            cuda_us[evt.name] += evt.cuda_time_total or 0.0
            cpu_us[evt.name] += evt.cpu_time_total or 0.0
            counts[evt.name] += 1

    n = NUM_ACTIVE
    step_cuda = cuda_us[STEP_TAG] / n
    step_cpu = cpu_us[STEP_TAG] / n
    print()
    print(f"Per-step (avg over {n} iters):")
    print(f"  total CPU wall:  {step_cpu / 1e3:8.2f} ms")
    print(f"  total CUDA time: {step_cuda / 1e3:8.2f} ms")
    print()
    print(
        f"{'region':<22}  {'calls/step':>10}  "
        f"{'cuda ms':>9}  {'cuda %':>7}  {'cpu ms':>9}  {'cpu %':>7}"
    )
    print("-" * 80)
    for t in tags:
        if t == STEP_TAG:
            continue
        n_calls = counts[t] / n
        c_us = cuda_us[t] / n
        p_us = cpu_us[t] / n
        c_pct = (100 * c_us / step_cuda) if step_cuda else 0
        p_pct = (100 * p_us / step_cpu) if step_cpu else 0
        print(
            f"{t:<22}  {n_calls:>10.1f}  "
            f"{c_us / 1e3:>9.2f}  {c_pct:>6.1f}%  "
            f"{p_us / 1e3:>9.2f}  {p_pct:>6.1f}%"
        )


if __name__ == "__main__":
    main()
