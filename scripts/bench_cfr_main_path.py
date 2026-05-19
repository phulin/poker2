#!/usr/bin/env python3
"""Benchmark the realistic ReBeL CFR data-generation + training loop.

This script has two modes:

* ``source``: runs a few production ``RebelCFRTrainer.train_step`` calls with
  torch profiler record-function tags around major components.
* ``micro``: prepares the same trainer/evaluator and times selected components
  with CUDA events.

It also pauses any active ``train_rebel`` process before each benchmark section
and resumes it afterwards, so background training does not distort timings.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

from p2.core.structured_config import Config
from p2.env.card_utils import suit_permutations_tensor
from p2.rl.cfr_trainer import RebelCFRTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "cfr_main_path_benchmark.json"
STEP_TAG = "train_step"

COMPONENT_TAGS = [
    STEP_TAG,
    "trainer_update_model",
    "data_generate",
    "cfr_evaluate",
    "cfr_init_subgame",
    "cfr_warm_start",
    "cfr_iter",
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
    "training_supervise",
    "training_model_fwd",
    "training_metrics",
    "aggression_metrics",
    "value_buffer_sample",
    "policy_buffer_sample",
    "value_buffer_add",
    "policy_buffer_add",
    "batch_permute",
    "suit_permutation",
]


def _iter_processes() -> list[tuple[int, str]]:
    procs: list[tuple[int, str]] = []
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return procs
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if not raw:
            continue
        cmd = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
        procs.append((pid, cmd))
    return procs


def _find_train_rebel_processes(pattern: str) -> list[tuple[int, str]]:
    self_pid = os.getpid()
    parent_pid = os.getppid()
    matches: list[tuple[int, str]] = []
    for pid, cmd in _iter_processes():
        if pid in {self_pid, parent_pid}:
            continue
        if pattern in cmd:
            matches.append((pid, cmd))
    return matches


@contextmanager
def pause_train_rebel(enabled: bool, pattern: str):
    paused: list[tuple[int, str]] = []
    if enabled:
        matches = _find_train_rebel_processes(pattern)
        for pid, cmd in matches:
            try:
                os.kill(pid, signal.SIGSTOP)
                paused.append((pid, cmd))
            except ProcessLookupError:
                continue
        if paused:
            print(
                "Paused train_rebel processes: "
                + ", ".join(str(pid) for pid, _ in paused),
                flush=True,
            )
            time.sleep(0.5)
        else:
            print(f"No running process matched pause pattern {pattern!r}.", flush=True)
    try:
        yield
    finally:
        for pid, _cmd in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                continue
        if paused:
            print(
                "Resumed train_rebel processes: "
                + ", ".join(str(pid) for pid, _ in paused),
                flush=True,
            )


def _wrap_class_call(cls: type, tag: str, patched: set[type]) -> None:
    if cls in patched:
        return
    patched.add(cls)
    orig_call = cls.__call__

    def wrapped(self, *args, **kwargs):
        with record_function(tag):
            return orig_call(self, *args, **kwargs)

    cls.__call__ = wrapped


def _wrap_method(obj: Any, attr: str, tag: str) -> None:
    orig = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        with record_function(tag):
            return orig(*args, **kwargs)

    setattr(obj, attr, wrapped)


def _wrap_class_method(cls: type, attr: str, tag: str, patched: set[tuple[type, str]]):
    key = (cls, attr)
    if key in patched:
        return
    patched.add(key)
    orig = getattr(cls, attr)

    def wrapped(self, *args, **kwargs):
        with record_function(tag):
            return orig(self, *args, **kwargs)

    setattr(cls, attr, wrapped)


def _patch_profiler_tags(trainer: RebelCFRTrainer) -> None:
    patched: set[type] = set()
    patched_methods: set[tuple[type, str]] = set()
    _wrap_class_call(trainer.model.__class__, "training_model_fwd", patched)
    _wrap_class_call(trainer.cfr_evaluator.model.__class__, "cfr_model_fwd", patched)

    _wrap_method(trainer, "_update_model", "trainer_update_model")
    _wrap_method(trainer, "_supervise", "training_supervise")
    _wrap_method(trainer, "_compute_metrics", "training_metrics")
    _wrap_method(trainer, "_compute_permutation_loss", "suit_permutation")
    _wrap_method(
        trainer.aggression_analyzer, "analyze_batch", "aggression_metrics"
    )
    _wrap_method(trainer.value_buffer, "sample", "value_buffer_sample")
    _wrap_method(trainer.policy_buffer, "sample", "policy_buffer_sample")
    _wrap_method(trainer.value_buffer, "add_batch", "value_buffer_add")
    _wrap_method(trainer.policy_buffer, "add_batch", "policy_buffer_add")

    from p2.rl.rebel_batch import RebelBatch

    _wrap_class_method(
        RebelBatch, "with_permuted_targets", "batch_permute", patched_methods
    )

    generator = trainer.data_generator
    _wrap_method(generator, "generate_data", "data_generate")

    ev = trainer.cfr_evaluator
    for attr, tag in (
        ("evaluate_cfr", "cfr_evaluate"),
        ("initialize_subgame", "cfr_init_subgame"),
        ("warm_start", "cfr_warm_start"),
        ("cfr_iteration", "cfr_iter"),
        ("set_leaf_values", "cfr_set_leaf"),
        ("_showdown_value_both", "cfr_leaf_showdown"),
        ("_set_model_values", "cfr_leaf_set_model_values"),
        ("compute_expected_values", "cfr_iter_expected_values"),
        ("update_policy", "cfr_iter_update_policy"),
        ("update_average_values", "cfr_iter_avg_values"),
        ("update_average_policy", "cfr_iter_avg_policy"),
        ("sample_leaves", "cfr_sample_leaves"),
        ("training_data", "cfr_training_data"),
    ):
        if hasattr(ev, attr):
            _wrap_method(ev, attr, tag)


def _apply_realistic_overrides(cfg: Config, args: argparse.Namespace) -> None:
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_steps = max(cfg.num_steps, args.warmup_steps + args.active_steps + 1)

    cfg.model.hidden_dim = 256
    cfg.model.range_hidden_dim = 128
    cfg.model.ffn_dim = 512
    cfg.model.num_hidden_layers = 3
    cfg.model.num_value_layers = 1
    cfg.model.num_policy_layers = 1

    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.iterations is not None:
        cfg.search.iterations = args.iterations
        cfg.search.iterations_final = args.iterations
    if args.depth is not None:
        cfg.search.depth = args.depth
    if args.no_compile:
        cfg.model.compile = "off"


def _load_config(args: argparse.Namespace) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dict_config: DictConfig = hydra.compose(
            config_name="config_rebel_cfr",
            overrides=args.hydra_override,
        )
    cfg = Config.from_dict_config(dict_config)
    _apply_realistic_overrides(cfg, args)
    return cfg


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _make_trainer(cfg: Config) -> RebelCFRTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    try:
        torch._dynamo.config.recompile_limit = 32
    except Exception:
        pass
    torch.manual_seed(cfg.seed)
    return RebelCFRTrainer(cfg=cfg, device=device)


def _event_device_us(evt) -> float:
    value = getattr(evt, "device_time_total", None)
    if value is not None:
        return float(value or 0.0)
    return float(getattr(evt, "cuda_time_total", 0.0) or 0.0)


def _profiler_summary(
    prof, active_steps: int, active_wall_s: list[float]
) -> dict[str, Any]:
    cuda_us: dict[str, float] = {tag: 0.0 for tag in COMPONENT_TAGS}
    cpu_us: dict[str, float] = {tag: 0.0 for tag in COMPONENT_TAGS}
    counts: dict[str, int] = {tag: 0 for tag in COMPONENT_TAGS}

    for evt in prof.events():
        if evt.name not in cuda_us:
            continue
        cuda_us[evt.name] += _event_device_us(evt)
        cpu_us[evt.name] += float(evt.cpu_time_total or 0.0)
        counts[evt.name] += 1

    wall_mean_us = (
        1e6 * sum(active_wall_s) / max(1, len(active_wall_s))
        if active_wall_s
        else 0.0
    )
    step_cuda = cuda_us[STEP_TAG] / max(1, active_steps)
    step_cpu = cpu_us[STEP_TAG] / max(1, active_steps)
    rows: list[dict[str, Any]] = []
    for tag in COMPONENT_TAGS:
        if tag == STEP_TAG:
            continue
        c_us = cuda_us[tag] / max(1, active_steps)
        p_us = cpu_us[tag] / max(1, active_steps)
        rows.append(
            {
                "component": tag,
                "calls_per_step": counts[tag] / max(1, active_steps),
                "cuda_ms_per_step": c_us / 1e3,
                "cuda_pct_of_wall": (
                    (100.0 * c_us / wall_mean_us) if wall_mean_us else 0.0
                ),
                "cpu_ms_per_step": p_us / 1e3,
                "cpu_pct_of_step": (100.0 * p_us / step_cpu) if step_cpu else 0.0,
            }
        )

    rows.sort(key=lambda r: r["cuda_ms_per_step"], reverse=True)
    return {
        "step_annotated_cuda_ms": step_cuda / 1e3,
        "step_cpu_ms": step_cpu / 1e3,
        "components": rows,
    }


def run_source_benchmark(
    trainer: RebelCFRTrainer, args: argparse.Namespace
) -> dict[str, Any]:
    _patch_profiler_tags(trainer)
    device = trainer.device

    warmup_wall: list[float] = []
    for step in range(args.warmup_steps):
        t0 = time.perf_counter()
        trainer.train_step(step)
        _sync(device)
        warmup_wall.append(time.perf_counter() - t0)
        print(f"[source warmup {step}] {warmup_wall[-1]:.3f}s", flush=True)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    active_wall: list[float] = []
    with profile(activities=activities, record_shapes=False) as prof:
        for i in range(args.active_steps):
            step = args.warmup_steps + i
            t0 = time.perf_counter()
            with record_function(STEP_TAG):
                trainer.train_step(step)
                _sync(device)
            active_wall.append(time.perf_counter() - t0)
            print(f"[source active {step}] {active_wall[-1]:.3f}s", flush=True)

    summary = _profiler_summary(prof, args.active_steps, active_wall)
    summary["warmup_wall_s"] = warmup_wall
    summary["active_wall_s"] = active_wall
    summary["active_wall_mean_s"] = sum(active_wall) / max(1, len(active_wall))
    return summary


def _cuda_event_time(
    device: torch.device,
    fn: Callable[[], Any],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)

    wall: list[float] = []
    cuda_ms: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            end.record()
            torch.cuda.synchronize()
            cuda_ms.append(float(start.elapsed_time(end)))
        else:
            _sync(device)
        wall.append(time.perf_counter() - t0)
    return {
        "iters": float(iters),
        "wall_ms_mean": 1e3 * sum(wall) / max(1, len(wall)),
        "cuda_ms_mean": sum(cuda_ms) / max(1, len(cuda_ms)) if cuda_ms else 0.0,
    }


def _ensure_training_batches(trainer: RebelCFRTrainer) -> None:
    while min(len(trainer.value_buffer), len(trainer.policy_buffer)) < trainer.batch_size:
        trainer.data_generator.generate_data(trainer.K_value)
        _sync(trainer.device)


def _prepare_evaluator_for_iter(trainer: RebelCFRTrainer) -> int:
    ev = trainer.cfr_evaluator
    if getattr(trainer.data_generator, "current_pbs", None) is None:
        trainer.data_generator.generate_data(1)
        _sync(trainer.device)
    ev.initialize_policy_and_beliefs()
    if ev.warm_start_iterations > 0:
        ev.warm_start()
    ev.set_leaf_values(0)
    ev.compute_expected_values()
    ev.values_avg[:] = ev.latest_values
    ev.t_sample = ev._get_sampling_schedule()
    if hasattr(ev, "_prepare_sample_update_table"):
        ev._prepare_sample_update_table()
    _sync(trainer.device)
    return max(ev.warm_start_iterations, 2)


def _make_supervise_callable(trainer: RebelCFRTrainer) -> Callable[[], Any]:
    _ensure_training_batches(trainer)
    value_batch = trainer.value_buffer.sample(trainer.batch_size).to(trainer.device)
    policy_batch = trainer.policy_buffer.sample(trainer.batch_size).to(trainer.device)
    suit_idxs = torch.randint(
        0,
        24,
        (len(value_batch),),
        generator=trainer.rng,
        device=trainer.device,
    )
    suit_perms = suit_permutations_tensor(device=trainer.device)[suit_idxs]
    permuted_batch, suit_idxs = value_batch.with_permuted_targets(
        suit_permutations=suit_perms,
        suit_permutation_idxs=suit_idxs,
        num_players=trainer.num_players,
    )

    def run():
        return trainer._supervise(
            value_batch,
            policy_batch,
            permuted_batch,
            suit_idxs,
            None,
            None,
            None,
        )

    return run


def _sample_training_inputs(trainer: RebelCFRTrainer):
    _ensure_training_batches(trainer)
    value_batch = trainer.value_buffer.sample(trainer.batch_size).to(trainer.device)
    policy_batch = trainer.policy_buffer.sample(trainer.batch_size).to(trainer.device)
    suit_idxs = torch.randint(
        0,
        24,
        (len(value_batch),),
        generator=trainer.rng,
        device=trainer.device,
    )
    suit_perms = suit_permutations_tensor(device=trainer.device)[suit_idxs]
    permuted_batch, suit_idxs = value_batch.with_permuted_targets(
        suit_permutations=suit_perms,
        suit_permutation_idxs=suit_idxs,
        num_players=trainer.num_players,
    )
    return value_batch, policy_batch, permuted_batch, suit_idxs


def _make_metrics_callable(trainer: RebelCFRTrainer) -> Callable[[], Any]:
    value_batch, policy_batch, permuted_batch, suit_idxs = _sample_training_inputs(
        trainer
    )
    (
        episode_stats,
        permuted_value_output,
        _value_output_orig,
        _policy_output,
        value_loss_update,
        policy_loss_update,
    ) = trainer._supervise(
        value_batch,
        policy_batch,
        permuted_batch,
        suit_idxs,
        None,
        None,
        None,
    )
    _sync(trainer.device)
    step_stats = {
        key: float(value.detach().reshape(()).cpu().item())
        for key, value in episode_stats.items()
    }

    def run():
        return trainer._compute_metrics(
            episodes=1,
            updates=1,
            step_stats=step_stats,
            value_batch=permuted_batch,
            policy_batch=policy_batch,
            value_output=permuted_value_output,
            policy_output=None,
            value_loss_all=value_loss_update,
            policy_loss_all=policy_loss_update,
            fresh_value_batch=None,
        )

    return run


def _make_buffer_add_callables(
    trainer: RebelCFRTrainer,
) -> tuple[Callable[[], Any], Callable[[], Any]]:
    fresh_value_batch, fresh_policy_batch = trainer.data_generator.generate_data(
        max(1, trainer.K_value)
    )
    _sync(trainer.device)
    if fresh_value_batch is None or fresh_policy_batch is None:
        raise RuntimeError("generate_data did not return fresh benchmark batches")

    def add_value():
        trainer.value_buffer.add_batch(fresh_value_batch)

    def add_policy():
        trainer.policy_buffer.add_batch(fresh_policy_batch)

    return add_value, add_policy


def run_microbenchmarks(
    trainer: RebelCFRTrainer, args: argparse.Namespace
) -> dict[str, Any]:
    device = trainer.device
    results: dict[str, Any] = {}

    micro_warmup = args.micro_warmup
    micro_iters = args.micro_iters

    results["generate_data"] = _cuda_event_time(
        device,
        lambda: trainer.data_generator.generate_data(max(1, args.micro_value_samples)),
        warmup=0,
        iters=max(1, min(micro_iters, 2)),
    )

    start_t = _prepare_evaluator_for_iter(trainer)
    t_box = {"t": start_t}

    def run_cfr_iter():
        trainer.cfr_evaluator.cfr_iteration(t_box["t"])
        t_box["t"] += 1

    results["cfr_iteration"] = _cuda_event_time(
        device, run_cfr_iter, warmup=micro_warmup, iters=micro_iters
    )

    results["set_leaf_values"] = _cuda_event_time(
        device,
        lambda: trainer.cfr_evaluator.set_leaf_values(t_box["t"]),
        warmup=micro_warmup,
        iters=micro_iters,
    )

    results["compute_expected_values"] = _cuda_event_time(
        device,
        lambda: trainer.cfr_evaluator.compute_expected_values(),
        warmup=micro_warmup,
        iters=micro_iters,
    )

    results["update_policy"] = _cuda_event_time(
        device,
        lambda: trainer.cfr_evaluator.update_policy(t_box["t"]),
        warmup=micro_warmup,
        iters=micro_iters,
    )

    supervise = _make_supervise_callable(trainer)
    results["training_supervise"] = _cuda_event_time(
        device,
        supervise,
        warmup=max(0, min(micro_warmup, 1)),
        iters=max(1, min(micro_iters, 3)),
    )

    _ensure_training_batches(trainer)
    results["value_buffer_sample"] = _cuda_event_time(
        device,
        lambda: trainer.value_buffer.sample(trainer.batch_size),
        warmup=micro_warmup,
        iters=micro_iters,
    )
    results["policy_buffer_sample"] = _cuda_event_time(
        device,
        lambda: trainer.policy_buffer.sample(trainer.batch_size),
        warmup=micro_warmup,
        iters=micro_iters,
    )

    value_batch, _policy_batch, _permuted_batch, _suit_idxs = _sample_training_inputs(
        trainer
    )

    def permute_batch():
        suit_idxs = torch.randint(
            0,
            24,
            (len(value_batch),),
            generator=trainer.rng,
            device=trainer.device,
        )
        suit_perms = suit_permutations_tensor(device=trainer.device)[suit_idxs]
        return value_batch.with_permuted_targets(
            suit_permutations=suit_perms,
            suit_permutation_idxs=suit_idxs,
            num_players=trainer.num_players,
        )

    results["batch_permute"] = _cuda_event_time(
        device, permute_batch, warmup=micro_warmup, iters=micro_iters
    )

    metrics = _make_metrics_callable(trainer)
    results["training_metrics"] = _cuda_event_time(
        device,
        metrics,
        warmup=max(0, min(micro_warmup, 1)),
        iters=max(1, min(micro_iters, 3)),
    )

    add_value, add_policy = _make_buffer_add_callables(trainer)
    results["value_buffer_add"] = _cuda_event_time(
        device, add_value, warmup=micro_warmup, iters=micro_iters
    )
    results["policy_buffer_add"] = _cuda_event_time(
        device, add_policy, warmup=micro_warmup, iters=micro_iters
    )

    return results


def _print_source_summary(source: dict[str, Any]) -> None:
    print("\nSource-of-truth profile:")
    print(f"  wall mean: {source['active_wall_mean_s'] * 1e3:.2f} ms/step")
    print(f"  annotated CUDA range: {source['step_annotated_cuda_ms']:.2f} ms/step")
    print(f"  cpu:       {source['step_cpu_ms']:.2f} ms/step")
    print("  top components:")
    for row in source["components"][:12]:
        print(
            f"    {row['component']:<26}"
            f" {row['cuda_ms_per_step']:>9.2f} ms cuda"
            f" {row['cuda_pct_of_wall']:>6.1f}% wall"
            f" calls/step={row['calls_per_step']:.1f}"
        )


def _print_micro_summary(micro: dict[str, Any]) -> None:
    print("\nMicrobenchmarks:")
    for name, row in micro.items():
        print(
            f"  {name:<24}"
            f" wall={row['wall_ms_mean']:>9.2f} ms"
            f" cuda={row['cuda_ms_mean']:>9.2f} ms"
            f" iters={int(row['iters'])}"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["source", "micro", "both"], default="both")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--active-steps", type=int, default=1)
    parser.add_argument("--micro-warmup", type=int, default=1)
    parser.add_argument("--micro-iters", type=int, default=3)
    parser.add_argument("--micro-value-samples", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("hydra_override", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    cfg = _load_config(args)
    trainer = _make_trainer(cfg)

    config_summary = {
        "num_envs": cfg.num_envs,
        "batch_size": cfg.train.batch_size,
        "replay_buffer_device": cfg.train.replay_buffer_device,
        "iterations": cfg.search.iterations,
        "iterations_final": cfg.search.iterations_final,
        "depth": cfg.search.depth,
        "sparse_fused": cfg.search.sparse_fused,
        "cfr_type": str(cfg.search.cfr_type),
        "model": {
            "hidden_dim": cfg.model.hidden_dim,
            "range_hidden_dim": cfg.model.range_hidden_dim,
            "ffn_dim": cfg.model.ffn_dim,
            "num_hidden_layers": cfg.model.num_hidden_layers,
            "num_value_layers": cfg.model.num_value_layers,
            "num_policy_layers": cfg.model.num_policy_layers,
            "compile": cfg.model.compile,
        },
    }
    print(json.dumps({"config": config_summary}, indent=2), flush=True)

    output: dict[str, Any] = {"config": config_summary}
    if args.mode in {"source", "both"}:
        with pause_train_rebel(not args.no_pause, args.pause_pattern):
            output["source"] = run_source_benchmark(trainer, args)
        _print_source_summary(output["source"])

    if args.mode in {"micro", "both"}:
        with pause_train_rebel(not args.no_pause, args.pause_pattern):
            output["micro"] = run_microbenchmarks(trainer, args)
        _print_micro_summary(output["micro"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
