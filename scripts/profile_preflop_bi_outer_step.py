#!/usr/bin/env python3
"""Time preflop backward-induction outer-loop phases with CUDA syncs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2.cli.train_rebel_preflop_buckets import _execution_config_from_config  # noqa: E402
from p2.config.rebel_load import load_rebel_config  # noqa: E402
from p2.rl.cfr_trainer import RebelCFRTrainer, _compile_kwargs  # noqa: E402
from p2.runtime.training_run import setup_torch_runtime  # noqa: E402
from p2.search.fused_cfr_triton import GraphedCFRIteration  # noqa: E402
from p2.stages.preflop_backward_induction import (  # noqa: E402
    PublicStateBucketReader,
    _build_validation_cache,
    _bucket_cfr_batch_size,
    _bucket_spec,
    _checkpoint_model_config,
    _copy_public_states_to_env,
    _evaluate_validation_set,
    _filter_batch_by_action_bucket,
    _load_model_weights,
    _make_env_from_manifest,
    _policy_only,
    _policy_train_batch_size,
    _random_beliefs,
    _train_policy_minibatches,
    _train_value_minibatches,
    _validation_to_device,
    _value_only,
)
from p2.stages.preflop_buckets import build_run_config  # noqa: E402


DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "outputs/preflop_backward_induction/"
    "gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5"
)
DEFAULT_STATE_DATASET = (
    REPO_ROOT
    / "outputs/preflop_policy_states/"
    "eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622"
)
DEFAULT_SOLVER_CHECKPOINT = DEFAULT_RUN_DIR / "actions_12_15/checkpoints/specialist_final.pt"
DEFAULT_TRAINER_CHECKPOINT = (
    DEFAULT_RUN_DIR / "actions_8_11/checkpoints/specialist_inprogress.pt"
)
DEFAULT_OUT = REPO_ROOT / "outputs/preflop_outer_step_profile/profile.json"
DEFAULT_PROFILER_DIR = REPO_ROOT / "outputs/preflop_outer_step_profile/torch_profiler"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


class PhaseTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.rows: list[dict[str, float | str | int]] = []

    @contextmanager
    def time(self, phase: str, *, step: int) -> Any:
        _sync(self.device)
        start = time.perf_counter()
        yield
        _sync(self.device)
        self.rows.append(
            {"step": step, "phase": phase, "wall_s": time.perf_counter() - start}
        )


@contextmanager
def _phase(
    timer: PhaseTimer,
    phase: str,
    *,
    step: int,
    use_timer: bool,
    use_profiler: bool,
) -> Any:
    record_ctx = record_function(phase) if use_profiler else nullcontext()
    timer_ctx = timer.time(phase, step=step) if use_timer else nullcontext()
    with record_ctx:
        with timer_ctx:
            yield


def _prof_device_us(evt: Any) -> float:
    value = getattr(evt, "device_time_total", None)
    if value is not None:
        return float(value or 0.0)
    return float(getattr(evt, "cuda_time_total", 0.0) or 0.0)


def _prof_self_device_us(evt: Any) -> float:
    value = getattr(evt, "self_device_time_total", None)
    if value is not None:
        return float(value or 0.0)
    return float(getattr(evt, "self_cuda_time_total", 0.0) or 0.0)


def _profiler_rows(prof: Any, steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evt in prof.key_averages():
        self_device_us = _prof_self_device_us(evt)
        device_us = _prof_device_us(evt)
        self_cpu_us = float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
        cpu_us = float(getattr(evt, "cpu_time_total", 0.0) or 0.0)
        count = int(getattr(evt, "count", 1) or 1)
        rows.append(
            {
                "name": evt.key,
                "count_total": count,
                "count_per_step": count / max(1, steps),
                "self_device_ms_per_step": self_device_us / 1e3 / max(1, steps),
                "device_ms_per_step": device_us / 1e3 / max(1, steps),
                "self_cpu_ms_per_step": self_cpu_us / 1e3 / max(1, steps),
                "cpu_ms_per_step": cpu_us / 1e3 / max(1, steps),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["self_device_ms_per_step"]),
            float(row["self_cpu_ms_per_step"]),
        ),
        reverse=True,
    )
    return rows


def _profiler_cpu_rows(prof: Any, steps: int) -> list[dict[str, Any]]:
    rows = _profiler_rows(prof, steps)
    rows.sort(key=lambda row: float(row["self_cpu_ms_per_step"]), reverse=True)
    return rows


def _load_cfg(args: argparse.Namespace):
    overrides = [
        "device=cuda",
        "use_wandb=false",
        "preflop_buckets.command=train_specialists",
        f"preflop_buckets.state_dataset={args.state_dataset}",
        f"preflop_buckets.base_checkpoint={args.solver_checkpoint}",
        f"preflop_buckets.output_dir={args.run_dir}",
        f"preflop_buckets.train_bucket={args.bucket}",
        "preflop_buckets.write_solved_shards=false",
        "preflop_buckets.validation_interval_steps=10",
        "preflop_buckets.validation_eval_batch_size=1024",
        "preflop_buckets.compile=default",
    ]
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_preflop_buckets",
            overrides=overrides,
        )
    cfg = load_rebel_config(dc)
    execution = _execution_config_from_config(cfg)
    return cfg, execution


def _summarize(rows: list[dict[str, float | str | int]]) -> dict[str, Any]:
    by_phase: dict[str, list[float]] = {}
    for row in rows:
        by_phase.setdefault(str(row["phase"]), []).append(float(row["wall_s"]))
    out: dict[str, Any] = {}
    for phase, values in sorted(by_phase.items()):
        out[phase] = {
            "count": len(values),
            "total_s": sum(values),
            "mean_s": sum(values) / max(1, len(values)),
            "min_s": min(values),
            "max_s": max(values),
        }
    return out


@torch.no_grad()
def _evaluate_cfr_timed(evaluator: Any, timer: PhaseTimer, *, step: int) -> None:
    evaluator._ensure_fused_attrs()
    evaluator.model.eval()

    with timer.time("cfr_initialize_policy_and_beliefs", step=step):
        evaluator.initialize_policy_and_beliefs()
    if evaluator.warm_start_iterations > 0:
        with timer.time("cfr_warm_start", step=step):
            evaluator.warm_start()
    with timer.time("cfr_initial_set_leaf_values", step=step):
        evaluator.set_leaf_values(0)
    with timer.time("cfr_initial_compute_expected_values", step=step):
        evaluator.compute_expected_values()
    with timer.time("cfr_initial_bookkeeping", step=step):
        evaluator.values_avg[:] = evaluator.latest_values
        if hasattr(evaluator, "_reset_deferred_average_values"):
            evaluator._reset_deferred_average_values()
        evaluator.t_sample = evaluator._get_sampling_schedule()
        if hasattr(evaluator, "_prepare_sample_update_table"):
            evaluator._prepare_sample_update_table()

    start = evaluator.warm_start_iterations
    end = evaluator.cfr_iterations
    stat_iters = evaluator._record_stats_percentile_ts()
    runners: dict[str, GraphedCFRIteration] = {}
    t = start
    capture_count = 0
    replay_count = 0
    eager_count = 0
    while t < end:
        evaluator.profiler_step()
        regime = evaluator._graph_capture_regime(t)
        can_capture = (
            regime is not None
            and regime not in runners
            and t + 1 < end
            and evaluator._graph_capture_regime(t + 1) == regime
            and t not in stat_iters
            and (t + 1) not in stat_iters
        )
        if can_capture:
            with timer.time(f"cfr_graph_capture_{regime}", step=step):
                runner = GraphedCFRIteration(evaluator)
                runner.capture(t_warmup=t, t_capture=t + 1)
            runners[regime] = runner
            capture_count += 1
            t += 1
            continue

        runner = runners.get(regime) if regime is not None else None
        if runner is not None and t not in stat_iters:
            with timer.time("cfr_graph_replay", step=step):
                runner.replay(t=t)
            replay_count += 1
        else:
            with timer.time("cfr_eager_iteration", step=step):
                evaluator.cfr_iteration(t)
            eager_count += 1
        t += 1

    with timer.time("cfr_finalize", step=step):
        if not evaluator.cfr_avg:
            evaluator._finalize_deferred_average_policy()
            evaluator.self_reach_avg[: evaluator.root_nodes] = 1.0
            evaluator._calculate_reach_weights(
                evaluator.self_reach_avg, evaluator.policy_probs_avg
            )
            evaluator._refresh_average_beliefs()

    timer.rows.append(
        {
            "step": step,
            "phase": "cfr_iteration_counts",
            "wall_s": 0.0,
            "capture_count": capture_count,
            "replay_count": replay_count,
            "eager_count": eager_count,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--state-dataset", type=Path, default=DEFAULT_STATE_DATASET)
    parser.add_argument("--solver-checkpoint", type=Path, default=DEFAULT_SOLVER_CHECKPOINT)
    parser.add_argument(
        "--trainer-checkpoint", type=Path, default=DEFAULT_TRAINER_CHECKPOINT
    )
    parser.add_argument("--bucket", default="actions_8_11")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--active-steps", type=int, default=3)
    parser.add_argument("--skip-rows", type=int, default=1_536_000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--profile-cpu-validation", action="store_true")
    parser.add_argument("--timed-cfr-internals", action="store_true")
    parser.add_argument("--compile-cfr-models", action="store_true")
    parser.add_argument("--fullgraph-cfr-models", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help=(
            "Profile active outer steps with torch.profiler. Active steps use "
            "record_function phase labels and only synchronize at step end."
        ),
    )
    parser.add_argument("--profiler-record-shapes", action="store_true")
    parser.add_argument("--profiler-with-stack", action="store_true")
    parser.add_argument("--profiler-out-dir", type=Path, default=DEFAULT_PROFILER_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    base_template, execution = _load_cfg(args)
    cfr_batch_size = _bucket_cfr_batch_size(execution, args.bucket)
    train_batch_size = max(1, int(execution.train_batch_size))
    policy_train_batch_size = _policy_train_batch_size(execution)
    bucket_index = ("actions_12_15", "actions_8_11", "actions_4_7", "actions_0_3").index(
        args.bucket
    )
    spec = _bucket_spec(args.bucket)
    train_value = args.bucket != "actions_0_3"

    cfg = build_run_config(
        base_template,
        execution,
        checkpoint_dir=args.run_dir / args.bucket / "checkpoints",
        num_steps=max(1, args.warmup_steps + args.active_steps),
        num_envs=cfr_batch_size,
        bucket_label=args.bucket,
    )
    setup_torch_runtime(cfg, device, recompile_limit=16)
    cutoff_cfg = _checkpoint_model_config(cfg, str(args.solver_checkpoint))

    print("initializing models", flush=True)
    solver = RebelCFRTrainer(cfg=cutoff_cfg, device=device, pregeneration_only=True)
    _load_model_weights(solver, str(args.solver_checkpoint))
    if args.compile_cfr_models and device.type == "cuda":
        print("compiling CFR inference models", flush=True)
        compile_kwargs = _compile_kwargs(cutoff_cfg)
        if args.fullgraph_cfr_models:
            compile_kwargs["fullgraph"] = True
            solver.cfr_evaluator._compile_kwargs = dict(compile_kwargs)
        cfr_models = [
            getattr(solver.cfr_evaluator, "model", None),
            getattr(solver.cfr_evaluator, "closing_leaf_value_model", None),
        ]
        for model in cfr_models:
            if model is not None and hasattr(model, "compile_forward_modes"):
                model.compile_forward_modes(**compile_kwargs)
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    loaded_step = trainer.load_checkpoint(str(args.trainer_checkpoint), load_optimizer=True)
    print(f"loaded trainer step={loaded_step}", flush=True)

    reader = PublicStateBucketReader(
        str(args.state_dataset),
        args.bucket,
        allow_partial=execution.allow_partial,
        seed=execution.seed + bucket_index * 10_000,
    )
    env = _make_env_from_manifest(
        reader.manifest,
        num_envs=cfr_batch_size,
        device=device,
        seed=execution.seed + bucket_index + 100,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(int(execution.seed))
    timer = PhaseTimer(device)

    if not args.skip_validation:
        print("loading validation cache", flush=True)
        validation_cpu = _build_validation_cache(
            args=execution,
            bucket_label=args.bucket,
            spec=spec,
            bucket_dir=args.run_dir / args.bucket,
            cutoff_checkpoint=str(args.solver_checkpoint),
            cfg=cutoff_cfg,
            reader=reader,
            device=device,
            bucket_index=bucket_index,
        )
        with timer.time("validation_to_device_once", step=-1):
            validation_gpu = _validation_to_device(validation_cpu, device)
        with timer.time("validation_gpu_resident_compile_warmup", step=-1):
            _evaluate_validation_set(
                trainer,
                validation_gpu,
                eval_batch_size=execution.validation_eval_batch_size,
            )
        with timer.time("validation_gpu_resident", step=-1):
            _evaluate_validation_set(
                trainer,
                validation_gpu,
                eval_batch_size=execution.validation_eval_batch_size,
            )
        if args.profile_cpu_validation:
            with timer.time("validation_cpu_transfer_each_time", step=-1):
                _evaluate_validation_set(
                    trainer,
                    validation_cpu,
                    eval_batch_size=execution.validation_eval_batch_size,
                )

    total_steps = args.warmup_steps + args.active_steps
    state_iter = reader.iter_state_batches(
        batch_size=cfr_batch_size,
        max_rows=args.skip_rows + total_steps * cfr_batch_size,
        seed=execution.seed + bucket_index * 10_000,
        skip_rows=args.skip_rows,
    )
    step_summaries: list[dict[str, Any]] = []
    profiler_obj = None
    profiler_ctx = None
    profiler_trace_path: Path | None = None
    profiler_summary_path: Path | None = None
    profiler_active_wall_s: list[float] = []
    for step in range(total_steps):
        active_step = step >= args.warmup_steps
        profile_active = bool(args.torch_profiler and active_step)
        if profile_active and profiler_ctx is None:
            args.profiler_out_dir.mkdir(parents=True, exist_ok=True)
            profiler_trace_path = args.profiler_out_dir / "trace.json"
            profiler_summary_path = args.profiler_out_dir / "summary.json"
            activities = [ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(ProfilerActivity.CUDA)
            profiler_ctx = profile(
                activities=activities,
                record_shapes=args.profiler_record_shapes,
                with_stack=args.profiler_with_stack,
                profile_memory=False,
            )
            profiler_obj = profiler_ctx.__enter__()
        states = next(state_iter)
        phase_step = step - args.warmup_steps if active_step else -1 - step
        step_start = time.perf_counter() if profile_active else 0.0
        step_record = record_function("outer_active_step") if profile_active else nullcontext()
        with step_record:
            with _phase(
                timer,
                "copy_public_states_to_env",
                step=phase_step,
                use_timer=not profile_active,
                use_profiler=profile_active,
            ):
                rows = _copy_public_states_to_env(env, states)
            with _phase(
                timer,
                "random_beliefs",
                step=phase_step,
                use_timer=not profile_active,
                use_profiler=profile_active,
            ):
                beliefs = _random_beliefs(
                    rows,
                    solver.num_players,
                    device=device,
                    rng=rng,
                    mode=execution.belief_mode,
                    profile=getattr(execution, "belief_profile", "actions_12_end"),
                    hand_dim=getattr(execution, "belief_hand_dim", 169),
                )
            roots = torch.arange(rows, device=device)
            evaluator = solver.cfr_evaluator
            with _phase(
                timer,
                "initialize_subgame",
                step=phase_step,
                use_timer=not profile_active,
                use_profiler=profile_active,
            ):
                evaluator.initialize_subgame(env, roots, beliefs)
            with _phase(
                timer,
                "evaluate_cfr_total",
                step=phase_step,
                use_timer=not profile_active,
                use_profiler=profile_active,
            ):
                if args.timed_cfr_internals and not profile_active:
                    _evaluate_cfr_timed(evaluator, timer, step=phase_step)
                else:
                    evaluator.evaluate_cfr(
                        training_mode=True,
                        sample_continuation=False,
                    )
            with _phase(
                timer,
                "training_data",
                step=phase_step,
                use_timer=not profile_active,
                use_profiler=profile_active,
            ):
                value_batch, _unused, policy_batch = evaluator.training_data(
                    include_pre_chance_value_batch=False,
                    include_policy_batch=True,
                )
            with _phase(
                timer,
                "filter_bucket",
                step=phase_step,
                use_timer=not profile_active,
                use_profiler=profile_active,
            ):
                value_batch = _filter_batch_by_action_bucket(
                    value_batch,
                    low=spec.low,
                    high=spec.high,
                )
                policy_batch = _filter_batch_by_action_bucket(
                    policy_batch,
                    low=spec.low,
                    high=spec.high,
                )
                value_stream = _value_only(value_batch) if train_value else None
                policy_stream = _policy_only(policy_batch)
            value_examples = 0 if value_stream is None else len(value_stream)
            policy_examples = 0 if policy_stream is None else len(policy_stream)
            trained_this_step = False
            if value_stream is not None and not args.skip_training:
                with _phase(
                    timer,
                    "train_value_minibatches",
                    step=phase_step,
                    use_timer=not profile_active,
                    use_profiler=profile_active,
                ):
                    _train_value_minibatches(
                        trainer,
                        value_stream,
                        step=int(loaded_step) + step,
                        batch_size=train_batch_size,
                        sync_inference_model=False,
                    )
                trained_this_step = True
            if policy_stream is not None and not args.skip_training:
                with _phase(
                    timer,
                    "train_policy_minibatches",
                    step=phase_step,
                    use_timer=not profile_active,
                    use_profiler=profile_active,
                ):
                    _train_policy_minibatches(
                        trainer,
                        policy_stream,
                        step=int(loaded_step) + step,
                        batch_size=policy_train_batch_size,
                        sync_inference_model=False,
                    )
                trained_this_step = True
            if trained_this_step:
                with _phase(
                    timer,
                    "sync_inference_model",
                    step=phase_step,
                    use_timer=not profile_active,
                    use_profiler=profile_active,
                ):
                    trainer.sync_inference_model()
        if profile_active:
            _sync(device)
            profiler_active_wall_s.append(time.perf_counter() - step_start)
        if active_step:
            step_summaries.append(
                {
                    "step": phase_step,
                    "rows": rows,
                    "total_nodes": int(evaluator.total_nodes),
                    "nodes_per_root": float(evaluator.total_nodes) / max(1, int(rows)),
                    "model_indices": int(evaluator.model_indices.numel()),
                    "showdown_indices": int(evaluator.showdown_indices.numel()),
                    "value_examples": value_examples,
                    "policy_examples": policy_examples,
                }
            )
        print(
            f"{'active' if active_step else 'warmup'} step={phase_step} "
            f"nodes/root={float(evaluator.total_nodes) / max(1, int(rows)):.1f} "
            f"value={value_examples} policy={policy_examples}",
            flush=True,
        )
    if profiler_ctx is not None:
        profiler_ctx.__exit__(None, None, None)
        assert profiler_obj is not None
        assert profiler_trace_path is not None
        assert profiler_summary_path is not None
        profiler_obj.export_chrome_trace(str(profiler_trace_path))
        steps = max(1, len(profiler_active_wall_s))
        profiler_summary = {
            "active_wall_s": profiler_active_wall_s,
            "wall_ms_per_step": 1e3
            * sum(profiler_active_wall_s)
            / max(1, len(profiler_active_wall_s)),
            "top_device": _profiler_rows(profiler_obj, steps)[:80],
            "top_cpu": _profiler_cpu_rows(profiler_obj, steps)[:80],
        }
        profiler_summary_path.write_text(
            json.dumps(profiler_summary, indent=2) + "\n"
        )
        print(f"torch profiler trace: {profiler_trace_path}", flush=True)
        print(f"torch profiler summary: {profiler_summary_path}", flush=True)
        print("top device ops:", flush=True)
        for row in profiler_summary["top_device"][:20]:
            print(
                f"  {row['self_device_ms_per_step']:>8.3f} ms/step "
                f"{row['count_per_step']:>8.1f}x {row['name']}",
                flush=True,
            )
        print("top CPU ops:", flush=True)
        for row in profiler_summary["top_cpu"][:20]:
            print(
                f"  {row['self_cpu_ms_per_step']:>8.3f} ms/step "
                f"{row['count_per_step']:>8.1f}x {row['name']}",
                flush=True,
            )

    active_rows = [row for row in timer.rows if int(row["step"]) >= 0]
    output = {
        "config": {
            "bucket": args.bucket,
            "cfr_batch_size": cfr_batch_size,
            "train_batch_size": train_batch_size,
            "value_train_batch_size": train_batch_size,
            "policy_train_batch_size": policy_train_batch_size,
            "warmup_steps": args.warmup_steps,
            "active_steps": args.active_steps,
            "skip_rows": args.skip_rows,
            "solver_checkpoint": str(args.solver_checkpoint),
            "trainer_checkpoint": str(args.trainer_checkpoint),
        },
        "step_summaries": step_summaries,
        "phase_rows": timer.rows,
        "active_phase_summary": _summarize(active_rows),
        "all_phase_summary": _summarize(timer.rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["active_phase_summary"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
