#!/usr/bin/env python3
"""Benchmark eager fused sparse CFR iterator work from saved spots.

This microbenchmark selects an equal number of roots from preflop/flop/turn/
river in ``outputs/spots.pt``, initializes one fused sparse evaluator subgame,
and profiles eager ``cfr_iteration`` segments. It intentionally bypasses
``evaluate_cfr`` CUDA graph capture so the iterator kernels and model leaf
work are visible.
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

from p2.cli.sample_spots import build_pbs_from_spots, load_spots
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.fused_cfr_triton import fused_model_values_writeback_
from p2.search.fused_cfr_triton import fused_parent_sum_divide_


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "cfr_iterator_spots_micro.json"
STREET_TO_ID = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}

COMPONENT_TAGS = [
    "iterator_segment",
    "cfr_iter",
    "cfr_iter_update_policy",
    "cfr_iter_expected_values",
    "cfr_iter_avg_policy",
    "cfr_iter_avg_values",
    "cfr_set_leaf",
    "cfr_leaf_showdown",
    "cfr_leaf_set_model_values",
    "cfr_model_fwd",
]


def _iter_processes() -> list[tuple[int, str]]:
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return []
    own = {os.getpid(), os.getppid()}
    out: list[tuple[int, str]] = []
    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in own:
            continue
        try:
            cmd = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        out.append((pid, cmd))
    return out


@contextmanager
def pause_train_rebel(enabled: bool, pattern: str):
    paused: list[int] = []
    if enabled:
        for pid, cmd in _iter_processes():
            if pattern not in cmd:
                continue
            try:
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
            except ProcessLookupError:
                continue
        if paused:
            print(f"Paused train_rebel processes: {paused}", flush=True)
            time.sleep(0.5)
        else:
            print(f"No process matched pause pattern {pattern!r}.", flush=True)
    try:
        yield
    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                continue
        if paused:
            print(f"Resumed train_rebel processes: {paused}", flush=True)


def _wrap_method(obj: Any, attr: str, tag: str) -> None:
    orig = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        with record_function(tag):
            return orig(*args, **kwargs)

    setattr(obj, attr, wrapped)


def _wrap_model_call(cls: type, tag: str) -> None:
    orig = cls.__call__

    def wrapped(self, *args, **kwargs):
        with record_function(tag):
            return orig(self, *args, **kwargs)

    cls.__call__ = wrapped


def _patch_profiler_tags(trainer: RebelCFRTrainer) -> None:
    _wrap_model_call(trainer.cfr_evaluator.model.__class__, "cfr_model_fwd")
    ev = trainer.cfr_evaluator
    for attr, tag in (
        ("cfr_iteration", "cfr_iter"),
        ("update_policy", "cfr_iter_update_policy"),
        ("update_average_policy", "cfr_iter_avg_policy"),
        ("update_average_values", "cfr_iter_avg_values"),
        ("set_leaf_values", "cfr_set_leaf"),
        ("_showdown_value_both", "cfr_leaf_showdown"),
        ("_set_model_values", "cfr_leaf_set_model_values"),
        ("compute_expected_values", "cfr_iter_expected_values"),
    ):
        if hasattr(ev, attr):
            _wrap_method(ev, attr, tag)


def _event_device_us(evt) -> float:
    value = getattr(evt, "device_time_total", None)
    if value is not None:
        return float(value or 0.0)
    return float(getattr(evt, "cuda_time_total", 0.0) or 0.0)


def _profiler_rows(prof, iters: int, wall_s: float) -> list[dict[str, float | str]]:
    rows = []
    for tag in COMPONENT_TAGS:
        cuda_us = 0.0
        cpu_us = 0.0
        calls = 0
        for evt in prof.events():
            if evt.name != tag:
                continue
            cuda_us += _event_device_us(evt)
            cpu_us += float(evt.cpu_time_total or 0.0)
            calls += 1
        if calls == 0:
            continue
        rows.append(
            {
                "component": tag,
                "calls": float(calls),
                "calls_per_iter": calls / max(1, iters),
                "cuda_ms_total": cuda_us / 1e3,
                "cuda_ms_per_iter": cuda_us / 1e3 / max(1, iters),
                "cpu_ms_total": cpu_us / 1e3,
                "cpu_ms_per_iter": cpu_us / 1e3 / max(1, iters),
                "cuda_pct_of_wall": (100.0 * cuda_us / 1e6 / wall_s)
                if wall_s > 0
                else 0.0,
            }
        )
    rows.sort(key=lambda r: float(r["cuda_ms_total"]), reverse=True)
    return rows


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


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


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_envs = args.per_street * 4
    cfg.model.hidden_dim = 256
    cfg.model.range_hidden_dim = 128
    cfg.model.ffn_dim = 512
    cfg.model.num_hidden_layers = 3
    cfg.model.num_value_layers = 1
    cfg.model.num_policy_layers = 1
    cfg.model.compile = not args.no_compile
    if args.compile_mode is not None:
        cfg.model.compile_mode = args.compile_mode
    cfg.search.depth = args.depth
    cfg.search.iterations = args.iterations
    cfg.search.iterations_final = None
    cfg.search.sparse = True
    cfg.search.sparse_fused = not args.unfused
    if args.dcfr_delay is not None:
        cfg.search.dcfr_plus_delay = args.dcfr_delay


def _load_cfg(args: argparse.Namespace) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_cfr",
            overrides=args.hydra_override,
        )
    cfg = Config.from_dict_config(dc)
    _apply_overrides(cfg, args)
    return cfg


def _select_even_spots(
    payload: dict[str, Any], per_street: int, randomize: bool, seed: int
) -> torch.Tensor:
    chunks = []
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    streets = payload["spots"]["street"]
    for street_id in STREET_TO_ID.values():
        candidates = torch.where(streets == street_id)[0].cpu()
        if candidates.numel() < per_street:
            raise ValueError(
                f"Need {per_street} spots for street {street_id}, "
                f"but only found {candidates.numel()}."
            )
        if randomize:
            candidates = candidates[torch.randperm(candidates.numel(), generator=g)]
        chunks.append(candidates[:per_street])
    return torch.cat(chunks, dim=0)


def _prepare_iterator_state(trainer: RebelCFRTrainer, pbs, skip_record_stats: bool):
    ev = trainer.cfr_evaluator
    ev.model.eval()
    with torch.no_grad():
        src_indices = torch.arange(pbs.env.N, device=trainer.device)
        ev.initialize_subgame(pbs.env, src_indices, pbs.beliefs)
        ev.initialize_policy_and_beliefs()
        if ev.warm_start_iterations > 0:
            ev.warm_start()
        ev.set_leaf_values(0)
        ev.compute_expected_values()
        ev.values_avg[:] = ev.latest_values
        ev.t_sample = ev._get_sampling_schedule()
        if hasattr(ev, "_prepare_sample_update_table"):
            ev._prepare_sample_update_table()
        if hasattr(ev, "_skip_record_stats"):
            ev._skip_record_stats = skip_record_stats
    _sync(trainer.device)


def _run_segment(
    trainer: RebelCFRTrainer,
    pbs,
    name: str,
    start_t: int,
    warmup_iters: int,
    active_iters: int,
    skip_record_stats: bool,
) -> dict[str, Any]:
    _prepare_iterator_state(trainer, pbs, skip_record_stats=skip_record_stats)
    ev = trainer.cfr_evaluator
    with torch.no_grad():
        for i in range(warmup_iters):
            ev.cfr_iteration(start_t + i)
    _sync(trainer.device)

    activities = [ProfilerActivity.CPU]
    if trainer.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    t0 = time.perf_counter()
    with torch.no_grad():
        with profile(activities=activities, record_shapes=False) as prof:
            with record_function("iterator_segment"):
                for i in range(active_iters):
                    ev.cfr_iteration(start_t + warmup_iters + i)
                _sync(trainer.device)
    wall_s = time.perf_counter() - t0
    return {
        "name": name,
        "start_t": int(start_t),
        "warmup_iters": int(warmup_iters),
        "active_iters": int(active_iters),
        "wall_s": wall_s,
        "wall_ms_per_iter": 1e3 * wall_s / max(1, active_iters),
        "rows": _profiler_rows(prof, active_iters, wall_s),
    }


def _run_component_benchmarks(
    trainer: RebelCFRTrainer,
    pbs,
    start_t: int,
    warmup_iters: int,
    active_iters: int,
    skip_record_stats: bool,
) -> dict[str, Any]:
    ev = trainer.cfr_evaluator
    results: dict[str, Any] = {}

    def time_component(name: str, setup: Callable[[], None], fn: Callable[[], Any]):
        setup()
        results[name] = _cuda_event_time(
            trainer.device,
            fn,
            warmup=warmup_iters,
            iters=active_iters,
        )

    def setup_state() -> None:
        _prepare_iterator_state(
            trainer,
            pbs,
            skip_record_stats=skip_record_stats,
        )

    t_box = {"t": start_t}

    def next_t() -> int:
        t = t_box["t"]
        t_box["t"] += 1
        return t

    time_component(
        "cfr_iteration",
        setup_state,
        lambda: ev.cfr_iteration(next_t()),
    )
    if all(
        hasattr(ev, name)
        for name in (
            "_prepare_sample_update_table",
            "_sample_update_rows",
            "_sample_update_counts",
            "_t_scalars",
        )
    ):
        from p2.search.fused_cfr_triton import fused_policy_sample_update_

        def setup_policy_sample_update() -> None:
            setup_state()
            ev.prepare_replay(start_t)
            ev._prepare_sample_update_table()

        def policy_sample_update():
            return fused_policy_sample_update_(
                ev.policy_probs,
                ev.policy_probs_sample,
                ev._sample_update_rows,
                ev._sample_update_counts,
                ev._t_scalars.t_tensor,
            )

        time_component(
            "policy_sample_update",
            setup_policy_sample_update,
            policy_sample_update,
        )
    time_component(
        "set_leaf_values",
        setup_state,
        lambda: ev.set_leaf_values(next_t()),
    )
    if hasattr(ev, "_leaf_beliefs_for_model_and_showdown") and hasattr(
        ev, "_model_features_for_beliefs"
    ):
        leaf_box: dict[str, Any] = {}

        def setup_leaf_parts() -> None:
            setup_state()
            ev.prepare_replay(start_t)
            beliefs = ev.beliefs_avg if ev.cfr_avg else ev.beliefs
            beliefs_at_model, showdown_beliefs = (
                ev._leaf_beliefs_for_model_and_showdown(beliefs)
            )
            features_at_model = ev._model_features_for_beliefs(beliefs_at_model)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                base_model = getattr(ev.model, "_orig_mod", ev.model)
                static_features = None
                if hasattr(base_model, "static_feature_base"):
                    static_features = getattr(ev, "_static_model_base_features", None)
                    if static_features is None:
                        static_features = base_model.static_feature_base(
                            features_at_model
                        )
                model_output = ev.model(
                    features_at_model,
                    include_policy=False,
                    apply_zero_sum=bool(ev.cfr_avg),
                    static_base_features=static_features,
                )
            leaf_box["beliefs"] = beliefs
            leaf_box["beliefs_at_model"] = beliefs_at_model
            leaf_box["showdown_beliefs"] = showdown_beliefs
            leaf_box["features_at_model"] = features_at_model
            leaf_box["static_features"] = getattr(
                ev, "_static_model_base_features", None
            )
            leaf_box["hand_values"] = model_output.hand_values.contiguous()

        def leaf_belief_gather():
            beliefs = leaf_box["beliefs"]
            return ev._leaf_beliefs_for_model_and_showdown(beliefs)

        def leaf_feature_assembly():
            return ev._model_features_for_beliefs(leaf_box["beliefs_at_model"])

        def leaf_model_forward():
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return ev.model(
                    leaf_box["features_at_model"],
                    include_policy=False,
                    apply_zero_sum=bool(ev.cfr_avg),
                    static_base_features=leaf_box["static_features"],
                )

        def leaf_model_writeback():
            hand_values = leaf_box["hand_values"]
            fused_model_values_writeback_(
                hand_values=hand_values,
                last_model_values=hand_values,
                beliefs=leaf_box["beliefs_at_model"].contiguous(),
                model_indices=ev.model_indices.contiguous(),
                latest_values=ev.latest_values,
                last_out=hand_values,
                old_plus_new_over_new=ev._t_scalars.mix_onon,
                old_over_new=ev._t_scalars.mix_oon,
                do_mix=False,
                enforce_zero_sum=bool(ev.model.enforce_zero_sum) and not ev.cfr_avg,
                store_last=False,
            )

        def leaf_showdown_ev():
            return ev._showdown_value_both(leaf_box["showdown_beliefs"])

        time_component("leaf_belief_gather", setup_leaf_parts, leaf_belief_gather)
        time_component("leaf_feature_assembly", setup_leaf_parts, leaf_feature_assembly)
        time_component("leaf_model_forward", setup_leaf_parts, leaf_model_forward)
        time_component("leaf_model_writeback", setup_leaf_parts, leaf_model_writeback)
        time_component("leaf_showdown_ev", setup_leaf_parts, leaf_showdown_ev)
    time_component(
        "compute_expected_values",
        setup_state,
        ev.compute_expected_values,
    )
    if hasattr(ev, "_regret_src_weights"):
        def setup_regret_src_weights() -> None:
            setup_state()
            ev._prepare_tree_slices()

        def regret_src_weights():
            beliefs = ev.beliefs_avg if ev.cfr_avg else ev.beliefs
            return ev._regret_src_weights(beliefs, ev._top)

        time_component(
            "regret_src_weights",
            setup_regret_src_weights,
            regret_src_weights,
        )
    time_component(
        "update_policy",
        setup_state,
        lambda: ev.update_policy(next_t()),
    )
    if hasattr(ev, "_ensure_positive_regrets_buf"):

        def setup_regret_matching_parent_sum_divide() -> None:
            setup_state()
            ev._prepare_tree_slices()
            positive_regrets = ev._ensure_positive_regrets_buf()
            torch.clamp(ev.cumulative_regrets, min=0.0, out=positive_regrets)
            bottom = ev._bottom
            parent_sum_box["positive_regrets"] = positive_regrets.contiguous()
            parent_sum_box["fallback"] = ev.uniform_policy[bottom:].contiguous()
            parent_sum_box["out"] = ev.policy_probs[bottom:]
            parent_sum_box["bottom"] = bottom
            parent_sum_box["child_offsets"] = ev._child_offsets_top
            parent_sum_box["child_count"] = ev._child_count_top

        parent_sum_box: dict[str, Any] = {}

        def regret_matching_parent_sum_divide():
            return fused_parent_sum_divide_(
                values=parent_sum_box["positive_regrets"],
                fallback=parent_sum_box["fallback"],
                child_offsets=parent_sum_box["child_offsets"],
                child_count=parent_sum_box["child_count"],
                out=parent_sum_box["out"],
                out_offset=parent_sum_box["bottom"],
                max_children=ev.num_actions,
            )

        time_component(
            "regret_matching_parent_sum_divide",
            setup_regret_matching_parent_sum_divide,
            regret_matching_parent_sum_divide,
        )
    time_component(
        "calculate_reach_weights",
        setup_state,
        lambda: ev._calculate_reach_weights(ev.self_reach, ev.policy_probs),
    )
    time_component(
        "propagate_beliefs",
        setup_state,
        lambda: ev._propagate_all_beliefs(ev.beliefs, ev.self_reach),
    )
    time_component(
        "update_average_policy",
        setup_state,
        lambda: ev.update_average_policy(next_t()),
    )
    time_component(
        "update_average_values",
        setup_state,
        lambda: ev.update_average_values(next_t()),
    )
    if hasattr(ev, "_finalize_deferred_average_policy"):
        def setup_deferred_average_policy() -> None:
            setup_state()
            ev.update_average_policy(next_t())
            _sync(trainer.device)

        time_component(
            "finalize_deferred_average_policy",
            setup_deferred_average_policy,
            ev._finalize_deferred_average_policy,
        )
    return results


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots", default="outputs/spots.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-street", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--dcfr-delay", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--active-iters", type=int, default=10)
    parser.add_argument("--component-warmup", type=int, default=2)
    parser.add_argument("--component-iters", type=int, default=10)
    parser.add_argument("--random-spots", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="Optional torch.compile mode, e.g. max-autotune.",
    )
    parser.add_argument(
        "--unfused",
        action="store_true",
        help="Use SparseCFREvaluator instead of FusedSparseCFREvaluator.",
    )
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument(
        "--record-stats",
        action="store_true",
        help="Include percentile iteration stats cloning/recording in cfr_iteration.",
    )
    parser.add_argument(
        "--no-component-bench",
        action="store_true",
        help="Skip isolated CUDA-event component microbenchmarks.",
    )
    parser.add_argument("hydra_override", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    payload = load_spots(args.spots, map_location="cpu")
    indices = _select_even_spots(
        payload,
        per_street=args.per_street,
        randomize=args.random_spots,
        seed=args.seed,
    )
    selected_streets = payload["spots"]["street"][indices]
    street_counts = {
        name: int((selected_streets == sid).sum().item())
        for name, sid in STREET_TO_ID.items()
    }

    cfg = _load_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    _patch_profiler_tags(trainer)
    pbs = build_pbs_from_spots(payload, device=device, indices=indices)

    ev = trainer.cfr_evaluator
    delay = int(ev.dcfr_delay)
    start_pre = max(2, int(ev.warm_start_iterations))
    start_post = max(start_pre, delay + 1)

    output: dict[str, Any] = {
        "spots": str(args.spots),
        "selected_indices": indices.tolist(),
        "street_counts": street_counts,
        "config": {
            "evaluator": trainer.cfr_evaluator.__class__.__name__,
            "num_envs": cfg.num_envs,
            "per_street": args.per_street,
            "depth": cfg.search.depth,
            "iterations": cfg.search.iterations,
            "dcfr_delay": delay,
            "warm_start_iterations": int(ev.warm_start_iterations),
            "model": {
                "hidden_dim": cfg.model.hidden_dim,
                "range_hidden_dim": cfg.model.range_hidden_dim,
                "ffn_dim": cfg.model.ffn_dim,
                "num_hidden_layers": cfg.model.num_hidden_layers,
                "num_value_layers": cfg.model.num_value_layers,
                "num_policy_layers": cfg.model.num_policy_layers,
                "compile": cfg.model.compile,
                "compile_mode": cfg.model.compile_mode,
            },
        },
    }

    print(json.dumps({"config": output["config"], "street_counts": street_counts}, indent=2))
    with pause_train_rebel(not args.no_pause, args.pause_pattern):
        _prepare_iterator_state(
            trainer, pbs, skip_record_stats=not args.record_stats
        )
        output["subgame"] = {
            "total_nodes": int(ev.total_nodes),
            "tree_depth": int(ev.tree_depth),
            "model_indices": int(ev.model_indices.numel()),
            "showdown_indices": int(ev.showdown_indices.numel()),
            "leaf_count": int(ev.leaf_mask.sum().item()),
        }
        print(f"Subgame: {output['subgame']}", flush=True)
        output["segments"] = [
            _run_segment(
                trainer,
                pbs,
                name="pre_dcfr_delay",
                start_t=start_pre,
                warmup_iters=args.warmup_iters,
                active_iters=args.active_iters,
                skip_record_stats=not args.record_stats,
            ),
            _run_segment(
                trainer,
                pbs,
                name="post_dcfr_delay",
                start_t=start_post,
                warmup_iters=args.warmup_iters,
                active_iters=args.active_iters,
                skip_record_stats=not args.record_stats,
            ),
        ]
        if not args.no_component_bench:
            output["component_microbenchmarks"] = _run_component_benchmarks(
                trainer,
                pbs,
                start_t=start_post,
                warmup_iters=args.component_warmup,
                active_iters=args.component_iters,
                skip_record_stats=not args.record_stats,
            )

    for seg in output["segments"]:
        print(
            f"\n{seg['name']}: wall={seg['wall_ms_per_iter']:.3f} ms/iter "
            f"(start_t={seg['start_t']})"
        )
        for row in seg["rows"][:12]:
            print(
                f"  {row['component']:<26}"
                f" cuda={row['cuda_ms_per_iter']:>8.3f} ms/iter"
                f" cpu={row['cpu_ms_per_iter']:>8.3f}"
                f" calls/iter={row['calls_per_iter']:.1f}"
            )

    if "component_microbenchmarks" in output:
        print("\nComponent microbenchmarks:")
        for name, row in output["component_microbenchmarks"].items():
            print(
                f"  {name:<34}"
                f" cuda={row['cuda_ms_mean']:>8.3f} ms"
                f" wall={row['wall_ms_mean']:>8.3f} ms"
                f" iters={int(row['iters'])}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
