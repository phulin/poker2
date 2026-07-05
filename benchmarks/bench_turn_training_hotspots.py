#!/usr/bin/env python3
"""Microbenchmarks for S_turn training hot paths.

This separates the two main suspects when random-turn S_turn training is slow:
the analytic turn range-equity baseline inside BetterFFN value forwards, and the
fused sparse CFR solve/model-value evaluation for random-turn roots.

Example:
    uv run python benchmarks/bench_turn_training_hotspots.py --num-envs 512 \
        --model-batch-size 2048 --cfr-iters 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.models.mlp.better_features import ValueScalarContext, context_length
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.cfr_trainer import RebelCFRTrainer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "turn_training_hotspots_microbench.json"
DEFAULT_CUDA_DRIVER_DIR = Path("/usr/lib/x86_64-linux-gnu")


def _ensure_cuda_driver_path() -> None:
    libcuda = DEFAULT_CUDA_DRIVER_DIR / "libcuda.so.1"
    if not libcuda.exists():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in current.split(":") if part]
    driver_dir = str(DEFAULT_CUDA_DRIVER_DIR)
    if driver_dir not in parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join([driver_dir, *parts])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(samples: list[float], *, prefix: str) -> dict[str, float]:
    if not samples:
        return {
            f"{prefix}_mean_ms": 0.0,
            f"{prefix}_min_ms": 0.0,
            f"{prefix}_max_ms": 0.0,
        }
    sorted_samples = sorted(samples)
    return {
        f"{prefix}_mean_ms": sum(samples) / len(samples),
        f"{prefix}_p50_ms": sorted_samples[len(sorted_samples) // 2],
        f"{prefix}_p90_ms": sorted_samples[
            min(len(sorted_samples) - 1, int(len(sorted_samples) * 0.9))
        ],
        f"{prefix}_min_ms": min(samples),
        f"{prefix}_max_ms": max(samples),
    }


def _time_call(
    device: torch.device,
    fn: Callable[[], Any],
    *,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type != "cuda":
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append(1e3 * (time.perf_counter() - start))
        return _summarize(samples, prefix="wall")

    cuda_samples: list[float] = []
    wall_samples: list[float] = []
    for _ in range(iters):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_wall = time.perf_counter()
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize(device)
        cuda_samples.append(float(start_ev.elapsed_time(end_ev)))
        wall_samples.append(1e3 * (time.perf_counter() - start_wall))
    out = _summarize(cuda_samples, prefix="cuda")
    out.update(_summarize(wall_samples, prefix="wall"))
    return out


def _load_cfg(args: argparse.Namespace) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name=args.config,
            overrides=args.hydra_override,
        )
    cfg = load_rebel_config(dc)
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_steps = 1
    cfg.num_envs = args.num_envs
    cfg.data.mode = "live"
    cfg.data.live_root_source = "random_turn"
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    cfg.search.iterations = args.cfr_iters
    cfg.search.iterations_final = None
    cfg.search.warm_start_iterations = args.warm_start_iters
    cfg.model.compile = "default" if args.compile else "off"
    if args.depth is not None:
        cfg.search.depth = args.depth
    if args.turn_equity_chunk_size is not None:
        cfg.model.value_turn_range_equity_chunk_size = args.turn_equity_chunk_size
    return cfg


def _make_trainer(cfg: Config) -> RebelCFRTrainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)
    return RebelCFRTrainer(cfg=cfg, device=device, pregeneration_only=True)


def _random_turn_features(
    *,
    batch_size: int,
    num_players: int,
    device: torch.device,
    pot: float,
) -> MLPFeatures:
    board4 = torch.rand(batch_size, 52, device=device).argsort(dim=1)[:, :4]
    board = torch.full((batch_size, 5), -1, dtype=torch.long, device=device)
    board[:, :4] = board4
    combos = hand_combos_tensor().to(device=device)
    card_a = combos[:, 0]
    card_b = combos[:, 1]
    board_ok = (
        (card_a[None, :] != board4[:, :, None])
        & (card_b[None, :] != board4[:, :, None])
    ).all(dim=1)
    beliefs = torch.rand(
        batch_size,
        num_players,
        NUM_HANDS,
        dtype=torch.float32,
        device=device,
    )
    beliefs = beliefs * board_ok[:, None, :].to(dtype=beliefs.dtype)
    beliefs = beliefs / beliefs.sum(dim=2, keepdim=True).clamp_min(1e-8)
    context = torch.zeros(
        batch_size,
        context_length(num_players),
        dtype=torch.float32,
        device=device,
    )
    context[:, ValueScalarContext.POT.value] = pot
    return MLPFeatures(
        context=context,
        street=torch.full((batch_size,), 2, dtype=torch.long, device=device),
        to_act=torch.zeros(batch_size, dtype=torch.long, device=device),
        board=board,
        beliefs=beliefs.view(batch_size, -1),
    )


def _iter_better_ffn_modules(obj: Any) -> Iterator[BetterFFN]:
    seen: set[int] = set()
    if obj is None:
        return
    module = getattr(obj, "_orig_mod", obj)
    if isinstance(module, BetterFFN) and id(module) not in seen:
        seen.add(id(module))
        yield module
    if hasattr(module, "modules"):
        for child in module.modules():
            child = getattr(child, "_orig_mod", child)
            if isinstance(child, BetterFFN) and id(child) not in seen:
                seen.add(id(child))
                yield child


@contextmanager
def _turn_equity_enabled(root: Any, enabled: bool) -> Iterator[None]:
    modules = list(_iter_better_ffn_modules(root))
    old_values = [
        bool(getattr(module, "value_turn_range_equity_baseline", False))
        for module in modules
    ]
    for module in modules:
        module.value_turn_range_equity_baseline = enabled
    try:
        yield
    finally:
        for module, old_value in zip(modules, old_values, strict=True):
            module.value_turn_range_equity_baseline = old_value


def _benchmark_model_forward(
    trainer: RebelCFRTrainer,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = trainer.device
    model = trainer.model
    model.eval()
    features = _random_turn_features(
        batch_size=args.model_batch_size,
        num_players=trainer.num_players,
        device=device,
        pot=args.pot,
    )
    beliefs = features.beliefs.view(args.model_batch_size, trainer.num_players, NUM_HANDS)
    autocast_enabled = device.type == "cuda"
    rows: list[dict[str, Any]] = []

    base_model = next(_iter_better_ffn_modules(model), None)
    if base_model is not None:
        for chunk_size in args.chunk_sizes:
            old_chunk = base_model.value_turn_range_equity_chunk_size
            base_model.value_turn_range_equity_chunk_size = chunk_size

            def equity_features() -> Any:
                with torch.no_grad():
                    return base_model._turn_range_equity_features(
                        beliefs,
                        features,
                        torch.float32,
                    )

            row: dict[str, Any] = {
                "kind": "turn_equity_features_only",
                "batch_size": args.model_batch_size,
                "chunk_size": chunk_size,
                "rank_bins": base_model.value_turn_range_equity_rank_bins,
                "blockers": bool(base_model.value_turn_range_equity_blockers),
            }
            row.update(
                _time_call(
                    device,
                    equity_features,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
            rows.append(row)
            print(_format_row(row), flush=True)
            base_model.value_turn_range_equity_chunk_size = old_chunk

    for enabled in (False, True):

        def forward_value() -> Any:
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ),
                _turn_equity_enabled(model, enabled),
            ):
                return model(features, include_policy=False, apply_zero_sum=False)

        row = {
            "kind": "model_forward_value",
            "turn_equity_baseline": enabled,
            "batch_size": args.model_batch_size,
        }
        row.update(
            _time_call(device, forward_value, warmup=args.warmup, iters=args.iters)
        )
        rows.append(row)
        print(_format_row(row), flush=True)

    summary = _delta_summary(rows, "model_forward_value", "turn_equity_baseline")
    return rows, summary


def _prepare_evaluator(trainer: RebelCFRTrainer, args: argparse.Namespace):
    evaluator = trainer.cfr_evaluator
    pbs = _sample_pbs(trainer, args)
    return _initialize_evaluator_from_pbs(trainer, evaluator, pbs)


def _sample_pbs(trainer: RebelCFRTrainer, args: argparse.Namespace):
    if trainer.data_generator is None:
        raise RuntimeError("benchmark requires live data_generator")
    return trainer.data_generator._sample_roots(args.num_envs)


def _initialize_evaluator_from_pbs(
    trainer: RebelCFRTrainer,
    evaluator: Any,
    pbs: Any,
):
    root_indices = torch.arange(pbs.env.N, device=trainer.device)
    evaluator.initialize_subgame(pbs.env, root_indices, pbs.beliefs)
    evaluator.initialize_policy_and_beliefs()
    _sync(trainer.device)
    return evaluator


def _benchmark_evaluator(
    trainer: RebelCFRTrainer,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    device = trainer.device
    if device.type != "cuda":
        raise RuntimeError("evaluator benchmark requires CUDA")
    try:
        from triton.runtime import driver

        driver.active.get_current_device()
    except Exception as exc:
        raise RuntimeError(
            "evaluator benchmark requires a working Triton CUDA driver "
            "(libcuda.so.1 visible to the process)"
        ) from exc
    rows: list[dict[str, Any]] = []

    leaf_pbs = _sample_pbs(trainer, args)
    for enabled in (False, True):
        evaluator = _initialize_evaluator_from_pbs(
            trainer,
            trainer.cfr_evaluator,
            leaf_pbs,
        )

        def leaf_values() -> Any:
            with _turn_equity_enabled(evaluator.value_model, enabled):
                with _turn_equity_enabled(evaluator.closing_leaf_value_model, enabled):
                    return evaluator.set_leaf_values(0)

        row: dict[str, Any] = {
            "kind": "evaluator_set_leaf_values",
            "turn_equity_baseline": enabled,
            "num_envs": args.num_envs,
            "model_indices": int(evaluator.model_indices.numel()),
            "showdown_indices": int(evaluator.showdown_indices.numel()),
            "allin_call_indices": int(evaluator.allin_call_indices.numel()),
            "total_nodes": int(evaluator.total_nodes),
        }
        row.update(
            _time_call(device, leaf_values, warmup=args.warmup, iters=args.iters)
        )
        rows.append(row)
        print(_format_row(row), flush=True)

    full_pbs = _sample_pbs(trainer, args)
    for enabled in (False, True):

        def full_solve() -> Any:
            evaluator = _initialize_evaluator_from_pbs(
                trainer,
                trainer.cfr_evaluator,
                full_pbs,
            )
            with _turn_equity_enabled(evaluator.value_model, enabled):
                with _turn_equity_enabled(evaluator.closing_leaf_value_model, enabled):
                    return evaluator.evaluate_cfr(
                        training_mode=True,
                        sample_continuation=False,
                    )

        row = {
            "kind": "full_random_turn_cfr_solve",
            "turn_equity_baseline": enabled,
            "num_envs": args.num_envs,
            "cfr_iters": args.cfr_iters,
            "warm_start_iters": args.warm_start_iters,
        }
        row.update(
            _time_call(device, full_solve, warmup=args.full_warmup, iters=args.full_iters)
        )
        rows.append(row)
        print(_format_row(row), flush=True)

    summary = {
        "set_leaf_values": _delta_summary(
            rows, "evaluator_set_leaf_values", "turn_equity_baseline"
        ),
        "full_solve": _delta_summary(
            rows, "full_random_turn_cfr_solve", "turn_equity_baseline"
        ),
    }
    return rows, summary


def _row_ms(row: dict[str, Any]) -> float:
    if "cuda_mean_ms" in row:
        return float(row["cuda_mean_ms"])
    return float(row["wall_mean_ms"])


def _delta_summary(
    rows: list[dict[str, Any]],
    kind: str,
    flag_key: str,
) -> dict[str, float] | dict[str, str]:
    off = next(
        (row for row in rows if row.get("kind") == kind and row.get(flag_key) is False),
        None,
    )
    on = next(
        (row for row in rows if row.get("kind") == kind and row.get(flag_key) is True),
        None,
    )
    if off is None or on is None:
        return {"error": f"missing rows for {kind}"}
    off_ms = _row_ms(off)
    on_ms = _row_ms(on)
    return {
        "off_mean_ms": off_ms,
        "on_mean_ms": on_ms,
        "delta_ms": on_ms - off_ms,
        "ratio": on_ms / max(off_ms, 1e-9),
    }


def _format_row(row: dict[str, Any]) -> str:
    ms_key = "cuda_mean_ms" if "cuda_mean_ms" in row else "wall_mean_ms"
    parts = [str(row["kind"])]
    for key in (
        "turn_equity_baseline",
        "batch_size",
        "num_envs",
        "chunk_size",
        "cfr_iters",
        "model_indices",
        "total_nodes",
    ):
        if key in row:
            parts.append(f"{key}={row[key]}")
    parts.append(f"{ms_key}={float(row[ms_key]):.3f}")
    return " ".join(parts)


def _parse_csv_ints(value: str) -> list[int]:
    out = [int(part) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_rebel_curriculum_turn")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--model-batch-size", type=int, default=2048)
    parser.add_argument("--cfr-iters", type=int, default=300)
    parser.add_argument("--warm-start-iters", type=int, default=15)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--pot", type=float, default=100.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--full-warmup", type=int, default=1)
    parser.add_argument("--full-iters", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--mode", choices=("all", "model", "evaluator"), default="all")
    parser.add_argument("--chunk-sizes", type=_parse_csv_ints, default=[32, 64, 128])
    parser.add_argument("--turn-equity-chunk-size", type=int, default=None)
    parser.add_argument("hydra_override", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    _ensure_cuda_driver_path()
    args = parse_args(argv)
    cfg = _load_cfg(args)
    trainer = _make_trainer(cfg)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if args.mode in {"all", "model"}:
        model_rows, model_summary = _benchmark_model_forward(trainer, args)
        rows.extend(model_rows)
        summary["model"] = model_summary
    if args.mode in {"all", "evaluator"}:
        evaluator_rows, evaluator_summary = _benchmark_evaluator(trainer, args)
        rows.extend(evaluator_rows)
        summary["evaluator"] = evaluator_summary

    payload = {
        "argv": argv,
        "device": str(trainer.device),
        "config": {
            "config": args.config,
            "num_envs": cfg.num_envs,
            "model_batch_size": args.model_batch_size,
            "cfr_iters": cfg.search.iterations,
            "warm_start_iters": cfg.search.warm_start_iterations,
            "depth": cfg.search.depth,
            "compile": cfg.model.compile,
            "turn_equity_baseline": cfg.model.value_turn_range_equity_baseline,
            "turn_equity_chunk_size": cfg.model.value_turn_range_equity_chunk_size,
            "turn_equity_rank_bins": cfg.model.value_turn_range_equity_rank_bins,
            "turn_equity_blockers": cfg.model.value_turn_range_equity_blockers,
        },
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
