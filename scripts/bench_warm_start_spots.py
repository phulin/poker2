#!/usr/bin/env python3
"""Microbenchmark fused sparse CFR warm-start work from saved spots.

This reconstructs a representative PBS from ``outputs/spots.pt``, initializes a
fused sparse evaluator subgame, and times ``warm_start()`` plus the same work
split into its main substeps. It pauses any live ``train_rebel`` process while
measuring so background training does not distort CUDA timings.
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

from p2.cli.sample_spots import build_pbs_from_spots, load_spots
from p2.core.structured_config import Config, WarmStartType
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.better_trm import BetterTRM
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.fused_cfr_triton import fused_model_values_writeback_


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "cfr_warm_start_spots_micro.json"
STREET_TO_ID = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _cuda_time_ms(device: torch.device, fn: Callable[[], Any]) -> tuple[float, float]:
    if device.type != "cuda":
        t0 = time.perf_counter()
        fn()
        return 0.0, 1e3 * (time.perf_counter() - t0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), 1e3 * (time.perf_counter() - t0)


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
    cfg.model.compile = "off" if args.no_compile else "default"
    cfg.search.depth = args.depth
    cfg.search.iterations = args.iterations
    cfg.search.iterations_final = None
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    cfg.search.warm_start_iterations = (
        args.warm_start
        if args.warm_start is not None
        else max(1, args.iterations // 20)
    )
    cfg.search.dcfr_plus_delay = (
        args.dcfr_delay
        if args.dcfr_delay is not None
        else int(round(args.iterations * 0.4))
    )


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


def _initialize_for_warm_start(trainer: RebelCFRTrainer, pbs) -> None:
    ev = trainer.cfr_evaluator
    with torch.no_grad():
        src_indices = torch.arange(pbs.env.N, device=trainer.device)
        ev.initialize_subgame(pbs.env, src_indices, pbs.beliefs)
        ev.initialize_policy_and_beliefs()
    _sync(trainer.device)


def _run_full_warm_start(trainer: RebelCFRTrainer, pbs) -> dict[str, float]:
    ev = trainer.cfr_evaluator
    _initialize_for_warm_start(trainer, pbs)
    with torch.no_grad():
        cuda_ms, wall_ms = _cuda_time_ms(trainer.device, ev.warm_start)
    return {"cuda_ms": cuda_ms, "wall_ms": wall_ms}


def _run_split_warm_start(trainer: RebelCFRTrainer, pbs) -> dict[str, float]:
    ev = trainer.cfr_evaluator
    _initialize_for_warm_start(trainer, pbs)
    out: dict[str, float] = {}

    def timed(name: str, fn: Callable[[], Any]) -> Any:
        result_box: dict[str, Any] = {}

        def run():
            result_box["value"] = fn()

        cuda_ms, wall_ms = _cuda_time_ms(trainer.device, run)
        out[f"{name}_cuda_ms"] = cuda_ms
        out[f"{name}_wall_ms"] = wall_ms
        return result_box.get("value")

    with torch.no_grad():
        if args_global_split_set_leaf():
            _run_split_set_leaf_values(ev, timed)
        else:
            timed("set_leaf_values", lambda: ev.set_leaf_values(0))
        timed(
            "compute_expected_values",
            lambda: ev.compute_expected_values(
                policy=ev.policy_probs,
                beliefs=ev.beliefs,
                leaf_values=ev.latest_values,
                values=ev.latest_values,
            ),
        )
        timed("record_initial_exploitability", ev._record_initial_exploitability)

        if ev.warm_start_type == WarmStartType.model:
            bottom = ev.depth_offsets[1]
            weight = float(ev.warm_start_iterations) * float(
                ev.warm_start_multiplier
            )
            regrets = timed(
                "compute_instantaneous_regrets",
                lambda: ev.compute_instantaneous_regrets(ev.latest_values),
            )

            def seed_regrets() -> None:
                weights = weight * regrets[bottom:].clamp(min=0.0).mean(dim=-1)
                ev.cumulative_regrets[bottom:] = (
                    ev.policy_probs[bottom:] * weights[:, None]
                )

            timed("seed_regrets", seed_regrets)
            timed("update_policy", lambda: ev.update_policy(ev.warm_start_iterations))
        else:
            zero_actor = torch.zeros_like(ev.env.to_act)
            one_actor = torch.ones_like(ev.env.to_act)
            br0 = timed(
                "best_response_p0",
                lambda: ev._best_response_values(
                    ev.policy_probs, ev.beliefs, ev.latest_values, zero_actor
                ),
            )
            br1 = timed(
                "best_response_p1",
                lambda: ev._best_response_values(
                    ev.policy_probs, ev.beliefs, ev.latest_values, one_actor
                ),
            )

            def br_regrets() -> torch.Tensor:
                values_br = torch.where(ev.prev_actor[:, None, None] == 0, br0, br1)
                return ev.compute_instantaneous_regrets(
                    values_achieved=values_br, values_expected=ev.latest_values
                )

            regrets = timed("compute_instantaneous_regrets", br_regrets)

            def seed_regrets() -> None:
                scale = float(ev.warm_start_iterations) * float(
                    ev.warm_start_multiplier
                )
                ev.cumulative_regrets += scale * regrets

            timed("seed_regrets", seed_regrets)
            timed("update_policy", lambda: ev.update_policy(ev.warm_start_iterations))

    out["split_sum_cuda_ms"] = sum(v for k, v in out.items() if k.endswith("_cuda_ms"))
    out["split_sum_wall_ms"] = sum(v for k, v in out.items() if k.endswith("_wall_ms"))
    return out


def args_global_split_set_leaf() -> bool:
    return bool(getattr(args_global_split_set_leaf, "enabled", False))


def _run_split_set_leaf_values(ev, timed: Callable[[str, Callable[[], Any]], Any]):
    beliefs = ev.beliefs_avg if ev.cfr_avg else ev.beliefs

    if ev.model_indices.numel() > 0:
        beliefs_at_model = timed(
            "set_leaf_gather_beliefs",
            lambda: beliefs[ev.model_indices],
        )
        features_at_model = timed(
            "set_leaf_feature_encode",
            lambda: ev._model_features_for_beliefs(beliefs_at_model),
        )
        showdown_beliefs = timed(
            "set_leaf_gather_showdown",
            lambda: beliefs[ev.showdown_indices],
        )
        timed(
            "set_leaf_model_forward",
            lambda: _model_forward_values(ev, features_at_model),
        )
        hand_values = getattr(ev, "_bench_last_hand_values")
        timed(
            "set_leaf_model_writeback",
            lambda: _model_values_writeback(ev, beliefs_at_model, hand_values),
        )
    else:
        empty_shape = (0, ev.num_players, NUM_HANDS)
        if ev._last_model_values_buf is None or (
            ev._last_model_values_buf.shape != empty_shape
        ):
            ev._last_model_values_buf = ev.latest_values.new_empty(empty_shape)
        ev.last_model_values = ev._last_model_values_buf
        showdown_beliefs = timed(
            "set_leaf_gather_showdown",
            lambda: beliefs[ev.showdown_indices],
        )

    showdown_values = timed(
        "set_leaf_showdown_values",
        lambda: ev._showdown_value_both(showdown_beliefs),
    )
    timed(
        "set_leaf_showdown_writeback",
        lambda: ev.latest_values.__setitem__(ev.showdown_indices, showdown_values),
    )


def _model_forward_values(ev, features_at_model: MLPFeatures) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if isinstance(ev.model, BetterTRM):
            model_output = ev.model(
                features_at_model, include_policy=False, latent=ev.latent
            )
            ev.latent = model_output.latent
        else:
            model_output = ev.model(features_at_model, include_policy=False)
    hand_values = model_output.hand_values.contiguous()
    ev._bench_last_hand_values = hand_values
    return hand_values


def _model_values_writeback(ev, beliefs_at_model: torch.Tensor, hand_values: torch.Tensor):
    do_mix = ev.cfr_avg and 0 > 1 and ev.last_model_values is not None
    store_last = bool(ev.cfr_avg)
    if store_last:
        last_shape = (hand_values.shape[0], ev.num_players, NUM_HANDS)
        if ev._last_model_values_buf is None or (
            ev._last_model_values_buf.shape != last_shape
        ):
            ev._last_model_values_buf = ev.latest_values.new_empty(last_shape)
        last_out = ev._last_model_values_buf
        last_model_values = ev.last_model_values.contiguous() if do_mix else hand_values
    else:
        last_out = hand_values
        last_model_values = hand_values
    fused_model_values_writeback_(
        hand_values=hand_values,
        last_model_values=last_model_values,
        beliefs=beliefs_at_model.contiguous(),
        model_indices=ev.model_indices.contiguous(),
        latest_values=ev.latest_values,
        last_out=last_out,
        old_plus_new_over_new=ev._t_scalars.mix_onon,
        old_over_new=ev._t_scalars.mix_oon,
        do_mix=do_mix,
        enforce_zero_sum=bool(ev.model.enforce_zero_sum) and do_mix,
        store_last=store_last,
    )
    ev.last_model_values = last_out if store_last else None
    return ev.latest_values, last_out


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows]
        out[f"{key}_mean"] = sum(vals) / len(vals)
        out[f"{key}_min"] = min(vals)
        out[f"{key}_max"] = max(vals)
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots", default="outputs/spots.pt")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-street", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--warm-start", type=int, default=None)
    parser.add_argument("--dcfr-delay", type=int, default=None)
    parser.add_argument("--compile-warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--split-first",
        action="store_true",
        help="Run the split warm-start before full warm_start in each repeat.",
    )
    parser.add_argument(
        "--split-set-leaf",
        action="store_true",
        help="Split set_leaf_values into feature/model/showdown substeps.",
    )
    parser.add_argument("--random-spots", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-pause", action="store_true")
    parser.add_argument("--pause-pattern", default="train_rebel")
    parser.add_argument("hydra_override", nargs="*")
    return parser.parse_args(argv)


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    args_global_split_set_leaf.enabled = bool(args.split_set_leaf)
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
    trainer.cfr_evaluator.model.eval()
    pbs = build_pbs_from_spots(payload, device=device, indices=indices)
    ev = trainer.cfr_evaluator

    config_summary = {
        "evaluator": ev.__class__.__name__,
        "spots": str(args.spots),
        "num_envs": cfg.num_envs,
        "per_street": args.per_street,
        "street_counts": street_counts,
        "depth": cfg.search.depth,
        "iterations": cfg.search.iterations,
        "warm_start_iterations": int(ev.warm_start_iterations),
        "warm_start_type": str(ev.warm_start_type),
        "warm_start_multiplier": float(ev.warm_start_multiplier),
        "dcfr_delay": int(ev.dcfr_delay),
        "compile": cfg.model.compile,
        "repeats": args.repeats,
        "compile_warmups": args.compile_warmups,
    }
    print(json.dumps({"config": config_summary}, indent=2), flush=True)

    with pause_train_rebel(not args.no_pause, args.pause_pattern):
        warmup_rows = []
        for i in range(args.compile_warmups):
            row = _run_full_warm_start(trainer, pbs)
            warmup_rows.append(row)
            print(
                f"[compile warmup {i}] warm_start cuda={row['cuda_ms']:.3f} ms "
                f"wall={row['wall_ms']:.3f} ms",
                flush=True,
            )

        full_rows = []
        split_rows = []
        subgame_summary: dict[str, int] | None = None
        for i in range(args.repeats):
            if args.split_first:
                split = _run_split_warm_start(trainer, pbs)
                full = _run_full_warm_start(trainer, pbs)
            else:
                full = _run_full_warm_start(trainer, pbs)
                split = _run_split_warm_start(trainer, pbs)
            full_rows.append(full)
            split_rows.append(split)
            subgame_summary = {
                "total_nodes": int(ev.total_nodes),
                "tree_depth": int(ev.tree_depth),
                "model_indices": int(ev.model_indices.numel()),
                "showdown_indices": int(ev.showdown_indices.numel()),
                "leaf_count": int(ev.leaf_mask.sum().item()),
            }
            print(
                f"[repeat {i}] full={full['cuda_ms']:.3f} ms cuda, "
                f"split_sum={split['split_sum_cuda_ms']:.3f} ms cuda",
                flush=True,
            )

    output = {
        "config": config_summary,
        "subgame": subgame_summary,
        "compile_warmups": warmup_rows,
        "full_warm_start": {
            "rows": full_rows,
            "summary": _summarize(full_rows),
        },
        "split_warm_start": {
            "rows": split_rows,
            "summary": _summarize(split_rows),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")

    print("\nSummary:")
    print(
        "  full warm_start: "
        f"{output['full_warm_start']['summary']['cuda_ms_mean']:.3f} ms cuda "
        f"({output['full_warm_start']['summary']['wall_ms_mean']:.3f} ms wall)"
    )
    split_summary = output["split_warm_start"]["summary"]
    for key in sorted(split_summary):
        if not key.endswith("_cuda_ms_mean") or key == "split_sum_cuda_ms_mean":
            continue
        print(f"  {key.removesuffix('_cuda_ms_mean'):<32} {split_summary[key]:>9.3f} ms cuda")
    print(f"  {'split_sum':<32} {split_summary['split_sum_cuda_ms_mean']:>9.3f} ms cuda")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
