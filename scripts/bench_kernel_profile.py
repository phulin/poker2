#!/usr/bin/env python3
"""Per-CUDA-kernel self-time profiler for the live CFR training config.

Matches the live run: preflop curriculum, 6 players, compact 169-hand,
depth-4 sparse_fused sapcfr, 400 iterations, BetterFFN h256 compile=default
transformer-169, num_envs=512.

Usage:
    uv run python scripts/bench_kernel_profile.py --no-pause \
        [--warmup-steps N] [--active-steps N] \
        [--out-dir outputs/opt_review/]

Saves:
    kernel_profile.json       - top kernels by self-time (JSON)
    kernel_trace.json         - Chrome trace
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2.config.rebel_load import load_rebel_config  # noqa: E402
from p2.core.structured_config import Config  # noqa: E402
from p2.rl.cfr_trainer import RebelCFRTrainer  # noqa: E402


def _find_train_rebel(pattern: str) -> list[int]:
    self_pid = os.getpid()
    parent_pid = os.getppid()
    pids = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in (self_pid, parent_pid):
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if pattern.encode() in raw:
            pids.append(pid)
    return pids


def _load_live_config() -> Config:
    overrides = [
        "curriculum.stages=[preflop]",
        "+curriculum.preflop.kind=train",
        "+curriculum.preflop.net=S_preflop",
        "+curriculum.preflop.num_steps=2000",
        "+curriculum.preflop.closing_net=E_preflop",
        f"+curriculum.preflop.closing_checkpoint={REPO_ROOT}/checkpoints-epreflop-distill-100k-lr2e4-sample256-from-sflop2000-169/promoted/E_preflop.pt",
        "+curriculum.preflop.value_checkpoint=off",
        "+curriculum.preflop.search_overrides.model_scope=mixed_street",
        "data.mode=live",
        "data.live_root_source=self_play",
        "data.include_pre_chance_value_batches=false",
        "+env.num_players=6",
        "num_envs=512",
        "train.batch_size=2048",
        "train.episodes_per_step=5",
        "train.replay_buffer_batches=128",
        "train.value_reuse_goal=2",
        "train.policy_capacity_factor=4",
        "model.name=BetterFFN",
        "model.hidden_dim=256",
        "model.range_hidden_dim=256",
        "model.ffn_dim=768",
        "model.board_interaction_dim=0",
        "model.policy_rank=96",
        "model.policy_hand_bias_rank=24",
        "model.street_value_heads=both",
        "model.shared_trunk=true",
        "model.num_hidden_layers=0",
        "model.num_value_layers=7",
        "model.num_policy_layers=6",
        "model.compile=default",
        "++model.preflop_hand_dim=169",
        "++model.preflop_model_type=transformer",
        "++model.preflop_transformer_heads=8",
        "search.depth=4",
        "search.bet_bins_by_depth=[[0.25,0.5,0.75,1.0,1.5],[0.5,1.0],[1.0],[]]",
        "search.allin_by_depth=[true,true,true,false]",
        "search.sparse=true",
        "search.sparse_fused=true",
        "search.value_targets_from_final_policy=false",
        "search.cfr_type=sapcfr",
        "search.sapcfr_alpha=2.0",
        "search.predictive_cfr_dcfr_hybrid=true",
        "search.predictive_cfr_delay=40",
        "search.cfr_avg=false",
        "search.cfr_plus=false",
        "search.iterations=400",
        "search.iterations_final=400",
        "search.warm_start_iterations=15",
        "search.warm_start_type=model_br",
        "search.warm_start_multiplier=2",
        "search.dcfr_plus_delay=80",
        "search.dcfr_alpha=3.0",
        "search.dcfr_alpha_final=4.0",
        "search.dcfr_beta=-1.0",
        "search.dcfr_beta_final=null",
        "search.dcfr_gamma=5.0",
        "search.dcfr_gamma_final=0.0",
        "+search.allin_call_terminal_abstraction=true",
        f"search.preflop_allin_table_path={REPO_ROOT}/outputs/preflop_allin_table.pt.zst",
        f"search.preflop_allin_169_cache_path={REPO_ROOT}/outputs/precompute/preflop_allin_169_u16.pt.zst",
        f"search.preflop_allin_169_model_checkpoint={REPO_ROOT}/outputs/checkpoints_allin_169_transformer/allin_64k_xf4_h192_hd128_mainlr0005_adamw0008_bs1024_maskfold_s16384_b1024_min4/latest.pt",
        "trueskill.enabled=false",
        "use_wandb=false",
        "num_steps=50",
        "device=cuda",
        # curriculum config path
        "config_name=config_rebel_curriculum_flop",
    ]
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dict_config: DictConfig = hydra.compose(
            config_name="config_rebel_curriculum_flop",
            overrides=[o for o in overrides if not o.startswith("config_name=")],
        )
    cfg = load_rebel_config(dict_config)
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_steps = 50
    return cfg


def _sync():
    torch.cuda.synchronize()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--active-steps", type=int, default=1)
    p.add_argument("--no-pause", action="store_true")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs" / "opt_review")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paused = []
    if not args.no_pause:
        pids = _find_train_rebel("train_rebel")
        for pid in pids:
            try:
                os.kill(pid, signal.SIGSTOP)
                paused.append(pid)
                print(f"Paused pid {pid}")
            except ProcessLookupError:
                pass

    try:
        cfg = _load_live_config()
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
        try:
            torch._dynamo.config.recompile_limit = 32
        except Exception:
            pass
        torch.manual_seed(cfg.seed)

        print("Building trainer...", flush=True)
        trainer = RebelCFRTrainer(cfg=cfg, device=device)
        print("Trainer ready.", flush=True)

        # Warmup steps
        for s in range(args.warmup_steps):
            t0 = time.perf_counter()
            trainer.train_step(s)
            _sync()
            print(f"[warmup {s}] {time.perf_counter() - t0:.2f}s", flush=True)

        # Profile active steps with per-kernel granularity
        active_walls = []
        trace_path = args.out_dir / "kernel_trace.json"

        print("Starting profiler...", flush=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
            on_trace_ready=None,
        ) as prof:
            for i in range(args.active_steps):
                step = args.warmup_steps + i
                t0 = time.perf_counter()
                with record_function("train_step"):
                    trainer.train_step(step)
                    _sync()
                active_walls.append(time.perf_counter() - t0)
                print(f"[active {i}] {active_walls[-1]:.2f}s", flush=True)

        wall_mean_ms = 1e3 * sum(active_walls) / max(1, len(active_walls))
        print(f"\nWall mean: {wall_mean_ms:.1f} ms/step", flush=True)

        # Export chrome trace
        prof.export_chrome_trace(str(trace_path))
        print(f"Chrome trace saved to {trace_path}")

        # Build per-kernel self-time table
        # key_averages groups by kernel name; self_cuda_time_total is self time
        avgs = prof.key_averages()

        rows = []
        for evt in avgs:
            self_cuda_us = float(
                getattr(evt, "self_device_time_total", None)
                or getattr(evt, "self_cuda_time_total", 0.0)
                or 0.0
            )
            cuda_total_us = float(
                getattr(evt, "device_time_total", None)
                or getattr(evt, "cuda_time_total", 0.0)
                or 0.0
            )
            cpu_self_us = float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
            count = int(getattr(evt, "count", 1) or 1)
            rows.append(
                {
                    "name": evt.key,
                    "count_total": count,
                    "count_per_step": count / max(1, args.active_steps),
                    "self_cuda_us_total": self_cuda_us,
                    "self_cuda_ms_per_step": self_cuda_us
                    / 1e3
                    / max(1, args.active_steps),
                    "cuda_total_us": cuda_total_us,
                    "cuda_total_ms_per_step": cuda_total_us
                    / 1e3
                    / max(1, args.active_steps),
                    "self_cpu_us_total": cpu_self_us,
                }
            )

        # Sort by self_cuda_ms_per_step
        rows.sort(key=lambda r: r["self_cuda_ms_per_step"], reverse=True)

        # Compute pct of wall
        for r in rows:
            r["self_cuda_pct_of_wall"] = (
                100.0 * r["self_cuda_ms_per_step"] / wall_mean_ms
                if wall_mean_ms
                else 0.0
            )

        top40 = rows[:40]

        # Also print the key_averages table
        print("\n=== torch profiler key_averages (sort by self_cuda_time_total) ===")
        try:
            tbl = avgs.table(sort_by="self_cuda_time_total", row_limit=40)
            print(tbl)
        except Exception as e:
            print(f"  (table() failed: {e})")
            try:
                tbl = avgs.table(sort_by="cuda_time_total", row_limit=40)
                print(tbl)
            except Exception as e2:
                print(f"  (cuda_time_total also failed: {e2})")

        print("\n=== Top 40 kernels by CUDA self-time ===")
        print(f"{'#':>3}  {'name':<65}  {'ms/step':>9}  {'%wall':>7}  {'cnt/step':>9}")
        print("-" * 105)
        for i, r in enumerate(top40):
            print(
                f"{i + 1:>3}  {r['name']:<65}  "
                f"{r['self_cuda_ms_per_step']:>9.3f}  "
                f"{r['self_cuda_pct_of_wall']:>6.1f}%  "
                f"{r['count_per_step']:>9.1f}"
            )

        output = {
            "config": {
                "warmup_steps": args.warmup_steps,
                "active_steps": args.active_steps,
                "num_envs": 512,
                "iterations": 400,
                "depth": 4,
                "model": "BetterFFN-h256-transformer-169",
            },
            "wall_mean_ms": wall_mean_ms,
            "active_wall_s": active_walls,
            "top_kernels": top40,
            "all_kernels_count": len(rows),
        }

        out_json = args.out_dir / "kernel_profile.json"
        out_json.write_text(json.dumps(output, indent=2) + "\n")
        print(f"\nKernel profile JSON saved to {out_json}")

    finally:
        for pid in paused:
            try:
                os.kill(pid, signal.SIGCONT)
                print(f"Resumed pid {pid}")
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    main()
