#!/usr/bin/env python3
"""Probe the bimodal `action_mix/allin` (and avg-policy entropy) seen in the
fresh-batch metrics.

Decisive experiment: hold the *inputs* fixed (same spots, same schedule) and
re-run ``evaluate_cfr`` K times. If action_mix / entropy flips between two
levels with identical inputs, the bimodality is solver-state / non-determinism
(CUDA-graph capture-replay, buffer reuse), not the data distribution.

A second mode (--cycle) re-roots like the data generator (feeds each solve's
sampled leaves back as the next roots) to see whether a *beat* against the
re-rooting cadence produces the two levels.

Loads the real checkpoint weights and matching architecture, so the policy is
the trained one. Uses the same `pause_train_rebel` mechanism as the bench
scripts to avoid contending with a live training run.

Example:
    .venv/bin/python scripts/probe_action_mix_bimodal.py \
        --checkpoint checkpoints-rebel/rebel_latest.pt \
        --spots outputs/spots.pt --per-street 256 --repeats 24 \
        --iterations 610 --pause-train
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from p2.cli.sample_spots import build_pbs_from_spots, load_spots
from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
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
            cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(
                errors="replace"
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


def _select_even_spots(payload, per_street: int, seed: int) -> torch.Tensor:
    chunks = []
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    streets = payload["spots"]["street"]
    for sid in STREET_TO_ID.values():
        cand = torch.where(streets == sid)[0].cpu()
        if cand.numel() < per_street:
            raise ValueError(f"street {sid}: need {per_street}, have {cand.numel()}")
        cand = cand[torch.randperm(cand.numel(), generator=g)]
        chunks.append(cand[:per_street])
    return torch.cat(chunks, dim=0)


def _load_cfg(hydra_overrides: list[str]) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_cfr", overrides=hydra_overrides
        )
    return load_rebel_config(dc)


def _set_schedule(ev, iterations: int) -> None:
    """Mirror RebelCFRTrainer._update_schedules' derived CFR schedule."""
    warm = max(1, iterations // 20)
    delay = int(round(iterations * 0.4))
    iterations = max(warm + 1, iterations)
    ev.cfr_iterations = iterations
    ev.warm_start_iterations = warm
    ev.dcfr_delay = delay


@torch.no_grad()
def _preflop_node_fraction(ev) -> float:
    """Fraction of non-leaf nodes whose root state is preflop (street 0)."""
    N = ev.root_nodes
    total = ev.total_nodes
    # Propagate each node's root-street down the tree. Parents always have a
    # lower index than children, so a single forward pass suffices.
    root_street = torch.empty(total, dtype=torch.long, device=ev.device)
    root_street[:N] = ev.env.street[:N]
    parent = ev.parent_index
    offs = ev.depth_offsets
    # Process depth by depth so each level's parents are already filled.
    for d in range(1, len(offs) - 1):
        lo, hi = offs[d], offs[d + 1]
        if hi > lo:
            root_street[lo:hi] = root_street[parent[lo:hi]]
    top = ev.depth_offsets[-2]
    nonleaf = ~ev.leaf_mask[:top]
    if nonleaf.sum() == 0:
        return float("nan")
    return float((root_street[:top][nonleaf] == 0).float().mean().item())


@torch.no_grad()
def _avg_policy_entropy(ev) -> float:
    top = ev.depth_offsets[-2]
    p = ev.policy_probs_avg[:top].clamp_min(1e-8)
    ent = -(p * p.log()).sum(dim=-1)
    nonleaf = ~ev.leaf_mask[:top]
    if nonleaf.sum() == 0:
        return float("nan")
    return float(ent[nonleaf].mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints-rebel/rebel_latest.pt")
    ap.add_argument("--spots", default="outputs/spots.pt")
    ap.add_argument("--per-street", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=24)
    ap.add_argument("--iterations", type=int, default=610)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cycle",
        action="store_true",
        help="re-root from each solve's sampled leaves (mimic the generator)",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="sweep preflop/river root ratio on fixed spots (saturation test)",
    )
    ap.add_argument("--total-roots", type=int, default=256)
    ap.add_argument("--pause-train", action="store_true")
    ap.add_argument("--pause-pattern", default="train_rebel")
    ap.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="hydra overrides; must match the run's model arch for the ckpt to load",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = _load_cfg(args.hydra_override)
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_envs = args.per_street * 4

    with pause_train_rebel(args.pause_train, args.pause_pattern):
        trainer = RebelCFRTrainer(cfg=cfg, device=device)
        # Load just the model weights (optimizer/EMA state are not needed and the
        # muon optimizer state isn't load-compatible here).
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        save_dtype_str = ckpt.get("save_dtype")
        model_state = ckpt["model"]
        if save_dtype_str is not None and save_dtype_str != str(trainer.float_dtype):
            model_state = {
                k: v.to(trainer.float_dtype) if v.dtype.is_floating_point else v
                for k, v in model_state.items()
            }
        # Prefer EMA weights if present: the data generator solves under
        # _eval_swap (EMA), so the averaged params best match the live metrics.
        if "model_avg" in ckpt and ckpt["model_avg"]:
            avg_state = ckpt["model_avg"]
            if save_dtype_str is not None and save_dtype_str != str(trainer.float_dtype):
                avg_state = {
                    k: v.to(trainer.float_dtype) if v.dtype.is_floating_point else v
                    for k, v in avg_state.items()
                }
            merged = dict(model_state)
            merged.update(avg_state)
            model_state = merged
            print("Using EMA (model_avg) weights where available", flush=True)
        trainer.model.load_state_dict(model_state, strict=cfg.strict_model_loading)
        print(f"Loaded checkpoint at step {ckpt.get('step', '?')}", flush=True)
        ev = trainer.cfr_evaluator
        ev.model.eval()

        payload = load_spots(args.spots, map_location="cpu")
        idx = _select_even_spots(payload, args.per_street, args.seed)
        pbs = build_pbs_from_spots(payload, device, indices=idx)
        _set_schedule(ev, args.iterations)
        print(
            f"schedule: cfr_iterations={ev.cfr_iterations} "
            f"warm_start={ev.warm_start_iterations} dcfr_delay={ev.dcfr_delay}",
            flush=True,
        )

        if args.sweep:
            streets = payload["spots"]["street"]
            g = torch.Generator(device="cpu"); g.manual_seed(args.seed)
            pre_pool = torch.where(streets == 0)[0]
            riv_pool = torch.where(streets == 3)[0]
            pre_pool = pre_pool[torch.randperm(pre_pool.numel(), generator=g)]
            riv_pool = riv_pool[torch.randperm(riv_pool.numel(), generator=g)]
            T = args.total_roots
            ratios = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]
            print(f"\n{'pre_frac':>9} {'n_pre':>6} {'allin':>8} {'fold':>8} "
                  f"{'bet':>8} {'pre_node_frac':>14} {'nodes/root':>11}")
            for r in ratios:
                n_pre = int(round(r * T))
                n_riv = T - n_pre
                if n_pre > pre_pool.numel() or n_riv > riv_pool.numel():
                    continue
                idx_mix = torch.cat([pre_pool[:n_pre], riv_pool[:n_riv]])
                sub = build_pbs_from_spots(payload, device, indices=idx_mix)
                with torch.no_grad():
                    src = torch.arange(sub.env.N, device=device)
                    ev.initialize_subgame(sub.env, src, sub.beliefs)
                    ev.evaluate_cfr(training_mode=True)
                am = ev.stats.get("action_mix", {})
                nodes_per_root = ev.total_nodes / max(1, ev.root_nodes)
                print(
                    f"{r:>9.2f} {n_pre:>6d} {am.get('allin', float('nan')):>8.4f} "
                    f"{am.get('fold', float('nan')):>8.4f} {am.get('bet', float('nan')):>8.4f} "
                    f"{_preflop_node_fraction(ev):>14.4f} {nodes_per_root:>11.1f}",
                    flush=True,
                )
            return

        cur = pbs
        print(f"\n{'k':>3} {'street_mean':>11} {'allin':>8} {'fold':>8} "
              f"{'bet':>8} {'avg_ent':>8} {'n_roots':>8}")
        for k in range(args.repeats):
            with torch.no_grad():
                src = torch.arange(cur.env.N, device=device)
                ev.initialize_subgame(cur.env, src, cur.beliefs)
                nxt = ev.evaluate_cfr(training_mode=True)
            am = ev.stats.get("action_mix", {})
            street_mean = float(ev.env.street[: ev.root_nodes].float().mean().item())
            print(
                f"{k:>3} {street_mean:>11.4f} {am.get('allin', float('nan')):>8.4f} "
                f"{am.get('fold', float('nan')):>8.4f} {am.get('bet', float('nan')):>8.4f} "
                f"{_avg_policy_entropy(ev):>8.4f} {ev.root_nodes:>8d}",
                flush=True,
            )
            if args.cycle and nxt is not None and nxt.env.N > 0:
                cur = nxt  # re-root like the generator
            # else: same fixed spots every repeat


if __name__ == "__main__":
    main()
