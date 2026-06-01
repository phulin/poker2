#!/usr/bin/env python3
"""Benchmark exact CFR showdown runners after one realistic subgame build."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import hydra
import torch
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.fused_cfr_triton import (
    ShowdownActiveCompactGraphRunner,
    ShowdownActiveCardBlockGraphRunner,
    ShowdownActiveGraphRunner,
    ShowdownFullyCompactDirectGraphRunner,
    ShowdownFullyCompactDualSlotGraphRunner,
    ShowdownFullyCompactGraphRunner,
    ShowdownFullyCompactNoBOppBothHeroCumGraphRunner,
    ShowdownFullyCompactNoBOppCardBlockCumGraphRunner,
    ShowdownFullyCompactNoBOppCardBlockFastEVGraphRunner,
    ShowdownFullyCompactNoBOppFastEVGraphRunner,
    ShowdownFullyCompactNoBOppGraphRunner,
    ShowdownFullyCompactOffsetGraphRunner,
    ShowdownFullyCompactScatterDirectGraphRunner,
    ShowdownGraphRunner,
    precompute_showdown_active_extras,
    precompute_showdown_extras,
)
from p2.search.postflop_spot_sampler import sample_postflop_start_roots


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cfg(args: argparse.Namespace) -> Config:
    with hydra.initialize_config_dir(
        config_dir=str(REPO_ROOT / "conf"), version_base=None
    ):
        dc: DictConfig = hydra.compose(
            config_name="config_rebel_cfr",
            overrides=args.hydra_override,
        )
    cfg = Config.from_dict_config(dc)
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.num_envs = args.num_envs
    cfg.data.live_root_source = "random_river"
    cfg.search.depth = args.depth
    cfg.search.iterations = args.iterations
    cfg.search.iterations_final = args.iterations
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    return cfg


def _time_cuda_us(fn: Callable[[], torch.Tensor], reps: int) -> list[float]:
    start = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    return [start[i].elapsed_time(end[i]) * 1000.0 for i in range(reps)]


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    n = len(ordered)
    return {
        "min_us": ordered[0],
        "median_us": ordered[n // 2],
        "p95_us": ordered[min(n - 1, int(n * 0.95))],
        "max_us": ordered[-1],
        "mean_us": sum(ordered) / n,
        "total_ms": sum(ordered) / 1000.0,
    }


def _print_summary(label: str, row: dict[str, float]) -> None:
    print(
        f"{label:<28} mean={row['mean_us']:8.2f} us  "
        f"median={row['median_us']:8.2f} us  min={row['min_us']:8.2f} us  "
        f"p95={row['p95_us']:8.2f} us  total={row['total_ms']:8.2f} ms",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("hydra_override", nargs="*")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    cfg = _load_cfg(args)
    if cfg.seed is not None:
        torch.manual_seed(int(cfg.seed))

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    pbs = sample_postflop_start_roots(
        trainer.env,
        batch_size=args.num_envs,
        street=3,
        generator=trainer.rng,
    )
    root_indices = torch.arange(pbs.env.N, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.cfr_evaluator.initialize_subgame(pbs.env, root_indices, pbs.beliefs)
    torch.cuda.synchronize()
    build_ms = (time.perf_counter() - t0) * 1000.0

    ev = trainer.cfr_evaluator
    if ev.showdown_indices.numel() == 0:
        raise RuntimeError("Subgame has no river showdown leaves.")
    beliefs = ev.beliefs[ev.showdown_indices].contiguous()
    num_hands = beliefs.shape[-1]
    m = beliefs.shape[0]

    root_index = ev._get_root_index()
    showdown_roots = root_index[ev.showdown_indices]
    rank_indices, active_extra_indices = torch.unique(
        showdown_roots, sorted=True, return_inverse=True
    )
    active_extras = precompute_showdown_active_extras(
        ev.hand_rank_data,
        ev.env,
        rank_indices,
        scale_indices=ev.showdown_indices,
        extra_indices=active_extra_indices,
    )
    legacy_extras = precompute_showdown_extras(
        ev.hand_rank_data,
        ev.env,
        rank_indices,
        scale_indices=ev.showdown_indices,
        extra_indices=active_extra_indices,
    )
    production_runner = ev._showdown_graph_runner
    if production_runner is None:
        production_runner = ShowdownActiveCardBlockGraphRunner(
            extras=active_extras,
            M=m,
            NUM_HANDS=num_hands,
            device=device,
        )
    active_runner = ShowdownActiveGraphRunner(
        extras=active_extras,
        M=m,
        NUM_HANDS=num_hands,
        device=device,
    )
    active_nozero_runner = ShowdownActiveGraphRunner(
        extras=active_extras,
        M=m,
        NUM_HANDS=num_hands,
        device=device,
        zero_output=False,
    )
    active_compact_runner = ShowdownActiveCompactGraphRunner(
        extras=active_extras,
        M=m,
        NUM_HANDS=num_hands,
        device=device,
    )
    fully_compact_runner = ShowdownFullyCompactGraphRunner(
        extras=active_extras,
        M=m,
        device=device,
    )
    fully_compact_offset_runner = ShowdownFullyCompactOffsetGraphRunner(
        extras=active_extras,
        M=m,
        device=device,
    )
    fully_compact_nobopp_runner = ShowdownFullyCompactNoBOppGraphRunner(
        extras=active_extras,
        M=m,
        device=device,
    )
    fully_compact_nobopp_both_cum_runner = (
        ShowdownFullyCompactNoBOppBothHeroCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
        )
    )
    fully_compact_nobopp_card2_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            card_block=2,
        )
    )
    fully_compact_nobopp_card4_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            card_block=4,
        )
    )
    fully_compact_nobopp_card8_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            card_block=8,
        )
    )
    fully_compact_nobopp_card16_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            card_block=16,
        )
    )
    fully_compact_nobopp_card32_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            card_block=32,
        )
    )
    fully_compact_nobopp_card64_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            card_block=64,
        )
    )
    fully_compact_nobopp_card64_b128_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            block_k=128,
            card_block=64,
        )
    )
    fully_compact_nobopp_card64_b256_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            block_k=256,
            card_block=64,
        )
    )
    fully_compact_nobopp_card64_b256_w8_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            block_k=256,
            card_block=64,
            finish_num_warps=8,
        )
    )
    fully_compact_nobopp_card64_b256_w8_fast_ev_runner = (
        ShowdownFullyCompactNoBOppCardBlockFastEVGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            block_k=256,
            card_block=64,
            finish_num_warps=8,
        )
    )
    fully_compact_nobopp_card64_b512_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            block_k=512,
            card_block=64,
        )
    )
    fully_compact_nobopp_card64_b1024_runner = (
        ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
            extras=active_extras,
            M=m,
            device=device,
            block_k=1024,
            card_block=64,
        )
    )
    fully_compact_nobopp_fast_ev_runner = ShowdownFullyCompactNoBOppFastEVGraphRunner(
        extras=active_extras,
        M=m,
        device=device,
    )
    fully_compact_dualslot_runner = ShowdownFullyCompactDualSlotGraphRunner(
        extras=active_extras,
        M=m,
        device=device,
    )
    fully_compact_direct_runner = ShowdownFullyCompactDirectGraphRunner(
        extras=active_extras,
        M=m,
        device=device,
    )
    fully_compact_scatter_runner = ShowdownFullyCompactScatterDirectGraphRunner(
        extras=active_extras,
        M=m,
        device=device,
    )
    baseline_runner = ShowdownGraphRunner(
        extras=legacy_extras,
        M=m,
        NUM_HANDS=num_hands,
        device=device,
    )

    gather_idx = active_extras["sorted_indices"][active_extras["extra_indices"]]
    compact_beliefs = beliefs.gather(2, gather_idx[:, None, :].expand(-1, 2, -1))

    for _ in range(20):
        baseline_runner(beliefs)
        production_runner(beliefs)
        active_runner(beliefs)
        active_nozero_runner(beliefs)
        active_compact_runner(beliefs)
        fully_compact_runner(compact_beliefs)
        fully_compact_offset_runner(compact_beliefs)
        fully_compact_nobopp_runner(compact_beliefs)
        fully_compact_nobopp_both_cum_runner(compact_beliefs)
        fully_compact_nobopp_card2_runner(compact_beliefs)
        fully_compact_nobopp_card4_runner(compact_beliefs)
        fully_compact_nobopp_card8_runner(compact_beliefs)
        fully_compact_nobopp_card16_runner(compact_beliefs)
        fully_compact_nobopp_card32_runner(compact_beliefs)
        fully_compact_nobopp_card64_runner(compact_beliefs)
        fully_compact_nobopp_card64_b128_runner(compact_beliefs)
        fully_compact_nobopp_card64_b256_runner(compact_beliefs)
        fully_compact_nobopp_card64_b256_w8_runner(compact_beliefs)
        fully_compact_nobopp_card64_b256_w8_fast_ev_runner(compact_beliefs)
        fully_compact_nobopp_card64_b512_runner(compact_beliefs)
        fully_compact_nobopp_card64_b1024_runner(compact_beliefs)
        fully_compact_nobopp_fast_ev_runner(compact_beliefs)
        fully_compact_dualslot_runner(compact_beliefs)
        fully_compact_direct_runner(compact_beliefs)
        fully_compact_scatter_runner(compact_beliefs)
    torch.cuda.synchronize()

    ref = baseline_runner(beliefs).clone()
    production_got = production_runner(beliefs).clone()
    got = active_runner(beliefs).clone()
    compact_got = active_compact_runner(beliefs).clone()
    fully_compact_got = fully_compact_runner(compact_beliefs).clone()
    fully_compact_offset_got = fully_compact_offset_runner(compact_beliefs).clone()
    fully_compact_nobopp_got = fully_compact_nobopp_runner(compact_beliefs).clone()
    fully_compact_nobopp_both_cum_got = fully_compact_nobopp_both_cum_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card2_got = fully_compact_nobopp_card2_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card4_got = fully_compact_nobopp_card4_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card8_got = fully_compact_nobopp_card8_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card16_got = fully_compact_nobopp_card16_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card32_got = fully_compact_nobopp_card32_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card64_got = fully_compact_nobopp_card64_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card64_b128_got = fully_compact_nobopp_card64_b128_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card64_b256_got = fully_compact_nobopp_card64_b256_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card64_b256_w8_got = (
        fully_compact_nobopp_card64_b256_w8_runner(compact_beliefs).clone()
    )
    fully_compact_nobopp_card64_b256_w8_fast_ev_got = (
        fully_compact_nobopp_card64_b256_w8_fast_ev_runner(compact_beliefs).clone()
    )
    fully_compact_nobopp_card64_b512_got = fully_compact_nobopp_card64_b512_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_card64_b1024_got = fully_compact_nobopp_card64_b1024_runner(
        compact_beliefs
    ).clone()
    fully_compact_nobopp_fast_ev_got = fully_compact_nobopp_fast_ev_runner(
        compact_beliefs
    ).clone()
    fully_compact_dualslot_got = fully_compact_dualslot_runner(compact_beliefs).clone()
    fully_compact_direct_got = fully_compact_direct_runner(compact_beliefs).clone()
    fully_compact_scatter_got = fully_compact_scatter_runner(compact_beliefs).clone()
    torch.cuda.synchronize()
    ref_compact = ref.gather(2, gather_idx[:, None, :].expand(-1, 2, -1))
    production_max_abs_diff = float((ref - production_got).abs().max().item())
    production_mean_abs_diff = float((ref - production_got).abs().mean().item())
    max_abs_diff = float((ref - got).abs().max().item())
    mean_abs_diff = float((ref - got).abs().mean().item())
    compact_max_abs_diff = float((ref_compact - compact_got).abs().max().item())
    compact_mean_abs_diff = float((ref_compact - compact_got).abs().mean().item())
    fully_compact_max_abs_diff = float(
        (ref_compact - fully_compact_got).abs().max().item()
    )
    fully_compact_mean_abs_diff = float(
        (ref_compact - fully_compact_got).abs().mean().item()
    )
    fully_compact_offset_max_abs_diff = float(
        (ref_compact - fully_compact_offset_got).abs().max().item()
    )
    fully_compact_offset_mean_abs_diff = float(
        (ref_compact - fully_compact_offset_got).abs().mean().item()
    )
    fully_compact_nobopp_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_got).abs().max().item()
    )
    fully_compact_nobopp_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_got).abs().mean().item()
    )
    fully_compact_nobopp_both_cum_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_both_cum_got).abs().max().item()
    )
    fully_compact_nobopp_both_cum_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_both_cum_got).abs().mean().item()
    )
    fully_compact_nobopp_card2_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card2_got).abs().max().item()
    )
    fully_compact_nobopp_card2_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card2_got).abs().mean().item()
    )
    fully_compact_nobopp_card4_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card4_got).abs().max().item()
    )
    fully_compact_nobopp_card4_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card4_got).abs().mean().item()
    )
    fully_compact_nobopp_card8_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card8_got).abs().max().item()
    )
    fully_compact_nobopp_card8_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card8_got).abs().mean().item()
    )
    fully_compact_nobopp_card16_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card16_got).abs().max().item()
    )
    fully_compact_nobopp_card16_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card16_got).abs().mean().item()
    )
    fully_compact_nobopp_card32_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card32_got).abs().max().item()
    )
    fully_compact_nobopp_card32_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card32_got).abs().mean().item()
    )
    fully_compact_nobopp_card64_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_got).abs().max().item()
    )
    fully_compact_nobopp_card64_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_got).abs().mean().item()
    )
    fully_compact_nobopp_card64_b128_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b128_got).abs().max().item()
    )
    fully_compact_nobopp_card64_b128_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b128_got).abs().mean().item()
    )
    fully_compact_nobopp_card64_b256_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b256_got).abs().max().item()
    )
    fully_compact_nobopp_card64_b256_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b256_got).abs().mean().item()
    )
    fully_compact_nobopp_card64_b256_w8_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b256_w8_got).abs().max().item()
    )
    fully_compact_nobopp_card64_b256_w8_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b256_w8_got).abs().mean().item()
    )
    fully_compact_nobopp_card64_b256_w8_fast_ev_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b256_w8_fast_ev_got)
        .abs()
        .max()
        .item()
    )
    fully_compact_nobopp_card64_b256_w8_fast_ev_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b256_w8_fast_ev_got)
        .abs()
        .mean()
        .item()
    )
    fully_compact_nobopp_card64_b512_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b512_got).abs().max().item()
    )
    fully_compact_nobopp_card64_b512_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b512_got).abs().mean().item()
    )
    fully_compact_nobopp_card64_b1024_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b1024_got).abs().max().item()
    )
    fully_compact_nobopp_card64_b1024_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_card64_b1024_got).abs().mean().item()
    )
    fully_compact_nobopp_fast_ev_max_abs_diff = float(
        (ref_compact - fully_compact_nobopp_fast_ev_got).abs().max().item()
    )
    fully_compact_nobopp_fast_ev_mean_abs_diff = float(
        (ref_compact - fully_compact_nobopp_fast_ev_got).abs().mean().item()
    )
    fully_compact_dualslot_max_abs_diff = float(
        (ref_compact - fully_compact_dualslot_got).abs().max().item()
    )
    fully_compact_dualslot_mean_abs_diff = float(
        (ref_compact - fully_compact_dualslot_got).abs().mean().item()
    )
    fully_compact_direct_max_abs_diff = float(
        (ref_compact - fully_compact_direct_got).abs().max().item()
    )
    fully_compact_direct_mean_abs_diff = float(
        (ref_compact - fully_compact_direct_got).abs().mean().item()
    )
    fully_compact_scatter_max_abs_diff = float(
        (ref_compact - fully_compact_scatter_got).abs().max().item()
    )
    fully_compact_scatter_mean_abs_diff = float(
        (ref_compact - fully_compact_scatter_got).abs().mean().item()
    )

    baseline_samples = _time_cuda_us(lambda: baseline_runner(beliefs), args.reps)
    production_samples = _time_cuda_us(lambda: production_runner(beliefs), args.reps)
    active_samples = _time_cuda_us(lambda: active_runner(beliefs), args.reps)
    active_nozero_samples = _time_cuda_us(
        lambda: active_nozero_runner(beliefs), args.reps
    )
    active_compact_samples = _time_cuda_us(
        lambda: active_compact_runner(beliefs), args.reps
    )
    fully_compact_samples = _time_cuda_us(
        lambda: fully_compact_runner(compact_beliefs), args.reps
    )
    fully_compact_offset_samples = _time_cuda_us(
        lambda: fully_compact_offset_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_both_cum_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_both_cum_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card2_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card2_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card4_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card4_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card8_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card8_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card16_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card16_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card32_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card32_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card64_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card64_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card64_b128_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card64_b128_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card64_b256_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card64_b256_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card64_b256_w8_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card64_b256_w8_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card64_b256_w8_fast_ev_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card64_b256_w8_fast_ev_runner(compact_beliefs),
        args.reps,
    )
    fully_compact_nobopp_card64_b512_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card64_b512_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_card64_b1024_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_card64_b1024_runner(compact_beliefs), args.reps
    )
    fully_compact_nobopp_fast_ev_samples = _time_cuda_us(
        lambda: fully_compact_nobopp_fast_ev_runner(compact_beliefs), args.reps
    )
    fully_compact_dualslot_samples = _time_cuda_us(
        lambda: fully_compact_dualslot_runner(compact_beliefs), args.reps
    )
    fully_compact_direct_samples = _time_cuda_us(
        lambda: fully_compact_direct_runner(compact_beliefs), args.reps
    )
    fully_compact_scatter_samples = _time_cuda_us(
        lambda: fully_compact_scatter_runner(compact_beliefs), args.reps
    )
    baseline = _summary(baseline_samples)
    production = _summary(production_samples)
    active = _summary(active_samples)
    active_nozero = _summary(active_nozero_samples)
    active_compact = _summary(active_compact_samples)
    fully_compact = _summary(fully_compact_samples)
    fully_compact_offset = _summary(fully_compact_offset_samples)
    fully_compact_nobopp = _summary(fully_compact_nobopp_samples)
    fully_compact_nobopp_both_cum = _summary(fully_compact_nobopp_both_cum_samples)
    fully_compact_nobopp_card2 = _summary(fully_compact_nobopp_card2_samples)
    fully_compact_nobopp_card4 = _summary(fully_compact_nobopp_card4_samples)
    fully_compact_nobopp_card8 = _summary(fully_compact_nobopp_card8_samples)
    fully_compact_nobopp_card16 = _summary(fully_compact_nobopp_card16_samples)
    fully_compact_nobopp_card32 = _summary(fully_compact_nobopp_card32_samples)
    fully_compact_nobopp_card64 = _summary(fully_compact_nobopp_card64_samples)
    fully_compact_nobopp_card64_b128 = _summary(
        fully_compact_nobopp_card64_b128_samples
    )
    fully_compact_nobopp_card64_b256 = _summary(
        fully_compact_nobopp_card64_b256_samples
    )
    fully_compact_nobopp_card64_b256_w8 = _summary(
        fully_compact_nobopp_card64_b256_w8_samples
    )
    fully_compact_nobopp_card64_b256_w8_fast_ev = _summary(
        fully_compact_nobopp_card64_b256_w8_fast_ev_samples
    )
    fully_compact_nobopp_card64_b512 = _summary(
        fully_compact_nobopp_card64_b512_samples
    )
    fully_compact_nobopp_card64_b1024 = _summary(
        fully_compact_nobopp_card64_b1024_samples
    )
    fully_compact_nobopp_fast_ev = _summary(fully_compact_nobopp_fast_ev_samples)
    fully_compact_dualslot = _summary(fully_compact_dualslot_samples)
    fully_compact_direct = _summary(fully_compact_direct_samples)
    fully_compact_scatter = _summary(fully_compact_scatter_samples)
    speedup = baseline["mean_us"] / active["mean_us"]
    speedup_production = baseline["mean_us"] / production["mean_us"]
    speedup_nozero = baseline["mean_us"] / active_nozero["mean_us"]
    speedup_compact = baseline["mean_us"] / active_compact["mean_us"]
    speedup_fully_compact = baseline["mean_us"] / fully_compact["mean_us"]
    speedup_fully_compact_offset = baseline["mean_us"] / fully_compact_offset["mean_us"]
    speedup_fully_compact_nobopp = baseline["mean_us"] / fully_compact_nobopp["mean_us"]
    speedup_fully_compact_nobopp_both_cum = (
        baseline["mean_us"] / fully_compact_nobopp_both_cum["mean_us"]
    )
    speedup_fully_compact_nobopp_card2 = (
        baseline["mean_us"] / fully_compact_nobopp_card2["mean_us"]
    )
    speedup_fully_compact_nobopp_card4 = (
        baseline["mean_us"] / fully_compact_nobopp_card4["mean_us"]
    )
    speedup_fully_compact_nobopp_card8 = (
        baseline["mean_us"] / fully_compact_nobopp_card8["mean_us"]
    )
    speedup_fully_compact_nobopp_card16 = (
        baseline["mean_us"] / fully_compact_nobopp_card16["mean_us"]
    )
    speedup_fully_compact_nobopp_card32 = (
        baseline["mean_us"] / fully_compact_nobopp_card32["mean_us"]
    )
    speedup_fully_compact_nobopp_card64 = (
        baseline["mean_us"] / fully_compact_nobopp_card64["mean_us"]
    )
    speedup_fully_compact_nobopp_card64_b128 = (
        baseline["mean_us"] / fully_compact_nobopp_card64_b128["mean_us"]
    )
    speedup_fully_compact_nobopp_card64_b256 = (
        baseline["mean_us"] / fully_compact_nobopp_card64_b256["mean_us"]
    )
    speedup_fully_compact_nobopp_card64_b256_w8 = (
        baseline["mean_us"] / fully_compact_nobopp_card64_b256_w8["mean_us"]
    )
    speedup_fully_compact_nobopp_card64_b256_w8_fast_ev = (
        baseline["mean_us"] / fully_compact_nobopp_card64_b256_w8_fast_ev["mean_us"]
    )
    speedup_fully_compact_nobopp_card64_b512 = (
        baseline["mean_us"] / fully_compact_nobopp_card64_b512["mean_us"]
    )
    speedup_fully_compact_nobopp_card64_b1024 = (
        baseline["mean_us"] / fully_compact_nobopp_card64_b1024["mean_us"]
    )
    speedup_fully_compact_nobopp_fast_ev = (
        baseline["mean_us"] / fully_compact_nobopp_fast_ev["mean_us"]
    )
    speedup_fully_compact_dualslot = (
        baseline["mean_us"] / fully_compact_dualslot["mean_us"]
    )
    speedup_fully_compact_direct = baseline["mean_us"] / fully_compact_direct["mean_us"]
    speedup_fully_compact_scatter = (
        baseline["mean_us"] / fully_compact_scatter["mean_us"]
    )

    print(
        json.dumps(
            {
                "num_envs": args.num_envs,
                "depth": args.depth,
                "iterations": args.iterations,
                "reps": args.reps,
                "build_ms": build_ms,
                "showdown_rows": int(m),
                "unique_boards": int(rank_indices.numel()),
                "num_hands": int(num_hands),
                "active_hands": int(active_extras["sorted_indices"].shape[1]),
                "active_cards": int(active_extras["card_positions"].shape[1]),
                "production_max_abs_diff": production_max_abs_diff,
                "production_mean_abs_diff": production_mean_abs_diff,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "compact_max_abs_diff": compact_max_abs_diff,
                "compact_mean_abs_diff": compact_mean_abs_diff,
                "fully_compact_max_abs_diff": fully_compact_max_abs_diff,
                "fully_compact_mean_abs_diff": fully_compact_mean_abs_diff,
                "fully_compact_offset_max_abs_diff": fully_compact_offset_max_abs_diff,
                "fully_compact_offset_mean_abs_diff": (
                    fully_compact_offset_mean_abs_diff
                ),
                "fully_compact_nobopp_max_abs_diff": fully_compact_nobopp_max_abs_diff,
                "fully_compact_nobopp_mean_abs_diff": (
                    fully_compact_nobopp_mean_abs_diff
                ),
                "fully_compact_nobopp_both_cum_max_abs_diff": (
                    fully_compact_nobopp_both_cum_max_abs_diff
                ),
                "fully_compact_nobopp_both_cum_mean_abs_diff": (
                    fully_compact_nobopp_both_cum_mean_abs_diff
                ),
                "fully_compact_nobopp_card2_max_abs_diff": (
                    fully_compact_nobopp_card2_max_abs_diff
                ),
                "fully_compact_nobopp_card2_mean_abs_diff": (
                    fully_compact_nobopp_card2_mean_abs_diff
                ),
                "fully_compact_nobopp_card4_max_abs_diff": (
                    fully_compact_nobopp_card4_max_abs_diff
                ),
                "fully_compact_nobopp_card4_mean_abs_diff": (
                    fully_compact_nobopp_card4_mean_abs_diff
                ),
                "fully_compact_nobopp_card8_max_abs_diff": (
                    fully_compact_nobopp_card8_max_abs_diff
                ),
                "fully_compact_nobopp_card8_mean_abs_diff": (
                    fully_compact_nobopp_card8_mean_abs_diff
                ),
                "fully_compact_nobopp_card16_max_abs_diff": (
                    fully_compact_nobopp_card16_max_abs_diff
                ),
                "fully_compact_nobopp_card16_mean_abs_diff": (
                    fully_compact_nobopp_card16_mean_abs_diff
                ),
                "fully_compact_nobopp_card32_max_abs_diff": (
                    fully_compact_nobopp_card32_max_abs_diff
                ),
                "fully_compact_nobopp_card32_mean_abs_diff": (
                    fully_compact_nobopp_card32_mean_abs_diff
                ),
                "fully_compact_nobopp_card64_max_abs_diff": (
                    fully_compact_nobopp_card64_max_abs_diff
                ),
                "fully_compact_nobopp_card64_mean_abs_diff": (
                    fully_compact_nobopp_card64_mean_abs_diff
                ),
                "fully_compact_nobopp_card64_b128_max_abs_diff": (
                    fully_compact_nobopp_card64_b128_max_abs_diff
                ),
                "fully_compact_nobopp_card64_b128_mean_abs_diff": (
                    fully_compact_nobopp_card64_b128_mean_abs_diff
                ),
                "fully_compact_nobopp_card64_b256_max_abs_diff": (
                    fully_compact_nobopp_card64_b256_max_abs_diff
                ),
                "fully_compact_nobopp_card64_b256_mean_abs_diff": (
                    fully_compact_nobopp_card64_b256_mean_abs_diff
                ),
                "fully_compact_nobopp_card64_b256_w8_max_abs_diff": (
                    fully_compact_nobopp_card64_b256_w8_max_abs_diff
                ),
                "fully_compact_nobopp_card64_b256_w8_mean_abs_diff": (
                    fully_compact_nobopp_card64_b256_w8_mean_abs_diff
                ),
                "fully_compact_nobopp_card64_b256_w8_fast_ev_max_abs_diff": (
                    fully_compact_nobopp_card64_b256_w8_fast_ev_max_abs_diff
                ),
                "fully_compact_nobopp_card64_b256_w8_fast_ev_mean_abs_diff": (
                    fully_compact_nobopp_card64_b256_w8_fast_ev_mean_abs_diff
                ),
                "fully_compact_nobopp_card64_b512_max_abs_diff": (
                    fully_compact_nobopp_card64_b512_max_abs_diff
                ),
                "fully_compact_nobopp_card64_b512_mean_abs_diff": (
                    fully_compact_nobopp_card64_b512_mean_abs_diff
                ),
                "fully_compact_nobopp_card64_b1024_max_abs_diff": (
                    fully_compact_nobopp_card64_b1024_max_abs_diff
                ),
                "fully_compact_nobopp_card64_b1024_mean_abs_diff": (
                    fully_compact_nobopp_card64_b1024_mean_abs_diff
                ),
                "fully_compact_nobopp_fast_ev_max_abs_diff": (
                    fully_compact_nobopp_fast_ev_max_abs_diff
                ),
                "fully_compact_nobopp_fast_ev_mean_abs_diff": (
                    fully_compact_nobopp_fast_ev_mean_abs_diff
                ),
                "fully_compact_dualslot_max_abs_diff": (
                    fully_compact_dualslot_max_abs_diff
                ),
                "fully_compact_dualslot_mean_abs_diff": (
                    fully_compact_dualslot_mean_abs_diff
                ),
                "fully_compact_direct_max_abs_diff": fully_compact_direct_max_abs_diff,
                "fully_compact_direct_mean_abs_diff": (
                    fully_compact_direct_mean_abs_diff
                ),
                "fully_compact_scatter_max_abs_diff": (
                    fully_compact_scatter_max_abs_diff
                ),
                "fully_compact_scatter_mean_abs_diff": (
                    fully_compact_scatter_mean_abs_diff
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    _print_summary("production_graph", baseline)
    _print_summary("new_production_graph", production)
    _print_summary("active_exact_graph", active)
    _print_summary("active_nozero_graph", active_nozero)
    _print_summary("active_compact_graph", active_compact)
    _print_summary("fully_compact_graph", fully_compact)
    _print_summary("fully_offset_graph", fully_compact_offset)
    _print_summary("fully_nobopp_graph", fully_compact_nobopp)
    _print_summary("fully_both_cum_graph", fully_compact_nobopp_both_cum)
    _print_summary("fully_card2_graph", fully_compact_nobopp_card2)
    _print_summary("fully_card4_graph", fully_compact_nobopp_card4)
    _print_summary("fully_card8_graph", fully_compact_nobopp_card8)
    _print_summary("fully_card16_graph", fully_compact_nobopp_card16)
    _print_summary("fully_card32_graph", fully_compact_nobopp_card32)
    _print_summary("fully_card64_graph", fully_compact_nobopp_card64)
    _print_summary("fully_card64_b128_graph", fully_compact_nobopp_card64_b128)
    _print_summary("fully_card64_b256_graph", fully_compact_nobopp_card64_b256)
    _print_summary("fully_card64_b256_w8_graph", fully_compact_nobopp_card64_b256_w8)
    _print_summary(
        "fully_card64_fast_graph", fully_compact_nobopp_card64_b256_w8_fast_ev
    )
    _print_summary("fully_card64_b512_graph", fully_compact_nobopp_card64_b512)
    _print_summary("fully_card64_b1024_graph", fully_compact_nobopp_card64_b1024)
    _print_summary("fully_fast_ev_graph", fully_compact_nobopp_fast_ev)
    _print_summary("fully_dualslot_graph", fully_compact_dualslot)
    _print_summary("fully_direct_graph", fully_compact_direct)
    _print_summary("fully_scatter_graph", fully_compact_scatter)
    print(f"speedup_mean={speedup:.3f}x", flush=True)
    print(f"speedup_new_production_mean={speedup_production:.3f}x", flush=True)
    print(f"speedup_nozero_mean={speedup_nozero:.3f}x", flush=True)
    print(f"speedup_compact_mean={speedup_compact:.3f}x", flush=True)
    print(f"speedup_fully_compact_mean={speedup_fully_compact:.3f}x", flush=True)
    print(
        f"speedup_fully_offset_mean={speedup_fully_compact_offset:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_nobopp_mean={speedup_fully_compact_nobopp:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_both_cum_mean={speedup_fully_compact_nobopp_both_cum:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card2_mean={speedup_fully_compact_nobopp_card2:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card4_mean={speedup_fully_compact_nobopp_card4:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card8_mean={speedup_fully_compact_nobopp_card8:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card16_mean={speedup_fully_compact_nobopp_card16:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card32_mean={speedup_fully_compact_nobopp_card32:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card64_mean={speedup_fully_compact_nobopp_card64:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card64_b128_mean="
        f"{speedup_fully_compact_nobopp_card64_b128:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card64_b256_mean="
        f"{speedup_fully_compact_nobopp_card64_b256:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card64_b256_w8_mean="
        f"{speedup_fully_compact_nobopp_card64_b256_w8:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card64_fast_mean="
        f"{speedup_fully_compact_nobopp_card64_b256_w8_fast_ev:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card64_b512_mean="
        f"{speedup_fully_compact_nobopp_card64_b512:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_card64_b1024_mean="
        f"{speedup_fully_compact_nobopp_card64_b1024:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_fast_ev_mean={speedup_fully_compact_nobopp_fast_ev:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_dualslot_mean={speedup_fully_compact_dualslot:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_direct_mean={speedup_fully_compact_direct:.3f}x",
        flush=True,
    )
    print(
        f"speedup_fully_scatter_mean={speedup_fully_compact_scatter:.3f}x",
        flush=True,
    )

    if args.json_out is not None:
        payload = {
            "metadata": {
                "num_envs": args.num_envs,
                "depth": args.depth,
                "iterations": args.iterations,
                "reps": args.reps,
                "build_ms": build_ms,
                "showdown_rows": int(m),
                "unique_boards": int(rank_indices.numel()),
                "num_hands": int(num_hands),
                "active_hands": int(active_extras["sorted_indices"].shape[1]),
                "active_cards": int(active_extras["card_positions"].shape[1]),
                "production_max_abs_diff": production_max_abs_diff,
                "production_mean_abs_diff": production_mean_abs_diff,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "speedup_new_production_mean": speedup_production,
                "speedup_mean": speedup,
                "speedup_nozero_mean": speedup_nozero,
                "speedup_compact_mean": speedup_compact,
                "speedup_fully_compact_mean": speedup_fully_compact,
                "speedup_fully_offset_mean": speedup_fully_compact_offset,
                "speedup_fully_nobopp_mean": speedup_fully_compact_nobopp,
                "speedup_fully_both_cum_mean": speedup_fully_compact_nobopp_both_cum,
                "speedup_fully_card2_mean": speedup_fully_compact_nobopp_card2,
                "speedup_fully_card4_mean": speedup_fully_compact_nobopp_card4,
                "speedup_fully_card8_mean": speedup_fully_compact_nobopp_card8,
                "speedup_fully_card16_mean": speedup_fully_compact_nobopp_card16,
                "speedup_fully_card32_mean": speedup_fully_compact_nobopp_card32,
                "speedup_fully_card64_mean": speedup_fully_compact_nobopp_card64,
                "speedup_fully_card64_b128_mean": (
                    speedup_fully_compact_nobopp_card64_b128
                ),
                "speedup_fully_card64_b256_mean": (
                    speedup_fully_compact_nobopp_card64_b256
                ),
                "speedup_fully_card64_b256_w8_mean": (
                    speedup_fully_compact_nobopp_card64_b256_w8
                ),
                "speedup_fully_card64_fast_mean": (
                    speedup_fully_compact_nobopp_card64_b256_w8_fast_ev
                ),
                "speedup_fully_card64_b512_mean": (
                    speedup_fully_compact_nobopp_card64_b512
                ),
                "speedup_fully_card64_b1024_mean": (
                    speedup_fully_compact_nobopp_card64_b1024
                ),
                "speedup_fully_fast_ev_mean": speedup_fully_compact_nobopp_fast_ev,
                "speedup_fully_dualslot_mean": speedup_fully_compact_dualslot,
                "speedup_fully_direct_mean": speedup_fully_compact_direct,
                "speedup_fully_scatter_mean": speedup_fully_compact_scatter,
            },
            "production_graph": baseline,
            "new_production_graph": production,
            "active_exact_graph": active,
            "active_nozero_graph": active_nozero,
            "active_compact_graph": active_compact,
            "fully_compact_graph": fully_compact,
            "fully_offset_graph": fully_compact_offset,
            "fully_nobopp_graph": fully_compact_nobopp,
            "fully_both_cum_graph": fully_compact_nobopp_both_cum,
            "fully_card2_graph": fully_compact_nobopp_card2,
            "fully_card4_graph": fully_compact_nobopp_card4,
            "fully_card8_graph": fully_compact_nobopp_card8,
            "fully_card16_graph": fully_compact_nobopp_card16,
            "fully_card32_graph": fully_compact_nobopp_card32,
            "fully_card64_graph": fully_compact_nobopp_card64,
            "fully_card64_b128_graph": fully_compact_nobopp_card64_b128,
            "fully_card64_b256_graph": fully_compact_nobopp_card64_b256,
            "fully_card64_b256_w8_graph": fully_compact_nobopp_card64_b256_w8,
            "fully_card64_fast_graph": fully_compact_nobopp_card64_b256_w8_fast_ev,
            "fully_card64_b512_graph": fully_compact_nobopp_card64_b512,
            "fully_card64_b1024_graph": fully_compact_nobopp_card64_b1024,
            "fully_fast_ev_graph": fully_compact_nobopp_fast_ev,
            "fully_dualslot_graph": fully_compact_dualslot,
            "fully_direct_graph": fully_compact_direct,
            "fully_scatter_graph": fully_compact_scatter,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
