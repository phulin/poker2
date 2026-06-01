#!/usr/bin/env python3
"""Profile the compact no-bopp exact showdown pipeline by Triton kernel."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import hydra
import torch
import triton
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.fused_cfr_triton import (
    SHOWDOWN_MAX_PER_CARD,
    _showdown_build_cum_compact_both_heroes_kernel,
    _showdown_build_cum_compact_card_block_kernel,
    _showdown_build_cum_compact_kernel,
    _showdown_ev_active_offset_compact_nobopp_kernel,
    _showdown_setup_P_compact_kernel,
    precompute_showdown_active_extras,
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


def _time_cuda_us(fn, reps: int) -> list[float]:
    start = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    return [start[i].elapsed_time(end[i]) * 1000.0 for i in range(reps)]


def _summarize(name: str, samples: list[float]) -> None:
    samples = sorted(samples)
    n = len(samples)
    mean = sum(samples) / n
    print(
        f"{name:<12} mean={mean:8.2f} us  "
        f"median={samples[n // 2]:8.2f} us  "
        f"p95={samples[min(n - 1, int(n * 0.95))]:8.2f} us",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--card-block", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--finish-warps", type=int, default=8)
    parser.add_argument("hydra_override", nargs="*")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This profiler requires CUDA.")
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
    beliefs = ev.beliefs[ev.showdown_indices].contiguous()
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
    gather_idx = active_extras["sorted_indices"][active_extras["extra_indices"]]
    compact_beliefs = beliefs.gather(2, gather_idx[:, None, :].expand(-1, 2, -1))

    h = active_extras["sorted_indices"].shape[1]
    num_cards = active_extras["card_positions"].shape[1]
    block_h = triton.next_power_of_2(h)
    block_k = args.block_k
    p_padded = torch.empty(m, 2, h + 1, device=device, dtype=torch.float32)
    cum_both = torch.empty(
        m, 2, num_cards, SHOWDOWN_MAX_PER_CARD, device=device, dtype=torch.float32
    )
    ev_out = torch.empty(m, 2, h, device=device, dtype=torch.float32)
    extra_indices = active_extras["extra_indices"]

    def setup() -> None:
        _showdown_setup_P_compact_kernel[(m, 2)](
            compact_beliefs,
            p_padded,
            h,
            BLOCK_H=block_h,
        )

    def build_cum() -> None:
        _showdown_build_cum_compact_kernel[(m, 2 * num_cards)](
            compact_beliefs,
            extra_indices,
            active_extras["card_positions"],
            cum_both,
            h,
            NUM_CARDS=num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )

    def build_cum_both() -> None:
        _showdown_build_cum_compact_both_heroes_kernel[(m, num_cards)](
            compact_beliefs,
            extra_indices,
            active_extras["card_positions"],
            cum_both,
            h,
            NUM_CARDS=num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )

    def build_cum_card_block() -> None:
        _showdown_build_cum_compact_card_block_kernel[
            (m, triton.cdiv(num_cards, args.card_block))
        ](
            compact_beliefs,
            extra_indices,
            active_extras["card_positions"],
            cum_both,
            h,
            NUM_CARDS=num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            CARD_BLOCK=args.card_block,
        )

    def finish() -> None:
        _showdown_ev_active_offset_compact_nobopp_kernel[(m, triton.cdiv(h, block_k))](
            compact_beliefs,
            p_padded,
            cum_both,
            extra_indices,
            active_extras["L_idx"],
            active_extras["R_idx"],
            active_extras["scale_factor"],
            active_extras["off_L_c1"],
            active_extras["off_L_c2"],
            active_extras["off_R_c1"],
            active_extras["off_R_c2"],
            active_extras["off_last_c1"],
            active_extras["off_last_c2"],
            ev_out,
            h,
            NUM_CARDS=num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=block_k,
            num_warps=args.finish_warps,
        )

    def pipeline() -> None:
        setup()
        build_cum()
        finish()

    def pipeline_both() -> None:
        setup()
        build_cum_both()
        finish()

    def pipeline_card_block() -> None:
        setup()
        build_cum_card_block()
        finish()

    for _ in range(20):
        pipeline_card_block()
    torch.cuda.synchronize()

    print(
        f"build_ms={build_ms:.2f} showdown_rows={m} "
        f"unique_boards={rank_indices.numel()} active_hands={h}",
        flush=True,
    )
    _summarize("setup", _time_cuda_us(setup, args.reps))
    _summarize("build_cum", _time_cuda_us(build_cum, args.reps))
    _summarize("build_both", _time_cuda_us(build_cum_both, args.reps))
    _summarize("build_card", _time_cuda_us(build_cum_card_block, args.reps))
    _summarize("finish", _time_cuda_us(finish, args.reps))
    _summarize("pipeline", _time_cuda_us(pipeline, args.reps))
    _summarize("pipe_both", _time_cuda_us(pipeline_both, args.reps))
    _summarize("pipe_card", _time_cuda_us(pipeline_card_block, args.reps))

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
    ) as prof:
        for _ in range(10):
            pipeline_card_block()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))


if __name__ == "__main__":
    main()
