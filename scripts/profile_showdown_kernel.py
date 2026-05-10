#!/usr/bin/env python3
"""Kernel-only profile of the production showdown EV pipeline.

Times the v15 Triton path and the CUDA-graph (v16) wrapper that ship in
`FusedSparseCFREvaluator._showdown_value_both`. Substitutes for NCU on
boxes without it: uses `torch.cuda.Event` to isolate per-call GPU time
and `torch.profiler` to attribute work across the three pipeline kernels.
"""

from __future__ import annotations

import time

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.fused_cfr_triton import (
    SHOWDOWN_MAX_PER_CARD,
    ShowdownGraphRunner,
    precompute_showdown_extras,
    showdown_ev_v15,
)


REPS = 100


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    cfg = Config.from_dict_config(dict_config)
    cfg.use_wandb = False
    cfg.num_steps = max(cfg.num_steps, 5)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    print("Building trainer + getting subgame with river leaves...")
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    eval_ = trainer.cfr_evaluator
    for step in range(20):
        trainer._update_model(step)
        if eval_.showdown_indices.numel() > 0:
            break

    M = eval_.showdown_indices.numel()
    NUM_HANDS = eval_.beliefs.shape[-1]
    print(f"M={M} showdown rows, NUM_HANDS={NUM_HANDS}")

    beliefs = torch.rand(M, 2, NUM_HANDS, device=device, dtype=torch.float32)
    beliefs /= beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    extras = precompute_showdown_extras(
        eval_.hand_rank_data, eval_.env, eval_.showdown_indices,
    )
    runner = ShowdownGraphRunner(
        extras=extras, M=M, NUM_HANDS=NUM_HANDS, device=device,
    )

    # Warm up.
    for _ in range(20):
        showdown_ev_v15(beliefs, extras)
        runner(beliefs)
    torch.cuda.synchronize()

    # Per-call GPU timing via cuda events.
    def _time(fn) -> list[float]:
        events_s = [torch.cuda.Event(enable_timing=True) for _ in range(REPS)]
        events_e = [torch.cuda.Event(enable_timing=True) for _ in range(REPS)]
        for i in range(REPS):
            events_s[i].record()
            fn()
            events_e[i].record()
        torch.cuda.synchronize()
        return sorted(events_s[i].elapsed_time(events_e[i]) * 1e3 for i in range(REPS))

    def _summary(label: str, ts: list[float]) -> None:
        print(
            f"  {label:<40} min={ts[0]:7.2f}  median={ts[REPS // 2]:7.2f}  "
            f"p95={ts[int(REPS * 0.95)]:7.2f}  max={ts[-1]:8.2f}  us"
        )

    print("\nGPU-side timing (us/call):")
    _summary("PyTorch baseline (compiled)",
             _time(lambda: eval_._showdown_value_both.__wrapped__(eval_, beliefs)
                   if hasattr(eval_._showdown_value_both, "__wrapped__")
                   else eval_._showdown_value_both(beliefs)))
    _summary("Triton v15 (eager wrapper)",
             _time(lambda: showdown_ev_v15(beliefs, extras)))
    _summary("Triton v15 + CUDA graph",
             _time(lambda: runner(beliefs)))

    # Wall-clock end-to-end (Python wrapper time included).
    def _wall(fn) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e6 / REPS

    print("\nWall-clock per call (us, includes Python overhead):")
    print(f"  PyTorch baseline:     {_wall(lambda: eval_._showdown_value_both(beliefs)):.2f}")
    print(f"  Triton v15 (eager):   {_wall(lambda: showdown_ev_v15(beliefs, extras)):.2f}")
    print(f"  Triton v15 + graph:   {_wall(lambda: runner(beliefs)):.2f}")

    # Per-kernel breakdown for the production graph path.
    print("\nGraph-replay kernel breakdown (torch.profiler):")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        for _ in range(20):
            with record_function("graph_replay"):
                runner(beliefs)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))

    # Memory bandwidth estimate.
    cum_bytes = runner.cum_both.numel() * runner.cum_both.element_size()
    slot_bytes = sum(
        extras[k].numel() * extras[k].element_size()
        for k in (
            "slot_L_c1", "slot_L_c2", "slot_R_c1", "slot_R_c2",
            "slot_last_c1", "slot_last_c2",
        )
    )
    out_bytes = runner.ev_out.numel() * runner.ev_out.element_size()
    print(f"\nMain-kernel tensor sizes (per call, MB):")
    print(f"  card_cumsum (read):  {cum_bytes / 1e6:.2f}")
    print(f"  slot tensors (R):    {slot_bytes / 1e6:.2f}")
    print(f"  ev_out (W):          {out_bytes / 1e6:.2f}")
    min_bytes = cum_bytes + slot_bytes + out_bytes
    print(f"  Theoretical min @ 1.5 TB/s: {min_bytes / 1.5e6:.2f} us")
    _ = SHOWDOWN_MAX_PER_CARD  # silence unused import


if __name__ == "__main__":
    main()
