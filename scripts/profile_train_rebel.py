#!/usr/bin/env python3
"""Profile a few iterations of train_rebel's main loop with torch.profiler.

Skips the first iteration (torch.compile warmup), then profiles the next
few. Prints a top-ops summary to stdout and writes a Chrome trace.
"""

from __future__ import annotations

import os
import sys
import time

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


NUM_WARMUP = 3     # iterations to run before profiler starts (torch.compile warmup)
NUM_ACTIVE = 3     # iterations to actively record


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    cfg = load_rebel_config(dict_config)
    cfg.use_wandb = False
    # Just ensure the loop doesn't try anything fancy in our few iters
    cfg.num_steps = max(cfg.num_steps, NUM_WARMUP + NUM_ACTIVE + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    print("Trainer initialized.")

    trace_dir = os.path.abspath("profiler_trace")
    os.makedirs(trace_dir, exist_ok=True)

    # Run warmup outside the profiler context to avoid recording compile.
    for s in range(NUM_WARMUP):
        t0 = time.time()
        trainer.train_step(s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[warmup {s}] {time.time() - t0:.2f}s")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print(f"Profiling next {NUM_ACTIVE} iterations...")
    with profile(
        activities=activities,
        record_shapes=False,
        with_stack=True,
        profile_memory=False,
        experimental_config=torch._C._profiler._ExperimentalConfig(
            verbose=True,
        ),
    ) as prof:
        for s in range(NUM_WARMUP, NUM_WARMUP + NUM_ACTIVE):
            t0 = time.time()
            trainer.train_step(s)
            if device.type == "cuda":
                torch.cuda.synchronize()
            print(f"[profile {s}] {time.time() - t0:.2f}s")

    print("\n========== TOP CUDA OPS (self time) ==========")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25
    ))
    print("\n========== TOP CPU OPS (self time) ==========")
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=20
    ))

    # Attribute aten::index calls back to user code by walking the events.
    print("\n========== aten::index call sites (top user-code frames) ==========")
    from collections import Counter
    counts: Counter = Counter()
    times: dict[str, float] = {}
    for evt in prof.events():
        if evt.name != "aten::index":
            continue
        stack = list(getattr(evt, "stack", None) or [])
        # Walk top-of-stack down to find the deepest user-code frame.
        attribution = None
        for frame in stack:
            if "/p2/" in frame and "/.venv/" not in frame:
                attribution = frame
                break
        if attribution is None and stack:
            attribution = stack[0]
        if attribution is None:
            attribution = "<no stack>"
        counts[attribution] += 1
        times[attribution] = times.get(attribution, 0.0) + (
            (evt.cpu_time_total or 0.0) / 1e3
        )
    for frame, n in counts.most_common(20):
        print(f"  {n:>6}  {times.get(frame, 0):.1f} ms  {frame}")
    print(f"\nTrace written to: {trace_dir}")


if __name__ == "__main__":
    main()
