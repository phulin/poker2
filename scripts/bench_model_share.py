#!/usr/bin/env python3
"""Quick benchmark: what fraction of GPU time in train_step is spent in
the model forward vs everything else?

Wraps `trainer.model.__call__` with a torch.profiler record_function so the
model-forward share is attributable, and wraps each train_step in another
record_function. Reports total CUDA self-time inside model forwards vs
total CUDA self-time per train_step, averaged over the profiled iters.
"""

from __future__ import annotations

import os
import time

import hydra
import torch
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, profile, record_function

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


NUM_WARMUP = int(os.environ.get("P2_BENCH_WARMUP", "3"))
NUM_ACTIVE = int(os.environ.get("P2_BENCH_ACTIVE", "3"))
MODEL_TAGS = (
    "train_model_fwd",
    "eval_model_fwd",
    "closing_model_fwd",
)
STEP_TAG = "train_step"


def _wrap_model_forward(model: torch.nn.Module | None, tag: str) -> None:
    """Patch one module instance so separate model roles remain attributable."""
    if model is None or not isinstance(model, torch.nn.Module):
        return
    orig_forward = model.forward

    def wrapped(*args, **kwargs):
        with record_function(tag):
            return orig_forward(*args, **kwargs)

    model.forward = wrapped  # type: ignore[method-assign]


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    cfg = load_rebel_config(dict_config)
    cfg.use_wandb = False
    cfg.num_steps = max(cfg.num_steps, NUM_WARMUP + NUM_ACTIVE + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    _wrap_model_forward(trainer.model, "train_model_fwd")
    _wrap_model_forward(trainer.cfr_evaluator.model, "eval_model_fwd")
    _wrap_model_forward(
        getattr(trainer.cfr_evaluator, "closing_leaf_value_model", None),
        "closing_model_fwd",
    )

    print(f"Using device: {device}; warmup={NUM_WARMUP} active={NUM_ACTIVE}")

    for s in range(NUM_WARMUP):
        t0 = time.time()
        trainer.train_step(s)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[warmup {s}] {time.time() - t0:.2f}s")

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    print("Profiling...")
    with profile(activities=activities, record_shapes=False) as prof:
        for s in range(NUM_WARMUP, NUM_WARMUP + NUM_ACTIVE):
            with record_function(STEP_TAG):
                trainer.train_step(s)
                if device.type == "cuda":
                    torch.cuda.synchronize()

    # Sum CUDA self time inside each tagged region using event durations.
    # Walk events: find the per-step CPU events for STEP_TAG and MODEL_TAG;
    # for each, sum cuda_time_total over child kernels.
    step_cpu_us = 0.0
    step_cuda_us = 0.0
    model_cpu_us = {tag: 0.0 for tag in MODEL_TAGS}
    model_cuda_us = {tag: 0.0 for tag in MODEL_TAGS}
    for evt in prof.events():
        if evt.name == STEP_TAG:
            step_cpu_us += evt.cpu_time_total or 0.0
            step_cuda_us += evt.cuda_time_total or 0.0
        elif evt.name in model_cpu_us:
            model_cpu_us[evt.name] += evt.cpu_time_total or 0.0
            model_cuda_us[evt.name] += evt.cuda_time_total or 0.0

    n = NUM_ACTIVE
    print()
    print(f"Per-step (avg over {n} iters):")
    print(f"  total CPU wall:    {step_cpu_us/n/1e3:8.2f} ms")
    print(f"  total CUDA time:   {step_cuda_us/n/1e3:8.2f} ms")
    total_model_cpu_us = sum(model_cpu_us.values())
    total_model_cuda_us = sum(model_cuda_us.values())
    for tag in MODEL_TAGS:
        print(f"  {tag} CPU: {model_cpu_us[tag]/n/1e3:8.2f} ms")
        print(f"  {tag} CUDA:{model_cuda_us[tag]/n/1e3:8.2f} ms")
    print(f"  total model CPU:   {total_model_cpu_us/n/1e3:8.2f} ms")
    print(f"  total model CUDA:  {total_model_cuda_us/n/1e3:8.2f} ms")
    if step_cuda_us > 0:
        print(
            f"  model share of CUDA time: "
            f"{100 * total_model_cuda_us / step_cuda_us:.1f}%"
        )
    if step_cpu_us > 0:
        print(
            f"  model share of CPU time:  "
            f"{100 * total_model_cpu_us / step_cpu_us:.1f}%"
        )


if __name__ == "__main__":
    main()
