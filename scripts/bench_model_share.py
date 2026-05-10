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

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer


NUM_WARMUP = 3
NUM_ACTIVE = 3
MODEL_TAG = "model_fwd"
STEP_TAG = "train_step"


def _wrap_models(*models: torch.nn.Module) -> None:
    """Patch every supplied model's class __call__ with a record_function tag.
    Pass both the trainer's model and the CFR evaluator's torch.compile-wrapped
    model so both forward paths get attributed."""
    seen_classes = set()
    for model in models:
        cls = model.__class__
        if cls in seen_classes:
            continue
        seen_classes.add(cls)
        orig_call = cls.__call__

        def make_wrapped(orig):
            def wrapped(self, *args, **kwargs):
                with record_function(MODEL_TAG):
                    return orig(self, *args, **kwargs)
            return wrapped

        cls.__call__ = make_wrapped(orig_call)


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    cfg = Config.from_dict_config(dict_config)
    cfg.use_wandb = False
    cfg.num_steps = max(cfg.num_steps, NUM_WARMUP + NUM_ACTIVE + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    _wrap_models(trainer.model, trainer.cfr_evaluator.model)

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
    model_cpu_us = 0.0
    model_cuda_us = 0.0
    for evt in prof.events():
        if evt.name == STEP_TAG:
            step_cpu_us += evt.cpu_time_total or 0.0
            step_cuda_us += evt.cuda_time_total or 0.0
        elif evt.name == MODEL_TAG:
            model_cpu_us += evt.cpu_time_total or 0.0
            model_cuda_us += evt.cuda_time_total or 0.0

    n = NUM_ACTIVE
    print()
    print(f"Per-step (avg over {n} iters):")
    print(f"  total CPU wall:    {step_cpu_us/n/1e3:8.2f} ms")
    print(f"  total CUDA time:   {step_cuda_us/n/1e3:8.2f} ms")
    print(f"  model fwd CPU:     {model_cpu_us/n/1e3:8.2f} ms")
    print(f"  model fwd CUDA:    {model_cuda_us/n/1e3:8.2f} ms")
    if step_cuda_us > 0:
        print(
            f"  model share of CUDA time: "
            f"{100 * model_cuda_us / step_cuda_us:.1f}%"
        )
    if step_cpu_us > 0:
        print(
            f"  model share of CPU time:  "
            f"{100 * model_cpu_us / step_cpu_us:.1f}%"
        )


if __name__ == "__main__":
    main()
