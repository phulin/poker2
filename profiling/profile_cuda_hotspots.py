"""CUDA hotspot profiling using torch.profiler.

This script runs a short self-play training loop under the PyTorch profiler
and exports a Chrome trace / TensorBoard log so that CUDA hotspots can be
inspected. The configuration mirrors the lightweight setup used by
``profile_transformer_forward.py`` but targets CUDA execution.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pathlib
import sys

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaholdem.rl.self_play import SelfPlayTrainer
from profiling.profile_transformer_forward import build_profiling_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CUDA hotspots during a PPO training step."
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=1,
        help="Profiler wait steps before warmup (not recorded).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Profiler warmup steps (not recorded, stabilizes metrics).",
    )
    parser.add_argument(
        "--active",
        type=int,
        default=4,
        help="Profiler active steps (recorded).",
    )
    parser.add_argument(
        "--logdir",
        type=pathlib.Path,
        default=pathlib.Path("profiler_logs"),
        help="Directory where profiler traces are stored.",
    )
    parser.add_argument(
        "--trace-prefix",
        type=str,
        default="cuda_hotspots",
        help="Prefix for the generated trace directory.",
    )
    parser.add_argument(
        "--include-cpu",
        action="store_true",
        help="Record CPU activities alongside CUDA kernels.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index to profile (default: 0).",
    )
    args = parser.parse_args()
    if args.wait < 0 or args.warmup < 0 or args.active <= 0:
        raise ValueError("wait >= 0, warmup >= 0, and active > 0 are required.")
    return args


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device not available. Please run on a GPU-enabled machine."
        )

    device = torch.device("cuda", args.device_index)
    torch.cuda.set_device(device)

    cfg = build_profiling_config(device=device)
    trainer = SelfPlayTrainer(cfg=cfg, device=device)

    # Prime the replay buffer so the profiled steps focus on the update pass.
    trainer._fill_replay_buffer(trainer.batch_size)

    logdir = (
        args.logdir
        / f"{args.trace_prefix}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    logdir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CUDA]
    if args.include_cpu:
        activities.append(ProfilerActivity.CPU)

    prof_schedule = schedule(
        wait=args.wait, warmup=args.warmup, active=args.active, repeat=1
    )

    total_steps = args.wait + args.warmup + args.active

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(str(logdir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(1, total_steps + 1):
            with record_function("train_step"):
                trainer.train_step(step)
            prof.step()

    torch.cuda.synchronize(device)
    print(
        f"Profiler trace written to {logdir}. Load in TensorBoard or Chrome trace viewer to inspect CUDA hotspots."
    )


if __name__ == "__main__":
    main()
