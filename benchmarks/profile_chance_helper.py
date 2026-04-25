#!/usr/bin/env python3
"""Profile ChanceNodeHelper.flop_chance_values and single_card_chance_values."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from hydra import compose, initialize_config_dir
from torch.profiler import ProfilerActivity, profile, schedule

from p2.core.structured_config import Config
from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.cfr_trainer import RebelCFRTrainer


def build_trainer(overrides: list[str]) -> RebelCFRTrainer:
    conf_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "conf")
    )
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        dict_cfg = compose(config_name="config_rebel_cfr", overrides=overrides)
    cfg = Config.from_dict_config(dict_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch._dynamo.config.recompile_limit = 16
    torch.manual_seed(cfg.seed if cfg.seed is not None else 0)
    return RebelCFRTrainer(cfg=cfg, device=device)


def synth_inputs(
    helper, B: int, num_context: int, num_players: int, device, dtype
):
    context = torch.randn(B, num_context, device=device, dtype=dtype)
    street = torch.full((B,), 1, device=device, dtype=torch.long)
    to_act = torch.zeros(B, device=device, dtype=torch.long)
    board = torch.full((B, 5), -1, device=device, dtype=torch.long)
    beliefs_struct = torch.rand(B, num_players, NUM_HANDS, device=device, dtype=dtype)
    beliefs_struct = beliefs_struct / beliefs_struct.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    beliefs_flat = beliefs_struct.reshape(B, num_players * NUM_HANDS)
    feats = MLPFeatures(
        context=context, street=street, to_act=to_act, board=board, beliefs=beliefs_flat
    )
    pre_beliefs = beliefs_struct.clone()
    root_indices = torch.arange(B, device=device, dtype=torch.long)
    # board_pre: previous-street board, used by single_card; for "turn from flop" pretend no flop yet
    board_pre = torch.full((B, 5), -1, device=device, dtype=torch.long)
    return root_indices, feats, pre_beliefs, board_pre


def warmup(helper, root_indices, feats, pre_beliefs, board_pre, n: int):
    for _ in range(n):
        _ = helper.flop_chance_values(root_indices, feats, pre_beliefs)
        _ = helper.single_card_chance_values(
            root_indices, feats, pre_beliefs, board_pre
        )
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=192,
                    help="number of envs / boundary states to enumerate")
    ap.add_argument("--iters", type=int, default=3,
                    help="profiled iterations per fn")
    ap.add_argument("--trace-out", type=str,
                    default="debugging/chance_helper_trace.json")
    ap.add_argument("--no-compile", action="store_true",
                    help="disable model.compile to compare")
    args = ap.parse_args()

    overrides = ["num_envs=8", "model.nonlinearity=silu"]  # silu so BetterFFN construct works
    if args.no_compile:
        overrides.append("model.compile=false")
    trainer = build_trainer(overrides)
    helper = trainer.cfr_evaluator.chance_helper
    device = trainer.device
    dtype = trainer.float_dtype
    from p2.models.mlp.better_features import context_length
    num_context = context_length(trainer.num_players)
    print(f"num_context={num_context}", flush=True)
    num_players = trainer.num_players

    root_indices, feats, pre_beliefs, board_pre = synth_inputs(
        helper, args.B, num_context, num_players, device, dtype
    )

    print(f"Warmup (compile + cudnn autotune)...", flush=True)
    warmup(helper, root_indices, feats, pre_beliefs, board_pre, n=2)

    # ---- Standalone timed runs (event timing) ----
    def time_fn(fn, *a, n: int = 3):
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
        for i in range(n):
            starts[i].record()
            _ = fn(*a)
            ends[i].record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return ts

    flop_ts = time_fn(helper.flop_chance_values, root_indices, feats, pre_beliefs,
                      n=args.iters)
    sc_ts = time_fn(helper.single_card_chance_values, root_indices, feats,
                    pre_beliefs, board_pre, n=args.iters)
    print(f"\nB={args.B}, num_envs in helper unused")
    print(f"flop_chance_values        ms: {flop_ts}")
    print(f"  per-call avg: {sum(flop_ts)/len(flop_ts):.2f} ms")
    print(f"single_card_chance_values ms: {sc_ts}")
    print(f"  per-call avg: {sum(sc_ts)/len(sc_ts):.2f} ms")

    # ---- torch.profiler ----
    os.makedirs(os.path.dirname(args.trace_out), exist_ok=True)

    sched = schedule(wait=0, warmup=1, active=args.iters, repeat=1)

    def run_profile(name: str, fn, *a):
        out = args.trace_out.replace(".json", f"_{name}.json")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            record_shapes=True,
            with_stack=False,
            with_flops=False,
        ) as prof:
            for _ in range(args.iters + 1):
                _ = fn(*a)
                prof.step()
        torch.cuda.synchronize()
        prof.export_chrome_trace(out)
        print(f"\n=== {name} top ops by CUDA time ===")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20))
        print(f"\nChrome trace -> {out}")

    run_profile("flop", helper.flop_chance_values, root_indices, feats, pre_beliefs)
    run_profile("single_card", helper.single_card_chance_values,
                root_indices, feats, pre_beliefs, board_pre)


if __name__ == "__main__":
    main()
