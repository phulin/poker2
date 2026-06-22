#!/usr/bin/env python3
"""Diagnose river CFR exploitability on the live (river-curriculum) distribution.

Samples N random_river spots from the CURRENT river-curriculum checkpoint's
model and compares, on the *same spots*:
  1. TRAINING PATH  -- the exact fused CUDA-graph `evaluate_cfr()` the trainer
     runs, reading back its logged `local_exploitability_mbbg`.
  2. EAGER PATH     -- the per-iteration eager loop, measuring the true
     average-strategy exploitability (leaf values recomputed under beliefs_avg).
Also reports raw (training-style) exploitability for the eager solve, and the
exact model-leaf / showdown-leaf counts so we can confirm whether river has any
model leaves at all.

  uv run python scripts/diagnose_river_convergence.py --num-spots 100 --iterations 300
"""
from __future__ import annotations
import argparse, time
import torch
from p2.config.rebel_load import load_rebel_config_file
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.postflop_spot_sampler import sample_river_start_roots


def load_weights(trainer: RebelCFRTrainer, checkpoint_path: str) -> None:
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.to(trainer.device)
    trainer._sync_inference_model()
    trainer.model.eval()


def measure_clean(ev):
    """True avg-strategy exploitability (recompute leaf values under beliefs_avg)."""
    sl, sv = ev.latest_values.clone(), ev.values_avg.clone()
    slm = ev.last_model_values.clone() if ev.last_model_values is not None else None
    ev.update_average_values_final()
    ev._exploitability_cache_key = None
    mbbg = ev._local_exploitability_mbbg(ev._compute_exploitability().local_exploitability)
    ev.latest_values, ev.values_avg, ev.last_model_values = sl, sv, slm
    ev._exploitability_cache_key = None
    return mbbg.detach().float().cpu()


def measure_raw(ev):
    """Training-style exploitability (raw running values_avg, no recompute)."""
    ev._exploitability_cache_key = None
    mbbg = ev._local_exploitability_mbbg(ev._compute_exploitability().local_exploitability)
    ev._exploitability_cache_key = None
    return mbbg.detach().float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="conf/config_rebel_cfr.yaml")
    ap.add_argument("--checkpoint", default="checkpoints-rebel-curriculum/river/rebel_latest.pt")
    ap.add_argument("--num-spots", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--mode", choices=["both", "train", "eager"], default="both",
                    help="train = fused graphed evaluate_cfr; eager = per-iter eager loop. "
                         "Run separately (different processes) to avoid cross-contamination.")
    ap.add_argument("--measure-every", type=int, default=0,
                    help="In eager mode, call measure_clean (discarded) every N iters to test "
                         "whether intermediate measurement mutates CFR state.")
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(f"checkpoint={args.checkpoint}  step={ck.get('step')}")
    cfg = load_rebel_config_file(args.config)
    cfg.resume_from = args.checkpoint
    cfg.num_envs = args.num_spots
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    cfg.search.iterations = args.iterations
    cfg.search.iterations_final = args.iterations
    cfg.model.compile = "off"
    cfg.trueskill.enabled = False
    cfg.train.replay_buffer_device = "cpu"
    cfg.train.replay_buffer_batches = 1
    print(f"live_root_source={cfg.data.live_root_source}  warm_start={cfg.search.warm_start_type}  "
          f"depth={cfg.search.depth}  iters={cfg.search.iterations}  delay={cfg.search.dcfr_plus_delay}")

    t0 = time.time()
    tr = RebelCFRTrainer(cfg=cfg, device=dev)
    load_weights(tr, args.checkpoint)
    ev = tr.cfr_evaluator
    print(f"trainer ready in {time.time()-t0:.0f}s")

    g = torch.Generator(device=dev).manual_seed(args.seed)
    pbs = sample_river_start_roots(tr.env, batch_size=args.num_spots, generator=g)
    src = torch.arange(args.num_spots, device=dev)

    with torch.no_grad():
        ev.initialize_subgame(pbs.env, src, pbs.beliefs)
        n_model = int(ev.model_indices.numel())
        n_leaf = int(ev.leaf_mask.sum())
        n_showdown = int(ev.showdown_indices.numel())
        print(f"\n--- leaf composition (river subgame) ---")
        print(f"total_nodes={ev.total_nodes}  leaves={n_leaf}  showdown_leaves={n_showdown}  "
              f"MODEL_LEAVES={n_model}")

        if args.mode in ("both", "train"):
            # ---- TRAINING PATH: exact fused evaluate_cfr (CUDA graphs) ----
            t1 = time.time()
            ev.evaluate_cfr(training_mode=False, sample_continuation=False)
            train_final = ev.stats.get("local_exploitability_mbbg")
            print(f"\n[TRAIN PATH] fused graphed evaluate_cfr in {time.time()-t1:.0f}s")
            print(f"   logged local_exploitability_mbbg (raw)  = {train_final:.2f}")
            clean_after_train = measure_clean(ev)
            print(f"   TRUE clean exploitability of result     = {clean_after_train.mean():.2f} "
                  f"(min {clean_after_train.min():.2f} max {clean_after_train.max():.2f})")

        if args.mode in ("both", "eager"):
            # ---- EAGER PATH: per-iteration eager loop ----
            ev.initialize_subgame(pbs.env, src, pbs.beliefs)
            ev.model.eval()
            ev.initialize_policy_and_beliefs()
            if ev.warm_start_iterations > 0:
                ev.warm_start()
            ev.set_leaf_values(0)
            ev.compute_expected_values()
            ev.values_avg[:] = ev.latest_values
            ev.t_sample = ev._get_sampling_schedule()
            ev._sample_leaf_enabled = False
            start, end = ev.warm_start_iterations, ev.cfr_iterations
            t2 = time.time()
            for t in range(start, end):
                ev.cfr_iteration(t)
                if args.measure_every and (t + 1) % args.measure_every == 0:
                    _ = measure_clean(ev)  # discard; tests for side effects
            eager_clean = measure_clean(ev)
            eager_raw = measure_raw(ev)
            print(f"\n[EAGER PATH] {end} eager iters in {time.time()-t2:.0f}s")
            print(f"   TRUE clean exploitability = {eager_clean.mean():.2f} "
                  f"(min {eager_clean.min():.2f} max {eager_clean.max():.2f})")
            print(f"   raw (training-style)      = {eager_raw.mean():.2f} "
                  f"(min {eager_raw.min():.2f} max {eager_raw.max():.2f})")


if __name__ == "__main__":
    main()
