# S_turn Structural Experiments Status

Started 2026-07-11 UTC.

## Active Holdout

Session: `sturn_holdout_300_current_20260711`

Output: `outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711`

Purpose: disjoint random-turn holdout with the same sampler, 300 CFR iterations, and promoted 300k `E_turn` used by the training dataset. Seed 777 is disjoint from training seed 42.

Log: `outputs/training_logs/turn_holdout_32768_300it_eturn300k_seed777_20260711.log`

## Paired And Structural Pipeline

Session: `sturn_structural_pipeline_20260711`

Launcher: `scripts/launch_sturn_structural_experiments_20260711.sh`

Log: `outputs/training_logs/sturn_structural_pipeline_20260711.log`

Stages:

1. Wait for the disjoint holdout manifest.
2. Smoke-test saved-root paired generation on 512 roots at 300 CFR iterations with two solve seeds.
3. Save 4,096 root PBS batches and solve identical roots at 300, 1,000, and 5,000 CFR iterations using solve seed 9001.
4. Re-solve the same roots at 300 iterations with solve seed 9002 to estimate solver-label noise.
5. Run 500-step value sweeps for blocker-corrected turn equity, turn-equity feature head, rank-64 cross-range interaction, second moment plus blockers, and direct pair-operator application.

Paired output: `outputs/rebel_postflop/paired_sturn_4096_300_1000_5000it_eturn300k_20260711`

Structural sweep output: `outputs/sturn_pregen_500step_structural_sweep_20260711`

All structural runs train on the original 1,024,000-example pregen epoch and validate only on the disjoint current-target holdout. In-sample matched validation is disabled.

## Deferred Reachable-PBS Experiment

The existing `self_play` pregeneration source produces mixed-street continuation states and is not a controlled reachable-turn sampler. A valid experiment requires rollouts with a complete frozen street-model registry and extraction of turn PBS roots. It is intentionally not mixed into the current queue.
