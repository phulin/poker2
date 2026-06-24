# Task Plan: Warm-Start Transformer Repro

## Goal
Rerun the `ifb0ifue` 12-15 warm-start transformer setup with the fixed supervised loss and the shared validation cache.

## Phases
- [x] Phase 1: Locate the old run configuration and checkpoint.
- [x] Phase 2: Confirm validation cache reuse and launch command.
- [x] Phase 3: Start the reproduction run and verify it is logging.
- [x] Phase 4: Report run id, output path, and caveats.

## Key Facts
- Reference run: W&B `ifb0ifue`.
- Reference checkpoint: `/home/user/poker2/checkpoints-rebel-curriculum-preflop_2000_p6_lr0p01_backupcons_actor_lam01_rb32_from2p_norb/preflop/rebel_latest.pt`.
- Shared validation cache: `outputs/preflop_backward_induction/validation_cache/actions_12_15/validation_n4096_cfr10000_0b306fcc1ac813c6.pt`.

## Status
**Complete** - run `bagzz3zs` is active in tmux and has reached the first training progress line.
