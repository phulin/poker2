# Task Plan: Preflop Backward-Induction Shuffle/Validation Restart

## Goal
Add deterministic bucket dataset shuffling, cached solved validation sets, periodic validation logging, multi-epoch deep-bucket training, and restart the full backward-induction run.

## Phases
- [x] Phase 1: Inspect existing reader/logging/run process
- [x] Phase 2: Implement seeded shuffle and value-step logging
- [x] Phase 3: Validate the script with a small smoke check
- [x] Phase 4: Stop the active run and restart with shuffle enabled
- [x] Phase 5: Report the new run details
- [x] Phase 6: Add cached 10k-iteration validation and 12-15 multi-epoch training
- [x] Phase 7: Stop current run and restart with validation enabled

## Key Questions
1. Does shuffling need to avoid loading whole 5M-row buckets into memory?
2. Should W&B global step remain unchanged?

## Decisions Made
- Keep W&B `step=global_step` unchanged and add `{bucket}/value_step` as a metric, because changing W&B's step axis mid-experiment would make comparisons harder.
- Always shuffle datasets. Use `--seed` only to make the unconditional shuffle reproducible.
- Validation caches should be keyed by the cutoff checkpoint signature so resuming a run with the same cutoff model reuses the 10k solved set.

## Errors Encountered
- `ty check .` reports existing repo-wide diagnostics unrelated to this change; targeted `ty` on the touched scripts passes.

## Status
**Complete** - Validation-enabled run started with 4 epochs for `actions_12_15`.
