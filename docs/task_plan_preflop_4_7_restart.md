# Task Plan: Preflop 4-7 Restart Monitor

## Goal
When the current preflop BI run finishes actions_8_11 and starts actions_4_7, stop that inherited actions_4_7 process and restart actions_4_7 with the new Hydra LR settings while keeping the rest of the run configuration unchanged.

## Phases
- [x] Phase 1: Identify active run, logs, and command
- [ ] Phase 2: Monitor actions_8_11 until actions_4_7 starts
- [x] Phase 3: Stop inherited actions_4_7 run
- [x] Phase 4: Start replacement actions_4_7 run with updated Hydra config
- [x] Phase 5: Watch replacement run at 10 minute intervals for 1-2 hours
- [x] Phase 6: Report outcome

## Key Questions
1. What exact command/environment is the current run using?
2. What checkpoint should seed the replacement actions_4_7 run?
3. Does the replacement run pick up `learning_rate=0.006`, `learning_rate_final=0.0009`, `adamw_learning_rate=0.004`, WSD `0.6`?

## Decisions Made
- Keep the current actions_8_11 run alive until the log shows the transition to actions_4_7.
- Use the committed Hydra config defaults for the replacement run rather than command-line LR overrides unless the original launch requires explicit overrides for reproducibility.
- Replacement should run only `preflop_buckets.train_bucket=actions_4_7` and set `preflop_buckets.base_checkpoint` to the completed actions_8_11 `specialist_final.pt`.
- Keep the original run's throughput overrides unless intentionally changing LR: `preflop_buckets.train_batch_size=256` and `preflop_buckets.policy_train_batch_size=null`.

## Errors Encountered
- `.codex/tasks` could not be created because that path is read-only in this sandbox; task files are stored in the repo root instead.

## Status
**Complete** - Replacement actions_4_7 run is live, training, checkpointing, and has been watched for about 1h46 from launch.
