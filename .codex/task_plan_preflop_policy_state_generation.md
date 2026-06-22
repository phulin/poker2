# Task Plan: Preflop Policy State Generation

## Goal
Generate a large compact dataset of multiplayer preflop public betting states sampled from a saved 6-player policy model.

## Phases
- [x] Phase 1: Confirm requested checkpoint availability
- [x] Phase 2: Inspect model/env rollout APIs
- [x] Phase 3: Implement compact policy-rollout generator
- [x] Phase 4: Smoke test and run generation
- [x] Phase 5: Report output and caveats

## Decisions Made
- Use `eroymcd2` because `v8yxyiya` has no local checkpoint or W&B checkpoint artifact.
- Store compact public-state tensors, not full environment objects.
- Roll out with belief-weighted policy sampling and Bayes-update the actor belief after each sampled action.

## Errors Encountered
- `v8yxyiya` checkpoint directory is empty locally and W&B files/artifacts do not include weights.

## Results
- Generated 3,000,000 rows into `outputs/preflop_policy_states/eroymcd2_policy_rollout_3m_20260621`.
- CUDA generation used eager bfloat16 policy inference, 65,536 envs, and 250,000-row shards.
- Validation streamed all rows: 0 non-preflop rows and 0 terminal rows.
- Added stratified frontier mode to target equal row counts in action-depth buckets.
- Generated 4,000,000 stratified rows into `outputs/preflop_policy_states/eroymcd2_policy_rollout_stratified_1m_buckets_20260621`.
- Validation streamed all stratified rows: every bucket has 1,000,000 rows, 0 non-preflop rows, 0 terminal rows, and 0 out-of-bucket action counts.

## Status
**Complete** - Plain and stratified datasets generated and validated.
