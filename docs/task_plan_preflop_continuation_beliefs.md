# Task Plan: Preflop Continuation Belief Cascade

## Goal
Generate 1024 action-depth 0-3 preflop roots, solve bucket by bucket through 4-7, 8-11, and 12-end, and save the propagated root beliefs/env states for distribution analysis.

## Phases
- [x] Phase 1: Inspect existing bucket solver and continuation sampling APIs
- [x] Phase 2: Add a reproducible cascade script
- [x] Phase 3: Run the 1024-root cascade
- [x] Phase 4: Verify saved payload and summarize output

## Key Questions
1. Can we reuse evaluator continuation sampling to pass leaf beliefs as next roots?
2. Which checkpoints should be used for each bucket solve?
3. Does the saved file include enough metadata/env state to reconstruct the states later?

## Decisions Made
- Use the packed `actions_0_3` public-state dataset for initial roots.
- Use the current `actions_4_7` specialist to solve 0-3 roots, `actions_8_11` to solve 4-7 roots, and `actions_12_15` to solve both 8-11 and 12-end roots.
- Save both root and sampled continuation PBS tensors for each stage.

## Errors Encountered
- `search.cfr_model_batch_size=None` in the checkpoint config caused execution-config construction to fail before solving. Treat unset as `0`, matching existing config semantics.
- The embedded `preflop_buckets` config does not contain W&B fields; use top-level `Config.wandb_project` and `Config.wandb_tags` in the script.
- The first completed run used the generic continuation fallback for later buckets because continuation depth bounds are clipped to tree depth; patch the script to sample continuation roots by absolute `actions_this_round` ranges.

## Status
**Complete** - corrected strict-boundary cascade saved to `outputs/preflop_continuation_beliefs/cascade_1024_actions0_3_to_12end_seed42_iter300.pt`.
