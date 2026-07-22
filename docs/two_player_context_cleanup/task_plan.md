# Task Plan: Two-Player Context Cleanup

## Goal
Track `last_aggressive_amount` in `HUNLTensorEnv` and remove constant or redundant two-player BetterFFN context fields without breaking tensor layouts or state propagation.

## Phases
- [x] Phase 1: Map environment state and feature-layout consumers
- [x] Phase 2: Define the compact two-player context schema and compatibility boundary
- [x] Phase 3: Implement environment propagation and encoder/model layout changes
- [x] Phase 4: Add focused regression tests
- [x] Phase 5: Run verification and write the implementation report
- [x] Phase 6: Commit the scoped changes and launch a fresh W&B run from step 0

## Key Questions
1. Which environment constructors, copies, gathers, resets, and action kernels must propagate `last_aggressive_amount`?
2. Which context fields can be removed without losing independent two-player information?
3. Are existing checkpoints expected to remain loadable after changing BetterFFN input dimensions?
4. How should the last run's checkpoint-embedded settings be launched without resume state?

## Decisions Made
- Remove fields only from two-player BetterFFN layouts; retain general multi-player schema definitions.
- Treat existing checkpoint shape compatibility as an explicit boundary rather than silently remapping weights.

## Errors Encountered
- A broader integration suite exposed eight pre-existing failures in compact
  preflop constructor keyword handling and bootstrap replay staging. The focused
  context/environment/model tests pass; none of those failures touch the changed
  heads-up schema or `last_aggressive_amount` paths.
- Two existing compact-value-loss tests fail with `NameError: output` inside
  `RebelSupervisedLoss._forward_compact_value`; this is unrelated to context size.
- The first fresh-run launch used resolved-config group paths for two Hydra
  overrides; `checkpoint_dir` is flat and the optional run name must be appended
  as `+wandb_name`. Corrected before trainer initialization, so no run or
  checkpoint was created.

## Status
**Complete** - Commit `dab97193`; fresh W&B run `ool717bi` completed step 0.
