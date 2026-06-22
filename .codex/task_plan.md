# Task Plan: Training Config Refactor Implementation

## Goal
Implement the ReBeL training/config refactor plan in committed increments.

## Phases
- [x] Phase 1: Re-read current state and identify safe first slice
- [x] Phase 2: Add shared ReBeL config/runtime helpers with tests
- [x] Phase 3: Migrate main ReBeL and curriculum lifecycle onto helpers
- [x] Phase 4: Add Hydra-first preflop staged schema and package CLI
- [ ] Phase 5: Expand validation/tests, update AGENTS/docs, and audit completion

## Key Questions
1. How can the new runtime helpers be introduced without breaking existing Hydra configs?
2. Which old defensive/compatibility paths can be removed once typed contracts exist?
3. Where should preflop staged bucket config live so Hydra is the source of truth?
4. What tests prove the refactor is behavior-preserving while moving toward the target architecture?

## Decisions Made
- Start with shared runtime/config infrastructure and main ReBeL migration before the larger preflop conversion.
- Do not commit unrelated untracked `.codex` or local metadata files.
- Per user direction, prioritize a clean internally consistent repo state over backward compatibility or checkpoint-config migration.
- Prefer typed contracts over defensive `getattr`/`isinstance` in newly refactored paths.
- Moved preflop backward-induction implementation into `p2.stages` and routed the top-level script through the Hydra CLI.
- Replaced the preflop Hydra CLI's `argparse.Namespace` adapter with `PreflopBucketExecutionConfig`.
- Converted `scripts/evaluate_rebel_value_loss.py` to Hydra-first config and removed checkpoint-embedded config loading.
- Tightened postflop pregeneration feature-encoder metadata and value-only checkpoint loading around the trainer/BetterSplitFFN contract.

## Errors Encountered
- `tests/test_preflop_backward_induction_config.py` initially imported `scripts` as a package, but pytest did not expose that namespace. Switched the test to import the script by file path.

## Status
**Currently in Phase 5** - Running focused validation and auditing remaining ReBeL utility cleanup.
