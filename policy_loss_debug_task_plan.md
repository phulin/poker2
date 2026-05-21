# Task Plan: Policy Loss Divergence Debug

## Goal
Identify why the latest W&B run shows policy loss divergence and produce a concrete diagnosis or next debugging step.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Inspect latest W&B run artifacts and metrics
- [x] Phase 3: Trace policy loss computation and config
- [x] Phase 4: Synthesize diagnosis and recommended fix

## Key Questions
1. Which run is the latest local W&B run, and which policy loss metric diverged?
2. Did other metrics change at the same time, such as value loss, gradient norm, entropy, advantage scale, or learning rate?
3. Is the divergence explained by code/config behavior or by data/search instability?

## Decisions Made
- Use task-specific plan and notes files to avoid overwriting existing root planning files.
- Create `policy_loss_debug_report.md` as the final on-disk report for this investigation.

## Errors Encountered

## Status
**Completed** - Diagnosis recorded and schedule/logging guardrail verified.
