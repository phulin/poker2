# Task Plan: Preflop Value Sweep Follow-Ups

## Goal
Run additional one-epoch value-training experiments on pregenerated `actions_12_15` data, including scheduler/AdamW variants and a 500-CFR target dataset, then identify the best validation loss.

## Phases
- [x] Phase 1: Cleanup and setup
- [x] Phase 2: Inspect optimizer/schedule implementation
- [x] Phase 3: Extend or reuse sweep harness for new experiment knobs
- [x] Phase 4: Generate 500-CFR presolved value dataset
- [x] Phase 5: Run schedule, AdamW, 500-CFR, and additional useful experiments
- [x] Phase 6: Summarize results and commit source changes

## Key Questions
1. Which scheduler names and AdamW/Muon fields does `RebelCFRTrainer` actually consume?
2. Does 500-CFR target generation improve fixed validation loss enough to justify extra solve cost?
3. Which extra low-cost experiments are most informative after the first LR/BS sweep?

## Decisions Made
- Store experiment bookkeeping in `.codex/` instead of adding planning artifacts to source directories.
- Treat the first sweep's `lr=0.01, bs=512` result as the incumbent and run focused follow-up sweeps around it.
- Compare schedules with `final_ratio=0.1`; compare WSD at two decay fractions (`0.2`, `0.5`).

## Errors Encountered
- None yet.

## Status
**Complete** - Experiments finished; source changes committed; report delivered in final response.
