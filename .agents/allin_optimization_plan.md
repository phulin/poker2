# Task Plan: All-In 3P Precompute Optimization

## Goal
Find and implement a large algorithmic speedup for exact 3-player preflop all-in tensor generation, then verify, commit, and regenerate the quantized artifact if code changes.

## Phases
- [x] Phase 1: Establish current profile baseline
- [x] Phase 2: Remove duplicate host-side work-batch generation
- [ ] Phase 3: Explore ambitious algorithmic rewrites
- [ ] Phase 4: Implement the best viable rewrite
- [ ] Phase 5: Verify, benchmark, commit, and regenerate artifact

## Key Questions
1. Can the board x concrete-triple loop be replaced with per-board rank histograms or class-pair aggregation?
2. Can we compute all caller-pair results for a hero class with dense tensor algebra instead of one Triton launch per tuple chunk?
3. Which rewrite preserves exactness and fits GPU memory?

## Decisions Made
- Incremental host-side cleanup is not enough; focus on reducing the algorithmic operation count in the accumulation kernel.

## Status
**Currently in Phase 3** - deriving and testing a histogram/class-pair aggregation strategy.
