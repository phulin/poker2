# Task Plan: Fused Sparse CFR Optimization Audit

## Goal
Try the five proposed fused sparse evaluator optimizations under realistic 400-iteration, 40/80 SAPDCFR benchmarks and commit only clear winners.

## Phases
- [x] Phase 1: Establish benchmark harness and CUDA access
- [x] Phase 2: Baseline benchmark under constrained GPU memory
- [ ] Phase 3: Implement fastest viable candidate versions
- [ ] Phase 4: Benchmark each candidate against baseline
- [ ] Phase 5: Commit winners and write final report

## Key Questions
1. Which ideas can be implemented as real hot-path launch/memory reductions without invasive correctness risk?
2. Which candidates improve end-to-end iterator wall time, not only one component timer?
3. Which changes are already present in the codebase and should be treated as baseline?

## Decisions Made
- Benchmark shape: use balanced saved spots with `--per-street 64` because a paused training run holds about 37 GB of the A100.
- Benchmark schedule: keep `search.cfr_type=sapcfr`, `search.predictive_cfr_dcfr_hybrid=true`, `search.predictive_cfr_delay=40`, `search.warm_start_iterations=40`, and `search.dcfr_plus_delay=80`.
- Harness setup: skip warm-start initial exploitability in `scripts/bench_cfr_iterator_spots.py` setup because it is not part of the timed iterator and OOMed under the paused training process.
- Idea 3 direct-policy prototype: rejected as not a clear end-to-end wall-clock winner at the measured size.

## Errors Encountered
- 512-root benchmark OOMed during warm-start initial exploitability while another run used ~37 GB. Resolution: skip that diagnostic in the benchmark setup and reduce to 256 roots.
- A first direct-policy kernel version included padded child lanes in the policy denominator. Resolution: fixed for testing, then removed because end-to-end timing was not a clear win.
- CUDA was invisible without escalation. Resolution: run CUDA commands with escalated permissions.

## Status
**Paused after Phase 3 trial work** - Idea 3 was implemented in two forms and rejected as no clear wall-clock winner. Ideas 1, 2, and 5 require larger semantic rewrites to test fairly; idea 4 is already present in the SAPDCFR baseline path.

## Pass 2 Goal
Try the three new fused sparse evaluator ideas: direct leaf-belief scatter, concurrent leaf-value streams, and shared belief-stat prep; commit only clear end-to-end winners.

## Pass 2 Phases
- [x] Phase 1: Map current leaf-belief gather/copy, leaf-value producer, and stat-prep paths
- [x] Phase 2: Implement opt-in fastest viable experiments with correctness checks
- [x] Phase 3: Run realistic 400-iteration 40/80 SAPDCFR benchmarks
- [x] Phase 4: Commit only benchmark-proven winners, otherwise remove experiments

## Pass 2 Decisions Made
- Idea 5 from the previous pass, alternating updates, remains out of scope because the user clarified it is a semantic change.
- Direct showdown leaf-belief path is the pass-2 winner: final post-delay wall 13.194 ms/iter versus old best baseline 13.670 ms/iter.
- Concurrent leaf streams were removed because the confirmation run was noisy/slower.
- Shared regret-stat prep was removed because it failed CUDA graph replay parity.

## Pass 2 Status
**Completed** - Committed direct showdown leaf-belief optimization as `a5e64e0`.

## Pass 3 Goal
Try the full direct leaf scatter path for model leaves: scatter packed model beliefs from the down-sweep and keep it if it improves a component microbenchmark.

## Pass 3 Phases
- [x] Phase 1: Confirm current direct-showdown baseline and dirty-worktree boundaries
- [x] Phase 2: Implement opt-out model leaf scatter in the fused down-sweep
- [x] Phase 3: Run correctness tests
- [x] Phase 4: Benchmark scatter on/off with the same build
- [x] Phase 5: Keep and commit only if a relevant microbenchmark improves

## Pass 3 Results
- Correctness: four focused CUDA tests passed after the depth-mask version.
- Component A/B: `set_leaf_values` improved from 5.119 ms to 5.021 ms CUDA (-1.92%).
- Component A/B: isolated `cfr_iteration` improved from 7.574 ms to 7.537 ms CUDA (-0.50%).
- Tradeoff: `update_policy` was effectively flat/slightly slower (2.016 ms to 2.021 ms), and `reach_beliefs_avg_fused` was flat (0.7340 ms to 0.7345 ms).
- End-to-end 400-iteration wall was favorable in this run, but still treated as noisy: post-delay 14.348 ms/iter off versus 14.028 ms/iter on.

## Pass 3 Status
**Completed** - Committed direct model leaf scatter as `feac84a`.
