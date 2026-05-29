# Task Plan: Website CFR Benchmark Optimization

## Goal
Make the `website/` CFR benchmark faster while preserving CFR outputs up to floating-point noise/accumulation-order differences, using interleaved benchmarks to reduce timing noise.

## Phases
- [x] Phase 1: Establish current benchmark/parity baseline
- [x] Phase 2: Identify CFR hot paths and existing experimental switches
- [x] Phase 3: Implement and tune faster CFR/runtime variants
- [x] Phase 4: Verify parity and interleaved benchmark speedup
- [x] Phase 5: Update AGENTS/docs if source layout or benchmark behavior changes

## Key Questions
1. Which benchmark spots and iteration/depth settings best expose the production CFR bottleneck?
2. Which existing runtime flags already select experimental CFR variants?
3. What output comparison gives strong evidence that the optimized path is unchanged except FP noise?

## Decisions Made
- Use `website/src/benchSpotsInterleaved.ts` for baseline-vs-candidate timing comparisons.
- Prefer production-compatible speedups, but allow high-risk experimental changes while tuning.
- Make command-buffer combining the default sparse CFR path; `P2_COMBINE_PREFIX_WITH_LEAF=0` restores the old path for A/B benchmarks.
- Remove the GPU-resident model warm-start experiment because it was faster but changed CFR parity beyond current tolerances.
- Cache prepared exact-belief phase-shift buffers by exact player/hand; `P2_CACHE_EXACT_PHASE_SHIFT=0` restores old rebuild behavior for A/B benchmarks.
- Skip BetterFFN board-interaction work for prepared empty-board batches; `P2_SKIP_EMPTY_BOARD_INTERACTION=0` restores the old explicit-zero path for A/B benchmarks.
- Release leaf prediction temporary buffers back to the BetterFFN buffer pool every 16 predictions; `P2_RELEASE_LEAF_TEMPS_EVERY=0` restores the old solve-end release behavior for A/B benchmarks.

## Errors Encountered
- `yarn exec tsx ...` consumed benchmark flags before `tsx`; use `./node_modules/.bin/tsx ...` for ad hoc runs.
- Sandbox blocked `tsx` IPC pipe creation under `/var/folders/.../T`; reran benchmark with approved escalation.
- Mixed street benchmark hit `production all-in table values require 1326 HUNL hands without permutations` on the flop spot, so current local assets/harness need either no-permutation states or all-in disablement for postflop spot timing.
- GPU-resident warm-start removed a CPU readback/writeback path but drifted CFR policy/action outputs too far versus the current default/PyTorch checks; removed from source.
- Batch-4 value-head subgroup selection microbenchmarked slightly faster but also moved CFR outputs outside tolerance, so it was reverted.
- Skipping unused average-buffer writes benchmarked slower, so that experiment was not kept.
- Two-input belief-shift add and model-uniform caching were both output-identical but slower, so both were reverted.
- Caching the belief-shift zero addend was output-identical but slower, so it was reverted.
- Batch-4 subgroup residual matvec selection was output-identical but slower, so it was reverted.
- Compute-pipeline bind-group-layout caching was output-identical but within timing noise, so it was reverted.
- Fusing exact-belief phase shifts into the BetterFFN half-norm path sped up root policy/action output but failed exact value fixture parity, so it was reverted.
- Sparse solve CPU scratch-array reuse was output-identical but within timing noise, so it was reverted.
- Exact-belief zero-sum value-head postprocess specialization was exact-value-safe but slower, so it was reverted.
- Exact-belief all-in table shader was output-identical but within timing noise, so it was reverted.
- Batch-3 subgroup value-head selection was output-identical and exact-value-safe but within timing noise, so it was reverted.
- Empty-board low-feature precompute skipping was output-identical but within timing noise, so it was reverted.

## Status
**Current pass complete; goal remains active** - Added chunked leaf temporary-buffer release for an additional output-identical speedup. Continue with more leaf/model kernel candidates next.
