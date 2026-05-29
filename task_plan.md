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
- Skip unused GPU `beliefsAvgBuffer` uploads when `cfrAvg=false`; `P2_SKIP_UNUSED_BELIEFS_AVG_UPLOAD=0` restores the old upload path for A/B benchmarks.
- Pool sparse CFR uniform parameter buffers across dispatches; `P2_POOL_SPARSE_UNIFORMS=0` restores the old create/destroy-per-dispatch path for A/B benchmarks.

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
- Leaf-temp release chunks 8 and 24 both lost against the current default chunk 16.
- Root-only policy readback was output-identical but slower than the single full-policy readback, so it was reverted.
- Contiguous root-policy slice readback was also output-identical but slower than the single full-policy readback, so it was reverted.
- Skipping the explicit queue wait before readback `mapAsync` was output-identical but slower, so it was reverted.
- A 128-thread all-in table shader workgroup was output-identical but slower than the current 64-thread shader, so it was reverted.
- Reference aggregate/apply kernels for regret weights and opponent policies were slower and changed CFR accumulation too much, so they were reverted.
- Preferring batch-3 subgroup linear-in kernels was output-identical but slower than the current batch-4 selection, so it was reverted.
- Skipping warm-start-overwritten initial policy-average/regret uploads was output-identical but within timing noise, so it was reverted.
- GPU-copy zero initialization for policy-sized buffers was output-identical but within timing noise, so it was reverted.
- Skipping initial policy-average upload was output-identical but lost on longer confirmation, so it was reverted.
- Skipping all unused average-policy uploads when no average policy is read was output-identical but lost on longer confirmation, so it was reverted.
- Skipping unused reach-buffer upload when no average policy is read was output-identical but slower, so it was reverted.
- Copying cached fold-terminal values from a static GPU buffer into the solve values buffer was output-identical but within noise/slightly slower on confirmation, so it was reverted.
- Using the batch-4 subgroup matvec for the biased value-head projection was output-identical and exact-value-safe but slightly slower, so it was reverted.
- Extending the leaky 1024 batch-2 subgroup shader to hidden 512x1024 linear-out projections was output-identical and exact-value-safe but slower in the full CFR benchmark, so it was reverted.
- Leaf-temp release chunk 12 was output-identical but within noise against chunk 16, so the default remains 16.
- Leaf-temp release chunk 4 was output-identical but slower against chunk 16, so the default remains 16.
- Caching the gather-node-beliefs bind group was output-identical but slower, so it was reverted.
- A parallel-reduction all-in table shader was slower and changed CFR output too much due to different accumulation order, so it was reverted.
- A pre-unpacked f32 all-in table lookup path was output-identical but stayed within noise on confirmation, so it was reverted.

## Status
**Current pass complete; goal remains active** - Added chunked leaf temporary-buffer release for an additional output-identical speedup. Continue with more leaf/model kernel candidates next.
