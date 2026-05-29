# Notes: Website CFR Benchmark Optimization

## Benchmark Commands
- Interleaved spot benchmark: `yarn bench:spots:interleaved` if present, otherwise run `tsx src/benchSpotsInterleaved.ts` with explicit flags from `website/`.
- Standard spot benchmark: `yarn bench:spots`.
- CFR parity/tests: `yarn test` from `website/` unless a narrower test is sufficient while iterating.

## Findings
- `benchSpotsInterleaved.ts` existed but `package.json` did not expose a script; added `bench:spots:interleaved`.
- `tsx` ad hoc invocations need escalation in this sandbox because its IPC pipe under `/var/folders/.../T` is blocked.
- Root PBS depth 6 / 128 iterations has 172 nodes, 31 model leaves, and 26 all-in leaves.
- Default profile with `P2_PROFILE=1` shows leaf value refresh dominates: roughly 665 ms over 117 per-iteration calls, with iteration prefix around 105 ms over 118 calls.
- GPU-resident model warm-start was tried and removed because it changed CFR outputs too much to enable.
- Batch-4 value-head subgroup selection was tested and reverted because CFR parity drift exceeded current tolerances.
- Combining each leaf-value/backup command buffer with the next iteration prefix preserves command order, reduces queue submissions, and is now the default. Set `P2_COMBINE_PREFIX_WITH_LEAF=0` to benchmark the old path.
- Prepared leaf features now cache exact-belief phase-shift GPU buffers per exact player/hand. Set `P2_CACHE_EXACT_PHASE_SHIFT=0` to benchmark the old rebuild/upload behavior.
- Prepared leaf features now mark all-empty boards so BetterFFN can skip the board-interaction branch when it is mathematically zero. Set `P2_SKIP_EMPTY_BOARD_INTERACTION=0` to benchmark the old explicit-zero path.
- Leaf prediction temporary buffers are now returned to the model buffer pool every 16 predictions instead of only at solve end. Set `P2_RELEASE_LEAF_TEMPS_EVERY=0` to benchmark the old retained-until-end path.
- Tried a two-input add kernel for belief phase shifts; it was output-identical but slightly slower, so it was reverted.
- Tried caching repeated model uniform buffers; key-building overhead made it slower, so it was reverted.
- Tried caching the zero input used by the three-input belief phase-shift add; it was output-identical but slower, so it was reverted.
- Tried batch-4 subgroup residual matvec selection; it was output-identical but slower, so it was reverted.
- Tried caching compute-pipeline bind-group layouts; it was output-identical but within noise, so it was reverted.
- Tried fusing exact-belief phase shifts into the half-norm path; it sped up root CFR policy/action output but failed the stronger exact value fixture, so it was reverted.
- Tried reusing sparse solve CPU scratch arrays; it was output-identical but within noise, so it was reverted.
- Tried an exact-belief zero-sum value-head postprocess; it was exact-value-safe but slower, so it was reverted.

## Measurements
- 2026-05-29 short no-op interleaved run, `bench_spots.json`, depth 4, iterations 32, runs 3:
  - Preflop spot completed: baseline 86.3 ms, candidate 92.4 ms, no-op speedup 0.935x.
  - Flop spot failed in all-in table values because the production path requires no permutations.
- 2026-05-29 root no-op interleaved run, `bench_spots_root.json`, depth 6, iterations 128, runs 5:
  - baseline 574.1 ms, no-op candidate 574.7 ms, speedup 0.999x.
- 2026-05-29 model kernel microbench, batch 31:
  - Value-head 2652x512 batch-4 subgroup was slightly faster than batch-2 subgroup in isolation, but not parity-safe for CFR.
- 2026-05-29 experimental GPU warm-start interleaved run, `bench_spots_root.json`, depth 6, iterations 128, warmups 1, runs 5:
  - baseline 523.5 ms, candidate 511.8 ms, speedup 1.023x.
  - Not production-safe yet because direct parity tests showed CFR policy/action drift beyond current tolerances.
- 2026-05-29 combined-prefix interleaved run, `bench_spots_root.json`, depth 6, iterations 128, warmups 2, runs 7, `--compare-outputs`:
  - old path (`P2_COMBINE_PREFIX_WITH_LEAF=0`) 625.3 ms, new default path 438.4 ms, speedup 1.426x.
  - Output comparison: `policyMaxAbs=0`, `actionProbsMaxAbs=0`.
- 2026-05-29 exact phase-shift cache interleaved run, `bench_spots_root.json`, depth 6, iterations 128, warmups 2, runs 7, `--compare-outputs`:
  - old rebuild path (`P2_CACHE_EXACT_PHASE_SHIFT=0`) 520.6 ms, cached path 512.1 ms, speedup 1.017x.
  - Output comparison: `policyMaxAbs=0`, `actionProbsMaxAbs=0`.
- 2026-05-29 current cumulative combined-prefix interleaved run after exact cache, `bench_spots_root.json`, depth 6, iterations 128, warmups 2, runs 7, `--compare-outputs`:
  - old command path (`P2_COMBINE_PREFIX_WITH_LEAF=0`) 562.3 ms, current default 446.6 ms, speedup 1.259x.
  - Output comparison: `policyMaxAbs=0`, `actionProbsMaxAbs=0`.
- 2026-05-29 empty-board interaction skip interleaved run, `bench_spots_root.json`, depth 6, iterations 128, warmups 2, runs 7, `--compare-outputs`:
  - explicit zero interaction path (`P2_SKIP_EMPTY_BOARD_INTERACTION=0`) 549.9 ms, skipped path 511.1 ms, speedup 1.076x.
  - Output comparison: `policyMaxAbs=0`, `actionProbsMaxAbs=0`.
- 2026-05-29 current cumulative combined-prefix interleaved run after empty-board skip, `bench_spots_root.json`, depth 6, iterations 128, warmups 2, runs 7, `--compare-outputs`:
  - old command path (`P2_COMBINE_PREFIX_WITH_LEAF=0`) 623.3 ms, current default 475.4 ms, speedup 1.311x.
  - Output comparison: `policyMaxAbs=0`, `actionProbsMaxAbs=0`.
- 2026-05-29 fused exact-belief phase-shift interleaved run, `bench_spots_root.json`, depth 6, iterations 128, warmups 2, runs 7, `--compare-outputs`:
  - explicit add path (`P2_FUSE_EXACT_BELIEF_SHIFT=0`) 673.0 ms, fused path 637.6 ms, speedup 1.056x.
  - Output comparison: `policyMaxAbs=0`, `actionProbsMaxAbs=0`.
- 2026-05-29 chunked leaf-temp release interleaved runs, `bench_spots_root.json`, depth 6, iterations 128, `--compare-outputs`:
  - release at solve end (`P2_RELEASE_LEAF_TEMPS_EVERY=0`) 534.9 ms, release every 16 predictions 525.6 ms, speedup 1.018x, exact output match.
  - longer confirmation with runs 9: release at solve end 747.6 ms, release every 16 predictions 727.2 ms, speedup 1.028x, exact output match.
  - release every 32 predictions was not stable: a direct 0 vs 32 run measured speedup 0.988x.
- 2026-05-29 failed experiments:
  - `P2_ADD2_BELIEF_SHIFT=1`: 503.2 ms baseline vs 504.1 ms candidate, speedup 0.998x, exact output match.
  - `P2_CACHE_MODEL_UNIFORMS=1`: 506.7 ms baseline vs 534.4 ms candidate, speedup 0.948x, exact output match.
  - `P2_CACHE_BELIEF_SHIFT_ZERO=1`: 487.1 ms baseline vs 532.0 ms candidate, speedup 0.916x, exact output match.
  - `P2_RESIDUAL_BATCH4=1`: 409.2 ms baseline vs 413.7 ms candidate, speedup 0.989x, exact output match.
  - `P2_CACHE_BIND_GROUP_LAYOUT=1`: 544.6 ms baseline vs 543.7 ms candidate, speedup 1.002x, exact output match.
  - `P2_FUSE_EXACT_BELIEF_SHIFT=1`: 673.0 ms baseline vs 637.6 ms candidate, speedup 1.056x, policy/action output match, but exact value fixture drifted by about 0.095; reverted.
  - `P2_REUSE_SPARSE_SOLVE_ARRAYS=1`: 539.8 ms baseline vs 540.2 ms candidate, speedup 0.999x, exact output match.
  - `P2_EXACT_ZERO_SUM=1`: 540.4 ms baseline vs 544.7 ms candidate, speedup 0.992x, exact output match and exact value fixture pass.

## Verification
- `yarn typecheck`: pass.
- `WEBGPU_BACKEND=metal node --test --test-concurrency=1 --import tsx tests/sparse_resolver.test.ts tests/sparse_cfr_kernels.test.ts`: pass, 17 tests.
- Added `combined sparse prefix command buffers match separate submissions` to compare old and new sparse solver paths with exact policy/action/belief equality.
- `WEBGPU_BACKEND=metal node --test --test-concurrency=1 --test-name-pattern "shifted exact-belief" --import tsx tests/split_checkpoint_fixture.test.ts`: pass.
- Added `empty-board interaction skip matches explicit zero interaction` to compare skipped and explicit-zero BetterFFN board-interaction paths exactly.
