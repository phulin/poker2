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
- Tried a two-input add kernel for belief phase shifts; it was output-identical but slightly slower, so it was reverted.
- Tried caching repeated model uniform buffers; key-building overhead made it slower, so it was reverted.
- Tried caching the zero input used by the three-input belief phase-shift add; it was output-identical but slower, so it was reverted.

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
- 2026-05-29 failed experiments:
  - `P2_ADD2_BELIEF_SHIFT=1`: 503.2 ms baseline vs 504.1 ms candidate, speedup 0.998x, exact output match.
  - `P2_CACHE_MODEL_UNIFORMS=1`: 506.7 ms baseline vs 534.4 ms candidate, speedup 0.948x, exact output match.
  - `P2_CACHE_BELIEF_SHIFT_ZERO=1`: 487.1 ms baseline vs 532.0 ms candidate, speedup 0.916x, exact output match.

## Verification
- `yarn typecheck`: pass.
- `WEBGPU_BACKEND=metal node --test --test-concurrency=1 --import tsx tests/sparse_resolver.test.ts tests/sparse_cfr_kernels.test.ts`: pass, 17 tests.
- Added `combined sparse prefix command buffers match separate submissions` to compare old and new sparse solver paths with exact policy/action/belief equality.
- `WEBGPU_BACKEND=metal node --test --test-concurrency=1 --test-name-pattern "shifted exact-belief" --import tsx tests/split_checkpoint_fixture.test.ts`: pass.
