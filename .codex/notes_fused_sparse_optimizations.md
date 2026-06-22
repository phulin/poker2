# Notes: Fused Sparse CFR Optimizations

## Benchmark Harness
- Script: `scripts/bench_cfr_iterator_spots.py`
- Realistic command shape:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/bench_cfr_iterator_spots.py --no-pause --no-compile --no-component-bench --per-street 64 --iterations 400 --dcfr-delay 80 --warmup-iters 40 --active-iters 400 ... search.cfr_type=sapcfr search.predictive_cfr_dcfr_hybrid=true search.predictive_cfr_delay=40 search.warm_start_iterations=40 search.dcfr_plus_delay=80`
- CUDA needs escalated permissions in this environment.
- Paused training process: PID 23502 using about 37116 MiB per `nvidia-smi`.

## Baseline Results
- `outputs/opt_review/baseline_iterator_p64_400it_40_80_sapdcfr.json`
  - Subgame: 256 roots, 16896 nodes, 11080 leaves.
  - Post-delay wall: 13.670 ms/iter.
  - Post-delay `cfr_iter_update_policy`: 1.717 ms/iter.
- `outputs/opt_review/baseline_rerun_iterator_p64_400it_40_80_sapdcfr.json`
  - Same subgame shape.
  - Post-delay wall: 13.970 ms/iter.
  - Post-delay `cfr_iter_update_policy`: 1.716 ms/iter.

## Idea 3 Trial
- Implemented a vectorized parent-centric kernel that did regret/DCFR update and wrote policy directly, skipping positive-regrets materialization and parent-sum-divide.
- Correctness tests passed after masking padded child lanes out of the denominator.
- Benchmark: `outputs/opt_review/direct_policy_iterator_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.926 ms/iter.
  - Post-delay `cfr_iter_update_policy`: 1.538 ms/iter.
- Decision: rejected for now. It improves the local update-policy component but does not clearly improve end-to-end wall time versus two baseline runs.

## Idea 3 Fast Static-Loop Epilogue Trial
- Implemented the intended fast version by extending `_fused_unblocked_regret_dcfr_update_kernel` itself:
  - Same one-parent/hand-block program shape as the current hot kernel.
  - First static child loop updates cumulative regrets and predictive last-regret state.
  - Accumulates positive policy mass in registers.
  - Second static child loop writes normalized `policy_probs` directly.
  - Skips `positive_regrets_out` and `fused_parent_sum_divide_` when enabled.
- Correctness:
  - `test_fused_unblocked_regret_dcfr_update_predictive_matches_pytorch` passed.
  - `P2_FUSED_REGRET_POLICY_EPILOGUE=1 test_fused_sparse_sapdcfr_runs_past_predictive_delay` passed.
- Benchmark: `outputs/opt_review/regret_policy_epilogue_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.964 ms/iter.
  - Post-delay `cfr_iter_update_policy`: 1.556 ms/iter.
- Decision: not a clear winner. It improves the local update-policy component but does not beat baseline end-to-end beyond run noise.

## Existing Baseline Features
- Predictive CFR/SAPCFR support is already implemented in the fused sparse evaluator and covered by `test_fused_sparse_sapdcfr_runs_past_predictive_delay`.
- Existing fused regret kernel already writes predictive positive regrets for policy extraction.
