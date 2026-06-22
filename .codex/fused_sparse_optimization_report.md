# Fused Sparse Optimization Report

## Benchmark Context
- CUDA access required escalated permissions.
- A paused training process used about 37 GB on the A100, so realistic saved-spot benchmarking used `--per-street 64` instead of `128`.
- All iterator benchmarks used 400 active iterations with 40 warmup iterations and SAPDCFR 40/80:
  - `search.cfr_type=sapcfr`
  - `search.predictive_cfr_dcfr_hybrid=true`
  - `search.predictive_cfr_delay=40`
  - `search.warm_start_iterations=40`
  - `search.dcfr_plus_delay=80`

## Results
- Baseline 1: `outputs/opt_review/baseline_iterator_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.670 ms/iter
  - Post-delay update-policy: 1.717 ms/iter
- Baseline 2: `outputs/opt_review/baseline_rerun_iterator_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.970 ms/iter
  - Post-delay update-policy: 1.716 ms/iter
- Idea 3 vectorized regret-policy trial:
  - `outputs/opt_review/direct_policy_iterator_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.926 ms/iter
  - Post-delay update-policy: 1.538 ms/iter
- Idea 3 static-loop epilogue trial:
  - `outputs/opt_review/regret_policy_epilogue_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.964 ms/iter
  - Post-delay update-policy: 1.556 ms/iter
- Idea 2 backup-regret epilogue trial:
  - `outputs/opt_review/backup_regret_epilogue_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 14.912 ms/iter
  - Post-delay update-policy: 1.700 ms/iter
- Idea 1 counterfactual-value/direct-regret trial:
  - `outputs/opt_review/cf_values_direct_regret_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.946 ms/iter
  - Post-delay update-policy: 1.712 ms/iter

## Decisions
- Idea 3 is not a commit winner. The fastest intended static-loop implementation improves the local update-policy component but does not produce a clear end-to-end wall-clock win.
- Idea 2 is not a commit winner. Folding regret work into the backup made the realistic iterator slower.
- Idea 1 is not a commit winner in the tested implementation. It reduced profiled CFR-iteration CUDA work, but the realistic end-to-end wall time did not beat the best baseline run.
- Idea 4 predictive SAPDCFR is already present in the baseline path and was the benchmark configuration.
- Idea 5 was not pursued after clarification because alternating updates are a semantic change.
- No evaluator optimization was committed from this pass.

## Pass 2 Results
- Direct showdown leaf-belief path:
  - `outputs/opt_review/direct_showdown_leaf_path_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.477 ms/iter
  - Removed showdown rows from the model leaf-belief gather and used the existing direct indexed showdown runner even when model leaves are present.
- Concurrent leaf streams trial:
  - `outputs/opt_review/leaf_streams_direct_showdown_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.339 ms/iter
  - Follow-up no-env/default run was slower at 13.725 ms/iter, so stream overlap was treated as noisy and removed.
- Direct showdown no-stream rerun:
  - `outputs/opt_review/direct_showdown_no_streams_rerun_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.381 ms/iter
- Final retained direct showdown path:
  - `outputs/opt_review/direct_showdown_final_p64_400it_40_80_sapdcfr.json`
  - Post-delay wall: 13.194 ms/iter
- Shared regret-stat prep trial:
  - Rejected before benchmarking. It passed eager state tests but failed CUDA graph replay parity because cross-iteration Python cache validity was not graph/snapshot-safe.

## Pass 2 Decisions
- Commit the direct showdown leaf-belief path. It is graph-compatible and repeatedly beat the best prior baseline.
- Do not commit concurrent leaf streams. The best run was good, but the no-env/default confirmation was not robust.
- Do not commit shared regret-stat prep without a device-side validity design.
