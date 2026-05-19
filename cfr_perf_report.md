# CFR Main-Path Performance Report

## Benchmark Harness
- Added `scripts/bench_cfr_main_path.py`.
- Runs realistic `RebelCFRTrainer.train_step` profiles with wandb/trueskill disabled.
- Applies requested model settings: `hidden_dim=256`, `range_hidden_dim=128`, `ffn_dim=512`, `num_hidden_layers=3`, `num_value_layers=1`, `num_policy_layers=1`.
- Pauses matching `train_rebel` processes before benchmark sections and resumes them afterward.
- Writes structured JSON output under `outputs/`.

## Baseline
Best source profile: `outputs/cfr_main_path_source_realistic_v2.json`.

- Active steady-state wall: `1.374s/step`.
- Largest regions:
  - Data generation: `718ms` inclusive CUDA-tagged.
  - CFR evaluate: `447ms`.
  - Supervised update: `146ms`.
  - Subgame init: `132ms`.
  - Metrics: `121ms`.
  - Replay buffer sampling: `248ms` CPU across value/policy sample tags.
  - Replay buffer add: `78ms` across value/policy add tags.

Expanded microbenchmarks: `outputs/cfr_main_path_micro_realistic_v2.json`.

## Main Findings
The next 2x is probably not mostly inside individual CFR iteration kernels. CUDA graph replay makes the visible per-iteration work small in the source profile. The larger costs are around the CFR loop and training loop: replay buffer CPU sampling/transfers, subgame setup/graph capture, metrics, and supervised-update batching.

## Recommended Work
1. Add a replay-buffer `sample_to_device` path with pinned CPU staging and preallocated GPU batch tensors.
2. Add a faster replay-buffer add path for GPU-produced training data, using pinned CPU destinations or asynchronous staging.
3. Reuse fused sparse evaluator buffers and bucket CUDA graphs by static tree shape/regime to avoid per-subgame graph capture/allocation work.
4. Move heavy metrics/aggression analysis to a lower cadence for training-fast runs.
5. Combine the two value forwards in `_supervise` into one doubled-batch value forward, and use `zero_grad(set_to_none=True)`.

## 2x Estimate
Baseline is `1.374s/step`; a 2x target is `~0.687s/step`.

Plausible output-preserving savings:
- Replay sample/add/transfer staging: `250-350ms`.
- CFR graph/subgame reuse: `250-350ms`.
- Lower-cadence metrics: `90-120ms`.
- Supervised update cleanup: `20-50ms`.

That totals `610-870ms`, which puts the loop in the `0.50-0.76s/step` range if the larger changes land.

## Caveats
- Source-profile component CUDA times are inclusive and nested; do not sum them.
- CUDA graph replay hides most per-iteration internals from Python record-function ranges.
- Microbench buffer-sample CUDA-event timings should be read as elapsed wall latency because the sampled tensors live on CPU.

## Latest Progress
- GPU replay buffers remove the CPU replay gather and transfer path in the ReBeL CFR config while keeping CPU as the dataclass default. Microbenchmarks improved value/policy sample from `23.26ms`/`36.48ms` on CPU replay to `0.36ms`/`0.48ms` on CUDA replay.
- Cached aggression hand groups and faster suit permutation reduce metrics/permutation overhead. `batch_permute` improved from `2.55ms` to `0.65ms`.
- Best clean one-step source run after these changes is `outputs/cfr_main_path_source_replay_device_cuda_fast_permute.json` at `0.703s/step`, about `1.95x` faster than the `1.374s/step` baseline. The strict 2x target is `~0.687s/step`, so the remaining gap is small but not conclusively closed.
- Added a Triton env row-gather fast path for sparse subgame expansion. On `outputs/spots.pt` with 1024 roots/street and depth 4, `construct_subgame` improved from `105.0/95.1/89.7/103.0ms` to `89.8/78.5/75.7/88.1ms` for preflop/flop/turn/river respectively. The clean source run `outputs/cfr_main_path_source_triton_env_gather_runtime_n.json` measured `0.703s/step`, with `cfr_init_subgame` at `129.19ms` vs `132.83ms` before.
- Added focused `spots.pt` microbenchmarks for CFR iterator, warm-start, and subgame initialization components. The subgame-init benchmark (`outputs/cfr_init_spots_uniform_parent_index.json`) found the init-time `uniform_policy` denominator expansion can use the already-built `parent_index` instead of another `repeat_interleave`, improving that component from `0.141ms` to `0.058ms` CUDA on average while asserting equality with the old formula.
- After the parent-index denominator change, `outputs/cfr_main_path_source_uniform_parent_index.json` measured `0.665s/step` on a realistic source run, crossing the strict `~0.687s/step` 2x target versus the `1.374s/step` baseline. In that run, `cfr_init_subgame` was `120.70ms/step`.
- Added an unfused mode to the balanced `spots.pt` iterator benchmark for direct core-loop progress checks. In matching short runs, `outputs/cfr_iterator_spots_unfused_baseline_ab4.json` averaged `70.76ms/iter` across pre/post DCFR segments, while `outputs/cfr_iterator_spots_fused_current_ab4.json` averaged `17.51ms/iter`, a `4.04x` speedup (`75.3%` lower wall time) on the requested `spots.pt` roots. The longer current run with BetterFFN fixed hand-index buffers, `outputs/cfr_iterator_spots_betterffn_hand_index_buffers_ab12.json`, averaged `14.52ms/iter`.
- BetterFFN now stores fixed hand rank/suit indices as non-persistent buffers. The targeted hand-embedding microbenchmark (`outputs/betterffn_hand_embedding_index_buffers_micro.json`) showed parity with the old formula and a small `~1.3%` CUDA reduction for that substep; the `spots.pt` iterator run above also showed a small model-forward reduction (`~4.92ms` vs prior `~4.99ms` CUDA per iteration segment).
- Corrected core-loop target is fused-to-fused, not unfused-to-fused: `outputs/cfr_iterator_spots_head_default_ab12.json` is the fused baseline at `15.01ms/iter`, so the 33% wall-clock reduction target is `<=10.0ms/iter`.
- BetterFFN exposes `static_feature_base`, and the fused evaluator caches that static board/street/context contribution per subgame while still using a compiled helper for compiled models. `outputs/cfr_iterator_spots_static_base_cache_compiled_ab12.json` averaged `13.95ms/iter`; `cfr_model_fwd` fell from `~4.92ms` to `~4.39ms` CUDA per iteration segment. This is a `7.1%` wall-clock reduction from the fused baseline, so the fused-to-fused target is still not met.
- The model-value writeback Triton launcher now uses 4 warps for the default no-mix/no-zero-sum writeback path while retaining 8 warps for reduction-heavy paths. `outputs/cfr_iterator_spots_writeback_4warps_ab12.json` averaged `13.65ms/iter`; a repeat with 20 active iterations (`outputs/cfr_iterator_spots_writeback_4warps_ab20.json`) averaged `13.21ms/iter`. The comparable `ab12` result is a `9.1%` wall-clock reduction from the fused baseline, still short of the `<=10.0ms/iter` target.
- A broader source-profile check after the static-base and writeback changes, `outputs/cfr_main_path_source_static_base_writeback.json`, measured `0.675s/step` with `cfr_iter` at `16.33ms/step` across the profiled train step.
- With `cfr_avg=false` and final-policy value targets, average-policy reach propagation is now deferred until the final average beliefs are needed. `outputs/cfr_iterator_spots_defer_avg_reach_ab12.json` averaged `13.37ms/iter`, and the longer `outputs/cfr_iterator_spots_defer_avg_reach_ab20.json` averaged `12.34ms/iter`. The comparable `ab12` result is a `10.9%` wall-clock reduction from the fused baseline; the target remains `<=10.0ms/iter`.
- The same final-policy path now also defers average-policy normalization itself, accumulating only numerator/denominator during CFR iterations and materializing `policy_probs_avg` once at finalization. `outputs/cfr_iterator_spots_defer_avg_policy_ab12.json` averaged `12.81ms/iter`; the longer `outputs/cfr_iterator_spots_defer_avg_policy_ab20.json` averaged `11.91ms/iter`. The comparable `ab12` result is a `14.6%` wall-clock reduction from the fused baseline; the target remains `<=10.0ms/iter`.
- The deferred average-policy accumulation path now uses 1024-hand Triton blocks. `outputs/cfr_iterator_spots_defer_avg_policy_block1024_ab12.json` averaged `12.34ms/iter`; `outputs/cfr_iterator_spots_defer_avg_policy_block1024_ab20.json` averaged `11.81ms/iter`. The comparable `ab12` result is a `17.8%` wall-clock reduction from the fused baseline; the target remains `<=10.0ms/iter`.
- A broader source-profile check after deferring average-policy finalization, `outputs/cfr_main_path_source_defer_avg_policy.json`, measured `0.652s/step`; `cfr_evaluate` was `410.69ms/step` and `cfr_iter` was `15.18ms/step`.
