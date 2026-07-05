## Directory summary
Focused microbenchmarks for tensor operations, CFR evaluator variants, Triton kernels, and poker-specific reductions.

### Source files
- `bench_advanced_indexing.py`: Advanced indexing benchmark.
- `bench_allin_datagen.py`: All-in data-generation throughput benchmark.
- `bench_better_ffn_step_hotpath.py`: BetterFFN forward-path benchmark.
- `bench_calculate_unblocked_mass.py`: Unblocked-mass calculation benchmark.
- `bench_combo_mask.py`: Combo masking benchmark.
- `bench_fused_cfr_triton.py`: Fused CFR Triton benchmark.
- `bench_fused_sparse_speedups.py`: Fused sparse evaluator speedup benchmark.
- `bench_heads_up_projection.py`: Heads-up closing-belief projection benchmark.
- `bench_indexed_add.py`: Indexed-add benchmark.
- `bench_preflop_allin_3p_kernel.py`: Native-169 3-player all-in kernel benchmark.
- `bench_preflop_allin_share3_denom.py`: Production 3-player all-in denominator benchmark.
- `bench_preflop_belief_projection.py`: Compact preflop belief-projection benchmark.
- `bench_preflop_eval_projection_cache.py`: Compact preflop eval-cache benchmark.
- `bench_preflop_ffn_epilogue.py`: Gated-token-mixer FFN epilogue benchmark.
- `bench_preflop_gate_projection_variants.py`: Gated-token-mixer gate-projection benchmark.
- `bench_preflop_gated_token_mixer_megakernel.py`: Compact gated-token-mixer megakernel benchmark.
- `bench_preflop_sample_snapshot.py`: Sparse sample snapshot benchmark.
- `bench_preflop_token_mixer_cross_norm.py`: Token-mixer cross-block normalization benchmark.
- `bench_preflop_token_mixer_cuda_graph.py`: Token-mixer CUDA graph benchmark.
- `bench_preflop_token_mixer_next_norm.py`: Token-mixer next-normalization benchmark.
- `bench_preflop_token_mixer_overlap.py`: Token-mixer staged-overlap benchmark.
- `bench_preflop_token_mixer_stage_tuning.py`: Token-mixer staged-kernel tuning benchmark.
- `bench_reach_weights.py`: Reach-weight computation benchmark.
- `bench_rules_triton.py`: Triton hand-rules benchmark.
- `bench_set_model_values_indexing.py`: Model-value indexing benchmark.
- `bench_showdown_active_runner.py`: Active-hand exact showdown benchmark.
- `bench_triton_pbs_env.py`: PBSEnv and TritonPBSEnv benchmark.
- `bench_triton_small_k_dot.py`: Small-K Triton dot benchmark.
- `bench_turn_training_hotspots.py`: S_turn training hotspot benchmark that separates turn equity baseline, value-forward, leaf-value writeback, and full random-turn CFR solve costs.
- `cfr_optimizations_bench.py`: CFR optimization comparison benchmark.
- `multiway_showdown_estimators.py`: Multiway showdown estimator validation harness.
- `profile_chance_helper.py`: Chance-node helper profiler.
- `profile_showdown_nobopp_components.py`: Compact showdown component profiler.

### Subdirectories
There are no child source directories.
