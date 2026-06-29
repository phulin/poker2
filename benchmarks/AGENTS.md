## Directory summary
Focused microbenchmarks for tensor operations, CFR evaluator variants, Triton kernels, and poker-specific reductions.

### Source files
- `bench_advanced_indexing.py`: Advanced indexing benchmark.
- `bench_allin_datagen.py`: All-in training data-generation microbenchmark; times random batch creation and Monte Carlo target estimation separately, reports boards/samples throughput, and can toggle persistent target workspaces, legacy board-allowed matrices, and folded-hero skipping for ablations.
- `bench_better_ffn_step_hotpath.py`: Benchmarks BetterFFN forward and chance-node value hot paths.
- `bench_calculate_unblocked_mass.py`: Unblocked-mass calculation benchmark.
- `bench_combo_mask.py`: Combo masking benchmark.
- `bench_fused_cfr_triton.py`: Fused CFR Triton benchmark.
- `bench_fused_sparse_speedups.py`: Fused sparse evaluator speedup benchmark.
- `bench_heads_up_projection.py`: Benchmarks the n-way-to-heads-up closing-belief projection, comparing the current PyTorch row-plus-player gather with the direct Triton row/player select kernel.
- `bench_indexed_add.py`: Indexed-add benchmark.
- `bench_preflop_allin_3p_kernel.py`: Benchmarks the exact native-169 3-player all-in share calculation, comparing the current materialized opponent outer-product matmul path with combined-GEMM, structural-denominator, packed-symmetry, and fused Triton candidates.
- `bench_preflop_gate_projection_variants.py`: CUDA microbenchmark isolating the gated-token-mixer gate projection, comparing seven separate token dots with token-combined M-dimension layouts in bf16 and fp32/TF32.
- `bench_preflop_allin_share3_denom.py`: Realistic actions_4_7 microbenchmark for the production exact 3-player all-in denominator path, with a dense-reference correctness check.
- `bench_preflop_gated_token_mixer_megakernel.py`: CUDA microbenchmark for compact preflop gated-token-mixer inference paths, comparing eager PyTorch, the current Triton mixer/gate residual helper, staged persistent mixer/residual launch variants, and the experimental RMSNorm+gate+mixer residual megakernel across large batch sizes derived from the preflop buckets Hydra config.
- `bench_preflop_ffn_epilogue.py`: CUDA microbenchmark for the compact gated-token-mixer FFN output boundary, comparing cuBLAS `linear_out` plus the Triton residual/next-token-RMSNorm epilogue against naive/compiled PyTorch and full-width Triton GEMM epilogue candidates.
- `bench_preflop_token_mixer_cuda_graph.py`: CUDA microbenchmark for fixed-shape CUDA Graph replay of the compact gated-token-mixer token path, comparing raw replay and input-copy-plus-replay against normal eager launches.
- `bench_preflop_token_mixer_cross_norm.py`: CUDA microbenchmark for stack-level compact gated-token-mixer fusion, comparing the old per-block loop, the wired cross-boundary residual-plus-next-token-RMSNorm runner, standalone boundary variants, and optional `torch.compile` naive/production stack baselines under CUDA graph replay.
- `bench_preflop_token_mixer_next_norm.py`: CUDA microbenchmark for fusing the compact gated-token-mixer residual with the following FFN RMSNorm, comparing against the old Triton residual path and an optional `torch.compile` naive PyTorch baseline under normal launch or CUDA graph-replay timing.
- `bench_preflop_token_mixer_overlap.py`: CUDA microbenchmark for a broader staged design that computes token mixer and gate projection as independent producers, including an overlapped two-stream version plus a fused final gate/residual consumer.
- `bench_preflop_token_mixer_stage_tuning.py`: CUDA microbenchmark for tuning only the precomputed gated-token-mixer residual stage (`x`, normalized tokens, and gate projection already materialized), comparing grid and persistent Triton launch shapes.
- `bench_preflop_sample_snapshot.py`: Realistic actions_4_7 microbenchmark for replacing full-tree sample policy/belief `torch.where` updates with a sparse fused snapshot copy.
- `bench_reach_weights.py`: Reach-weight computation benchmark.
- `bench_rules_triton.py`: Triton hand-rules benchmark.
- `bench_triton_pbs_env.py`: PBSEnv versus TritonPBSEnv microbenchmarks for legal, step, reset, copy, gather, repeat, and persistent `*_into` materialization paths.
- `bench_triton_small_k_dot.py`: CUDA microbenchmark comparing `tl.dot` variants against manual fp32 accumulation for small-K matrix products used by the gated-token-mixer experiments.
- `multiway_showdown_estimators.py`: GPU validation harness for n-way showdown equity estimators, including naive rejection sampling, sequential importance sampling, restricted-support brute force, batched single-board belief-update benchmarking, and exact inclusion-exclusion prototypes.
- `bench_set_model_values_indexing.py`: Model-value indexing benchmark.
- `bench_showdown_active_runner.py`: One-build/300-replay CUDA benchmark comparing the production exact CFR showdown graph runner with an exact active-hand/local-card runner inspired by tier-2 showdown compaction.
- `cfr_optimizations_bench.py`: CFR optimization comparison benchmark.
- `profile_showdown_nobopp_components.py`: Component-level profiler for the compact no-opponent-materialization showdown pipeline.
- `profile_chance_helper.py`: Chance-node helper profiler.

### Subdirectories
There are no child source directories.
