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
- `bench_indexed_add.py`: Indexed-add benchmark.
- `bench_reach_weights.py`: Reach-weight computation benchmark.
- `bench_rules_triton.py`: Triton hand-rules benchmark.
- `bench_triton_pbs_env.py`: PBSEnv versus TritonPBSEnv microbenchmarks for legal, step, reset, copy, gather, repeat, and persistent `*_into` materialization paths.
- `multiway_showdown_estimators.py`: GPU validation harness for n-way showdown equity estimators, including naive rejection sampling, sequential importance sampling, restricted-support brute force, batched single-board belief-update benchmarking, and exact inclusion-exclusion prototypes.
- `bench_set_model_values_indexing.py`: Model-value indexing benchmark.
- `bench_showdown_active_runner.py`: One-build/300-replay CUDA benchmark comparing the production exact CFR showdown graph runner with an exact active-hand/local-card runner inspired by tier-2 showdown compaction.
- `cfr_optimizations_bench.py`: CFR optimization comparison benchmark.
- `profile_showdown_nobopp_components.py`: Component-level profiler for the compact no-opponent-materialization showdown pipeline.
- `profile_chance_helper.py`: Chance-node helper profiler.

### Subdirectories
There are no child source directories.
