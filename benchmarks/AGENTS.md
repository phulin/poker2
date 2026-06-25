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
- `bench_preflop_allin_live2_entries.py`: Realistic actions_4_7 microbenchmark for exact preflop all-in value routing/writeback, including live-entry routing and sparse net-value writeback variants.
- `bench_preflop_cutoff_belief_reuse.py`: Realistic actions_4_7 microbenchmark for reusing cutoff model-input belief rows during value writeback instead of gathering the same rows twice.
- `bench_preflop_defer_avg_policy.py`: Realistic actions_4_7 microbenchmark for deferring average-policy materialization and renormalization until the end of a default preflop solve.
- `bench_preflop_model_leaf_scatter.py`: Realistic actions_4_7 microbenchmark for compact preflop model-leaf belief scatter, comparing direct model-index gathers against the fused propagation side-output while pausing live preflop training process groups.
- `bench_preflop_sample_snapshot.py`: Realistic actions_4_7 microbenchmark for replacing full-tree sample policy/belief `torch.where` updates with a sparse fused snapshot copy.
- `bench_preflop_skip_model_leaf_belief_store.py`: Realistic actions_4_7 microbenchmark for skipping full reach/belief tensor writes for model leaves when the compact model-leaf scatter side-output is active.
- `bench_preflop_selected_hu_writeback.py`: Realistic actions_4_7 microbenchmark for HU closing value writeback from already-selected live-player beliefs instead of a full six-player belief gather.
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
