## Directory summary
Reusable profiling and benchmark scripts that are broader than a single repro but are not package modules.

### Source files
- `profile_train_rebel.py`: Profiles the ReBeL training loop.
- `profile_showdown_kernel.py`: Profiles showdown kernel behavior.
- `bench_showdown.py`: Benchmarks showdown evaluation.
- `bench_showdown_tiers.py`: Runs interleaved tier 1-3 multiway showdown benchmarks with optional SIGSTOP/SIGCONT pausing of a live training process.
- `bench_parent_expand.py`: Benchmarks parent expansion behavior.
- `bench_breakdown.py`: Produces benchmark breakdowns for selected paths.
- `bench_model_share.py`: Benchmarks model sharing or reuse behavior.
- `bench_cfr_main_path.py`: Runs realistic ReBeL CFR train-step source profiling and component microbenchmarks with optional train_rebel pause/resume handling.
- `bench_write_children_kernel.py`: Replays real fused-subgame child-writer inputs to compare legacy and optimized `write_children_same_street` Triton kernels with CUDA events.
- `bench_cfr_init_spots.py`: Microbenchmarks fused sparse CFR subgame initialization and init-time tensor expansions from evenly balanced saved spots.
- `bench_cfr_iterator_spots.py`: Profiles eager sparse/fused-sparse CFR iterator segments and isolated CFR components from evenly balanced saved spots across preflop/flop/turn/river roots.
- `bench_warm_start_spots.py`: Microbenchmarks fused sparse CFR warm-start total time and substeps from evenly balanced saved spots.
- `precompute_preflop_allin_table.py`: Streams preflop five-card boards to build a zstd-compressed int16 `[1326, 1326]` all-in matchup payoff table for preflop all-in-call terminal abstraction.
- `probe_cfr_fp_precision.py`: Probes fp32 precision loss in fused sparse CFR average-policy/value updates from saved spots and checkpoints.
- `test_survey_runner.py`: Runs pytest files or node ids with per-invocation timeouts and records structured audit results.

### Subdirectories
There are no child source directories.
