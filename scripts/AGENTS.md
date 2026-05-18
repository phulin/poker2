## Directory summary
Reusable profiling and benchmark scripts that are broader than a single repro but are not package modules.

### Source files
- `profile_train_rebel.py`: Profiles the ReBeL training loop.
- `profile_showdown_kernel.py`: Profiles showdown kernel behavior.
- `bench_showdown.py`: Benchmarks showdown evaluation.
- `bench_parent_expand.py`: Benchmarks parent expansion behavior.
- `bench_breakdown.py`: Produces benchmark breakdowns for selected paths.
- `bench_model_share.py`: Benchmarks model sharing or reuse behavior.
- `bench_cfr_main_path.py`: Runs realistic ReBeL CFR train-step source profiling and component microbenchmarks with optional train_rebel pause/resume handling.
- `bench_cfr_init_spots.py`: Microbenchmarks fused sparse CFR subgame initialization and init-time tensor expansions from evenly balanced saved spots.
- `bench_cfr_iterator_spots.py`: Profiles eager fused sparse CFR iterator segments from evenly balanced saved spots across preflop/flop/turn/river roots.
- `bench_warm_start_spots.py`: Microbenchmarks fused sparse CFR warm-start total time and substeps from evenly balanced saved spots.
- `probe_cfr_fp_precision.py`: Probes fp32 precision loss in fused sparse CFR average-policy/value updates from saved spots and checkpoints.
- `test_survey_runner.py`: Runs pytest files or node ids with per-invocation timeouts and records structured audit results.

### Subdirectories
There are no child source directories.
