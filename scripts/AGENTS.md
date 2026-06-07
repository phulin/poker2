## Directory summary
Reusable profiling and benchmark scripts that are broader than a single repro but are not package modules.

### Source files
- `profile_train_rebel.py`: Profiles the ReBeL training loop.
- `profile_showdown_kernel.py`: Profiles showdown kernel behavior.
- `bench_showdown.py`: Benchmarks showdown evaluation.
- `bench_showdown_tiers.py`: Runs interleaved tier 1-3 multiway showdown benchmarks with optional SIGSTOP/SIGCONT pausing of a live training process and all-sample plus steady-state timing summaries.
- `bench_parent_expand.py`: Benchmarks parent expansion behavior.
- `bench_breakdown.py`: Produces benchmark breakdowns for selected paths.
- `bench_model_share.py`: Benchmarks model sharing or reuse behavior.
- `bench_cfr_main_path.py`: Runs realistic ReBeL CFR train-step source profiling and component microbenchmarks with optional train_rebel pause/resume handling and configurable starting CFR iteration for post-delay kernel timing.
- `bench_write_children_kernel.py`: Replays real fused-subgame child-writer inputs to compare legacy and optimized `write_children_same_street` Triton kernels with CUDA events.
- `bench_cfr_init_spots.py`: Microbenchmarks fused sparse CFR subgame initialization and init-time tensor expansions from evenly balanced saved spots.
- `bench_cfr_iterator_spots.py`: Profiles eager sparse/fused-sparse CFR iterator segments and isolated CFR components from evenly balanced saved spots across preflop/flop/turn/river roots.
- `bench_warm_start_spots.py`: Microbenchmarks fused sparse CFR warm-start total time and substeps from evenly balanced saved spots.
- `bench_turn_allin_values.py`: Microbenchmarks random-turn fused sparse CFR all-in-call leaf value writeback, comparing legacy per-node turn payoff dot kernels with grouped turn-board fused kernels across block-size candidates.
- `run_rebel_hp_bandit.py`: Runs bounded pregenerated ReBeL hyperparameter trials with a simple UCB multi-armed-bandit allocation over architecture, LR, and schedule candidates.
- `precompute_preflop_allin_table.py`: Streams preflop five-card boards to build a zstd-compressed int16 `[1326, 1326]` all-in matchup payoff table for preflop all-in-call terminal abstraction.
- `probe_cfr_fp_precision.py`: Probes fp32 precision loss in fused sparse CFR average-policy/value updates from saved spots and checkpoints.
- `river_cfr_exploitability_trajectory.py`: Samples random river roots, runs sparse/fused CFR, and logs per-spot exploitability trajectories for convergence diagnostics with warm-start, CFR-type, and delayed DCFR-hybrid PCFR/SAPCFR override flags.
- `diagnose_river_convergence.py`: Compares train-path and eager river CFR evaluation details, including leaf composition and value-source diagnostics.
- `independent_river_exploitability.py`: River CFR exploitability diagnostic harness; prints evaluator-consistent exact-final exploitability and a legacy from-scratch checker that is not authoritative for PBS terminal values.
- `evaluate_rebel_value_loss.py`: Evaluates a promoted ReBeL checkpoint's value head on a solved dataset value stream and reports weighted supervised value loss.
- `eturn_distill_mab_search.py`: Runs sequential W&B-enabled MAB-style E_turn distillation hyperparameter trials, including initial and follow-up presets, while recording planned/running experiments to YAML and results to JSONL.
- `eflop_distill_lr_sweep.py`: Runs fixed sequential W&B-enabled E_flop distillation LR sweeps from a promoted S_turn checkpoint, overriding curriculum substep train overrides and recording logs/results.
- `test_survey_runner.py`: Runs pytest files or node ids with per-invocation timeouts and records structured audit results.

### Subdirectories
There are no child source directories.
