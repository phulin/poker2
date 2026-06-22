## Directory summary
Reusable profiling and benchmark scripts that are broader than a single repro but are not package modules. Scripts that compose ReBeL Hydra configs should use the shared ReBeL config loader rather than parsing directly into `Config`.

### Source files
- `profile_train_rebel.py`: Profiles the ReBeL training loop.
- `profile_showdown_kernel.py`: Profiles showdown kernel behavior.
- `bench_showdown.py`: Benchmarks showdown evaluation.
- `bench_showdown_tiers.py`: Runs interleaved tier 1-3 multiway showdown benchmarks with optional SIGSTOP/SIGCONT pausing of a live training process and all-sample plus steady-state timing summaries.
- `bench_parent_expand.py`: Benchmarks parent expansion behavior.
- `bench_breakdown.py`: Produces benchmark breakdowns for selected paths.
- `bench_model_share.py`: Benchmarks model sharing or reuse behavior.
- `bench_cfr_main_path.py`: Runs realistic ReBeL CFR train-step source profiling and component microbenchmarks with optional train_rebel pause/resume handling and configurable starting CFR iteration for post-delay kernel timing.
- `bench_kernel_profile.py`: Profiles the live fused preflop CFR training configuration with PyTorch CPU/CUDA profiler, exports a Chrome trace, and writes per-kernel self-time JSON summaries.
- `bench_write_children_kernel.py`: Replays real fused-subgame child-writer inputs to compare legacy and optimized `write_children_same_street` Triton kernels with CUDA events.
- `bench_cfr_init_spots.py`: Microbenchmarks fused sparse CFR subgame initialization and init-time tensor expansions from evenly balanced saved spots.
- `bench_cfr_iterator_spots.py`: Profiles eager sparse/fused-sparse CFR iterator segments and isolated CFR components from evenly balanced saved spots across preflop/flop/turn/river roots.
- `bench_warm_start_spots.py`: Microbenchmarks fused sparse CFR warm-start total time and substeps from evenly balanced saved spots.
- `bench_turn_allin_values.py`: Microbenchmarks random-turn fused sparse CFR all-in-call leaf value writeback, comparing legacy per-node turn payoff dot kernels with grouped turn-board fused kernels across block-size candidates.
- `run_rebel_hp_bandit.py`: Runs bounded pregenerated ReBeL hyperparameter trials with a simple UCB multi-armed-bandit allocation over architecture, LR, and schedule candidates.
- `precompute_preflop_allin_table.py`: Streams preflop five-card boards to build a zstd-compressed int16 `[1326, 1326]` all-in matchup payoff table for preflop all-in-call terminal abstraction.
- `precompute_board_ranks.py`: Streams suit-canonical five-card board representatives, ranks all 1326 private hands per board, converts comparable scores to dense per-board `uint16` rank ids, and writes a raw memmap plus JSON metadata for all-in sampler rank-cache experiments.
- `convert_allin_dataset_to_169.py`: Converts pregenerated all-in datasets from 1326-combo beliefs/targets to native 169-class tensors, preserving shard boundaries and writing updated manifest metadata.
- `probe_cfr_fp_precision.py`: Probes fp32 precision loss in fused sparse CFR average-policy/value updates from saved spots using current ReBeL Hydra config plus checkpoint weights.
- `river_cfr_exploitability_trajectory.py`: Samples random river roots using current ReBeL Hydra config plus checkpoint weights, runs sparse/fused CFR, and logs per-spot exploitability trajectories for convergence diagnostics with warm-start, CFR-type, and delayed DCFR-hybrid PCFR/SAPCFR override flags.
- `diagnose_river_convergence.py`: Compares train-path and eager river CFR evaluation details using current ReBeL Hydra config plus checkpoint weights, including leaf composition and value-source diagnostics.
- `independent_river_exploitability.py`: River CFR exploitability diagnostic harness using current ReBeL Hydra config plus checkpoint weights; prints evaluator-consistent exact-final exploitability and a legacy from-scratch checker that is not authoritative for PBS terminal values.
- `evaluate_rebel_value_loss.py`: Hydra-first evaluator for a ReBeL checkpoint value head on a solved dataset value stream; `resume_from` provides weights and `validation_set.dataset` provides data, so checkpoint-embedded config is ignored.
- `pregenerate_preflop_policy_states.py`: Rolls compact 169-hand multiplayer preflop PBSEnv states forward with a checkpointed policy head, Bayes-updates actor beliefs after sampled legal actions, and writes compact sharded public-state tensors plus a manifest; optional stratified frontier mode balances action-depth buckets, and unique-frontier mode retries stochastic continuations while saving at most one successor frontier per root, with a streaming variant for large root counts and capped per-frontier writes.
- `pack_preflop_state_buckets.py`: Rewrites pregenerated preflop public-state bucket datasets into one `states.pt` shard per bucket while preserving manifest metadata for full-bucket DataLoader shuffling.
- `sweep_preflop_value_lr_bs.py`: Trains one value-only epoch over a pregenerated preflop solved dataset for each learning-rate/batch-size pair, evaluates against a fixed validation cache, writes per-trial JSON results, and saves the best checkpoint.
- `preflop_backward_induction.py`: Thin Hydra wrapper that delegates to `p2.cli.train_rebel_preflop_buckets`; preflop bucket settings now come from `conf/config_rebel_preflop_buckets.yaml` and overrides.
- `eturn_distill_mab_search.py`: Runs sequential W&B-enabled MAB-style E_turn distillation hyperparameter trials, including initial and follow-up presets, while recording planned/running experiments to YAML and results to JSONL.
- `eflop_distill_lr_sweep.py`: Runs fixed sequential W&B-enabled E_flop distillation LR sweeps from a promoted S_turn checkpoint, overriding curriculum substep train overrides and recording logs/results.
- `test_survey_runner.py`: Runs pytest files or node ids with per-invocation timeouts and records structured audit results.

### Subdirectories
There are no child source directories.
