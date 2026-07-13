## Directory summary
Reusable profiling, benchmark, data-prep, and diagnostic scripts. ReBeL Hydra scripts should use the shared ReBeL config loader.

### Source files
- `analyze_active_preflop_open_grids.py`: Analyzes preflop open-node policies from an active specialist checkpoint.
- `bench_breakdown.py`: Produces benchmark breakdowns.
- `bench_cfr_init_spots.py`: Benchmarks sparse CFR subgame initialization.
- `bench_cfr_iterator_spots.py`: Profiles sparse CFR iterator segments from saved spots.
- `bench_cfr_main_path.py`: Profiles ReBeL CFR train, solve, generate, and microbenchmark paths.
- `bench_kernel_profile.py`: Profiles fused preflop CFR kernels with PyTorch profiler.
- `bench_model_share.py`: Benchmarks model sharing or reuse behavior.
- `bench_parent_expand.py`: Benchmarks parent expansion behavior.
- `bench_preflop_evaluate_cfr_loop.py`: Benchmarks production compact preflop CFR solves.
- `bench_preflop_full_loop_profile.py`: Profiles compact preflop CFR full-loop timing.
- `bench_preflop_fused_hotloop.py`: Benchmarks compact preflop fused hot-loop kernels.
- `bench_showdown.py`: Benchmarks showdown evaluation.
- `bench_showdown_tiers.py`: Benchmarks multiway showdown approximation tiers.
- `bench_turn_allin_values.py`: Benchmarks turn all-in leaf value writeback.
- `bench_warm_start_spots.py`: Benchmarks fused sparse CFR warm-starts.
- `bench_write_children_kernel.py`: Benchmarks same-street child-writer kernels.
- `build_eturn_validation_set.py`: Builds fixed E-turn value validation sets by averaging legal river-card `S_river` outputs.
- `check_dynamic_compile_recompiles.py`: Checks dynamic compile recompilation behavior.
- `combine_rebel_solved_datasets.py`: Combines compatible ReBeL solved postflop datasets into one manifest/shard set using symlinks, hardlinks, or copies.
- `convert_allin_dataset_to_169.py`: Converts pregenerated all-in datasets to native 169-class tensors.
- `diagnose_river_convergence.py`: Diagnoses river CFR convergence details.
- `diagnose_river_value_fidelity.py`: Decomposes river value-target variance into canonical strength-bin and card-agnostic scalar components.
- `diagnose_turn_value_fidelity.py`: Fits E-turn range-equity positive/negative baseline coefficients against S-river chance targets.
- `distill_epreflop_6p_live_pair.py`: Runs 6-player compact E-preflop live-pair distillation.
- `eturn_distill_lr_sweep.py`: Runs fixed E-turn distillation learning-rate sweeps.
- `eflop_distill_lr_sweep.py`: Runs E-flop distillation learning-rate sweeps.
- `eturn_distill_mab_search.py`: Runs E-turn distillation bandit searches.
- `evaluate_rebel_value_loss.py`: Evaluates value loss for ReBeL checkpoints.
- `evaluate_rebel_value_loss_from_resolved.py`: Evaluates value loss for ReBeL checkpoints using a run's saved `resolved_config.json` to reconstruct the exact model/config.
- `extract_preflop_street_closed_states.py`: Extracts preflop street-closed public states.
- `fit_sturn_root_equity_baseline.py`: Fits blocker-aware turn-equity baseline coefficients directly to solved S-turn root targets.
- `generate_paired_sturn_targets.py`: Saves random-turn PBS roots and solves identical roots at multiple CFR iteration budgets and solve seeds.
- `independent_river_exploitability.py`: Runs an independent river exploitability diagnostic.
- `launch_sturn_structural_experiments_20260711.sh`: Waits for the disjoint turn holdout, then runs paired-root generation and the structural S-turn sweep.
- `launch_sturn_blocker_replications_20260712.sh`: Replicates baseline and blocker-aware S-turn variants across independent fixed-data seeds.
- `launch_sturn_equity_input_experiments_20260712.sh`: Runs blocker-aware learned turn-equity input-head experiments on fixed turn data.
- `launch_sturn_3epoch_optimizer_sweep_20260712.sh`: Queues three-epoch S-turn LR, schedule, warmup, and batch-size experiments on the fixed pregen dataset.
- `resume_sturn_3epoch_optimizer_sweep_20260712.sh`: Resumes the unfinished cheap S-turn optimizer sweep after completed trials, then runs the batch-size matrix.
- `queue_sturn_3epoch_lr10_20260712.sh`: Waits for the active cheap S-turn sweep, then runs the three-epoch LR 0.01 cosine trial.
- `pack_preflop_state_buckets.py`: Packs preflop public-state bucket datasets.
- `precompute_board_ranks.py`: Precomputes board-rank caches.
- `precompute_preflop_allin_table.py`: Precomputes preflop all-in payoff tables.
- `preflop_8_11_pregen_lr_sweep.py`: Runs preflop 8-11 pregeneration LR sweeps.
- `preflop_backward_induction.py`: Hydra wrapper for preflop bucket training.
- `pregenerate_preflop_policy_states.py`: Pregenerates compact preflop policy states.
- `probe_action_mix_bimodal.py`: Probes bimodality in action mixes.
- `probe_cfr_fp_precision.py`: Probes CFR fp32 precision behavior.
- `probe_eos_in_domain_beliefs.py`: Probes EOS values under in-domain beliefs.
- `profile_preflop_bi_outer_step.py`: Profiles preflop backward-induction outer steps.
- `profile_showdown_kernel.py`: Profiles showdown kernels.
- `profile_train_rebel.py`: Profiles the ReBeL training loop.
- `river_cfr_exploitability_trajectory.py`: Logs river CFR exploitability trajectories.
- `run_rebel_hp_bandit.py`: Runs ReBeL hyperparameter bandit trials.
- `run_sturn_pregen_sweep.py`: Runs controlled 500-step S-turn value experiments on a shared pregenerated turn dataset.
- `smoke_sturn_streaming_epoch_pregen.py`: Simulates one double-buffered S-turn streaming epoch cycle from pregenerated GPU data and checks exact coverage/swap invariants.
- `run_value_arch_proposal.py`: Runs value-only S_river BetterFFN architecture proposal tests on pregenerated postflop data.
- `sample_preflop_continuation_beliefs.py`: Samples preflop continuation belief cascades.
- `shuffle_preflop_street_closed_states.py`: Shuffles preflop street-closed state datasets.
- `sweep_preflop_value_lr_bs.py`: Runs preflop value LR/batch-size sweeps.
- `test_survey_runner.py`: Runs pytest files or node ids with per-invocation timeouts and records structured audit results.
- `time_fixed_value_checkpoints.py`: Times fixed value-architecture checkpoints with GPU-resident pregenerated batches and no-grad 4096-example value forwards.
- `time_value_forward_ablations.py`: Times value-forward ablations by monkey-patching checkpointed value-network components before reduce-overhead compilation.
- `validate_epreflop_6p_live_pair.py`: Validates 6-player compact E-preflop live-pair checkpoints.

### Subdirectories
There are no child source directories.
