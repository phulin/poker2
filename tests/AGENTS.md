## Directory summary
Python unit and integration tests for environments, rules, models, training utilities, CFR/search, ReBeL data generation, and RL support.

### Source files
- `test_*env*.py`, `test_rules*.py`, `test_card_utils.py`, `test_legal_actions.py`, `test_winner_reward_sign.py`: Environment, rule, reward, and card-combo coverage.
- `test_multiway_env.py`: Scalar multiway NLHE, vectorized PBS, and public-state parity coverage.
- `test_triton_pbs_env.py`: CUDA/Triton PBS parity against the PyTorch PBSEnv implementation.
- `test_showdown_package.py`: Package export and explicit triangle-weight checks for reusable showdown evaluators.
- `test_showdown_per_hand_equity.py`: Per-hand exact, A+xB, tiered approximation, and blocked-hand showdown equity vector coverage.
- `test_allin_equity_model.py`: Preflop all-in random batch generation, browser-friendly LeakyReLU/RMSNorm model shape checks, pregenerated dataset helpers, and random/exhaustive all-in target sampler smoke coverage.
- `test_*cfr*.py`, `test_allin_payoff.py`, `test_chance_node_helper.py`, `test_rebel_data_generator.py`, `test_rebel_pipeline.py`, `test_sparse_cfr_evaluator.py`, `test_high_exploitability_save.py`: Sparse CFR/search, all-in payoff kernels, and ReBeL pipeline coverage.
- `test_end_of_street_distillation.py`: End-of-street `E_X` distillation target-batch coverage for pre-chance features and chance-mode validation.
- `test_postflop_spot_sampler.py`: Random postflop public-root sampler invariants for flop/turn/river street-start and legal-prefix roots, closed-street chance-target roots with randomized pre-chance beliefs, board texture stratification, randomized board-legal beliefs, and conservative pot/SPR templates.
- `test_rebel_data_source.py`: ReBeL data-source abstraction coverage for live generator-backed, pregenerated solved-dataset-backed, and hybrid live-plus-holdout training data, including pregenerated step windows, shuffled sampling, direct dataset sampling, and checkpoint manifest-state validation.
- `test_rebel_solved_dataset.py`: Tensor-only solved ReBeL dataset serialization, manifest validation and street/depth/target-source/root-source plus leaf-source coverage counts, wrapped reads, random sampling, compressed float storage, and async prefetch coverage.
- `test_pregenerate_postflop_rebel.py`: Bounded postflop pregeneration CLI coverage for live-mode validation, trimmed solved-batch writer calls, per-row root-source tags, root-street metadata, feature-encoder metadata, quality metadata, and closing-leaf/distillation-source checkpoint provenance.
- `test_train_rebel_curriculum.py`: Curriculum CLI orchestration coverage for train/distill substep routing, per-substep checkpoint dirs, promotion state, metadata, S_i policy warm-starts, value-only E_i ownership, and validated substep-aware resume.
- `test_model_*.py`, `test_transformer_model.py`, `test_mlp_features.py`, `test_structured_embedding_data.py`, `test_kv_caching.py`, `test_encoders.py`, `test_activation_utils.py`, `test_state_encoder_perspective.py`: Model, encoder, and feature coverage.
- `test_street_model_registry.py`: Runtime street-to-net dispatch coverage for promoted postflop model registries.
- `test_losses.py`, `test_rl.py`, `test_rebel_replay.py`, `test_rebel_batch.py`, `test_vectorized_replay.py`, `test_kbest*.py`, `test_dred_pool.py`, `test_kmedoids.py`, `test_elo_calculator.py`: RL, replay, loss, opponent pool, and rating coverage.
- `test_rebel_loop.py`: Shared ReBeL loop runner control-flow coverage for checkpoints, stats, optional preflop analyzer printing, final save, and TrueSkill snapshots without invoking real CFR.
- `test_trueskill_tracker.py`: TrueSkill snapshot evaluator opponent sampling, game-budget sizing, and public-belief river payoff batching coverage.
- `test_ema*.py`, `test_model_context.py`, `test_model_utils.py`, `test_training_utils.py`, `test_trainer_config_build.py`, `test_schedules.py`, `test_mps_autocast.py`, `test_alignment.py`, `test_bins_configurable.py`: Utility, config, schedule, and platform coverage.

### Subdirectories
There are no child source directories.
