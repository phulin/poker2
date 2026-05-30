## Directory summary
Python unit and integration tests for environments, rules, models, training utilities, CFR/search, ReBeL data generation, and RL support.

### Source files
- `test_*env*.py`, `test_rules*.py`, `test_card_utils.py`, `test_legal_actions.py`, `test_winner_reward_sign.py`: Environment, rule, reward, and card-combo coverage.
- `test_multiway_env.py`: Scalar multiway NLHE, vectorized PBS, and public-state parity coverage.
- `test_triton_pbs_env.py`: CUDA/Triton PBS parity against the PyTorch PBSEnv implementation.
- `test_showdown_package.py`: Package export and explicit triangle-weight checks for reusable showdown evaluators.
- `test_showdown_per_hand_equity.py`: Per-hand exact, A+xB, tiered approximation, and blocked-hand showdown equity vector coverage.
- `test_allin_equity_model.py`: Preflop all-in random batch generation, browser-friendly LeakyReLU/RMSNorm model shape checks, and small Monte Carlo target sampler smoke coverage.
- `test_*cfr*.py`, `test_allin_payoff.py`, `test_chance_node_helper.py`, `test_rebel_data_generator.py`, `test_rebel_pipeline.py`, `test_sparse_cfr_evaluator.py`, `test_high_exploitability_save.py`: Sparse CFR/search, all-in payoff kernels, and ReBeL pipeline coverage.
- `test_model_*.py`, `test_transformer_model.py`, `test_mlp_features.py`, `test_structured_embedding_data.py`, `test_kv_caching.py`, `test_encoders.py`, `test_activation_utils.py`, `test_state_encoder_perspective.py`: Model, encoder, and feature coverage.
- `test_losses.py`, `test_rl.py`, `test_rebel_replay.py`, `test_rebel_batch.py`, `test_vectorized_replay.py`, `test_kbest*.py`, `test_dred_pool.py`, `test_kmedoids.py`, `test_elo_calculator.py`: RL, replay, loss, opponent pool, and rating coverage.
- `test_trueskill_tracker.py`: TrueSkill snapshot evaluator opponent sampling, game-budget sizing, and public-belief river payoff batching coverage.
- `test_ema*.py`, `test_model_context.py`, `test_model_utils.py`, `test_training_utils.py`, `test_trainer_config_build.py`, `test_schedules.py`, `test_mps_autocast.py`, `test_alignment.py`, `test_bins_configurable.py`: Utility, config, schedule, and platform coverage.

### Subdirectories
There are no child source directories.
