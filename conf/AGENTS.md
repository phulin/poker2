## Directory summary
Hydra configuration files for ReBeL training, staged curricula, preflop buckets, postflop pregeneration/evaluation, and legacy transformer PPO/K-best training.

### Source files
- `README.md`: Configuration catalog, usage examples, and parameter reference.
- `config_transformer.yaml`: Legacy transformer PPO/K-best training configuration for `train_kbest.py`.
- `config_rebel_cfr.yaml`: Main ReBeL CFR training configuration.
- `config_rebel_curriculum_flop.yaml`: Flop-stage postflop curriculum configuration.
- `config_rebel_curriculum_postflop.yaml`: Full postflop curriculum orchestration configuration.
- `config_rebel_curriculum_river.yaml`: River-stage postflop curriculum configuration.
- `config_rebel_curriculum_turn.yaml`: Turn-stage postflop curriculum configuration.
- `config_rebel_debug.yaml`: Faster ReBeL debug configuration.
- `config_rebel_evaluate_value_loss.yaml`: Hydra-first value-loss evaluation configuration.
- `config_rebel_postflop_hybrid_holdout.yaml`: Postflop hybrid live/holdout training configuration.
- `config_rebel_preflop_buckets.yaml`: Preflop backward-induction bucket training configuration.
- `config_rebel_pregenerate_postflop.yaml`: Bounded postflop solved-example pregeneration configuration.
- `rebel_hp_trials.yaml`: Trial specifications for pregenerated ReBeL hyperparameter runs.

### Subdirectories
- `allin/`: Hydra defaults for standalone preflop all-in equity model training.
