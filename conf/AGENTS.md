## Directory summary
Hydra configuration files for PPO/K-best training, transformer variants, and ReBeL CFR training.

### Source files
- `README.md`: Configuration catalog, usage examples, and parameter reference.
- `config.yaml`: Default balanced PPO/K-best configuration.
- `config_fast.yaml`: Small CPU-oriented development configuration.
- `config_high_perf.yaml`: High-performance CUDA PPO/K-best configuration.
- `config_transformer.yaml`: Transformer PPO training configuration.
- `config_transformer_hp.yaml`: High-performance transformer configuration.
- `config_transformer_cfr.yaml`: Transformer configuration with CFR-related settings.
- `config_rebel_cfr.yaml`: Main ReBeL CFR training configuration, including data mode selection, per-depth sparse search bet schedule, warm-start seed tuning, CFR/predictive-CFR variant selection, actor-backup consistency loss defaults, and preflop all-in table path.
- `rebel_hp_trials.yaml`: YAML trial specifications for the pregenerated ReBeL HP runner; each trial's `params` mapping can contain scalar values or arrays that expand as a local grid.
- `config_rebel_curriculum_river.yaml`: Initial postflop curriculum config for the implemented live random-river `S_river` train stage.
- `config_rebel_curriculum_postflop.yaml`: Full fixed-schedule postflop curriculum config that runs river, turn, flop, and compact preflop-handoff substeps in one ordered orchestrator pass with per-train-stage live root overrides, S_i policy warm-starts from S_{i+1}, tuned 0.08 linear LR / 1024 batch distillation overrides, compact `E_preflop` model overrides and validation.
- `config_rebel_curriculum_turn.yaml`: Postflop curriculum config for value-only `distill_E_turn` from promoted `S_river`, then live random-turn `S_turn` training with policy initialized from `S_river`, `E_turn` closing leaves, and tuned 0.08 linear LR / 1024 batch distillation overrides.
- `config_rebel_curriculum_flop.yaml`: Postflop curriculum config for value-only `distill_E_flop` from promoted `S_turn`, live random-flop `S_flop` training with policy initialized from `S_turn` and `E_flop` closing leaves, then compact value-only `distill_E_preflop` with 169-hand model overrides and validation, and tuned 0.08 linear LR / 1024 batch distillation overrides.
- `config_rebel_pregenerate_postflop.yaml`: Bounded postflop solved-example pregeneration config for HP sweeps/holdouts using live random postflop CFR roots, sparse-fused search by default, and optional compressed float storage.
- `config_rebel_evaluate_value_loss.yaml`: Hydra-first value-loss evaluation config; `resume_from` supplies the checkpoint and `validation_set.dataset` supplies the solved dataset while model/runtime settings come from current config.
- `config_rebel_postflop_hybrid_holdout.yaml`: Live random postflop training config wired to a fixed bounded pregenerated holdout dataset for fresh validation metrics.
- `config_rebel_preflop_buckets.yaml`: Hydra-first preflop backward-induction bucket specialist/distillation config; checkpoints provide weights only while run settings live under `preflop_buckets`.
- `config_rebel_debug.yaml`: Faster ReBeL debug configuration.

### Subdirectories
- `allin/`: Hydra defaults for standalone preflop all-in equity model training.
