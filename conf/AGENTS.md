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
- `config_rebel_cfr.yaml`: Main ReBeL CFR training configuration, including data mode selection, per-depth sparse search bet schedule, and preflop all-in table path.
- `config_rebel_curriculum_river.yaml`: Initial postflop curriculum config for the implemented live random-river `S_river` train stage with legacy pre-chance value augmentation disabled.
- `config_rebel_curriculum_postflop.yaml`: Full fixed-schedule postflop curriculum config that runs river, turn, flop, and preflop-handoff substeps in one ordered orchestrator pass with per-train-stage live root overrides and legacy pre-chance value augmentation disabled.
- `config_rebel_curriculum_turn.yaml`: Postflop curriculum config for `distill_E_turn` from promoted `S_river`, then live random-turn `S_turn` training with `E_turn` closing leaves and legacy pre-chance value augmentation disabled.
- `config_rebel_curriculum_flop.yaml`: Postflop curriculum config for `distill_E_flop` from promoted `S_turn`, live random-flop `S_flop` training with `E_flop` closing leaves, then `distill_E_preflop`, with legacy pre-chance value augmentation disabled.
- `config_rebel_pregenerate_postflop.yaml`: Bounded postflop solved-example pregeneration config for HP sweeps/holdouts using live random postflop CFR roots with optional compressed float storage and legacy pre-chance value augmentation disabled.
- `config_rebel_postflop_hybrid_holdout.yaml`: Live random postflop training config wired to a fixed bounded pregenerated holdout dataset for fresh validation metrics, with legacy pre-chance value augmentation disabled.
- `config_rebel_debug.yaml`: Faster ReBeL debug configuration.

### Subdirectories
- `allin/`: Hydra defaults for standalone preflop all-in equity model training.
