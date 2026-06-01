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
- `config_rebel_cfr.yaml`: Main ReBeL CFR training configuration, including the per-depth sparse search bet schedule and preflop all-in table path.
- `config_rebel_curriculum_river.yaml`: Initial postflop curriculum config for the implemented live random-river `S_river` train stage.
- `config_rebel_curriculum_turn.yaml`: Initial postflop curriculum config for the live random-turn `S_turn` train stage placeholder, pending `E_turn` leaf routing.
- `config_rebel_curriculum_flop.yaml`: Initial postflop curriculum config for the live random-flop `S_flop` train stage placeholder, pending `E_flop` leaf routing.
- `config_rebel_debug.yaml`: Faster ReBeL debug configuration.

### Subdirectories
- `allin/`: Hydra defaults for standalone preflop all-in equity model training.
