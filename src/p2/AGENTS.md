## Directory summary
This is the main Python package for P2. It contains the HUNL environments, neural models, RL trainers, CFR/search evaluators, and CLI entry points.

### Source files
- `__init__.py`: Package marker.
- `K_BEST_README.md`: Design and usage notes for K-best self-play.

### Subdirectories
- `allin/`: Preflop all-in equity model, random data generation, Monte Carlo terminal-value target sampler, and standalone training script.
- `cli/`: Hydra and command-line entry points for training, tuning, sampling, and demos.
- `core/`: Shared interfaces and structured configuration dataclasses/enums.
- `encoding/`: Action-bin encoding and legal-action mask helpers.
- `env/`: HUNL game state, tensorized environments, hand rules, and analysis helpers.
- `models/`: Shared model utilities plus CNN, MLP/TRM, and transformer model families.
- `rl/`: PPO/self-play, ReBeL training buffers, opponent pools, losses, and rating helpers.
- `search/`: CFR/DCFR evaluators, public belief states, chance handling, and Triton fused kernels.
- `stages/`: Shared staged-training helpers for ReBeL workflows, including typed preflop bucket configuration.
- `showdown/`: Reusable exact, approximate, and Monte Carlo multiway showdown equity evaluators copied from benchmark prototypes.
- `utils/`: Training, model, config, EMA, KL, profiling, and context-manager utilities.
