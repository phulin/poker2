# P2: Fast NN-Based HUNL Solver

P2 is a fast, neural-network-driven solver for Heads-Up No-Limit Texas Hold'em (HUNL). The current training path focuses on ReBeL-style public-belief-state CFR with high-throughput tensorized environments.

## Project Goals
- Train strong HUNL policies quickly with tensorized environments and efficient ReBeL data generation.
- Support current MLP/TRM ReBeL model families under Hydra configs.
- Run staged postflop curriculum, preflop bucket specialist training, and main ReBeL CFR training from one ReBeL-first config surface.
- Keep experiments reproducible via Hydra configs, resolved-config artifacts, W&B metadata, and checkpoint management.

## Main Training Entry Points
- `src/p2/cli/train_rebel.py`: ReBeL-style CFR training with search supervision. Used for MLP/TRM models. Uses search to the end of the street with neural value function approximation at the cutoff. Represents games as public belief states (2x1326 range vectors).
- `src/p2/cli/train_rebel_curriculum.py`: Staged postflop ReBeL curriculum runner.
- `src/p2/cli/train_rebel_preflop_buckets.py`: Preflop bucket specialist training and distillation.
- `src/p2/cli/train_kbest.py`: Legacy PPO self-play with K-Best/DReD opponent pools.

## Current Model Families
1. **MLP (BetterFFN)**: Flat feature encoders with feed-forward policy/value heads for CFR supervision.
2. **TRM (BetterTRM)**: Recursive trunk model with iterative refinement for CFR-based training.

Hydra-based configuration lives in `conf/`. ReBeL entry points default to explicit `config_rebel_*` files, and `model.name` selects the current ReBeL architecture.

## Quickstart (Training)
```bash
# ReBeL CFR training (train_rebel defaults to config_rebel_cfr.yaml)
uv run python -m p2.cli.train_rebel

# Staged postflop ReBeL curriculum
uv run python -m p2.cli.train_rebel_curriculum --config-name=config_rebel_curriculum_postflop

# Preflop backward-induction bucket stages
uv run python -m p2.cli.train_rebel_preflop_buckets \
  preflop_buckets.state_dataset=/path/to/states \
  preflop_buckets.base_checkpoint=/path/to/base.pt

# Legacy PPO self-play, kept outside the current ReBeL config surface
uv run python src/p2/cli/train_kbest.py --config-name=config_transformer
```

## Repository Structure
- `src/p2/`: Core library (envs, models, RL trainers, CFR/search, CLI).
- `conf/`: Hydra configs for training and model variants.
- `docs/`: Architecture plans and design notes.
- `scripts/`: Benchmarks, profiling, and conversion utilities.
- `tests/`: Unit/integration tests.

## Additional Docs
- `src/p2/K_BEST_README.md`: K-Best self-play design and usage.
- `conf/README.md`: Config catalog and override examples.
- `docs/multiway_rebel_plan.md`: Plan for multiway ReBeL/CFR training, sampled-depth value targets, and model changes.
