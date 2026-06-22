# P2: Fast NN-Based HUNL Solver

P2 is a fast, neural-network-driven solver for Heads-Up No-Limit Texas Hold'em (HUNL). The codebase focuses on high-throughput training loops, tensorized environments, and multiple model families for self-play and CFR-style search.

## Project Goals
- Train strong HUNL policies quickly with tensorized environments and efficient data generation.
- Support multiple neural architectures (CNN, transformer, MLP, TRM) under a unified config system.
- Provide both RL self-play (PPO + K-best/DReD pools) and ReBeL-style CFR training pipelines.
- Keep experiments reproducible via Hydra configs and checkpoint management.

## Main Training Entry Points
- `src/p2/cli/train_kbest.py`: PPO self-play with K-Best/DReD opponent pools. Used for CNN and transformer models. Model returns a policy (no search) at any given game state.
- `src/p2/cli/train_rebel.py`: ReBeL-style CFR training with search supervision. Used for MLP/TRM models. Uses search to the end of the street with neural value function approximation at the cutoff. Represents games as public belief states (2x1326 range vectors).

## Model Architectures (4 Families)
1. **CNN (SiameseConvNetV1)**: Convolutional encoders over card/action tensors for fast self-play.
2. **Transformer (PokerTransformerV1)**: Sequence model over action histories and public state for richer temporal context.
3. **MLP (RebelFFN / BetterFFN)**: Flat feature encoders with feed-forward heads for CFR supervision.
4. **TRM (BetterTRM)**: Recursive trunk model with iterative refinement for CFR-based training.

Hydra-based configuration lives in `conf/`, and `model.name` selects the architecture (see `conf/README.md`).

## Quickstart (Training)
```bash
# PPO self-play (CNN or transformer, via Hydra configs)
uv run python src/p2/cli/train_kbest.py --config-name=config
uv run python src/p2/cli/train_kbest.py --config-name=config_transformer

# ReBeL CFR training (MLP/TRM)
uv run python -m p2.cli.train_rebel --config-name=config_rebel_cfr

# Staged postflop ReBeL curriculum
uv run python -m p2.cli.train_rebel_curriculum --config-name=config_rebel_curriculum_postflop

# Preflop backward-induction bucket stages
uv run python -m p2.cli.train_rebel_preflop_buckets \
  preflop_buckets.state_dataset=/path/to/states \
  preflop_buckets.base_checkpoint=/path/to/base.pt
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
