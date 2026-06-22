# P2 Hydra Configs

The current training configs use explicit `config_rebel_*` files. ReBeL entry
points should use the shared loader in
`p2.config.rebel_load`, which applies ReBeL defaults, rejects legacy PPO/K-best
top-level fields, and projects the resolved payload into the typed ReBeL config
view used for artifacts and W&B.

## Current ReBeL Configs

- `config_rebel_cfr.yaml`: Main live ReBeL CFR training config.
- `config_rebel_debug.yaml`: Small ReBeL debug config.
- `config_rebel_curriculum_postflop.yaml`: Full river-to-turn-to-flop staged postflop curriculum.
- `config_rebel_curriculum_river.yaml`: River-only staged curriculum.
- `config_rebel_curriculum_turn.yaml`: Turn staged curriculum with E_turn distillation.
- `config_rebel_curriculum_flop.yaml`: Flop staged curriculum with downstream E_preflop distillation.
- `config_rebel_preflop_buckets.yaml`: Preflop bucket specialist training and distillation.
- `config_rebel_pregenerate_postflop.yaml`: Bounded solved-example pregeneration.
- `config_rebel_evaluate_value_loss.yaml`: Hydra-first checkpoint value-loss evaluation.
- `config_rebel_postflop_hybrid_holdout.yaml`: Live training with pregenerated holdout validation.
- `rebel_hp_trials.yaml`: Trial definitions for the bounded ReBeL HP runner.

`allin/config.yaml` is the standalone preflop all-in equity model config.

## Commands

```bash
# Main ReBeL CFR training; train_rebel defaults to config_rebel_cfr.yaml.
uv run python -m p2.cli.train_rebel

# Explicit main config.
uv run python -m p2.cli.train_rebel --config-name=config_rebel_cfr

# Staged postflop curriculum.
uv run python -m p2.cli.train_rebel_curriculum \
  --config-name=config_rebel_curriculum_postflop

# Preflop bucket specialist training or distillation.
uv run python -m p2.cli.train_rebel_preflop_buckets \
  --config-name=config_rebel_preflop_buckets \
  preflop_buckets.state_dataset=/path/to/states \
  preflop_buckets.base_checkpoint=/path/to/base.pt

# Value-loss evaluation. The checkpoint supplies weights only.
uv run python scripts/evaluate_rebel_value_loss.py \
  --config-name=config_rebel_evaluate_value_loss \
  resume_from=/path/to/rebel_final.pt \
  validation_set.dataset=/path/to/solved_dataset \
  device=cpu

# Standalone all-in equity model.
uv run python -m p2.allin.train
```

## Legacy PPO/K-Best

`config_transformer.yaml` remains for `src/p2/cli/train_kbest.py`. It is not
part of the current ReBeL config surface, and ReBeL loaders reject PPO/K-best
fields such as `opponent_pool_type` and `k_best_pool_size`.
