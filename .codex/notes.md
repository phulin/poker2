# Notes: Training Config Rationalization Review

## Scope
- Main ReBeL Hydra training: `src/p2/cli/train_rebel.py`, `conf/config_rebel_cfr.yaml`.
- Postflop staged curriculum: `src/p2/cli/train_rebel_curriculum.py`, `conf/config_rebel_curriculum*.yaml`, `src/p2/rl/rebel_curriculum.py`.
- Preflop staged buckets: `scripts/preflop_backward_induction.py`, packed bucket scripts/config-adjacent args.

## Findings

### Entry-Point Split
- Main ReBeL uses Hydra via `src/p2/cli/train_rebel.py` and config files under `conf/`.
- Postflop staged training uses Hydra but mutates a deep-copied global `Config` per stage in `src/p2/cli/train_rebel_curriculum.py`.
- Preflop staged bucket training uses standalone `argparse` in `scripts/preflop_backward_induction.py`.
- Preflop bucket config is loaded from a checkpoint, then patched with selected CLI fields. Unpatched fields silently inherit from the checkpoint.

### Lifecycle Duplication
- Device setup, matmul precision, seed setup, W&B init, model parameter summaries, checkpoint metadata, run-id handling, and run loops are spread across main ReBeL, curriculum, preflop buckets, and HP scripts.
- `run_training_loop` covers normal train stages only. Curriculum distill and preflop bucket loops reimplement logging/checkpoint cadence.

### WandB Inconsistency
- Main ReBeL logs full `asdict(cfg)` and recovers W&B run id from checkpoint.
- Curriculum reuses main `_init_wandb` but creates a new run per substep, using stage-specific names/groups.
- Preflop buckets log both `args` and `trainer_config`, do not use the same run-id resume logic, and use metric key namespaces unlike the main loop.

### Config Schema Issues
- `Config` combines PPO/K-best settings, ReBeL search, pregenerated data, curriculum, validation, all-in, TrueSkill, and checkpoint/logging fields.
- `TrainingConfig` includes PPO-era fields (`gamma`, `gae_lambda`, `ppo_*`, KL controllers, LR scaling) and ReBeL-specific fields in the same namespace.
- `Config.from_dict` has compatibility migrations for old model fields and special handling for direct `curriculum.<stage>` keys.
- Curriculum YAML uses both strict-looking dataclass config and permissive direct stage keys, plus scalar helper fields not represented in `CurriculumConfig`.

### Legacy/Compatibility Fields
- Some compatibility fields are still needed for old checkpoints (`legacy_context_features`, old board interaction names, missing preflop model type).
- These should move into explicit checkpoint migration code instead of remaining first-class current experiment settings.
