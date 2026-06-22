# Training Config Refactor Plan

## Diagnosis

The current system has three incompatible configuration surfaces for closely related work:

- Main ReBeL training is Hydra-native and reasonably compact.
- Postflop staged training is Hydra-native but uses ad hoc deep-copy stage mutation and a separate value-only distillation loop.
- Preflop staged buckets are argparse-native, seeded from checkpoint config, then patched with selected CLI args.

This makes experiments hard to reproduce because the source of truth is unclear: YAML, checkpoint config, CLI patching, or in-code defaults may all be active.

## Target Architecture

Create a ReBeL-specific experiment layer with these modules:

- `p2.config.rebel_schema`: Typed current schema for ReBeL training only.
- `p2.config.rebel_load`: Hydra/CLI/checkpoint loading, migration, validation, and frozen resolved config export.
- `p2.runtime.training_run`: Device setup, precision setup, seeding, W&B init, run naming, checkpoint run-id recovery, and summary logging.
- `p2.runtime.stage_runner`: Shared train loop contract for normal train stages, value-only distill stages, preflop bucket solve/train stages, and teacher distill stages.
- `p2.stages.curriculum`: Stage graph parsing, promotion state, resume validation, and source/closing checkpoint resolution.
- `p2.stages.preflop_buckets`: Bucket dataset readers, validation cache, solver/student stage definitions, and preflop-specific metadata.

CLI files should become thin wrappers that select a config and call the shared orchestrator.

## Refactoring Plan

### Phase 1: Normalize ReBeL Config

- Introduce `RebelExperimentConfig` separate from the PPO/K-best `Config`.
- Split top-level concerns into `run`, `checkpoint`, `logging`, `trainer`, `model`, `env`, `data`, `search`, `evaluation`, and `stages`.
- Keep `Config` as a compatibility adapter initially: `Config -> RebelExperimentConfig` for current ReBeL callers.
- Move checkpoint compatibility migrations into `p2.config.checkpoint_migrations`; keep fields like old board interaction names and missing compact-preflop model type out of the current schema.
- Add validation before trainer construction:
  - `data.mode` compatible with selected stage type.
  - `live_root_source` compatible with player count and fused evaluator.
  - compact preflop model requirements are explicit, not inferred by scattered code.
  - all checkpoint paths and promoted net references resolve before a long run starts.

### Phase 2: Make Every Training Mode Hydra-First

- Convert `scripts/preflop_backward_induction.py` into a package entry point, e.g. `src/p2/cli/train_rebel_staged.py`.
- Represent preflop specialists and distillation as stages in YAML:
  - `kind: preflop_bucket_specialist`
  - `kind: preflop_bucket_distill`
  - explicit bucket specs, state dataset, batch sizes, validation cache settings, teacher/source checkpoints, write-solved-shards settings.
- Allow argparse only as a compatibility shim that converts old flags into Hydra overrides, prints the resolved config path/container, and warns it is deprecated.
- Do not seed training config from checkpoints except for checkpoint migration/resume. Checkpoints should provide weights and provenance, not hidden defaults.

### Phase 3: Centralize Runtime/W&B/Checkpointing

- Replace `_init_wandb` copies with a single `TrainingRun` context manager.
- Standardize W&B config payload:
  - `resolved_config`: resolved dataclass/OmegaConf container.
  - `stage`: name/type/index/source checkpoints.
  - `provenance`: git SHA, checkpoint signatures, dataset manifest signatures.
- Standardize W&B run policy:
  - main ReBeL: one run.
  - staged curriculum: one parent group, one run per stage by default, optional single-run mode.
  - preflop buckets: either one run per bucket or one grouped run, but encoded in config.
- Standardize metric keys:
  - `train/loss`, `train/value_loss`, `train/policy_loss`
  - `search/local_exploitability_mbbg`, `search/nodes_per_root`
  - `validation/...`
  - `stage/<stage_name>/...` only when multiple stages share one W&B run.
- Centralize checkpoint naming and metadata:
  - `latest.pt`, `step_{n}.pt`, `final.pt` or keep `rebel_*` names behind a configurable policy.
  - consistent `metadata.stage`, `metadata.kind`, `metadata.source_checkpoints`, `metadata.resolved_config_hash`, `metadata.wandb_run_id`.

### Phase 4: Unify Stage Execution

- Define a common stage interface:
  - `prepare(context) -> StageState`
  - `step(state, step) -> metrics`
  - `save(state, checkpoint_policy)`
  - `promote(state) -> artifact`
- Implement existing stage types behind the interface:
  - live ReBeL train stage using current `RebelCFRTrainer.train_step`.
  - postflop end-of-street value distill stage.
  - preflop bucket specialist solve/train stage.
  - preflop teacher distill stage.
  - pregenerated HP/holdout stage if needed.
- Reuse the same checkpoint/log/validation cadence for all stage types.
- Move curriculum promotion state out of the CLI file and into a reusable stage graph state store.

### Phase 5: Clean Legacy Config

- Create a field inventory with status: current, compatibility, deprecated, remove.
- First removals/quarantine:
  - PPO/K-best-only fields from the ReBeL schema: `opponent_pool_type`, `k_best_pool_size`, `exploiter`, PPO loss knobs, KL beta controller, LR scaling controller, `offload_opponent_models`.
  - Old model aliases handled only by checkpoint migration: `rank_board_interaction_dim`, `suit_board_interaction_dim`, `policy_factor_scale`, missing `preflop_model_type`.
  - Ambiguous global fields that should be scoped: `num_steps`, `num_envs`, `checkpoint_interval`, `wandb_name`, `resume_from`.
- Keep real active ReBeL/search fields, but group them by evaluator concern: action schedule, CFR variant/schedules, warm start, leaf model/closing model, all-in terminal abstraction, continuation target sampling.

### Phase 6: Migration and Safety

- Add config resolution tests for all shipped YAML files.
- Add golden tests that old argparse preflop commands map to the same resolved staged config.
- Add smoke tests for:
  - main ReBeL debug config.
  - one river curriculum train stage.
  - one value-only distill step.
  - one tiny preflop bucket specialist pass with a fixture dataset.
- Preserve old entry points for one transition window with deprecation warnings.
- Update `AGENTS.md` and `conf/README.md` with the new directory/source summaries.

## Preferred End State

One canonical command family:

```bash
uv run python -m p2.cli.train_rebel config=config_rebel/main
uv run python -m p2.cli.train_rebel config=config_rebel/postflop_curriculum
uv run python -m p2.cli.train_rebel config=config_rebel/preflop_buckets
```

All three commands should produce a resolved config artifact, consistent W&B payloads, consistent checkpoints, and explicit stage metadata.
