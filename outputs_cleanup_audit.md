# Outputs Cleanup Audit

At audit time, `outputs/` occupied about 115 GiB. The approved Tier A, street-copy,
and short-sweep checkpoint cleanup was subsequently executed on 2026-07-13.

## Execution result

- `outputs/`: 115 GiB to 72 GiB.
- Repository: 139 GiB to 96 GiB.
- Filesystem free space: 61 GiB to 104 GiB.
- Filesystem usage: 79% to 64%.
- Composite river dataset: zero broken symlinks after cleanup.
- Protected current datasets, V5 checkpoint chain, and newest validation caches verified present.

## Tier A: Strong deletion candidates (~32 GiB)

### Preflop backward-induction bulk tensors (~15.1 GiB)

Preserve checkpoint directories, especially the referenced V5 chain. Remove bulk tensors that are either completed presolve products or superseded validation generations:

- Every `solved/` directory under `outputs/preflop_backward_induction/`: 7.25 GiB across 12 directories.
- Every embedded `validation/` directory under individual run directories: 3.54 GiB across 14 directories. Keep compact metrics/configs and the canonical cache.
- Older canonical cache generations, retaining the newest per action bucket:
  - Remove action 4-7 hashes `60d821...` and `6ebad4...`; retain `f37c52...`.
  - Remove action 8-11 hashes `2d4dde...` and `1a9b30...`; retain referenced `6bff18...`.
  - Remove action 12-15 hashes `74e58c...`, `d0c68e...`, and `0b306f...`; retain newest `24bc61...`.
- Remove `validation_cache_experiments/` after retaining its experimental conclusions: 332 MiB.

### Completed value-architecture trial checkpoints (~4.16 GiB)

Delete the 98 `.pt` files while preserving JSON, JSONL, Markdown, and timing results in:

- `outputs/value_arch_proposals_500step_20260630/`
- `outputs/value_arch_compound_500step_20260703/`
- `outputs/value_arch_proposals_100step_20260630/`
- `outputs/value_arch_perhand_500step_20260703/`
- `outputs/value_arch_canonical_only_500step_20260703/`
- `outputs/value_arch_generalization_2k_20260703/`

These are 100-2,000-step architecture probes, not long-running restart checkpoints. Their results are already summarized in repository Markdown.

### Old E-preflop trials (~3.36 GiB)

Under `outputs/epreflop_6p_live_pair/`, delete the 34 old `smoke_*`, `timing_*`, `hp_*`, and `sweep_*` directories. Preserve:

- `full_b16384_s5260_muon2e-2_adamw2e-3_warmup10_linear_shuffled_per_player_val/` because current configs load its final checkpoint.
- `validation/` because current validation scripts use it.

### Superseded preflop policy-state data (~3.62 GiB)

- Delete `eroymcd2_unique_buckets_20m_n5_cap5m_20260621/` (2.49 GiB): the packed successor is self-contained and is the version current configs/scripts use.
- Delete the unreferenced old rollout and stratified-rollout datasets (about 1.13 GiB).
- Preserve `eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622/`.

### Old postflop river targets, retries, and smoke data (~4.10 GiB)

Delete the outdated context-15 and superseded experimental data:

- `river_value_100k_300it/` (2.1 GiB)
- `river_value_400steps_409600_300it_depth5_sapcfr_20260630/` (1.1 GiB; non-pred40 variant)
- `river_value_100k/` (274 MiB)
- Old non-FP32 turn validation (86 MiB)
- `river_val_8192_10k_sapdcfr_2000` and retry1/retry2 (about 223 MiB total); retain the referenced `nowarm_ctx41` validation set.
- All `rebel_postflop/*smoke*` directories (about 473 MiB).

Do not remove the 100-step + pred40 + composed 500-step chain.

### Superseded all-in variants (~1.8 GiB)

Delete older small-batch training datasets, superseded/noise validation sets, and the explicit smoke checkpoint:

- `allin_training_data_169_p6_min4_2048000_s65536_b8192_bc512/`
- `allin_training_data_169_p6_min4_2048000_s16384_b8192_bc512/`
- `allin_validation_data_8192_s1048576_b131072_bc64/`
- `allin_validation_noise_s2097152/`
- `allin_validation_data_8192_s4194304_b524288_bc64/`
- `checkpoints_allin_169_smoke/`

Retain the currently configured allboards validation data, transformer checkpoint, and main `checkpoints_allin/` tree. Treat the non-smoke 169 P6 checkpoint as Tier B rather than deleting it automatically.

### Miscellaneous disposable output (<0.5 GiB)

Delete top-level `_tmp_*`, old smoke/profile directories, `experiments/preflop_smoke_dryrun_checkpoints/`, and empty/failed Hydra output shells. All 182 top-level directories smaller than 10 MiB total only about 84 MiB, so this is primarily hygiene.

## Tier B: Valuable space, but requires a retention decision (~27 GiB)

### Original unshuffled street-closed dataset (3.31 GiB)

`live_pair_all_shuffled_seed20260628/` is a self-contained successor to `live_pair_all/`. The original can be removed, but `scripts/validate_epreflop_6p_live_pair.py` still defaults to the original path. Change that default to the shuffled dataset first. Keep `live_pair_5m/` for fast diagnostics.

### Recent S-turn short sweeps (7.67 GiB)

The July 11-12 S-turn sweep families contain 183 `.pt` files. They are short-run checkpoints and compact results exist, but the investigation is less than two days old. Once winners and any desired comparison checkpoints are promoted, delete trial `.pt` files and retain results/configuration only.

### Current 169-player all-in training dataset (15.65 GiB)

`allin_training_data_169_p6_min4_2048000_s16384_b1024_bc512/` has no current textual consumer; the trained transformer checkpoint is referenced instead. It is a strong space candidate if retraining from the exact examples is not planned, but it is not actually superseded by an equivalent dataset, so keep it unless regeneration is acceptable.

### Non-smoke 169 P6 all-in checkpoint (113 MiB)

`checkpoints_allin_169_p6_min4_s16384_b1024/` appears superseded by the configured transformer checkpoint, but retain it if it is still useful as an architecture comparison or initializer.

## Protected outputs

- Current 21 GiB S-turn value dataset and 297 MiB top-up.
- Recent 4.2 GiB S-turn joint policy/value dataset, holdout, and paired validation datasets.
- Composite river chain: 2.1 GiB 100-step source, 8.3 GiB pred40 source, and 4 MiB composed manifest/symlinks.
- Current E-turn validation set.
- Packed preflop-policy states.
- Shuffled full and 5M street-closed datasets.
- Current final E-preflop checkpoint and its validation set.
- `allin_cache/`, current allboards validations, `checkpoints_allin/`, and configured 169 transformer checkpoints.
- Referenced V5 preflop backward-induction checkpoint chain.

## Expected outcome

- Tier A alone: approximately 32 GiB reclaimed while retaining long-running/current checkpoints and compact experiment records.
- Tier A plus the unshuffled street data: approximately 35 GiB.
- After the recent S-turn sweep is closed: approximately 43 GiB.
- If exact all-in retraining data is also released: approximately 59 GiB total.
