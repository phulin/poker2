# Notes: Outputs Retention Audit

## Baseline
- `outputs/` currently occupies about 115 GiB.
- Known composite river dataset depends on the 100-step and pred40 400-step source datasets via symlinks.

## Audit Criteria
- **Protect:** current config/script dependency, current initialization chain, recent long-running job, unique expensive validation/comparison asset.
- **Review:** old but potentially reproducible or scientifically useful data with no current reference.
- **Delete candidate:** failed/smoke/profile output, superseded variant, redundant copy, regenerated cache, or completed sweep bulk tensors with compact results retained.

## Dependency Findings
- The 500-step river dataset is the only composite manifest. Its 1,000 symlinks require the 100-step source (200) and pred40 400-step source (800); protect all three.
- Current configs reference the packed preflop-policy dataset, the final full E-preflop checkpoint, its validation tensor, and the 169-player transformer all-in checkpoint.
- The packed policy dataset is self-contained; its `source_dataset` field is provenance, not a symlink dependency.
- The shuffled street-closed dataset is also self-contained; deleting the original would require changing a validation script whose default still names the original.

## Supersession Findings
- Preflop backward induction contains 7.25 GiB of old `solved/` tensors, 3.54 GiB of embedded `validation/` tensors, and about 4.34 GiB of older canonical/experimental validation caches.
- Old E-preflop smoke/timing/sweep trials occupy 3.36 GiB; current configs use only the final full run and the 532 MiB validation directory.
- Six completed value-architecture families contain 4.16 GiB across 98 `.pt` checkpoint files; compact JSON/JSONL/Markdown results can remain.
- Raw policy states and old rollouts plus the original unshuffled street-closed copy occupy 6.94 GiB; packed/shuffled successors are self-contained.
- Clearly superseded all-in variants occupy 1.92 GiB, excluding current configured checkpoints and validation data.
- Old river targets/retries and smoke outputs occupy about 4.10 GiB, excluding the active composite chain and recent turn datasets.
- Recent July 11-12 S-turn sweeps occupy 7.67 GiB in 183 `.pt` files; they are short-run checkpoints but too recent to call obsolete without closing the current investigation.
- The 15.65 GiB current all-in training dataset has no current text reference, but is useful for retraining and is not superseded by an equivalent dataset.

## Execution
- On 2026-07-13, the user approved Tier A, the original street-state copy, and short-sweep checkpoint deletion.
- Actual reclaimed space was about 43 GiB; `outputs/` fell from 115 GiB to 72 GiB.
- The validation script default was changed to the preserved shuffled street-state dataset.
- Post-cleanup verification found no broken composite-dataset symlinks.
