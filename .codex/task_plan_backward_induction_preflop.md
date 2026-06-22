# Task Plan: Preflop Backward Induction

## Goal
Generate and train depth-cutoff specialist models from bucketed preflop frontier states, then provide a distillation path into one full preflop model.

## Phases
- [x] Phase 1: Stream the 20M/N=5 unique-frontier bucket state generation run.
- [x] Phase 2: Commit the policy-state generator changes.
- [x] Phase 3: Inspect existing ReBeL trainer, dataset, and CFR target APIs.
- [x] Phase 4: Add backward-induction training code for depth=4 bucket specialists.
- [x] Phase 5: Add single-model distillation code from the depth specialists.
- [x] Phase 6: Run compile/lint/smoke checks and report run status.

## Decisions Made
- Use streaming unique-frontier generation for 20M roots because full-root GPU pools do not fit in memory.
- Commit only the generator source and scripts directory summary; leave unrelated untracked notes/files untouched.
- Specialist order is deepest to shallowest: 12-15, 8-11, 4-7, 0-3.
- The 0-3 bucket participates in policy distillation/training but does not need a specialist value model.
- Unique bucket generation writes action-range directories `actions_0_3`, `actions_4_7`, `actions_8_11`, and `actions_12_15`; within each bucket, per-action cap defaults to one quarter of the bucket cap.
- Backward-induction script stores solved `RebelBatch` shards while training one pass over them online; policy targets are filtered to the current bucket action range.
- Distillation script loads four specialist checkpoints and trains one student; it includes value+policy for 12-15/8-11/4-7 and policy-only for 0-3.

## Errors Encountered
- Initial 20M launch wrote exact frontiers `0/4/8/12` instead of action-count buckets. Stopped the job, removed its partial output, and added `--unique-frontier-buckets` streaming mode.
- Distillation smoke initially failed because `torch.inference_mode()` produced inference tensors for targets; switched target generation to `torch.no_grad()`.

## Status
**Complete** - code is committed, 20M bucket dataset finished, and the 100k-per-bucket W&B specialist run is detached as `p2_preflop_bi_100k`.
