# Task Plan: 6-Player All-In LR Sweep

## Goal
Create a manifest for the existing 6-player all-in training shards and run 1,000-step bs=512 experiments to identify good learning-rate and decay settings.

## Phases
- [x] Phase 1: Inspect shard set and build manifest
- [x] Phase 2: Smoke-test 6-player training from manifest
- [x] Phase 3: Run 1,000-step LR/cosine sweep jobs
- [x] Phase 4: Summarize results and recommendation
- [x] Phase 5: Add and test linear decay schedules
- [x] Phase 6: Add and test stable warmdown schedules
- [x] Phase 7: Run 2,000-step cosine 1000/2000 comparison

## Key Questions
1. How many complete 6-player examples exist right now?
2. Does the standard trainer load and train from the custom manifest cleanly?
3. Which LR and cosine settings give the best 1,000-step loss behavior at bs=512?

## Decisions Made
- Keep this plan in `allin_lr_sweep/` because the repository root already has an unrelated `task_plan.md`.
- Treat existing complete shard files as immutable inputs.
- Write a sidecar manifest named `manifest_players6_existing.json` instead of `manifest.json`, so live generation can still finish normally.
- Use a 200-shard train / 31-shard validation split for LR search metrics.
- For the shortened 1,000-step budget, recommend LR 0.015 / AdamW LR 0.024 / cosine floor 0.015 / cosine decay steps 1000.
- Added `lr_decay` selector to test linear decay while preserving current cosine defaults.
- Linear decay does not improve the recommendation; linear 1000 is close but slightly worse on MAE.
- Added `stable_warmdown` schedule mode for flat LR followed by cosine warmdown.
- Stable warmdown starting at step 800 did not improve over cosine 1000 in the 1,000-step budget.
- In a 2,000-step run, cosine 2000 slightly beats cosine 1000 at the final eval, but cosine 1000 is much better at step 1000.

## Errors Encountered
- `.codex/tasks/...` directory creation failed with read-only filesystem, so the task files live in `allin_lr_sweep/`.
- Initial sweep was stopped after user requested testing cosine schedules further out into training; candidate matrix was updated before collecting final results.
- Second sweep attempt was stopped after PyTorch warned TF32 matmul was not enabled; fixed `device == "cuda"` to `device.type == "cuda"` in `p2.allin.train`.
- 2,000-step sweep was stopped per user request; harness now runs 1,000-step candidates with eval every 250 steps.

## Status
**Complete** - Warmdown and 2,000-step cosine comparisons are in the report.
