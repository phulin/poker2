# Task Plan: 8k All-In HP Comparison

## Goal
Compare recent MLP and player-transformer all-in training hyperparameters through 8,000 steps on the regenerated high-quality validation set.

## Phases
- [x] Phase 1: Confirm regenerated validation manifest
- [x] Phase 2: Define compact MLP vs transformer schedule matrix
- [ ] Phase 3: Run 8k trials and save logs/results
- [ ] Phase 4: Summarize MSE/MAE comparison

## Key Questions
1. Does the transformer still beat the MLP against the lower-noise validation targets?
2. Is cosine decay over 4k, cosine decay over 8k, or linear decay over 8k best at the 8k-step budget?
3. Do the relative conclusions change by live-player bucket?

## Decisions Made
- Use the completed high-quality validation manifest: `outputs/allin_validation_data_4096_s16777216_b2097152_bc64/manifest.json`.
- Keep the training source aligned with recent W&B all-in runs: `outputs/allin_training_data_512k_s65536_b4096_bc64/manifest.json`.
- Use `target_mode=eligible_pot_share` for both MLP and transformer so architecture is the main comparison.
- Keep the recent transformer shape fixed at `hidden_dim=1024`, `hand_dim=512`, `layers=10`, `transformer_heads=8`.
- Run six trials: MLP and transformer for cosine-4k, cosine-8k, and linear-8k schedules.

## Errors Encountered
- None yet.

## Status
**Currently in Phase 3** - Creating and running the 8k sweep harness.
