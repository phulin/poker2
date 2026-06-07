# Compact 169-Hand Preflop Deliverable

## Implemented
- Added 169 preflop rank-class utilities in `card_utils.py`.
- Generalized `MLPFeatures` to carry `hand_dim` while preserving 1326 as the default.
- Added compact `BetterPreflopValueFFN` and `BetterPreflopPolicyFFN` model variants.
- Routed curriculum `E_preflop`/`S_preflop`/`S_0` stages to compact preflop models with `model.preflop_hand_dim = 169`.
- Added compact value and policy loss branches using preflop class multiplicity.
- Changed closed-street-0 distillation so compact `E_preflop` features/targets are 169-wide, with 1326 expansion only around frozen `S_flop` target generation.
- Added native compact 169 preflop belief sampling for `E_preflop` distillation roots.
- Added periodic compact `E_preflop` validation metrics during curriculum distillation using exact raw-flop enumeration as the canonical-orbit-equivalent reference.
- Added non-fused `PreflopSparseCFREvaluator` support for native 169 tensors, multiplicity priors, 1326-to-169 root belief collapse, and exact class-level blocker projection for class-constant strategies.
- Added focused tests for mapping, expand/collapse, loss equivalence, compact model shapes, and compact distillation batches.
- Updated progressive disclosure `AGENTS.md` files for touched directories.

## Verification
- `uv run python -m compileall -q ...` on touched modules/tests.
- `uv run pytest tests/test_sparse_cfr_evaluator.py`
- `uv run pytest tests/test_preflop_169.py tests/test_end_of_street_distillation.py tests/test_postflop_spot_sampler.py tests/test_trainer_config_build.py tests/test_train_rebel_curriculum.py tests/test_model_forward.py tests/test_sparse_cfr_evaluator.py -q`
- `uv run pytest tests/test_preflop_169.py tests/test_sparse_cfr_evaluator.py -q`
- `uv run pytest tests/test_preflop_169.py tests/test_sparse_cfr_evaluator.py tests/test_losses.py -q`
- `uv run pytest tests/test_preflop_169.py tests/test_end_of_street_distillation.py tests/test_postflop_spot_sampler.py tests/test_trainer_config_build.py tests/test_train_rebel_curriculum.py tests/test_model_forward.py tests/test_sparse_cfr_evaluator.py tests/test_losses.py -q`

## Not Completed
- Fused/Triton native 169 preflop evaluator conversion.
- Compact all-in terminal payoff resolution.
- Checkpoint promotion compatibility for old 1326 preflop checkpoints; new compact preflop models will be trained fresh.
