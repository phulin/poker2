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
- Added compact-only `PreflopSparseCFREvaluator` support for native 169 tensors, multiplicity priors, optional 1326-to-169 root belief collapse at the boundary, and exact class-level blocker projection for class-constant strategies.
- Added compact-only `FusedPreflopSparseCFREvaluator` routing that initializes 169-wide sparse tree tensors and keeps only the fused kernels already parameterized by hand dimension.
- Added focused tests for mapping, expand/collapse, loss equivalence, compact model shapes, and compact distillation batches.
- Updated progressive disclosure `AGENTS.md` files for touched directories.

## Verification
- `uv run python -m compileall -q src/p2/search/fused_preflop_sparse_cfr_evaluator.py src/p2/search/preflop_sparse_cfr_evaluator.py src/p2/rl/cfr_trainer.py tests/test_sparse_cfr_evaluator.py tests/test_rebel_pipeline.py`
- `uv run pytest tests/test_sparse_cfr_evaluator.py`
- `uv run pytest tests/test_sparse_cfr_evaluator.py tests/test_rebel_pipeline.py::test_rebel_cfr_trainer_constructs_multiway_pbs_env tests/test_rebel_pipeline.py::test_rebel_cfr_trainer_routes_multiway_pbs_env_to_fused_preflop -q`
- `uv run pytest tests/test_preflop_169.py tests/test_end_of_street_distillation.py tests/test_postflop_spot_sampler.py tests/test_trainer_config_build.py tests/test_train_rebel_curriculum.py tests/test_model_forward.py tests/test_sparse_cfr_evaluator.py tests/test_losses.py tests/test_rebel_pipeline.py::test_rebel_cfr_trainer_constructs_multiway_pbs_env tests/test_rebel_pipeline.py::test_rebel_cfr_trainer_routes_multiway_pbs_env_to_fused_preflop -q`

## Not Completed
- Dedicated Triton kernels for 169-class blocker projection; the fused preflop evaluator currently uses torch matmul for exact class blocker math.
- Compact all-in terminal payoff resolution.
- Checkpoint promotion compatibility for old 1326 preflop checkpoints; new compact preflop models will be trained fresh.
