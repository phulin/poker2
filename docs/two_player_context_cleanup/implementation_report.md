# Two-Player Context Cleanup Implementation Report

## Implemented

- Added `last_aggressive_amount` to `HUNLTensorEnv`, including reset, ordinary
  stepping, street closure, copies, repeats, CPU/CUDA gathers, fused active views,
  and optimized/legacy Triton child construction.
- Added a compact heads-up BetterFFN schema: 11 scalar plus 10 fields per player,
  reducing context width from 41 to 31.
- Removed heads-up folded, can-call, acted-this-round, actor scalar, actor-position
  scalar, unopened/check-to-actor, and relative-position-to-actor features.
- Preserved the original 15 scalar plus 13 fields-per-player schema for
  multi-player models.

## Verification

- `tests/test_mlp_features.py`, `tests/test_tensor_env.py`, and
  `tests/test_model_forward.py`: 68 passed.
- Focused CUDA/Triton optimized writer, legacy writer, and environment gather:
  3 passed.
- Ruff on all touched Python files: passed (ignoring the pre-existing duplicate
  `forward` definitions in `better_ffn.py`).

Existing 41-wide heads-up BetterFFN checkpoints are intentionally incompatible;
the new architecture must start from step 0.
