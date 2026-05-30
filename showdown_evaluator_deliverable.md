# Showdown Evaluator Deliverable

## Implemented
- Added `src/p2/showdown/` as a reusable package.
- Copied `benchmarks/multiway_showdown_estimators.py` to `src/p2/showdown/multiway_showdown_estimators.py`.
- Copied `benchmarks/compare_multiway_showdown_tiers.py` to `src/p2/showdown/compare_multiway_showdown_tiers.py`.
- Added package-relative import fixes in the copied tier module only.
- Added `exact.py` wrappers, including `exact_nway_ie_fast` and explicit `tri` helpers.
- Added `approximate.py` exports for tier 1-4 calculators.
- Added `monte_carlo.py` exports for the alias tuple-reject MC path.
- Added `tests/test_showdown_package.py` for package exports and the `tri` identity check.
- Replaced `exact_nway_ie_fast` for up to 4 players with a rank-prefix `A+xB` exact evaluator.
- The A+xB path batches all hero hands in a rank class and uses exact scalar/pair/Delta/tri cluster polynomials.
- Added shared `PerHandEquityResult` and by-hand outputs for the exact oracle, A+xB exact path, tier 1-4 approximations, and a PyTorch conditional MC reference estimator.

## Verification
- `uv run python -c "import p2.showdown as s; from p2.showdown.approximate import tier4_third_degree_card_collision; from p2.showdown.monte_carlo import make_batched_alias_tuple_reject_workspace; print(s.tri_identity_smoke_check())"`
- `uv run pytest tests/test_showdown_package.py`
- `uv run pytest tests/test_showdown_package.py tests/test_showdown_per_hand_equity.py`
- `uv run python -m py_compile src/p2/showdown/__init__.py src/p2/showdown/exact.py src/p2/showdown/approximate.py src/p2/showdown/monte_carlo.py src/p2/showdown/compare_multiway_showdown_tiers.py src/p2/showdown/multiway_showdown_estimators.py`
- n=4 fixed-board check: oracle and A+xB matched exactly after float32 output conversion; sample CPU timings were oracle 4.25s, A+xB 1.60s.

## Notes
- For more than 4 players, `exact_nway_ie_fast` still falls back to the copied chunked IE prototype until size-4 and size-5 clusters are implemented for A+xB.
