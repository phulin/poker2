# Plan: Per-Hand Showdown Equity Vectors

## Goal
Rewrite the showdown evaluator examples so every implementation can return a per-hero-hand equity vector in `R^H`, not only aggregate player equity.

Here `H = NUM_HANDS = 1326` in public APIs. Invalid board-blocked hero hands should be present and set to zero unless a caller explicitly asks for active-hand compact output.

## Target Contract
Add a shared result type:

```python
@dataclass
class PerHandEquityResult:
    equity_by_hand: torch.Tensor  # [players, H] or [1, players, H]
    aggregate_equity: torch.Tensor  # [1, players]
    denominator_by_hand: torch.Tensor  # [players, H]
    seconds: float
```

Conventions:
- `equity_by_hand[p, h] = numerator[p, h] / denominator[p, h]` for hero player `p` holding hand `h`.
- `aggregate_equity[p] = sum_h belief[p,h] * numerator[p,h] / sum_h belief[p,h] * denominator[p,h]`.
- Board-blocked hands and zero-denominator hands return zero in `equity_by_hand`.
- Keep numerator/denominator internally available during tests even if not public at first.

## Phase 1: Shared Utilities
- Add `src/p2/showdown/results.py` for result dataclasses and aggregation helpers.
- Add helpers:
  - `safe_divide_by_hand(numerator, denominator)`.
  - `aggregate_from_num_denom(hero_belief, numerator, denominator)`.
  - `scatter_active_to_full(active_values, active_ids, players)`.
- Update package exports in `src/p2/showdown/__init__.py`.

## Phase 2: Exact Oracle IE
Target: `src/p2/showdown/multiway_showdown_estimators.py`.

Current state:
- `exact_nway_ie` already computes `numerator` and `denominator` vectors per hero internally.
- It only appends aggregate equity to the returned `ExactResult`.

Rewrite:
- Add `exact_nway_ie_by_hand(...) -> PerHandEquityResult`.
- Keep `exact_nway_ie(...) -> ExactResult` as a compatibility wrapper.
- Inside the hero loop, store:
  - `equity_by_hand[hero]`.
  - `denominator_by_hand[hero]`.
  - optionally `numerator_by_hand[hero]` for debug/tests.
- Do not duplicate the oracle computation. The aggregate-only wrapper should call or share the by-hand implementation.

Validation:
- Aggregate from `PerHandEquityResult` must equal current `exact_nway_ie`.
- For 3-way, compare selected hands against `_threeway_ie_for_hero`.

## Phase 3: A+xB Exact
Target: `src/p2/showdown/exact.py`.

Current state:
- `exact_nway_ie_axb` accumulates weighted numerator/denominator totals directly.
- Per-hand numerator and denominator are computed transiently per rank class.

Rewrite:
- Allocate full active arrays:
  - `numerator_active = torch.zeros(players, active_count)`.
  - `denominator_active = torch.zeros(players, active_count)`.
- In the rank-class loop, after computing `numerators` and `denominators`, assign them into the active hand positions for that hero:
  - `numerator_active[hero, rank_ids] = numerators`.
  - `denominator_active[hero, rank_ids] = denominators`.
- After the sweep, scatter active vectors back to `[players, NUM_HANDS]`.
- Compute aggregate with the shared helper.
- Make `exact_nway_ie_fast` return aggregate-only for compatibility, and add `exact_nway_ie_fast_by_hand`.

Optimization note:
- This also prepares denominator precompute. Once per-hand vectors exist, denominator can be computed once per hero hand and reused across rank classes.

Validation:
- `exact_nway_ie_fast_by_hand(...).aggregate_equity` equals `exact_nway_ie_fast(...)`.
- For n=4, aggregate and per-hand vectors match oracle by-hand output.

## Phase 4: Tier 1-4 Approximations
Target: `src/p2/showdown/compare_multiway_showdown_tiers.py` and `approximate.py`.

Current state:
- Tier functions compute `numerator` and `denominator` by hero hand, then immediately aggregate.

Rewrite:
- Add by-hand variants:
  - `tier1_hero_removal_by_hand`.
  - `tier2_first_order_opp_collision_by_hand`.
  - `tier3_second_order_opp_collision_by_hand`.
  - `tier4_third_degree_card_collision_by_hand`.
- Keep existing aggregate wrappers.
- Return `PerHandEquityResult` from the by-hand variants.

Implementation pattern:
- In each hero loop, assign `equity_by_hand[hero]`, `denominator_by_hand[hero]`.
- Reuse existing `numerator` and `denominator` tensors.
- Avoid recomputing relation matrices in wrappers.

Validation:
- Existing tier aggregate outputs remain unchanged.
- By-hand aggregate helper reproduces wrapper aggregate exactly.
- Tier 1 by-hand can be sanity checked against independent lower/tie products for a few hands.

## Phase 5: Monte Carlo Samplers
Targets:
- `src/p2/showdown/monte_carlo.py`.
- Copied Triton sampler paths in `multiway_showdown_estimators.py`.

Important distinction:
- Existing MC samplers estimate player aggregate equity from sampled complete deals.
- A true per-hand equity vector needs conditional estimates given each hero hand.

Two options:

1. Conditional per-hand MC, accurate but more work:
   - For each hero player and sampled hero hand, sample opponent hands conditional on hero cards being removed.
   - Accumulate `sum_wx[hero, hand]`, `sum_w[hero, hand]`, and `sum_wx2`.
   - Output `[players, H]`.

2. Deal-sample attribution, cheaper but not a full conditional vector:
   - Sample full valid deals as now.
   - Attribute observed share to the sampled hand for each player.
   - This estimates conditional equity only for hands visited often enough and needs per-hand ESS.

Plan:
- Implement option 1 first in a PyTorch reference sampler for correctness.
- Add Triton kernels later after the API and tests stabilize.
- Name clearly:
  - `alias_tuple_reject_aggregate` for current aggregate sampler.
  - `conditional_tuple_reject_by_hand` for the new per-hand estimator.

Validation:
- On restricted tiny supports, MC by-hand means converge to exact by-hand vectors.
- Aggregate of by-hand MC approximates current aggregate MC within sampling error.

## Phase 6: Tests
Add `tests/test_showdown_per_hand_equity.py`.

Test cases:
- n=2 exact by-hand matches direct pairwise lower/tie/disjoint calculation.
- n=3 oracle by-hand aggregate equals current `exact_threeway_ie`.
- n=4 A+xB by-hand matches oracle by-hand to tolerance.
- Tier 1-4 by-hand aggregate equals existing tier aggregate wrappers.
- Invalid board-blocked hands are zero.

Keep expensive tests small:
- Use one deterministic board.
- Use CPU for deterministic exact tests.
- Mark MC convergence tests as slow or keep them on tiny restricted supports.

## Phase 7: API Cleanup
After all implementations have by-hand variants:
- Update `ExactResult` or replace it with `PerHandEquityResult` where callers are internal.
- Keep backward-compatible aggregate wrappers for benchmark scripts.
- Update README/docs with examples:
  - exact aggregate.
  - exact by-hand vector.
  - tier by-hand vector.
  - MC aggregate vs conditional by-hand.

## Recommended Order
1. Exact oracle by-hand first, because it is the reference.
2. A+xB by-hand second, because its transient per-hand values already exist.
3. Tier 1-4 by-hand third, because it is mostly return-shape plumbing.
4. MC conditional by-hand last, because it changes estimator semantics and needs separate ESS handling.
