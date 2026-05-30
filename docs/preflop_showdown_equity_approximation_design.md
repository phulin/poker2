# Fast Preflop Multiway Showdown Equity Approximation

## Goal
Design a fast preflop all-in/showdown equity approximation for multiway public-belief search. The target use case is the preflop-only `PBSEnv` bootstrap path: when preflop betting reaches a called all-in or a terminal no-more-betting row, the evaluator needs per-seat, per-hand values without expanding postflop betting and without building an `H^P` private-hand table.

The design is based on the tier-2 multiway showdown evaluator experiments in `src/p2/showdown/compare_multiway_showdown_tiers.py` and the benchmark notes in `.codex/notes_equity_speedups.md`, `.codex/notes_tier2_by_card.md`, `.codex/tier2_by_card_report.md`, `.codex/tier2_mode3_architecture_report.md`, and `.codex/tier2_10ideas_scorecard.md`.

## Summary
Use a board-sampled or board-streamed preflop wrapper around the fastest four-player tier-2 by-hand river-board evaluator.

For each sampled complete board:

1. Build or reuse a board context for the 1081 board-legal hands.
2. Mask and normalize player beliefs to board-legal active hands.
3. Run the tier-2 p4 sparse direct-finish evaluator to get numerator and denominator by hero hand.
4. Accumulate numerator and denominator across boards.
5. Convert accumulated by-hand shares into chip EV through side-pot layers.

The important rule is to accumulate numerator and denominator, not per-board equity ratios. This keeps low-denominator hero hands and board-blocked hands from biasing the average.

The default implementation should support `P=4` first because the optimized path is specialized for four players. Add `P=3` by padding a zero-mass dummy player or by a smaller specialized kernel. Add `P>4` through seat-subset approximations, not by trying to generalize to `H^P`.

## Why Tier 2
Tier 2 is the best current speed/quality point for a live preflop resolver.

Tier 1 assumes independent opponent ranges after hero-card removal. It ignores opponent-opponent private-card collisions, so it is cheap but too loose in multiway pots.

Tier 2 starts with the tier-1 independent-opponent numerator/denominator and subtracts first-order opponent-pair collision corrections. For four players, each hero faces three opponents and there are three opponent-pair collision terms per hero. Across all heroes there are six unordered player pairs.

Tier 3 adds second-order collision terms. It is useful for validation and calibration, but the experiments show its wedge stage dominates runtime. The current tier-3 path is several times slower than the best tier-2 path.

Exact A+xB by-hand evaluation is the correctness oracle for small batches, but current timings were orders of magnitude slower than tier-2/tier-3 and unsuitable for a preflop inner loop.

## Experiment Lessons To Carry Forward

### Kept Ideas
- Sparse by-card cumsums beat dense `card_all` materialization.
- Direct-finish kernels beat separate pair-event materialization plus finish.
- Loading lower/tie/total player vectors once and reusing them across pair corrections is a meaningful win.
- Compact board lookup tensors help, especially packed lower/tie slot lookups and packed rank flags.
- Persistent static shapes and alignment-stable pointers are important for CUDA graph replay.
- Vectorized aggregate reductions are cheap and should be used, but the live preflop resolver should primarily consume by-hand numerator/denominator tensors.

### Rejected Ideas
- Dense rank-group `card_all` paths are bandwidth-heavy and slower.
- Direct prefix recomputation avoided `card_all` but recomputed too much per hero/card.
- Pair-level four-term PIE descriptors are mathematically correct, but descriptor construction was slower than the sparse direct-finish path for the tested board-batch sizes.
- A rank-group matrix-prefix wedge rewrite for tier 3 would allocate too much memory.
- Mixed precision for tier-3 `card_all` did not help the measured kernel.
- Current exact A+xB rank sweep is too slow to replace tier 2 in the live path.

### Relevant Timings
Measured on CUDA in the existing experiment notes:

- Dense tier-2 override at `B=512`: about `7.64 ms`.
- Sparse by-card tier-2 at `B=512`: about `5.72 ms`.
- Sparse direct-finish default at `B=512`: about `3.78 ms`.
- Reuse-vector direct-finish at `B=512`: about `3.25 ms`.
- Reuse-vector direct-finish at `B=2048`: about `15.09 ms`.
- CUDA graph replay over static repeated spots improved the direct path modestly, around `3.57 ms` versus `3.78 ms` in one scorecard run.

These numbers are for complete-board p4 by-hand tier-2 evaluation, not preflop board integration. The preflop design should batch many board samples so it stays near these efficient regimes.

## Target Contract
Add a preflop approximation API separate from the existing single-board showdown APIs:

```python
@dataclass
class PreflopApproxEquityResult:
    equity_by_hand: torch.Tensor        # [B, P, H]
    aggregate_equity: torch.Tensor      # [B, P]
    numerator_by_hand: torch.Tensor     # [B, P, H]
    denominator_by_hand: torch.Tensor   # [B, P, H]
    board_count: torch.Tensor           # [B]
    board_weight_sum: torch.Tensor      # [B]
    seconds: float
    diagnostics: dict[str, torch.Tensor]
```

The public entry point should look like:

```python
def approximate_preflop_showdown_equity(
    beliefs: torch.Tensor,              # [B, P, 1326]
    *,
    players: int,
    board_sampler: BoardSampler,
    side_pot: SidePotSpec | None = None,
    method: str = "tier2_p4_sparse_direct",
    board_count: int = 256,
    chunk_size: int = 512,
    generator: torch.Generator | None = None,
) -> PreflopApproxEquityResult:
    ...
```

For all-in CFR leaf writeback, add a chip-EV wrapper:

```python
def approximate_preflop_allin_values(
    env: PBSEnv,
    node_indices: torch.Tensor,
    beliefs: torch.Tensor,              # [M, P, H]
    *,
    board_count: int,
    method: str = "tier2_p4_sparse_direct",
) -> torch.Tensor:                      # [M, P, H]
    ...
```

## Semantics
For each hero player `p` and hero hand `h`, return:

```text
equity_by_hand[p, h] = numerator_by_hand[p, h] / denominator_by_hand[p, h]
```

The numerator is expected pot share under showdown rules. The denominator is compatible opponent-assignment mass under the tier approximation and sampled boards.

For chip EV at all-in leaves:

```text
value[p, h] = (stack_after_commit[p] + expected_payout[p, h] - starting_stack[p]) / scale
```

For side pots, compute expected payout per contribution layer and sum layers before converting to normalized value.

## Board Sampling Strategy

### First Version: Monte Carlo Complete Boards
Sample full five-card boards directly from the 52-card deck:

- input roots are preflop, so no public board cards are fixed;
- sample `S` boards per source row;
- reject duplicate boards only within each sampled board, which top-k random sampling already avoids;
- do not condition boards on private hands before tier evaluation. Hero/opponent hand blockers are handled inside the per-board active-hand masks and denominator.

Represent the expanded work batch as `[M * S, P, H]` plus `[M * S, 5]` boards, then process in chunks.

Recommended defaults:

- training resolver: `S=128` or `S=256`;
- validation resolver: `S=4096` or exhaustive chunks if feasible;
- smoke tests: `S=16`;
- chunk size: tune to hit `512-2048` board rows per tier-2 call.

### Later Version: Stratified Board Sampling
Uniform board sampling is simple but noisy. Add stratification after the first implementation:

- rank texture buckets: paired board, monotone, two-tone, high-card composition;
- suit texture buckets;
- low/high rank buckets;
- optional canonical board classes with precomputed sampling weights.

The output accumulation API should already support board weights:

```text
numerator += board_weight * numerator_board
denominator += board_weight * denominator_board
```

### Exhaustive Mode
There are `C(52, 5) = 2,598,960` full boards preflop. Exhaustive preflop evaluation is too expensive for online CFR leaves but useful for:

- offline calibration tables by coarse range class;
- regression tests on small batches;
- estimating Monte Carlo error.

The exhaustive path should stream board chunks through the same tier-2 evaluator. It should not write a `[board, player, hand]` table to disk by default.

## Core Algorithm

### Inputs
- `beliefs`: `[M, P, 1326]`, nonnegative marginal ranges.
- optional `live_mask`: `[M, P]`, seats eligible for showdown.
- optional `folded_mask`: `[M, P]`.
- optional side-pot contribution data from `PBSEnv.chips_placed`.

### Step 1: Expand Boards
Generate `S` complete boards per root:

```text
expanded_root = repeat_interleave(arange(M), S)
boards = sample_boards(M * S)
beliefs_expanded = beliefs[expanded_root]
```

If using stratified boards, also produce `board_weight`.

### Step 2: Build PreparedShowdown Chunks
For each chunk:

1. Compute `allowed = board_allowed_hands(board)`.
2. Mask beliefs by `allowed`.
3. Normalize or keep raw weights depending on the tier evaluator contract.
4. Rank hands for all boards.
5. Build `PreparedShowdown`.
6. Call the p4 tier-2 direct path.

The current `tier2_first_order_opp_collision_by_hand` returns full `[B, P, 1326]` by-hand tensors. For preflop throughput, add an internal variant that returns active tensors and active ids without scattering to full hands until the final accumulation step.

### Step 3: Accumulate By Source Root
For each chunk result:

```text
root_idx = expanded_root[chunk_rows]
num_accum.index_add_(0, root_idx, board_weight * numerator_by_hand)
den_accum.index_add_(0, root_idx, board_weight * denominator_by_hand)
weight_accum.index_add_(0, root_idx, board_weight)
```

If using active-only tensors, scatter once into a root accumulator or add an active-id gather/scatter kernel. Avoid full-hand scatter per board when possible.

### Step 4: Divide Once
After all boards:

```text
equity = safe_divide_by_hand(num_accum, den_accum)
aggregate = aggregate_all_players_from_num_denom(beliefs, num_accum, den_accum)
```

Do not average per-board equity tensors.

### Step 5: Convert To Chip Values
For plain winner-take-pot equity:

```text
expected_payout[p, h] = pot * equity[p, h]
value[p, h] = (stack[p] + expected_payout[p, h] - starting_stack[p]) / scale
```

For side pots, use the side-pot layer algorithm below.

## Tier-2 Inner Evaluator Shape

The default inner evaluator should mirror the fastest current p4 path:

1. Build active board context:
   - active ids for 1081 board-legal hands;
   - local card ids in `[0, 47)`;
   - sorted hand order by rank;
   - packed card-position slots;
   - lower/tie slot LUT;
   - packed rank flags.
2. Gather sorted beliefs.
3. Build sparse prefixes:
   - scalar prefix `[B, 4, H_active + 1]`;
   - pair prefix `[B, 6, H_active + 1]`;
   - player card cumsum `[B, 4, 47, 64]`;
   - pair card cumsum `[B, 6, 47, 64]`.
4. Build scalar/same terms and local belief matrix:
   - scalar lower/tie/total for each player and hero hand;
   - compact same-pair storage `[6, 3, B, H_active]`;
   - local belief matrix `[B, 4, 47, 47]`.
5. Run sparse direct-finish/reuse kernel:
   - loads lower/tie/total once per player vector;
   - applies all six pair corrections;
   - writes all four hero outputs.

This path corresponds to the current `P2_SHOWDOWN_TIER2_BY_CARD=3` default with vector reuse enabled, or explicit mode `5`.

## Precomputation Plan

### Process-Local Static Tensors
Cache on device:

- `hand_combos`: `[1326, 2]`;
- combo card incidence;
- local-card pair lookup helpers;
- player-pair ids for p4;
- rank/tie coefficient constants;
- reusable `torch.arange` buffers;
- optional CUDA graph workspaces for static batch sizes.

### Board Context Cache
The board context contains rank- and board-dependent tensors. Building it per sampled board is not free. Add an LRU cache for repeated boards:

```python
class BoardContextCache:
    def get(self, boards: torch.Tensor) -> TierBoardContextBatch:
        ...
```

For random preflop Monte Carlo, exact board repeats are rare inside a single call, but repeats occur across training over time. Start with a small CPU metadata cache and promote to GPU batch tensors only for the current chunk.

Do not cache all full-board contexts on GPU:

- `2,598,960` preflop boards is too many for full context tensors;
- active ids, ranks, slot LUTs, and rank flags would be large;
- random training does not need exact exhaustive board reuse.

### Canonical Board Cache
A later optimization can canonicalize boards by suit:

- store canonical board descriptors on CPU;
- store suit-remap ids;
- remap beliefs or active ids per sampled board.

This is likely useful for exhaustive validation or repeated board batches, but not required for the first implementation.

## Side Pots
A preflop all-in can have uneven contributions. The resolver should support side pots from the start because `PBSEnv` already tracks `chips_placed`.

Algorithm:

1. For each row, sort contribution levels.
2. For each positive layer width:
   - `participants = chips_placed >= level`;
   - `eligible = participants & ~folded`;
   - `layer_amount = width * participant_count`;
3. Evaluate showdown equity among eligible seats for that layer.
4. Add `layer_amount * equity_layer[p, h]` to each eligible focal seat.
5. Nonparticipants get zero payout from that layer.
6. Sum layers into `expected_payout[p, h]`.

For four or fewer eligible seats, call the tier-2 p4 path:

- if eligible count is 4, use as-is;
- if eligible count is 3, either use a p3 path or pad a zero-mass dummy player;
- if eligible count is 2, use the production heads-up exact/prefix evaluator if available, or tier-2 with two dummy players as a fallback.

Layer batching should group rows by `eligible_count` so kernels see stable player counts.

## Handling Player Counts

### P = 4
This is the primary target. Use the existing p4 tier-2 direct path.

### P = 3
Two options:

1. Pad to four players with a dummy range of zero mass and adapt denominators carefully.
2. Add a p3 direct-finish kernel.

Recommended first implementation: add a p3 path at the Python API level, but internally pad to p4 only if tests prove numerator/denominator semantics match. Otherwise write a simpler p3 kernel because there are only three player pairs and two opponents per hero.

### P = 2
Use the existing heads-up exact showdown/all-in machinery. Tier-2 is unnecessary.

### P > 4
Do not generalize tier-2 naively to all pairs in a p6 game as a first step. The number of opponent-pair corrections per hero grows as `C(P-1, 2)`, and an all-hero direct kernel becomes more complex.

Recommended approximation for `P > 4`:

- evaluate each side-pot layer using at most four highest-mass or highest-commitment eligible players;
- or sample four-player eligible subsets and average;
- log `player_subset_approx=true`;
- treat this as a deliberate approximation for rare many-way all-in leaves.

For the preflop bootstrap plan, forced-fold-to-heads-up reduces most closed-round rows before flop. Called all-in rows with more than four live eligible seats should be either sampled or discouraged by action scheduling until exact support is needed.

## Accuracy Controls

Expose these knobs:

```yaml
preflop:
  showdown_approx:
    method: tier2_p4_sparse_direct
    board_count: 256
    chunk_size: 1024
    stratified_boards: false
    calibration:
      enabled: true
      exact_sample_frac: 0.001
      tier3_sample_frac: 0.01
    p_gt_4_mode: subset_sample
    p3_mode: specialized
```

Diagnostics:

- `showdown_approx/board_count`;
- `showdown_approx/chunk_size`;
- `showdown_approx/method`;
- `showdown_approx/eligible_players`;
- `showdown_approx/denom_min`;
- `showdown_approx/zero_denom_frac`;
- `showdown_approx/tier3_delta_mean` on sampled calibration rows;
- `showdown_approx/exact_delta_mean` on tiny calibration rows;
- `showdown_approx/seconds`;
- `showdown_approx/boards_per_second`.

## Calibration Plan

### Against Exact A+xB
Use exact A+xB by-hand only on very small batches:

- choose `B <= 2`;
- use one full river board;
- compare tier-2 by-hand and aggregate outputs;
- record error distribution by hand class and board texture.

This validates single-board tier-2 behavior but not preflop board sampling variance.

### Against Tier 3
Tier 3 is slower but feasible for periodic calibration:

- sample a small fraction of preflop all-in leaves;
- reuse the same sampled board set;
- compute tier-2 and tier-3 accumulated outputs;
- log tier-3 minus tier-2.

### Against Exhaustive Heads-Up
For `P=2`, compare preflop board-streaming accumulation against the existing exact `[1326, 1326]` preflop table or direct exact script output. This validates board sampling and accumulation mechanics independent of multiway approximation.

### Monte Carlo Error
For board sampling:

- run repeated seeds for the same roots;
- measure per-hand and aggregate standard error;
- bucket by denominator and hand class;
- use this to set the default `board_count`.

## Integration With Preflop PBS

### Called All-In Leaves
In `PreflopSparseCFREvaluator.set_leaf_values`:

1. Partition all-in leaves by live/eligible player count.
2. Gather `beliefs[leaf_indices]`.
3. Call `approximate_preflop_allin_values`.
4. Write values back to `latest_values[leaf_indices]`.

Use exact heads-up resolver for two-player leaves and tier-2 approximation for four-player leaves.

### Closed Preflop Handoff Scoring
The forced-fold-to-heads-up handoff needs a fast range-strength estimate. This design can provide two versions:

- cheap static hand-class strength for every row;
- sampled preflop approximate equity for a smaller candidate set.

Do not run full `S=256` board approximation just to choose forced-fold survivors for every closed preflop row unless profiling shows it is affordable. The all-in resolver path has stricter accuracy requirements than survivor scoring.

### Value Targets
For all-in leaves, the resolver returns terminal value targets. For non-all-in closed-preflop leaves, use postflop model expectation as described in `docs/preflop_multiway_pbs_bootstrap_plan.md`, not this showdown-only approximation, unless deliberately using a cold-start proxy.

## Memory Budget
For p4, active hands per full board are 1081.

Typical key tensors per board batch:

- scalar prefix: `[B, 4, 1082]`;
- pair prefix: `[B, 6, 1082]`;
- player card cumsum: `[B, 4, 47, 64]`;
- pair card cumsum: `[B, 6, 47, 64]`;
- local belief matrix: `[B, 4, 47, 47]`;
- same terms: `[6, 3, B, 1081]`;
- output active numerator/denominator/equity: `[B, 4, 1081]`.

The design should reuse buffers across chunks. Avoid keeping per-board full-hand outputs after accumulation. For preflop roots, the long-lived accumulators are only:

- `num_accum`: `[M, P, 1326]`;
- `den_accum`: `[M, P, 1326]`;
- optional `payout_accum`: `[M, P, 1326]`.

## Performance Targets

Initial CUDA targets:

- p4 single-board chunk inner tier-2 remains within 20 percent of current direct-finish benchmark for comparable `B`.
- `S=128`, `M=128` should complete in a small number of large chunks rather than many tiny launches.
- All-in resolver should stay below preflop CFR model-evaluation cost for normal all-in leaf counts.
- No allocation should scale beyond `O(B_chunk * P^2 * H_active + M * P * H)`.

Practical first benchmark matrix:

- `M in {32, 128, 512}`;
- `S in {32, 128, 256}`;
- `P in {3, 4}`;
- `chunk_size in {512, 1024, 2048}`;
- compare normal eager and CUDA graph replay for static `M*S`.

## Implementation Phases

### Phase 1: API And Reference
- Add `preflop_approx.py` under `src/p2/showdown`.
- Implement a simple board-sampled wrapper around `tier2_first_order_opp_collision_by_hand`.
- Accumulate full-hand numerator/denominator with `index_add_`.
- Add tests for shapes, finite outputs, deterministic seeds, and heads-up sanity.

Acceptance:
- Works on CPU through existing fallback paths.
- Works on CUDA for p4.
- No `H^P` allocation.

### Phase 2: Active-Only Fast Path
- Expose an internal tier-2 p4 function returning active numerator/denominator/equity plus active ids.
- Avoid active-to-full scatter per board chunk.
- Scatter or add into full root accumulators once per chunk.
- Add persistent scratch buffers for fixed chunk shapes.

Acceptance:
- Matches Phase 1 outputs to float noise.
- Improves preflop wrapper throughput measurably.

### Phase 3: Side-Pot EV
- Add `SidePotSpec` from `PBSEnv` rows.
- Group by eligible player count and layer shape.
- Convert accumulated equity shares to chip EV.
- Add scalar `NLEnv` parity tests on restricted supports.

Acceptance:
- Side-pot payouts conserve chips in aggregate.
- Folded players cannot win layers.
- Uneven all-in commitments match reference cases.

### Phase 4: Calibration
- Add tier-3 calibration mode on sampled rows.
- Add exact A+xB calibration for tiny batches.
- Add repeated-seed board sampling error reports.

Acceptance:
- Diagnostics report tier-2 error estimates and Monte Carlo variance.
- Default board count is chosen from measured variance, not guessed.

### Phase 5: Integration
- Wire into `PreflopSparseCFREvaluator` all-in leaves.
- Add config and logging.
- Add benchmark script, likely `scripts/bench_preflop_showdown_approx.py`.

Acceptance:
- Preflop CFR all-in leaves get finite `[node, P, H]` values.
- Training runs without calling heads-up preflop all-in table for multiway leaves.
- Benchmarks show stable memory use across player counts and board counts.

## Test Plan

### Unit Tests
- `test_preflop_approx_shapes_p4_cuda_or_cpu`
- `test_preflop_approx_deterministic_seed`
- `test_preflop_approx_accumulates_num_denom_not_equity`
- `test_preflop_approx_no_h_power_allocation`
- `test_preflop_approx_heads_up_matches_preflop_table_small_sample`
- `test_preflop_approx_p3_padding_or_specialized_matches_reference`
- `test_preflop_approx_sidepot_conserves_chips`

### Correctness Tests
- Single fixed board: wrapper output matches direct tier-2 call.
- Multiple repeated identical boards: accumulated result equals direct result.
- Tiny restricted hand supports: compare against brute-force valid private assignments and board runouts.
- Folded/side-pot rows: compare against `NLEnv` reward accounting where private hands are fixed.

### Performance Tests
- Inner tier-2 benchmark remains close to current p4 direct-finish timings.
- Wrapper throughput scales roughly linearly in `M*S`.
- Chunk-size sweep identifies stable defaults.
- CUDA graph replay benchmark for static repeated all-in leaf batches.

## Open Questions
- Should p3 get a true specialized direct-finish kernel immediately, or is p4 padding acceptable?
- What default board count gives acceptable variance for CFR leaf values?
- Should board sampling be uniform first, or should we start with simple texture stratification?
- Should forced-fold survivor scoring use this approximate equity, or should it stay on a much cheaper static range-strength score?
- How much tier-3 calibration is affordable during training?

## Recommended First Implementation
Start with the simplest correct wrapper:

1. `P=4`, CUDA-first, using `tier2_first_order_opp_collision_by_hand`.
2. `S=128`, `chunk_size=1024`.
3. Full-hand numerator/denominator accumulation with `index_add_`.
4. No side pots for the first smoke test.
5. Add side-pot layer support before integrating into `PreflopSparseCFREvaluator`.
6. Only after correctness is locked, expose active-only outputs and persistent workspaces.

This path reuses the experiments that already worked, avoids new descriptor math, and gives a direct route to a usable multiway preflop all-in resolver.
