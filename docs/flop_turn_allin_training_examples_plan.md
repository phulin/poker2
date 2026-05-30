# Flop And Turn All-In Training Examples Plan

## Goal
Extend `p2.allin` from preflop-only all-in value training to generate flop and turn all-in training examples with board-aware beliefs, terminal value targets, sharded datasets, and a trainer path that can mix streets without corrupting the existing preflop workflow.

## Current Starting Point
`p2.allin` currently owns a preflop-specific contract:

- `PreflopAllInBatch` stores `[B, P, 1326]` beliefs and per-seat stack/commit/all-in/fold state, but no public board or street.
- `make_random_preflop_allin_batch` samples random preflop ranges and all-in/fold/covering-caller configurations.
- `estimate_preflop_allin_values` estimates `[B, P, 1326]` targets through preflop full-board sampling, with an exact two-player preflop fast path.
- `training_data.py` can pregenerate and shard examples, then apply suit/player augmentation.
- `train.py` trains one preflop-shaped model against those examples.

The adjacent `p2.search.allin_payoff` module already has useful postflop infrastructure:

- exact or quantized heads-up postflop payoff matrices for fixed boards;
- canonical flop table caching;
- batched flop and turn all-in value helpers;
- fused Triton writeback paths, including turn river integration.

The first implementation should reuse that machinery instead of building a new postflop showdown evaluator inside `p2.allin`.

## Scope
Build flop and turn example generation for all-in terminal states. Do not expand postflop betting or build a full postflop CFR data generator in this plan.

First target:

- heads-up flop and turn all-in examples;
- board-aware ranges masked to public cards;
- exact or table-backed terminal all-in targets;
- offline pregeneration and normal trainer consumption.

Second target:

- multiway flop/turn side-pot examples using the existing preflop side-pot accounting plus a multiway showdown approximation or exact oracle;
- mixed-street training with a street-aware model.

## Phase 1: Generalize The Data Contract
Add a street-aware batch dataclass, either by replacing `PreflopAllInBatch` with a compatible superset or by adding a sibling type.

Recommended shape:

```python
@dataclass
class AllInBatch:
    street: torch.Tensor             # [B], 0 preflop, 1 flop, 2 turn
    board: torch.Tensor              # [B, 5], missing cards are -1
    beliefs: torch.Tensor            # [B, P, 1326]
    starting_stacks: torch.Tensor    # [B, P]
    committed: torch.Tensor          # [B, P]
    stacks_after: torch.Tensor       # [B, P]
    allin_mask: torch.Tensor         # [B, P]
    folded_mask: torch.Tensor        # [B, P]
    scale: torch.Tensor              # [B]
```

Keep `PreflopAllInBatch` as an alias or narrow wrapper until the trainer and tests are migrated. The model and sampler should not infer street from `board >= 0`; street must be explicit for mixed data.

Implementation details:

- Add board masks with `board_allowed_hands(board)` from `p2.env.card_utils`.
- Ensure every generated belief is zero on board-blocked hands and normalized on legal hands.
- Store board and street in pregenerated shards.
- Update `FEATURE_KEYS`, `batch_to_tensors`, `tensors_to_batch`, concat/slice helpers, and pinned prefetch support.
- Extend suit permutation augmentation to permute both `board` and hand indices together.

## Phase 2: Add Board-Aware Random State Generation
Add generators for flop and turn terminal all-in states.

Proposed APIs:

```python
def make_random_postflop_allin_batch(
    batch_size: int,
    players: int,
    *,
    street: int,
    bb: int,
    stack_config: ...,
    range_config: ...,
    device: torch.device,
    generator: torch.Generator | None,
) -> AllInBatch:
    ...
```

Responsibilities:

- sample unique public boards: three cards for flop, four for turn;
- create board-legal per-seat belief distributions;
- optionally bias ranges by board texture and simple strength features rather than pure exponential random mass;
- sample stack/commit/all-in/fold status using the current all-in terminal conventions;
- preserve the covering-caller rule where useful: all live players except a unique deepest caller are all-in;
- keep folded dead money bounded so folded players do not create impossible side-pot layers.

Range generation should start simple but be board-aware:

- version 1: exponential random mass over board-legal combos;
- version 2: mixture of uniform, pair/draw-heavy, made-hand-heavy, and low-equity tails;
- version 3: ranges produced by actual postflop PBS/CFR states.

## Phase 3: Build Heads-Up Flop/Turn Target Resolvers
For heads-up postflop, use `p2.search.allin_payoff` as the target engine.

Flop path:

- Use canonical flop payoff tables when available.
- For each row, map actual flop to canonical table and apply hand permutation.
- Compute both-player `[B, 2, 1326]` all-in EV vectors with blocker-aware denominators through the existing table matvec path.
- Convert showdown shares to normalized chip values through the same side-pot/payout conversion used by the preflop sampler.

Turn path:

- Use the batched turn helper or `write_turn_allin_values_triton_` style river integration.
- Avoid materializing per-row Python loops for batches.
- Cache repeated turn boards only as a later optimization; first prefer batched kernels.

Add a street dispatcher:

```python
def estimate_allin_values(
    batch: AllInBatch,
    *,
    postflop_resolver: HeadsUpPostflopAllInResolver,
    preflop_config: AllInDataGenConfig,
    compute_stats: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

The dispatcher should reject unsupported combinations loudly:

- heads-up flop/turn: supported first;
- preflop: existing path;
- multiway flop/turn: unsupported until Phase 6.

## Phase 4: Pregeneration And Dataset Format
Version the dataset format before adding board/street fields.

Recommended manifest progression:

- keep existing `p2.allin.training_data.v1` readable for preflop-only datasets;
- add `p2.allin.training_data.v2` with `street` and `board`;
- include `street_counts`, `players`, `hands`, target key, feature keys, and generator config;
- keep shard tensors CPU-contiguous and compatible with pinned-memory prefetch.

Add CLI/config options:

```yaml
allin_data:
  streets: [flop, turn]
  street_weights: [0.5, 0.5]
  players: 2
  examples: ...
  shard_size: ...
  generation_batch_size: ...
  range_generator: board_legal_random
```

The pregenerator should support separate datasets per street and mixed-street datasets. Separate datasets are better for early debugging; mixed datasets are better once training is stable.

## Phase 5: Model And Trainer Changes
The current `PreflopAllInEquityModel` has no board input, so it cannot learn board-conditioned flop/turn values correctly.

Add a new model instead of overloading the preflop model:

```python
class StreetAllInEquityModel(nn.Module):
    def forward(
        self,
        street,
        board,
        beliefs,
        starting_stacks,
        committed,
        stacks_after,
        allin_mask,
        folded_mask,
    ) -> torch.Tensor:
        ...
```

Minimum architecture additions:

- street embedding;
- board card/rank/suit/texture features;
- per-hand board compatibility mask;
- hand-vs-board features: pair with board, made hand bucket proxy, flush draw/open-ended draw proxies for flop and turn;
- existing range summaries, player features, side-pot features, and blocker features.

Keep `PreflopAllInEquityModel` for preflop-only training until mixed-street quality is proven. The trainer can select the model by config:

- `model=preflop_allin` for existing datasets;
- `model=street_allin` for v2 flop/turn or mixed datasets.

## Phase 6: Multiway Flop/Turn Support
Do this after heads-up targets are tested.

Required pieces:

- reuse `_side_pot_layers` and folded/all-in eligibility logic from `p2.allin.sampler`;
- add a multiway board-fixed showdown evaluator returning per-seat per-hand equity vectors;
- for turn, stream river cards and accumulate numerator/denominator before converting to chip EV;
- for flop, either enumerate/sample turn-river runouts or use a board-completion sampler;
- validate against exact restricted-support enumerations before trusting large random ranges.

Do not build `1326 ** P` payoff tables. Use per-hand numerator/denominator accumulation and side-pot payout composition.

## Phase 7: Tests And Benchmarks
Add correctness tests before large-scale pregeneration.

Core tests:

- board sampling produces unique public cards and valid street/card counts;
- beliefs are zero on board-blocked combos and normalize per seat;
- suit permutation remaps board, beliefs, and targets consistently;
- heads-up flop target matches direct `compute_postflop_payoff_quantized` on small CPU cases;
- heads-up turn target matches explicit river enumeration on tiny support;
- side-pot payout conversion matches hand-written uneven all-in examples;
- v1 pregenerated datasets still load;
- v2 wrapped reads, pinned memory, and async prefetch preserve board/street tensors.

Benchmarks:

- examples/sec for flop and turn pregeneration;
- target kernel seconds by street;
- loader seconds with pinned prefetch;
- trainer step seconds by batch size and street mix.

## Suggested Implementation Order
1. Add `AllInBatch` with `street` and `board`, while preserving preflop compatibility.
2. Extend dataset serialization to v2 and add load tests.
3. Add board-aware heads-up flop/turn random batch generation.
4. Add heads-up postflop target dispatcher using `p2.search.allin_payoff`.
5. Add `pregenerate.py` support for `streets=[flop, turn]`.
6. Add `StreetAllInEquityModel` and trainer config selection.
7. Add flop/turn correctness tests and one microbenchmark.
8. Only then add mixed preflop/flop/turn datasets and multiway support.

## Open Decisions
- Whether `PreflopAllInBatch` becomes an alias of `AllInBatch` or remains a separate type.
- Whether flop targets should use full exact turn-river enumeration by default or sampled completions for throughput.
- Whether the first model should train flop and turn together or use separate street-specific heads.
- How realistic random postflop ranges need to be before replacing them with ranges from solved PBS states.
