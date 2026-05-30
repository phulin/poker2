# Multiway Preflop PBS Bootstrap Plan

## Goal
Use `PBSEnv` to solve and train multiway preflop public-belief states, then hand off only heads-up flop roots to the existing postflop ReBeL/CFR stack. The preflop model and main postflop model should be separate:

- Preflop model: trained only on preflop public states, including arbitrary-depth cutoffs.
- Main model: trained only on flop, turn, river, and showdown-adjacent heads-up states.
- Handoff boundary: a two-player flop `PublicBeliefState` generated from a multiway preflop solve.

This is intentionally narrower than a full multiway ReBeL trainer. It uses multiway only where the real game starts multiway, then converts to heads-up before postflop play.

## Non-Goals
- Do not train the main model on preflop examples.
- Do not make fused sparse CFR multiway in the first implementation.
- Do not build an `H^P` private-hand payoff table for called all-ins.
- Do not require preflop search to expand through flop, turn, and river.
- Do not pretend forced-fold handoff is real poker semantics. It is an explicit abstraction and must be logged.

## Current Repo Facts
- `PBSEnv` already supports public multi-player betting state, per-seat stacks and commitments, folds, all-ins, side-pot payout accounting, public board dealing, row gather/copy, and `[B, P, 1326]` marginal beliefs.
- `SparseCFREvaluator` already accepts `PBSEnv` and allocates `beliefs`, `values`, and `self_reach` as `[node, num_players, NUM_HANDS]`.
- Several evaluator paths are still heads-up biased:
  - `_showdown_value_both` and `_showdown_value` are heads-up only.
  - `AllInPayoffResolver` and Triton all-in writeback kernels are heads-up only.
  - `_best_response_values`, exploitability diagnostics, and some reach weighting assumptions are heads-up oriented.
  - `FusedSparseCFREvaluator` and `subgame_constructor_triton.py` are explicitly two-player in layout and kernels.
- `BetterFFN` already accepts `num_players` and uses flattened beliefs with shape `[B, P * 1326]`, but the current `BetterStreetValueFFN` is designed around street-boundary pre/post chance heads rather than arbitrary preflop interior states.
- `RebelSupervisedLoss` already has partially generalized multiway policy/value weighting using products of pairwise unblocked masses. This can be reused for a first preflop model, but should be validated as an approximation.
- `p2.showdown.exact` and `p2.showdown.approximate` contain multiway by-hand equity machinery that is a better foundation for called all-in resolution than the current heads-up all-in payoff tables.

## Target Architecture

### Components
Add three separate subsystems:

1. `PreflopDataGenerator`
   - Owns multiway preflop roots.
   - Runs `PreflopSparseCFREvaluator`.
   - Emits preflop policy/value batches for the preflop model.
   - Emits two-player flop roots for the postflop trainer.

2. `PreflopSparseCFREvaluator`
   - Uses `PBSEnv` only.
   - Expands only street-0 betting.
   - Stops at configurable depth, stochastic cutoffs, closed preflop rounds, folds, and all-in-call leaves.
   - Uses the preflop value model for nonterminal preflop cutoffs.
   - Uses a multiway all-in resolver for called all-in leaves.

3. `PreflopHandoffBuilder`
   - Converts multiway preflop terminal or closed-round rows into heads-up flop roots.
   - Natural path: if exactly two players remain live, keep them.
   - Forced-fold path: if more than two remain live, choose two survivors and fold the rest before dealing or materializing flop roots.

The existing postflop trainer should receive roots from `PreflopHandoffBuilder` and train as a heads-up trainer with `env.num_players == 2`.

### Data Flow
1. Start from a fresh multiway `PBSEnv` reset with uniform preflop beliefs.
2. Solve a bounded preflop subgame.
3. Store preflop policy and value examples from solved preflop nodes.
4. Sample preflop continuations:
   - some stop at arbitrary-depth cutoffs for preflop value training;
   - some continue until preflop closes or reaches a terminal fold/all-in state.
5. For rows that can produce a flop:
   - force or naturally reduce live seats to exactly two;
   - deal or sample a flop;
   - produce a heads-up `PublicBeliefState`.
6. Feed those heads-up flop roots into the postflop data generator.
7. Train models independently:
   - preflop optimizer reads preflop replay;
   - postflop optimizer reads postflop replay only.

## Configuration

Add a dedicated config subtree instead of overloading every existing `search.*` field:

```yaml
preflop:
  enabled: false
  num_players: 6
  target_flop_players: 2
  evaluator: sparse
  search:
    depth: 5
    iterations: 80
    warm_start_iterations: 8
    cfr_type: linear
    cfr_avg: true
    sample_epsilon: 0.15
    bet_bins_by_depth:
      - [0.5, 1.0, 2.0]
      - [0.5, 1.0, 2.0, 4.0]
      - [0.75, 1.5, 3.0]
      - [1.0, 2.0]
      - []
    allin_by_depth: [true, true, true, false, false]
  cutoff:
    min_depth: 1
    force_depth: 5
    stratified_quotas: [0.05, 0.15, 0.25, 0.30, 0.25]
    include_root_values: true
    target_kind: avg_policy
  handoff:
    mode: forced_fold_to_hu
    survivor_policy: deterministic_score
    survivor_temperature: 0.0
    flop_samples_per_closed_row: 1
    preserve_button_relative_order: true
  allin:
    resolver: streaming_by_hand
    board_sample_count: 256
    exact_when_live_players_lte: 4
    chunk_size: 128
  replay:
    value_capacity_batches: 512
    policy_capacity_batches: 512
    depth_stratify_sample: true
```

Keep postflop under the current `env` and `search` config, but make the root source explicit:

```yaml
postflop:
  root_source: preflop_handoff
  min_street: flop
  reject_preflop_batches: true
```

## Public Interfaces And Types

### Preflop Batch Metadata
Add statistics to preflop replay batches:

- `street`: always `0`.
- `node_depth`: same-street search depth.
- `cutoff_kind`: enum-like int for `root`, `sampled_depth`, `force_depth`, `fold_terminal`, `allin_terminal`, `closed_round`.
- `live_count`: number of non-folded seats.
- `eligible_count`: number of seats that can still bet.
- `actor`: current `to_act`.
- `button`: source button.
- `forced_fold_applied`: bool.
- `handoff_survivors`: optional `[B, 2]` seat ids for rows that become postflop roots.
- `value_target_source`: enum-like int for `preflop_cfr`, `postflop_model_expectation`, `allin_resolver`, or `fold_reward`.

### Handoff Output
`PreflopHandoffBuilder` should return:

```python
@dataclass
class PreflopHandoffBatch:
    pbs: PublicBeliefState          # PBSEnv with num_players == 2, street == 1
    source_env_indices: torch.Tensor
    source_seat_ids: torch.Tensor   # [B, 2], original multiway seat ids
    forced_fold_mask: torch.Tensor  # [B]
    forced_folded_seats: torch.Tensor | None
    statistics: dict[str, torch.Tensor]
```

The handoff `pbs.beliefs` shape is `[B, 2, NUM_HANDS]`. The handoff env should be a `PBSEnv`, not `HUNLTensorEnv`, because private cards remain represented by beliefs.

### Model Output
The preflop model should use the existing `ModelOutput` shape conventions:

- `policy_logits`: `[B, NUM_HANDS, A]`
- `hand_values`: `[B, P, NUM_HANDS]`
- `value`: `[B, P]` optional aggregate for logging only

## Preflop Sparse CFR Evaluator

### Construction
Implement a new evaluator class rather than mutating the generic evaluator heavily:

- `src/p2/search/preflop_sparse_cfr_evaluator.py`
- subclass `SparseCFREvaluator` if practical, but override leaf classification, terminal value setting, exploitability/stat recording, and sample-leaf behavior.
- require `isinstance(src_env, PBSEnv)`.
- require all source roots have `street == 0`.
- reject `cfg.search.sparse_fused == true`.
- reject postflop roots.

Tree construction should still use `PBSEnv.gather_rows`, `legal_bins_mask`, and `step_bins`, but classify leaves differently:

- `fold_terminal`: `env.done` and exactly one live player.
- `allin_terminal`: betting cannot continue and at least two live/all-in players need showdown EV.
- `closed_preflop`: `street` advanced from `0` to `1`.
- `depth_cutoff`: depth reached `preflop.cutoff.force_depth`.
- `sampled_cutoff_candidate`: nonterminal street-0 nodes at allowed depths.

For `closed_preflop`, do not expand flop actions in this evaluator. Mark as a handoff leaf.

### Belief Propagation
At each child node:

1. Fan out parent beliefs to child.
2. Identify previous actor from `prev_actor`.
3. Multiply only that actor's belief slice by the policy probability for the chosen action and private hand.
4. Leave non-actor belief slices unchanged.
5. For folded actors, keep their belief tensor for value accounting but mark them inactive in masks and losses.
6. Normalize each non-folded seat independently.
7. Apply public board blockers only when a flop is materialized for handoff, not during preflop betting.

The current generic `_propagate_level_beliefs` already does most of step 1-4, but the preflop evaluator should add explicit tests around folded-seat behavior and live-seat masks.

### CFR Backup And Regrets
Use actor-local regrets:

- Policy and regret tensors stay `[node, NUM_HANDS]` for child-action rows, matching the existing sparse evaluator layout.
- At a decision node, update regrets only for `env.to_act[node]`.
- Do not compute heads-up exploitability for `P > 2`.
- For non-actor value backup, reuse the current opponent-conditioned policy approximation initially:
  - project actor action probabilities through blocker-compatible mass;
  - apply the same action distribution to non-actor hand values.
- Log this as `multiway_backup_approximation=pairwise_unblocked_product`.

This is good enough for a first implementation because the preflop model is a bootstrap model, but add correctness tests against tiny restricted supports before trusting training results.

### Leaf Value Sources
Leaf value priority:

1. Fold terminal:
   - use `PBSEnv` reward rows, broadcast to `[P, H]`.
2. Called all-in terminal:
   - use `MultiwayAllInShowdownResolver`.
3. Closed preflop:
   - either use postflop model flop expectation for preflop target generation, or mark for handoff only depending on call site.
4. Depth/sampled cutoff:
   - use preflop value model.

For CFR iterations, closed-preflop leaves need values so preflop actions know what reaching the flop is worth. First version should call the current postflop value model through a sampled flop expectation helper. If no postflop model is available during cold start, use a simple showdown/equity proxy and tag the targets as bootstrap-quality.

### Sampling Cutoff Nodes
Replace `sample_leaves` for preflop with two outputs:

- preflop cutoff samples for training the preflop model;
- closed-preflop rows for heads-up handoff.

Use stratified quotas rather than a simple Bernoulli hazard:

1. Build candidate masks by depth for nonterminal street-0 nodes.
2. For each root, sample one path under `policy_probs_sample`.
3. Along each path, record visited nodes by depth.
4. Draw cutoff nodes to match `preflop.cutoff.stratified_quotas`.
5. Always include forced-depth nodes when no earlier cutoff is selected.
6. Continue a configurable fraction of paths to closed preflop for handoff generation.

This avoids a replay buffer dominated by shallow opening states.

## Preflop Model Architecture

### Recommended First Model
Add `BetterPreflopFFN` and optionally `BetterPreflopSplitFFN`.

Reuse the useful parts of `BetterFFN`:

- hand embedding over 1326 combos;
- belief projection using per-player `belief @ hand_emb`;
- low-rank policy head;
- per-hand value output.

Change the context:

- remove board embedding from the main preflop model path;
- add depth embedding;
- add live-count embedding;
- add action-history summary features;
- add per-seat status features.

### Context Features
Scalar features:

- current actor id;
- button-relative current actor position;
- same-street depth;
- actions this round;
- number of live seats;
- number of all-in seats;
- pot / scale;
- min raise / scale;
- max committed / scale;
- to-call / scale for current actor;
- log stack depth in bb;
- log pot in bb;
- cutoff phase flag: root/interior/closed-preflop/all-in.

Per-seat features, flattened seat-major:

- stack / scale;
- committed / scale;
- chips placed / scale;
- SPR;
- to-call / scale;
- folded flag;
- all-in flag;
- acted-this-round flag;
- live flag;
- can-act flag;
- button-relative position;
- blind role flags: button, small blind, big blind;
- current actor flag;
- last aggressor flag if added to env state.

Range features:

- `[B, P, 1326]` beliefs.
- shared per-player hand embedding projection.
- optional seat embedding added before flattening player belief summaries.
- optional per-seat range-card-mass summaries `[B, P, 52]` for fold-gate and all-in resolver reuse.

### Outputs
Policy:

- one actor-local policy head;
- output `[B, 1326, A]`;
- mask legal actions outside the model;
- action schedule can differ from postflop.

Value:

- output `[B, P, 1326]`;
- train every live and folded seat target when chip accounting defines a value;
- mask folded seat policy loss;
- keep folded seat value loss only for rows where terminal or cutoff value is meaningful.

Do not reuse `BetterStreetValueFFN` pre/post chance heads directly for preflop interior states. If the implementation wants maximum reuse, create a `PreflopValueFeatureEncoder` and a single-head value model first, then split heads only if diagnostics show target interference.

## Forced-Fold Handoff

### When It Runs
Run forced-fold only on preflop rows where:

- betting round is closed;
- `street` would advance to flop;
- live non-folded count is greater than 2;
- no unresolved all-in side-pot showdown requires immediate terminal evaluation.

Rows with exactly two live seats skip forced folding. Rows with one live seat are fold terminals and do not produce postflop roots.

### First Survivor Scoring Algorithm
Use a deterministic vectorized score per live seat:

```text
score =
  range_strength
  + commitment_pressure
  + stack_playability
  + position_bonus
  - allin_penalty
```

Where:

- `range_strength`: belief-weighted preflop hand-class score from a static `[1326]` or `[169]` lookup.
- `commitment_pressure`: `committed / max_committed` or `committed / pot`.
- `stack_playability`: `log1p(stack / bb)` clipped.
- `position_bonus`: button-relative bonus, small by default.
- `allin_penalty`: configurable, usually zero if all-in seats should be eligible to continue.

Apply `score.masked_fill_(folded, -inf)`, then choose top 2. For stochastic diversity, sample without replacement from `softmax(score / temperature)` when `temperature > 0`.

### Accounting
After selecting survivors:

- mark non-survivor live seats as folded in a temporary source row;
- preserve their committed chips in the pot;
- compact the two survivor seats into a new two-player `PBSEnv`;
- recompute `scale` as the effective or min starting stack policy chosen for heads-up training;
- preserve `pot`, survivor `stacks`, `committed`, `chips_placed`, `button`, and `to_act` semantics for the flop start;
- set `street = 1`, `actions_this_round = 0`, `committed = 0` for the new betting round unless the existing postflop code expects another convention;
- store original seat ids in handoff metadata.

Important implementation detail: do not mutate the original preflop evaluator env in place. Build handoff rows into a fresh `PBSEnv(num_players=2)`.

### Belief Compaction
For survivors `s0, s1`:

1. Gather `beliefs[:, [s0, s1], :]`.
2. Mask flop-blocked combos after flop cards are known.
3. Normalize each survivor range independently.
4. If a range becomes empty, fall back to uniform over board-legal combos and log `handoff_empty_range_fallback`.

The first version should not condition out cross-player private-card collisions during handoff. The postflop CFR and losses already use blocker-aware weights. Add a later enhancement for collision-conditioned reweighting if diagnostics show artifacts.

## Flop Sampling And Postflop Value Feedback

Preflop closed-round leaves need an estimate of the value of reaching the flop.

Add `PreflopToFlopValueHelper`:

1. Takes closed preflop PBSEnv rows and multiway beliefs.
2. Applies natural or forced-fold handoff to two players.
3. Samples `S` flops per row, or enumerates flops in validation mode.
4. Masks survivor beliefs by each flop.
5. Encodes heads-up postflop value features.
6. Calls the current postflop value model.
7. Aggregates expected values back to original survivor seats.
8. Assigns folded non-survivor values from chip accounting at the forced-fold abstraction.

Default `S=1` for throughput during training, `S=64` or exhaustive for eval/debug. Store `flop_sample_count` in statistics so noisy targets can be filtered or reweighted.

Cold start options:

- Option A: initialize postflop model first from existing heads-up training, then enable preflop.
- Option B: use a static equity proxy for preflop closed leaves until postflop model has enough replay.
- Recommended first rollout: Option A. It avoids bootstrapping two untrained value models at once.

## Called All-In Resolver Without `H^P`

### Contract
Add a resolver with this shape:

```python
class MultiwayAllInShowdownResolver:
    def values(
        self,
        env: PBSEnv,
        node_indices: torch.Tensor,
        beliefs: torch.Tensor,  # [M, P, H]
    ) -> torch.Tensor:          # [M, P, H]
        ...
```

It should return per-seat per-hand values in stack-normalized units, matching evaluator value target conventions.

### Strategy
Avoid `H^P` by computing per-hero-hand equity vectors with inclusion-exclusion/rank-prefix methods and streaming board runouts:

- For river boards:
  - use exact rank-prefix by-hand evaluator directly.
- For turn boards:
  - stream 44 or fewer river cards;
  - evaluate by-hand equity per river chunk;
  - accumulate numerator and denominator.
- For flop boards:
  - stream turn-river combinations in chunks;
  - reuse board/rank descriptors and combo-card incidence.
- For preflop all-ins:
  - sample or stream five-card boards;
  - cache board descriptors, not private-hand joint outcomes;
  - aggregate per-hand values over board chunks.

Use existing `p2.showdown.exact.exact_nway_ie_axb_by_hand` when live players are `<= 4`. For more live all-in players, begin with approximate tier or sampled conditional by-hand estimator, with a clear statistic `allin_resolver_exact=false`.

### Side Pots
Resolve side pots layer by layer:

1. Use `PBSEnv.chips_placed` to derive contribution levels.
2. For each layer, compute participant mask.
3. Eligible winners are participants who have not folded.
4. Run equity only among eligible live players for that layer.
5. Add layer payout EV to each eligible seat.
6. Convert final stack plus expected payouts minus starting stack to the same normalized value scale as `PBSEnv._reward_rows`.

Do not use `PBSEnv.expected_showdown_rewards` as the training resolver because it documents that private-card collisions between seats are not conditioned out. Keep it as a fast diagnostic/reference for marginal-product sanity checks.

### Precomputation
Cache these structures:

- combo cards `[1326, 2]`;
- combo-card incidence or card sums;
- board-allowed hand masks;
- rank vectors for each materialized board chunk;
- sorted hand indices and rank group spans;
- canonical flop/turn suit remap metadata where useful;
- hand-class strength lookup for forced-fold scoring.

Do not cache:

- `[H, H, H]`, `[H, H, H, H]`, or larger tables;
- per-player joint private assignments;
- full preflop board by private-tuple outcomes.

## Training Loops

### Preflop Trainer
Add `PreflopCFRTrainer` or a preflop mode inside `RebelCFRTrainer` with separate model/optimizer/replay:

- env: `PBSEnv(num_players=preflop.num_players)`;
- evaluator: `PreflopSparseCFREvaluator`;
- model: `BetterPreflopSplitFFN`;
- replay: preflop-only value and policy buffers;
- batches: reject `street != 0`;
- diagnostics: depth coverage, forced-fold rate, all-in resolver mode, live count, action mix by depth.

### Postflop Trainer
Keep current heads-up evaluator where possible, but accept `PBSEnv(num_players=2)` roots:

- batches: reject `street == 0`;
- root source: preflop handoff queue;
- fallback root source for debugging: existing random heads-up postflop sampler;
- model: current main model architecture.

If `HUNLTensorEnv` assumptions block `PBSEnv(2)` postflop training, add a small adapter layer rather than contaminating the preflop path with private-card env state.

### Scheduling
Recommended rollout schedule:

1. Train or load a usable postflop heads-up value model.
2. Enable preflop evaluator with closed-preflop leaves valued by the postflop model.
3. Train preflop model from arbitrary-depth cutoffs.
4. Feed preflop handoff roots back into postflop replay.
5. Alternate updates:
   - `K_preflop` preflop batches;
   - `K_postflop` postflop batches from handoff queue;
   - sync inference copies.

Avoid fully circular bootstrapping until each side has a stable checkpoint.

## Implementation Phases

### Phase 1: Safety Rails And Config
- Add config fields.
- Add validation that preflop mode requires `PBSEnv`, sparse evaluator, and `street == 0`.
- Add postflop batch validation that rejects preflop rows.
- Disable heads-up exploitability stats for `num_players > 2` in the preflop evaluator.
- Add docs and smoke tests only.

Acceptance:
- Config loads.
- Existing heads-up tests still pass.
- Preflop mode cannot accidentally route through fused sparse CFR.

### Phase 2: Handoff Builder
- Implement natural two-player handoff.
- Implement forced-fold top-2 score handoff.
- Implement flop sampling and belief masking.
- Add metadata and diagnostics.

Acceptance:
- Handoff conserves chips across compacted rows.
- Every output row has `num_players == 2`, `street == 1`, and normalized beliefs.
- Forced-fold rows log original folded seats.

### Phase 3: Preflop Model And Feature Encoder
- Add preflop context enums and encoder.
- Add `BetterPreflopFFN` or split policy/value modules.
- Add replay/loss smoke tests for `P in {3, 4, 6}`.

Acceptance:
- Forward shapes match contract.
- Loss can train policy-only, value-only, and both.
- No postflop board dependency exists in preflop model.

### Phase 4: Preflop Sparse Evaluator
- Add preflop-specific leaf classification.
- Add sampled and forced depth cutoffs.
- Add preflop training data assembly.
- Add closed-preflop leaf value helper using postflop value model or proxy.

Acceptance:
- Evaluator produces preflop policy/value batches.
- Cutoff depth histogram follows configured quotas.
- Closed-preflop leaves do not expand flop actions.

### Phase 5: All-In Resolver
- Implement exact by-hand resolver for up to 4 live players.
- Add side-pot layer accumulation.
- Add sampled/approx fallback for larger live counts.
- Integrate with preflop evaluator all-in leaves.

Acceptance:
- Tiny-support brute force parity.
- Side-pot parity with scalar `NLEnv`.
- Memory use does not scale as `H^P`.

### Phase 6: Integrated Bootstrap
- Wire preflop generator to postflop handoff queue.
- Train preflop model and postflop model independently.
- Add checkpointing for both models and both replay buffers.
- Add diagnostics and benchmark scripts.

Acceptance:
- Main model replay contains no preflop rows.
- Preflop replay contains no postflop rows.
- Handoff queue has stable throughput.
- Training step can run end to end on a small CUDA batch.

## Tests

### Unit Tests
- `test_preflop_handoff_natural_two_players`
- `test_preflop_handoff_forced_fold_top2`
- `test_preflop_handoff_beliefs_mask_flop_blockers`
- `test_preflop_handoff_chip_conservation`
- `test_preflop_feature_encoder_shapes_p3_p4_p6`
- `test_preflop_model_forward_policy_value_shapes`
- `test_preflop_cutoff_depth_quotas_deterministic`
- `test_preflop_evaluator_rejects_postflop_roots`
- `test_postflop_replay_rejects_preflop_rows`
- `test_multiway_allin_resolver_tiny_support_bruteforce`
- `test_multiway_allin_sidepot_matches_nl_env`

### Integration Tests
- Small 3-player preflop solve with depth 2 and no all-in abstraction.
- 4-player preflop solve with forced-fold handoff into two-player flop roots.
- Cold-start proxy leaf values produce finite targets.
- Postflop value model leaf feedback produces finite preflop values.
- One training step each for preflop and postflop models.

### Performance Tests
- Preflop sparse construction nodes/sec by player count and depth.
- Cutoff sampling throughput.
- Handoff rows/sec with `S=1`, `S=8`, and `S=64` flop samples.
- All-in resolver throughput for 2, 3, and 4 live players.
- Peak memory for all-in resolver to prove no `H^P` allocation.

## Diagnostics

Log these under clear namespaces:

Preflop evaluator:

- `preflop/nodes_total`
- `preflop/tree_depth`
- `preflop/live_count_mean`
- `preflop/cutoff_depth_frac/*`
- `preflop/action_mix_by_depth/*`
- `preflop/allin_leaf_frac`
- `preflop/closed_round_frac`
- `preflop/value_target_std`
- `preflop/value_sum_drift`

Handoff:

- `handoff/rows`
- `handoff/forced_fold_frac`
- `handoff/natural_hu_frac`
- `handoff/flop_sample_count`
- `handoff/empty_range_fallback_frac`
- `handoff/survivor_position_hist`
- `handoff/forced_fold_committed_chips_mean`

All-in:

- `allin/live_players`
- `allin/exact_frac`
- `allin/sample_count`
- `allin/chunk_size`
- `allin/seconds`
- `allin/peak_memory_mb`

Postflop:

- `postflop/root_source_preflop_handoff_frac`
- `postflop/root_street_flop_frac`
- `postflop/rejected_preflop_rows`

## Risks And Mitigations

- Risk: preflop values chase an unstable postflop model.
  - Mitigation: start from a postflop checkpoint, freeze it for initial preflop training, then alternate slowly.

- Risk: forced-fold abstraction creates biased flop ranges.
  - Mitigation: log forced-fold rate and survivor distributions; compare deterministic top-2 against stochastic survivor sampling; keep natural heads-up closed rows separate in metrics.

- Risk: multiway regret weighting is approximate.
  - Mitigation: validate on tiny restricted-support games and monitor exploitability-like local regret metrics rather than claiming exact multiway exploitability.

- Risk: all-in resolver is too slow.
  - Mitigation: exact only for small live counts, sampled fallback for larger counts, cache board descriptors, and benchmark before enabling high all-in frequencies.

- Risk: replay contamination between preflop and postflop models.
  - Mitigation: hard assertions on `street`, separate buffers, separate checkpoint keys, and statistics for rejected rows.

## Recommended Initial Defaults
- `preflop.num_players = 4` for first real experiments, then scale to 6.
- `preflop.search.depth = 4`.
- `preflop.search.iterations = 64`.
- `preflop.cutoff.stratified_quotas = [0.10, 0.20, 0.35, 0.35]`.
- `preflop.handoff.survivor_policy = deterministic_score`.
- `preflop.handoff.flop_samples_per_closed_row = 1` during training.
- `preflop.allin.board_sample_count = 128` for preflop called all-ins at first.
- Disable fused sparse CFR and all heads-up all-in payoff tables in preflop mode.

## Open Decisions
- Whether first production preflop experiments target 4-way or 6-way.
- Whether forced-fold survivor selection should prefer strongest ranges or most committed ranges when those disagree.
- Whether closed-preflop preflop value targets should use a frozen postflop model for a fixed number of steps.
- Whether postflop roots should be generated only from average policy handoffs or also final policy handoffs.
- Whether all-in resolver approximation is acceptable for more than 4 live all-in players, or whether such spots should be discouraged by action schedule until exact support exists.
