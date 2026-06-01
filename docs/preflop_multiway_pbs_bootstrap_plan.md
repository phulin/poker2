# Multiway Preflop PBS Bootstrap Plan

## Goal
Use `PBSEnv` to solve and train multiway preflop public-belief states, then hand off only heads-up flop roots to the existing postflop ReBeL/CFR stack. The preflop model and main postflop model should be separate:

- Preflop model (`S_preflop` / `S_0`): trained only on preflop public states, including arbitrary-depth cutoffs. Unlike postflop `S_X` nets, `S_0` is an arbitrary preflop public-state model, not only a start-of-street model.
- Main model: trained only on flop, turn, river, and showdown-adjacent heads-up states.
- Handoff boundary: a two-player flop `PublicBeliefState` generated from a multiway preflop solve.

This is intentionally narrower than a full multiway ReBeL trainer. It uses multiway only where the real game starts multiway, then converts to heads-up before postflop play.

## Non-Goals
- Do not train the main model on preflop examples.
- Do not make fused sparse CFR multiway in the first implementation.
- Do not build an `H^P` private-hand payoff table for called all-ins.
- Do not require preflop search to expand through flop, turn, and river.
- Do not pretend the heads-up reduction is real poker semantics. The legal-action invariant (squeeze-or-fold past two matched seats) is an explicit abstraction and must be logged.

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
   - Converts closed-round preflop rows into heads-up flop roots.
   - Always natural heads-up: the legal-action invariant (enforced in the evaluator) guarantees ≤ 2 non-folded seats reach any flop, so betting-reached closed rows have exactly two live seats. No survivor selection.

The existing postflop trainer should receive roots from `PreflopHandoffBuilder` and train as a heads-up trainer with `env.num_players == 2`.

### Data Flow
1. Start from a fresh multiway `PBSEnv` reset with uniform preflop beliefs.
2. Solve a bounded preflop subgame under the legal-action invariant (≤ 2 non-folded seats reach any flop).
3. Store preflop policy and value examples from solved preflop nodes.
4. Sample preflop continuations:
   - some stop at arbitrary-depth cutoffs for preflop value training;
   - some continue until preflop closes or reaches a terminal fold/all-in state.
5. For rows that can produce a flop (already heads-up by the invariant):
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
    continuation_value_target_sampling: true
    continuation_value_target_streets: [0]
    continuation_value_target_min_depth: 0
    continuation_value_target_max_depth: 5
    continuation_value_targets_replace_roots: true
    bet_bins_by_depth:
      - [0.5, 1.0, 2.0]
      - [0.5, 1.0, 2.0, 4.0]
      - [0.75, 1.5, 3.0]
      - [1.0, 2.0]
      - []
    allin_by_depth: [true, true, true, false, false]
  invariant:
    max_players_to_flop: 2         # legal-action cap: ≤2 non-folded seats (all-in included) reach any flop
  handoff:
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
- `call_restricted`: bool — whether the legal-action invariant masked a flat-call at this node (for diagnosing how often the cap binds).
- `handoff_seats`: optional `[B, 2]` live seat ids for rows that become postflop roots.
- `value_target_source`: enum-like int for `preflop_cfr`, `postflop_model_expectation`, `allin_resolver`, or `fold_reward`.

### Handoff Output
`PreflopHandoffBuilder` should return:

```python
@dataclass
class PreflopHandoffBatch:
    pbs: PublicBeliefState          # PBSEnv with num_players == 2, street == 1
    source_env_indices: torch.Tensor
    source_seat_ids: torch.Tensor   # [B, 2], original multiway seat ids of the two live seats
    folded_seats: torch.Tensor | None  # seats that voluntarily folded preflop (for accounting/diagnostics)
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

Tree construction should still use `PBSEnv.gather_rows`, `legal_bins_mask`, and `step_bins`, but with two changes:

**Legal-action invariant (the forced-fold mechanism).** After `legal_bins_mask`, additionally mask out the flat-call/check-to-stay action at any node where calling would keep ≥ 3 non-folded seats — counting all-in seats — in at the current bet level (i.e., `non_folded_in_count == max_players_to_flop`). The acting player is then left with re-raise / all-in / fold only. Track the non-folded-in count at the current level dynamically as part of node state — it resets whenever a raise re-opens the action. This guarantees every flop is reached by exactly two non-folded players (and any flop with an all-in seat is resolved at all-in time, not played). Record `call_restricted` in node stats wherever the mask binds. (See **Forced Fold Via A Legal-Action Invariant** for rationale.)

Then classify leaves:

- `fold_terminal`: `env.done` and exactly one live player.
- `allin_terminal`: betting cannot continue and at least two live/all-in players need showdown EV.
- `closed_preflop`: `street` advanced from `0` to `1` (always exactly 2 non-folded seats by the invariant).
- `depth_cutoff`: depth reached `preflop.search.depth`.
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
3. Closed preflop (reaches flop): always `L == 2` (heads-up) by the legal-action invariant below; value with `E_preflop`. See **Forced Fold Via A Legal-Action Invariant**.
4. Depth/sampled cutoff:
   - use preflop value model.

### Forced Fold Via A Legal-Action Invariant
Forced fold is implemented as a **constraint on the legal-action set inside CFR**, not as a post-hoc reduction or a value assignment. This is what keeps it unbiased: every fold is a voluntary, EV-maximizing choice by the player, so there is no artificial payoff to assign and no incentive distortion to correct for.

**Invariant: at most two players (all-in seats included) may remain non-folded going to any flop.** A live player facing the action may **not** flat-call/check-to-stay if doing so would leave a 3rd non-folded player still in at the current bet level — counting all-in seats. Their legal set is then restricted to re-raise / all-in / fold. Raising re-opens the action (the previously-in players must re-decide); folding removes them.

This subsumes the all-in case: behind an all-in plus one caller, a third player must jam or fold, never flat-call. Consequently every flop is reached by **exactly two non-folded players**, which gives a crisp split with no hybrid:

- a flop reached with **any all-in seat involved** is an all-in situation → resolved at all-in time by `MultiwayAllInShowdownResolver` (showdown over runouts, side pots), no postflop betting;
- a flop reached **by betting** has exactly two chips-behind seats → a pure single-pot heads-up PBS, in-distribution for `S_flop`/`E_preflop`.

So the postflop nets only ever see clean heads-up flops, and the resolver (needed anyway for multiway all-in terminals) absorbs every multiway-at-showdown case. There is no main-pot-3-way-under-HU-betting case to special-case — the reason we do **not** allow all-in + 2 callers (that would be an irreducible 3-way main-pot showdown the HU net cannot represent).

Implementation: enforce in `legal_bins_mask` — mask the flat-call/check-to-stay action whenever it would keep ≥ 3 non-folded players (matched or all-in) in at the current bet level — tracking the dynamic non-folded-at-current-level count, **not** a static "first two" set (which breaks under fold dynamics: if one of the first two later folds to a 3-bet, a later seat may return to the pair).

**Boundary values become trivial.** The only closed-preflop (betting) boundary is `L == 2`, valued by the postflop model's flop expectation (`E_preflop` = expected-over-flops value of `S_flop`), zero-sum over the pair. There is no `L >= 3` boundary to value — the mask makes it unreachable. Cold start: if no postflop model exists yet, value the `L == 2` boundary with a showdown-equity proxy and tag targets bootstrap-quality; switch to the HU net once a postflop checkpoint exists (see Scheduling).

**What this costs (the abstraction, logged):** restricting flat-call for the 3rd+ player (including behind an all-in) turns multiway entry into squeeze-or-fold, so the solved game has no multiway limped/called pots, no overcalling a jam, and somewhat inflated jam/3-bet frequencies relative to real poker. The HU flops handed off therefore come from a slightly more aggressive preflop game. This is a deliberate, logged abstraction (Non-Goal: do not pretend it is real poker semantics), acceptable because the postflop model is HU-only.

**Multiway all-in showdowns are still allowed** and resolved by `MultiwayAllInShowdownResolver` with side pots — the invariant restricts *calling*, not jamming, so 3+ players can still be all-in and reach showdown without any postflop play.

### Sampling Cutoff Nodes
Replace `sample_leaves` for preflop with two outputs:

- preflop cutoff samples for training the preflop model;
- closed-preflop rows for heads-up handoff.

Use target-depth continuation sampling rather than a simple Bernoulli hazard:

1. For each root, sample a target depth uniformly from the configured `continuation_value_target_*` range.
2. Descend one path under `policy_probs_sample`, with the existing epsilon-uniform exploration mix.
3. Before continuing from a node at the sampled target depth, abort that path and emit that node as a preflop value target.
4. If a path reaches a fold/all-in/closed-preflop terminal before the target depth, do not turn that terminal into an arbitrary-state `S_preflop` target; route it to the terminal/handoff machinery instead.
5. When `continuation_value_targets_replace_roots` is enabled, use these generated abort nodes as the value batch instead of the original roots. This keeps value examples balanced by generation-time depth sampling rather than replay-buffer stratification.
6. Continue non-aborted paths to closed preflop for handoff generation.

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

## Heads-Up Handoff

Because the legal-action invariant (see **Forced Fold Via A Legal-Action Invariant**) caps non-folded seats reaching any flop at 2, every closed-preflop row that reaches a flop by betting is **already heads-up**. There is no survivor selection or forced-fold step here — the handoff just compacts the two live seats into a two-player `PBSEnv`. (The folds that produced the heads-up situation already happened, voluntarily, during preflop betting.)

### When It Runs
Produce a postflop root from preflop rows where:

- betting round is closed;
- `street` would advance to flop;
- exactly two non-all-in live seats remain (guaranteed by the invariant);
- no all-in seat is involved (any all-in line is resolved at all-in time by the resolver, never handed off as a flop).

Rows with one live seat are fold terminals and do not produce postflop roots.

### Seat Compaction
For the two live seats:

- compact them into a new two-player `PBSEnv` (do **not** mutate the preflop evaluator env in place);
- recompute `scale` as the effective or min starting-stack policy chosen for heads-up training;
- preserve `pot`, `stacks`, `committed`, `chips_placed`, `button`, and `to_act` semantics for the flop start;
- preserve folded/all-in seats' committed chips in the pot;
- set `street = 1`, `actions_this_round = 0`, `committed = 0` for the new betting round unless the existing postflop code expects another convention;
- store original seat ids in handoff metadata.

### Belief Compaction
For the two live seats `s0, s1`:

1. Gather `beliefs[:, [s0, s1], :]`.
2. Mask flop-blocked combos after flop cards are known.
3. Normalize each range independently.
4. If a range becomes empty, fall back to uniform over board-legal combos and log `handoff_empty_range_fallback`.

The first version should not condition out cross-player private-card collisions during handoff. The postflop CFR and losses already use blocker-aware weights. Add a later enhancement for collision-conditioned reweighting if diagnostics show artifacts.

## Flop Sampling And Postflop Value Feedback

Preflop closed-round leaves need an estimate of the value of reaching the flop. By the legal-action invariant these are always heads-up (`L == 2`), so this helper only needs the HU branch.

Add `PreflopToFlopValueHelper`:

1. Takes closed preflop PBSEnv rows (exactly two non-all-in live seats) and their beliefs.
2. Sample `S` flops per row (or enumerate in validation mode).
3. Mask the two seats' beliefs by each flop.
4. Encode heads-up postflop value features and call the postflop value model.
5. Aggregate expected-over-flops values back to the two live seats; folded/all-in seats already have terminal/resolver values.

Default `S=1` for throughput during training, `S=64` or exhaustive for eval/debug. Store `flop_sample_count` in statistics so noisy targets can be filtered or reweighted.

(There is no `L >= 3` branch: such closed-preflop nodes are unreachable. Multiway all-in *showdowns* are handled separately by `MultiwayAllInShowdownResolver` at all-in terminals.)

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
- hand-class strength lookup (for diagnostics / range summaries).

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
- diagnostics: depth coverage, call-restriction rate, all-in resolver mode, live count, action mix by depth.

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

### Phase 2: Legal-Action Invariant + Handoff Builder
- Implement the legal-action invariant (mask flat-call past `max_players_to_flop` non-folded seats, all-in included) in the preflop evaluator's masking, with dynamic non-folded-count tracking.
- Implement natural two-player handoff (seat + belief compaction).
- Implement flop sampling and belief masking.
- Add metadata and diagnostics.

Acceptance:
- No solved tree ever closes a non-all-in round with ≥ 3 matched seats.
- Handoff conserves chips across compacted rows.
- Every output row has `num_players == 2`, `street == 1`, and normalized beliefs.
- Forced-fold rows log original folded seats.

### Phase 3: Preflop Model And Feature Encoder
- Add preflop context enums and encoder.
- Add `BetterPreflopFFN` or split policy/value modules for `S_preflop` / `S_0`.
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
- Cutoff depth histogram follows the configured target-depth sampling range.
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
- `test_preflop_legal_actions_cap_called_players` (flat-call masked when it would create a 3rd matched seat; matched count resets after a re-raise)
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
- 4-player preflop solve under the legal-action invariant, producing heads-up two-player flop roots (assert no closed round has ≥3 matched seats).
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

Invariant / handoff:

- `preflop/call_restricted_frac` — how often the legal-action cap masks a flat-call.
- `preflop/closed_round_live_count_hist` — should have zero mass at ≥3 non-all-in.
- `handoff/rows`
- `handoff/flop_sample_count`
- `handoff/empty_range_fallback_frac`
- `handoff/live_seat_position_hist` — positions of the two seats reaching the flop.
- `handoff/folded_committed_chips_mean` — committed chips left in the pot by voluntary preflop folds.

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

- Risk: the legal-action invariant (squeeze-or-fold for the 3rd+ player) distorts preflop ranges away from real poker — no multiway limped/called pots, inflated 3-bet frequencies.
  - Mitigation: this is the accepted, logged abstraction; every fold under it is voluntary so there is no value-assignment bias. Monitor `call_restricted_frac` and 3-bet frequencies; if the distortion is unacceptable, the fallback is full multiway postflop (a separate, larger project) rather than re-introducing post-hoc forced folds.
- Risk: forcing jam-or-fold behind an all-in (the ≤2-total rule) inflates jam frequency and removes overcall-a-jam lines.
  - Mitigation: accepted — it keeps every postflop flop a pure heads-up PBS and avoids the irreducible 3-way main-pot showdown that allowing all-in + 2 callers would create. Monitor `call_restricted_frac` and jam frequency; the fallback if the distortion is unacceptable is full multiway postflop, not allowing 3-way showdowns into the HU net.

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
- `search.continuation_value_target_sampling = true`.
- `search.continuation_value_target_streets = [0]`.
- `search.continuation_value_target_min_depth = 0`.
- `search.continuation_value_target_max_depth = 4`.
- `search.continuation_value_targets_replace_roots = true`.
- `preflop.invariant.max_players_to_flop = 2` (all-in seats included; jam-or-fold behind an all-in).
- `preflop.handoff.flop_samples_per_closed_row = 1` during training.
- `preflop.allin.board_sample_count = 128` for preflop called all-ins at first.
- Disable fused sparse CFR and all heads-up all-in payoff tables in preflop mode.

## Open Decisions
- Whether first production preflop experiments target 4-way or 6-way.
- Whether the squeeze-or-fold distortion of preflop ranges (now including jam-or-fold behind an all-in) is acceptable, or whether it eventually justifies full multiway postflop.
- Whether closed-preflop preflop value targets should use a frozen postflop model for a fixed number of steps.
- Whether postflop roots should be generated only from average policy handoffs or also final policy handoffs.
- Whether all-in resolver approximation is acceptable for more than 4 live all-in players, or whether such spots should be discouraged by action schedule until exact support exists.
