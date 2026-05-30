# Multiway ReBeL Trainer and CFR Plan

## Goal
Extend the ReBeL-style public-belief trainer from heads-up HUNL to 3- and 4-way no-limit Hold'em without assuming that every local search can be expanded to the end of the street.

The main design shift is that the value model must learn arbitrary public states at every position and every depth within a street. In heads-up, the current path can often treat the value head as a street-boundary model and rely on same-street CFR to reach an end-of-street cutoff. In 3- and 4-way, the branching factor makes that assumption too expensive and too brittle, so the data generator must sometimes abort while descending sampled leaves and train value targets at those interior nodes.

## Current Heads-Up Assumptions
- `src/p2/rl/cfr_trainer.py` sets `self.num_players = 2`, builds `HUNLTensorEnv`, and sizes models/replay buffers around two ranges.
- `src/p2/search/sparse_cfr_evaluator.py` and `src/p2/search/fused_sparse_cfr_evaluator.py` set `num_players = 2` and allocate beliefs, reaches, and values as `[node, 2, 1326]`.
- `src/p2/models/mlp/mlp_features.py`, `src/p2/models/mlp/better_feature_encoder.py`, and `src/p2/rl/rebel_replay.py` flatten beliefs as `2 * NUM_HANDS`.
- `src/p2/rl/losses.py` policy weighting assumes one actor and one opponent range.
- `BetterStreetValueFFN` separates pre-chance and post-chance value heads and is currently used as a street-boundary value model.
- The all-in and showdown acceleration paths are heads-up oriented and should not be assumed correct for multiway side-pot or multi-winner cases.

## Target Shape
The multiway path should keep internal tensors seat-major and only flatten at module boundaries:

- Beliefs: `[B, P, 1326]`
- Hand values: `[B, P, 1326]`
- Scalar legal masks: `[B, A]` for the current actor
- Policy targets: `[B, 1326, A]` for the current actor's private hand
- CFR policy/regret tensors: `[node, 1326, A]` because each node has one actor
- CFR value/reach tensors: `[node, P, 1326]`

`P` should be configured, with initial support for `P in {2, 3, 4}`. Keep heads-up compatibility as the regression baseline.

## Phase 1: Environment and Config
1. Add `env.num_players` to the ReBeL config and thread it through `RebelCFRTrainer`, model constructors, replay buffers, feature encoders, and CFR evaluators.
2. Use `PBSEnv` or `TritonPBSEnv` as the multiway public environment instead of extending `HUNLTensorEnv`.
3. Keep `HUNLTensorEnv` as the heads-up fast path until the multiway path is verified.
4. Audit `PBSEnv` for full poker accounting before relying on it for training:
   - side pots after uneven all-ins,
   - multiway showdown winners and ties,
   - folded/all-in player eligibility,
   - public-card dealing after sampled re-rooting.
5. Add a `search.multiway_enabled` or equivalent guard so early multiway code cannot silently run with heads-up-only evaluators.

## Phase 2: Public Belief State Generalization
1. Make `PublicBeliefState` environment-agnostic enough to hold either `HUNLTensorEnv` or `PBSEnv`.
2. Store beliefs as `[N, P, 1326]` everywhere inside search and data generation.
3. Replace all `2 * NUM_HANDS` constants with `num_players * NUM_HANDS` in:
   - `MLPFeatures`,
   - Better/ReBeL feature encoders,
   - replay buffers,
   - suit permutation helpers,
   - exporter/reference code that serializes model features.
4. Track active seats explicitly. Folded players still need value targets, but policy and reach updates should only apply to live, non-all-in actors.
5. Normalize each seat range independently after action and chance updates, masking impossible hands against the public board and known sampled cards.

## Phase 3: CFR Evaluator
The CFR evaluator can remain actor-local for policy and regret, but all belief and value movement must become multi-seat.

1. Tree construction:
   - Expand children using `PBSEnv.legal_bins_mask`.
   - Store `prev_actor`/`to_act` for any seat in `[0, P)`.
   - Keep per-depth action schedules, but add lower default depth and width for 3- and 4-way.
2. Belief propagation:
   - When actor `a` takes an action, multiply only `beliefs[:, a, :]` by that actor's per-hand policy.
   - Carry non-actor beliefs unchanged, then renormalize when needed.
   - Preserve zero mass for folded or impossible hands.
3. Regret updates:
   - Compute counterfactual values for the acting seat only at each decision node.
   - Update regrets for the acting seat's hand/action table.
   - Do not impose heads-up zero-sum identities when computing non-actor values.
4. Terminal values:
   - Replace heads-up all-in payoff tables with a multiway resolver.
   - First implementation can use exact tensorized showdown for small batches or sampled private-hand rollouts; optimize later with cached tables only after parity tests exist.
   - Handle side pots before optimizing all-in leaves.
5. Statistics:
   - Replace exploitability-style heads-up metrics with local regret, policy entropy, actor mix, value-target scale, and per-seat value-sum diagnostics.

## Phase 4: Sampled-Depth Value Training
Do not require the sampled continuation path to reach an end-of-street leaf. While descending sampled leaves, each active path should have a configurable probability of stopping at the current public state and producing a value target.

Recommended controls:

- `search.abort_depth_probs_by_street`: optional per-street hazards by same-street depth.
- `search.abort_min_depth`: avoid aborting immediately at every root unless deliberately training roots.
- `search.abort_force_by_depth`: force abort when depth/width budgets are reached.
- `train.value_depth_stratify_probs`: replay sampling weights for value targets by street and depth.

Sampling algorithm:

1. Start from roots after CFR finishes.
2. For each active sampled path, draw the next action from the current sampling policy.
3. Before applying the sampled action, decide whether to abort at the current node using a tensorized hazard mask.
4. Stop aborted paths and emit a value example for that node.
5. Continue non-aborted paths until terminal, forced cutoff, new-street cutoff, or sampled abort.
6. For paths that close the betting round, optionally emit both the pre-chance boundary target and a post-chance re-root sample.

The abort hazard must not bias training toward shallow states only. Use one of these:

- Stratified quotas: choose a target count per `(street, same_street_depth)` bucket.
- Inverse-probability weights: store `1 / stop_probability` in batch statistics and use it in value loss.
- Replay rebalancing: maintain separate value-buffer partitions by street/depth.

The simplest first version is stratified quotas. It is easier to debug, avoids large loss weights, and matches the existing policy-depth stratification pattern.

## Phase 5: Model Changes
The multiway value model should predict every seat at arbitrary public states, not just the actor at a street boundary.

1. Feature layout:
   - Keep scalar public features shared across seats.
   - Encode per-seat features as `[B, P, seat_feature_dim]`: stack, committed, SPR, folded, all-in, acted-this-round, button-relative position, and current actor flag.
   - Encode beliefs as `[B, P, 1326]` through a shared range encoder.
   - Add a seat/position embedding so the same cards in different positions are distinguishable.
2. Policy head:
   - Condition on the current actor.
   - Output `[B, 1326, A]` for that actor only.
   - Keep action masking outside the model.
3. Value head:
   - Output `[B, P, 1326]`.
   - Add to actions_this_round an additional feature bit for round closed.
   - Keep separate heads only if diagnostics show boundary/interior targets interfere; otherwise prefer one arbitrary-state value head with target-kind conditioning.
4. Constant-sum handling:
   - Do not reuse the current heads-up zero-sum projection blindly.
   - Multiway no-rake payoffs are constant-sum at the joint private-state level, but enforcing that requires blocker-aware weighting across all seats.
   - Initial implementation should disable `enforce_zero_sum` for multiway and log the belief-weighted sum of predicted seat EVs.
   - Add a soft penalty later if value-sum drift is large.
5. Suit permutation:
   - Generalize permutation code from two belief slices to `P` slices.
   - Apply the same combo remap to every seat's belief and value target.

## Phase 6: Trainer and Replay
1. Split heads-up and multiway construction paths in `RebelCFRTrainer` until the multiway stack is stable.
2. Size replay buffers by `num_players * NUM_HANDS`.
3. Store batch statistics for `street`, `same_street_depth`, `abort_kind`, `target_kind`, `num_live_players`, `actor`, and optional value loss weight.
4. Generalize `RebelSupervisedLoss`:
   - Value loss compares `[B, P, 1326]`.
   - Policy loss weights actor hands by actor belief and a blocker-compatible aggregate mass over all non-actor live ranges.
   - Folded seats should contribute value loss but not policy loss.
5. Update diagnostics to report per-seat value MSE, per-depth value MSE, actor-position policy KL, and abort-depth coverage.

## Phase 7: WebGPU CFR Evaluator
The browser/WebGPU evaluator currently exports and runs heads-up BetterFFN sparse CFR. Multiway support should come after Python parity.

1. Extend the model manifest with `numPlayers`, feature dimensions, value target kind support, and multiway CFR defaults.
2. Update exported BetterFFN weights only after the Python model layout is stable.
3. Update `webgpu_cfr/src/beliefs.ts`, model feature kernels, and sparse CFR buffers from two ranges to `P` ranges.
4. Port multiway tree construction and regret update only after Python tests lock down:
   - legality,
   - belief propagation,
   - terminal values,
   - sampled-depth cutoff behavior.
5. Keep heads-up artifacts and tests as the compatibility baseline.

## Test Plan
1. Environment parity:
   - Expand `tests/test_multiway_env.py` for side pots, all-in closures, and tied showdowns.
2. Shape tests:
   - Run feature encoders, replay buffers, and model forward passes for `P = 2, 3, 4`.
3. CFR invariants:
   - Beliefs stay finite and normalized for live players.
   - Only the acting seat's range changes on action propagation.
   - Regret updates touch only nodes for the acting seat.
4. Sampled-depth coverage:
   - With deterministic hazards, verify exact abort buckets.
   - With stochastic hazards, verify bucket frequencies within tolerance.
5. Regression:
   - Heads-up config produces the same tensor shapes and comparable policy/value losses as before.
6. Performance:
   - Benchmark tree construction, CFR iteration, sampled-leaf descent, and replay transfer for `P = 2, 3, 4`.

## Suggested Milestones
1. Add multiway config plumbing and `PBSEnv`-backed root generation with no CFR changes.
2. Generalize features/replay/model shapes and keep heads-up tests passing.
3. Implement a non-fused Python sparse multiway evaluator.
4. Add sampled-depth value targets and train a small 3-way debug run.
5. Add multiway terminal resolver correctness tests.
6. Optimize fused/Triton paths after the non-fused path is correct.
7. Export to WebGPU only after Python multiway CFR and model artifacts are stable.

## Open Questions
- Should early 3-way and 4-way training share one model with `num_players` conditioning, or keep separate fixed-`P` checkpoints?
- How much terminal exactness is required before useful value training starts: exact side-pot showdown, sampled showdown, or hybrid?
- Should abort-depth quotas be global per batch or enforced per street to avoid river domination?
- Is one arbitrary-state value head sufficient, or do pre-chance/post-chance/interior targets need separate heads?
