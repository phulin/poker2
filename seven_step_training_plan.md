# Seven-Stage Value Target ExecPlan

## Purpose and Outcome

Stakeholders want supervised training data that describes seven public-belief stages (showdown, start-of-river, end-of-turn, start-of-turn, end-of-flop, start-of-flop, end-of-preflop) instead of the current four. After completing this work, a trainer can sample value targets for each of these stages, confirm that every start-of-street sample has a corresponding end-of-street partner in the replay buffers, and observe them influence training. Validation involves running `source venv/bin/activate`, executing `pytest tests/search/test_rebel_data_generator.py::test_end_of_street_augmentation`, and running the integration script described below to ensure value batches include the expected pairs.

## Context

ReBeL-style Counterfactual Regret Minimization (CFR) uses `RebelCFREvaluator` to expand a poker search tree where every node represents a post-chance state on a single street. A "street" is the number of public board cards revealed: preflop (0 cards), flop (3 cards), turn (4 cards), river (5 cards). A Public Belief State (PBS) records the environment (`HUNLTensorEnv`) and the belief tensor shaped `[batch, players, 1326]`, where 1326 is the number of distinct two-card combinations in a 52-card deck. The feature encoders already support a `pre_chance_node=True` flag that yields the public information just before the latest chance event.

We will rely on the fact that every tree built by `RebelCFREvaluator.construct_subgame` remains on the same street as the root nodes. When we transition to the next street in `sample_leaf`, the resulting PBS becomes the root of a new evaluator, and the constructor recomputes masks for that street. That allows us to use the existing tensors to reconstruct the end-of-street state without modifying core evaluator structures.

## Functional Scope

This effort augments value-training data so that, for every start-of-street example we already collect, we also record the matching end-of-street example that immediately precedes the chance event. Concretely:

1. Trace how `RebelCFREvaluator.training_data` surfaces start-of-street samples and map each to the public state that existed right before the next board card was revealed, keeping the evaluator’s belief tensor in a pre-chance form.
2. When `RebelCFREvaluator.training_data` assembles the value batch, derive the corresponding end-of-street features and value targets (including chance expectations such as enumerating river cards or exploiting flop suit symmetry) and return them alongside the existing start-of-street samples.
3. Provide supporting utilities, configuration hooks, and tests so a novice can verify the augmentation end-to-end.

Out of scope: architectural changes to tree construction, new CFR algorithms, or rewriting the trainer loop.

## Implementation Strategy

### 1. Preserve Pre-Chance Beliefs Throughout the Tree

Ensure `RebelCFREvaluator` keeps `self.beliefs` in the pre-chance format across the entire tree without introducing a separate post-chance tensor. In `construct_subgame`, compute `allowed_hands` only for the root batch and distribute it to every node using a new helper `_fan_out_deep`, which takes a root-level tensor and repeats it to match the total node count regardless of depth. Replace existing `_fan_out` usages that previously fanned level-by-level so the masking logic stays consistent with the single-street assumption explained in the Context section.

Document `_fan_out_deep` within the file, including its assumptions (tree arity equals number of actions, consistent depth offsets) so future contributors can reason about it without re-reading the entire search implementation.

### 2. Track Pre-Chance Beliefs Across Evaluator Transitions

Extend `PublicBeliefState` to carry both `post_chance_beliefs` (the current field, unchanged) and a new `pre_chance_beliefs` tensor that stores the ranges immediately before the chance event. When `sample_leaf` constructs `next_pbs`, populate both tensors by copying the pre-chance beliefs from the parent evaluator alongside the post-chance beliefs already used today.

During `initialize_search`, seed `self.beliefs` with `pbs.pre_chance_beliefs` so the entire evaluator operates on pre-chance data. Store the post-chance view in a new attribute such as `self.root_post_chance_beliefs` for the handful of places that still need it (e.g., start-of-street targets). Add comments clarifying the lifecycle: `pre_chance_beliefs` flow down the tree, while `post_chance_beliefs` are held only for reporting and training output.

### 3. Derive End-of-Street Value Targets

Implement utilities inside `RebelCFREvaluator` (or a dedicated helper module) that accept start-of-street indices and produce the corresponding end-of-street value targets. Because `self.beliefs` already hold pre-chance data, the helper can feed those beliefs into `feature_encoder.encode(..., pre_chance_node=True)` without additional transformations.

For each street transition, spell out how to average over hidden chance outcomes:

* River: enumerate the 48 legal river cards for every start-of-river node, use the value network to estimate the post-chance value, weight by uniform probability, and produce the end-of-turn value target.
* Turn: enumerate the 45 legal turn cards consistent with the flop (52 cards minus 3 flop cards minus 4 hole cards). Use the value network to evaluate the resulting start-of-river values, then average them to obtain end-of-flop targets.
* Flop: enumerate canonical flops using suit symmetry. For each canonical flop, evaluate once, remap via suit permutations to cover its orbit, and weight by the number of raw flops represented. This yields the exact end-of-preflop expectation without Monte Carlo variance.

Emit assertions that the resulting tensors share the same shape as the original value targets so downstream code can rely on them without branching.

### 4. Return Augmented Value Batches From `training_data`

Extend `RebelCFREvaluator.training_data` so it returns both the existing `value_batch` (start-of-street samples) and a second batch containing the derived end-of-street examples. Internally, call the utilities from step 3 to build the additional features, value targets, and statistics. Update callers—primarily `RebelDataGenerator.generate_data`—to accept the new tuple `(value_batch, end_batch, policy_batch)` and push both value batches into the replay buffer.

Ensure the statistics in the augmented batch (pot size, actor to act, etc.) match the pre-chance view by copying or recomputing them before the chance event. If certain statistics are undefined before the chance event, document how to fill them (for example, reuse `pot` but zero out `bet_amounts`).

Write a regression test (for example `tests/search/test_rebel_data_generator.py::test_end_of_street_augmentation`) that constructs a tiny evaluator, runs one data-generation step, and asserts that the value buffer receives twice as many entries while the new entries correspond to the expected streets.

Update any trainer logging or metric computations that assume value batches only contain start-of-street samples so aggregated statistics remain meaningful.

### 5. Validation Steps

Create a script `scripts/check_street_targets.py` that runs a single CFR iteration, prints the count of start-of-street versus end-of-street samples from the value buffer, and exits. It should be safe to run repeatedly.

Add instructions to `README` or `AGENTS.md` describing how to run:

    source venv/bin/activate
    python scripts/check_street_targets.py

Expect to see identical counts for start-of-street and end-of-street entries at each street boundary and totals matching buffer sizes.

## Validation

1. Activate the virtual environment and run the augmentation test:

       source venv/bin/activate
       pytest tests/search/test_rebel_data_generator.py::test_end_of_street_augmentation

   It should fail before implementing the augmentation helper and pass afterward.

2. Run the integration script:

       source venv/bin/activate
       python scripts/check_street_targets.py

   Confirm the printed output shows paired counts for start-of-street and end-of-street samples and that the end-of-street values for start-of-river nodes match the averaged expectations computed in step 2.

3. Optionally run `pytest tests/` to check for regressions once core functionality works; expect all tests to pass.

## Progress

- [x] ExecPlan authored
- [ ] `_fan_out_deep` implemented and documented
- [ ] Pre-chance belief handoff across evaluators wired up
- [ ] End-of-street value derivation utilities implemented
- [ ] `training_data` returns paired value batches and tests cover it
- [ ] Validation script implemented and documented
- [ ] Tests and integration checks passing

## Risks and Mitigations

Enumerating canonical flops and full turn/river runouts may be expensive for large batches. Mitigate by caching canonical representatives, precomputing their multiplicities, vectorising evaluations, and short-circuiting when the evaluator depth is insufficient to reach a given street. Confirm via profiling that the helpers add acceptable overhead; if not, fall back to a configurable bounded sample count with clear warnings.

## Rollback Strategy

The changes are localized. If issues arise, revert the commits touching `RebelCFREvaluator`, `RebelDataGenerator`, the replay buffer helpers, or the new validation script. No schema or data migrations are required. Removing the new unit test and script restores the repository to its previous state.

## Decision Log

2024-05-30: Adopted the assumption that all nodes in an evaluator tree share the root street, allowing `allowed_hands` to be computed once per tree. This keeps the evaluator architecture unchanged while enabling access to pre-chance beliefs for the new training stages.
2024-05-31: Narrowed the scope to augment existing value batches with end-of-street counterparts generated during data collection, leaving evaluator node annotations and feature encoders unchanged.
2024-05-31: Chose to pass `pre_chance_beliefs` alongside standard beliefs in `PublicBeliefState` so each evaluator root retains the end-of-previous-street ranges needed for augmentation.
2024-05-31: Chose to integrate flop suit symmetry by enumerating canonical flops instead of Monte Carlo sampling, eliminating variance while keeping computation tractable.
