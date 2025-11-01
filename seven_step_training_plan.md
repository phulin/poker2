# Seven-Stage Value Target ExecPlan

## Purpose and Outcome

Stakeholders want supervised training data that describes seven public-belief stages (showdown, start-of-river, end-of-turn, start-of-turn, end-of-flop, start-of-flop, end-of-preflop) instead of the current four. After completing this work, a trainer can sample value targets for each of these stages, confirm that every start-of-street sample has a corresponding end-of-street partner in the replay buffers, and observe them influence training. Validation involves running `source venv/bin/activate`, executing `pytest tests/search/test_rebel_data_generator.py::test_end_of_street_augmentation`, and running the integration script described below to ensure value batches include the expected pairs.

## Context

ReBeL-style Counterfactual Regret Minimization (CFR) uses `RebelCFREvaluator` to expand a poker search tree where every node represents a post-chance state on a single street. A "street" is the number of public board cards revealed: preflop (0 cards), flop (3 cards), turn (4 cards), river (5 cards). A Public Belief State (PBS) records the environment (`HUNLTensorEnv`) and the belief tensor shaped `[batch, players, 1326]`, where 1326 is the number of distinct two-card combinations in a 52-card deck. The feature encoders already support a `pre_chance_node=True` flag that yields the public information just before the latest chance event.

We will rely on the fact that every tree built by `RebelCFREvaluator.construct_subgame` remains on the same street as the root nodes. When we transition to the next street in `sample_leaf`, the resulting PBS becomes the root of a new evaluator, and the constructor recomputes masks for that street. That allows us to use the existing tensors to reconstruct the end-of-street state without modifying core evaluator structures.

## Functional Scope

This effort augments value-training data so that, for every start-of-street example we already collect, we also record the matching end-of-street example that immediately precedes the chance event. Concretely:

1. Trace how `RebelCFREvaluator.training_data` surfaces start-of-street samples and map each to the public state that existed right before the next board card was revealed, keeping the evaluator’s belief tensor in a pre-chance form.
2. When `RebelCFREvaluator.training_data` assembles the value batch, derive the corresponding end-of-street features and value targets (including chance expectations such as enumerating river cards or exploiting flop suit symmetry enforced by the encoder’s suit-symmetry loss) and return them alongside the existing start-of-street samples.
3. Provide supporting utilities, configuration hooks, and tests so a novice can verify the augmentation end-to-end.

Out of scope: architectural changes to tree construction, new CFR algorithms, or rewriting the trainer loop.

## Implementation Strategy

### 1. Track Root Pre-Chance Beliefs Inside The Evaluator

Keep `self.beliefs` / `self.beliefs_avg` as the post-chance ranges used for CFR, but add a dedicated `root_pre_chance_beliefs` buffer that persists the end-of-street ranges supplied by the caller. Introduce `_fan_out_deep` to broadcast root-aligned masks (e.g., `allowed_hands`, uniform fallbacks, reach initialisation) across every node in the tree while leaving the stored pre-chance snapshot untouched so it can be reused during data augmentation.

### 2. Keep PBS Lightweight

Leave `PublicBeliefState` with a single `beliefs` tensor; when PBS instances represent street-end nodes those beliefs are already pre-chance and can be fed straight back into the evaluator. Update `initialize_search`, `self_play_iteration`, `sample_leaf`, and the replay/data-generator plumbing to populate the evaluator’s `root_pre_chance_beliefs` directly instead of expanding the PBS payload.

### 3. Derive End-of-Street Value Targets

Implement utilities inside `RebelCFREvaluator` (or a dedicated helper module) that accept start-of-street indices and produce the corresponding end-of-street value targets. Because `self.beliefs` already hold pre-chance data, the helper can feed those beliefs into `feature_encoder.encode(..., pre_chance_node=True)` without additional transformations.

For each street transition, spell out how to average over hidden chance outcomes:

* River: enumerate the 48 legal river cards for every start-of-river node, use the value network to estimate the post-chance value, weight by uniform probability, and produce the end-of-turn value target.
* Turn: enumerate the 47 legal turn cards consistent with the flop (52 cards minus 3 flop cards). Use the value network to evaluate the resulting start-of-river values, then average them to obtain end-of-flop targets.
* Flop: enumerate canonical flops and evaluate each once, weighting by the number of raw flops in its orbit. Because the encoder is trained with the suit-symmetry loss (`alphaholdem/rl/losses.py`), the model respects suit permutations and no explicit hand remapping is required. This yields the exact end-of-preflop expectation without Monte Carlo variance.

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
- [x] Root-level bookkeeping (indices, pre/post beliefs) propagated through tree
- [x] Pre-chance belief handoff across evaluators wired up
- [x] End-of-street value derivation utilities implemented
- [x] `training_data` returns paired value batches and tests cover it
- [ ] Validation script implemented and documented
- [ ] Tests and integration checks passing

## Risks and Mitigations

Enumerating canonical flops and full turn/river runouts may be expensive for large batches. Mitigate by caching canonical representatives, precomputing their multiplicities, vectorising evaluations, and short-circuiting when the evaluator depth is insufficient to reach a given street. Confirm via profiling that the helpers add acceptable overhead; if not, fall back to a configurable bounded sample count with clear warnings.

## Rollback Strategy

The changes are localized. If issues arise, revert the commits touching `RebelCFREvaluator`, `RebelDataGenerator`, the replay buffer helpers, or the new validation script. No schema or data migrations are required. Removing the new unit test and script restores the repository to its previous state.

## Decision Log

2024-05-30: Adopted the assumption that all nodes in an evaluator tree share the root street, allowing `allowed_hands` to be computed once per tree. This keeps the evaluator architecture unchanged while enabling access to pre-chance beliefs for the new training stages.
2024-05-31: Narrowed the scope to augment existing value batches with end-of-street counterparts generated during data collection, leaving evaluator node annotations and feature encoders unchanged.
2024-06-01: Kept PBS payloads minimal while storing a dedicated `root_pre_chance_beliefs` snapshot inside the evaluator for chance-node augmentation.
2024-05-31: Chose to integrate flop suit symmetry by enumerating canonical flops instead of Monte Carlo sampling, eliminating variance while keeping computation tractable.
