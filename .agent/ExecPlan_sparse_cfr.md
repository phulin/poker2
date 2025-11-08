# Align SparseCFREvaluator With RebelCFREvaluator Behavior

This ExecPlan is a living document, to be maintained in accordance with `/Users/phulin/Documents/Projects/poker2/.agent/PLANS.md`.

## Purpose / Big Picture

After this change, anyone can instantiate `SparseCFREvaluator` and expect it to mirror the search loop implemented by `RebelCFREvaluator`: tree construction by depth, belief propagation, regret updates (including CFR+ warm start), policy averaging, and replay extraction. The sparse evaluator currently contains placeholders and shortcuts; this work makes it functionally equivalent so researchers can toggle between sparse and dense storage while obtaining the same behaviour. Success is demonstrated by running a focused test suite that exercises both evaluators on the same subgame and comparing key tensors (policy, values, regrets, replay batches).

## Progress

- [x] (2025-11-08 17:47Z) Document current Sparse vs Rebel state and confirm helper coverage (noted absence of `valid_mask`, `allowed_hands`, reach propagation helpers, and replay parity hooks).
- [x] (2025-11-08 18:18Z) Bring tree construction, bookkeeping, and masks in `SparseCFREvaluator.initialize_subgame` in line with Rebel semantics (added sparse depth expansion, parent/action wiring, board-blocked hand masks, and reach scaffolding).
- [ ] (2025-11-08 18:32Z) Re-implement policy and belief propagation using `_fan_out`, `_push_down`, and `_pull_back_sum` (completed: added sparse equivalents of `_initialize_with_copy`, `_block_beliefs`, `_calculate_reach_weights`, rewrote `initialize_policy_and_beliefs`, `_propagate_all_beliefs`, and integrated them into `update_policy`/`compute_expected_values`; remaining: verify reach/averaging parity and port warm-start & sampling paths).
- [ ] (2025-11-08 17:40Z) Align CFR loop pieces (`warm_start`, `set_leaf_values`, `compute_expected_values`, `compute_instantaneous_regrets`, `update_policy`, `update_average_policy`, `sample_leaf`) with Rebel implementation.
- [x] (2025-11-08 19:05Z) Port replay extraction (`training_data`) to match Rebel batches (implemented `_pull_back`, `_best_response_values`, `_compute_exploitability`, and full replay batching using sparse helpers).
- [x] (2025-11-09 12:05Z) Add regression tests comparing sparse vs rebel outputs on deterministic seeds (added CPU-only parity tests for tree structure and initial policy/beliefs, covering Sparse vs Rebel alignment).
- [ ] (2025-11-08 17:40Z) Validate with pytest (targeted module) and document results in Outcomes section.

## Surprises & Discoveries

- None yet.

## Decision Log

- Decision: Keep policy, regrets, and related tensors canonically stored on destination nodes while porting helper logic, mirroring RebelCFREvaluator semantics without introducing sparse-only scatter paths.  
  Rationale: Destination alignment matches the dominant access pattern (belief propagation, reach weights, replay sampling) and avoids extra scatter/gather work compared to parent-aligned storage.  
  Date/Author: 2025-11-08 / Assistant.

## Outcomes & Retrospective

- Empty until milestones complete.

## Context and Orientation

`SparseCFREvaluator` in `src/alphaholdem/search/sparse_cfr_evaluator.py` is intended to store the game tree sparsely (each node has a parent pointer and action label). At the moment its public methods are skeletal, with placeholder belief propagation and missing replay extraction. `RebelCFREvaluator` in `src/alphaholdem/search/rebel_cfr_evaluator.py` is the authoritative implementation of the CFR search loop, using dense arrays indexed by depth offsets. The sparse evaluator already defines helpers `_fan_out`, `_push_down`, and `_pull_back_sum` analogous to Rebel’s reshaping routines; these must be used consistently to keep sparse storage efficient.

Key modules:

- `alphaholdem.env.hunl_tensor_env.HUNLTensorEnv`: vectorised poker environment.
- `alphaholdem.models.mlp.rebel_ffn.RebelFFN` and `alphaholdem.models.mlp.better_ffn.BetterFFN`: neural models providing policy and value heads.
- `alphaholdem.rl.rebel_replay.RebelBatch`: replay container used by trainers.
- `alphaholdem.search.chance_node_helper.ChanceNodeHelper`: handles chance-node rollouts for pre-chance value estimation.

We must ensure the sparse evaluator reproduces features such as warm-start regrets, reach-weight normalisation, belief blocking by board cards, and replay/statistics generation.

## Plan of Work

First, reconcile tree construction by ensuring `initialize_subgame` creates the same node set and masks (`valid_mask`, `leaf_mask`, `new_street_mask`), allowed-hand tensors, and previous-actor bookkeeping as `RebelCFREvaluator`, while respecting sparsity via parent/action arrays. Compute `allowed_hands` and `allowed_hands_prob` by fanning out from the root beliefs.

Next, bring policy and belief propagation in line: implement `_calculate_reach_weights`, `_normalize_beliefs`, `_block_beliefs`, `_propagate_level_beliefs`, `_propagate_all_beliefs`, and update `initialize_policy_and_beliefs` plus `_update_beliefs_from_policy`. Use `_push_down` to map parent policy slices to children and `_pull_back_sum` to accumulate reach weights.

Align the CFR iteration by porting Rebel’s logic for `warm_start`, `set_leaf_values`, `compute_expected_values`, `compute_instantaneous_regrets`, `update_policy`, `update_average_policy`, and `sample_leaf`, adapting indexing to sparse helpers. Preserve CFR+ behaviour (clamping, regret weighting) and ensure tensors remain destination-aligned.

Implement statistics and replay batching by mirroring Rebel’s `training_data`, `ExploitabilityStats`, and related helpers, adjusting computations to sparse storage while avoiding redundant materialisation.

Finally, add tests under `tests/search/test_sparse_cfr_evaluator.py` that construct deterministic subgames, run both evaluators for a small number of iterations, and assert closeness of key tensors (policy, values, regrets). Include comparisons of `RebelBatch` outputs to ensure replay parity.

## Concrete Steps

1. Activate the virtual environment from the repository root `/Users/phulin/Documents/Projects/poker2` with `source venv/bin/activate`.
2. Audit `SparseCFREvaluator` to catalogue methods needing parity updates and confirm helper coverage.
3. Modify `initialize_subgame` to build masks and tensors mirroring Rebel’s layout, using `_fan_out`, `_push_down`, and `_pull_back_sum`.
4. Implement helper routines for belief and reach propagation, updating `initialize_policy_and_beliefs` and `_update_beliefs_from_policy`.
5. Port CFR-loop components (`warm_start`, `set_leaf_values`, `compute_expected_values`, `compute_instantaneous_regrets`, `update_policy`, `update_average_policy`, `sample_leaf`) to match Rebel semantics.
6. Implement `training_data` and supporting statistics helpers so sparse evaluator produces the same replay tuples.
7. Create new tests under `tests/search/test_sparse_cfr_evaluator.py` that compare sparse vs rebel evaluators on deterministic seeds.
8. Run `pytest tests/search/test_sparse_cfr_evaluator.py` to validate parity, then extend to full `pytest` as needed.
9. Update this plan’s Progress, Decision Log, Surprises, and Outcomes sections as milestones complete.

## Validation and Acceptance

Validation passes when `pytest tests/search/test_sparse_cfr_evaluator.py` succeeds, demonstrating that sparse and rebel evaluators agree on key tensors within tolerance. Manual inspection or additional asserts should confirm belief normalisation and reach weights match the dense implementation. Full `pytest` should continue to pass.

## Idempotence and Recovery

Tree building and tensor allocations depend only on inputs, so re-running initialisation is safe. Code edits are confined to `src/alphaholdem/search/` and new tests; if problems arise, revert files via version control. Re-activate the virtual environment and rerun pytest to recover from environment drift.

## Artifacts and Notes

Populate this section with relevant command outputs or tensor diagnostics as work progresses.

## Interfaces and Dependencies

Ensure `SparseCFREvaluator.initialize_subgame` maintains its public contract while populating tensors equivalent to Rebel’s (`valid_mask`, `leaf_mask`, `allowed_hands`, `prev_actor`, etc.). Implement helper methods mirroring Rebel’s signatures, including `_calculate_reach_weights`, `_normalize_beliefs`, `_block_beliefs`, `_propagate_all_beliefs`, `_propagate_level_beliefs`, and `_get_mixing_weights`. CFR iteration methods must accept the same parameters as Rebel’s counterparts, notably `set_leaf_values(self, t: int)` and `compute_expected_values(self, policy=None, values=None)`. `training_data` must return three `RebelBatch` instances aligned with `RebelCFREvaluator.training_data`.

Revision history for this plan must be appended below with date and rationale whenever updates occur.
