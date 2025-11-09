# Align SparseCFREvaluator With RebelCFREvaluator Behavior

This ExecPlan is a living document, to be maintained in accordance with `/Users/phulin/Documents/Projects/poker2/.agent/PLANS.md`.

## Purpose / Big Picture

After this change, anyone can instantiate `SparseCFREvaluator` and expect it to mirror the search loop implemented by `RebelCFREvaluator`: tree construction by depth, belief propagation, regret updates (including CFR+ warm start), policy averaging, and replay extraction. The sparse evaluator currently contains placeholders and shortcuts; this work makes it functionally equivalent so researchers can toggle between sparse and dense storage while obtaining the same behaviour. Success is demonstrated by running a focused test suite that exercises both evaluators on the same subgame and comparing key tensors (policy, values, regrets, replay batches).

## Progress

- [x] (2025-11-08 17:47Z) Document current Sparse vs Rebel state and confirm helper coverage (noted absence of `valid_mask`, `allowed_hands`, reach propagation helpers, and replay parity hooks).
- [x] (2025-11-08 18:18Z) Bring tree construction, bookkeeping, and masks in `SparseCFREvaluator.initialize_subgame` in line with Rebel semantics (added sparse depth expansion, parent/action wiring, board-blocked hand masks, and reach scaffolding).
- [x] (2025-11-08 18:32Z) Re-implement policy and belief propagation using `_fan_out`, `_push_down`, and `_pull_back_sum` (completed: added sparse equivalents of `_initialize_with_copy`, `_block_beliefs`, `_calculate_reach_weights`, rewrote `initialize_policy_and_beliefs`, `_propagate_all_beliefs`, and integrated them into `update_policy`/`compute_expected_values`; remaining: verify reach/averaging parity and port warm-start & sampling paths).
- [x] (2025-11-08 17:40Z) Align CFR loop pieces (`warm_start`, `set_leaf_values`, `compute_expected_values`, `compute_instantaneous_regrets`, `update_policy`, `update_average_policy`, `sample_leaf`) with Rebel implementation.
- [x] (2025-11-08 19:05Z) Port replay extraction (`training_data`) to match Rebel batches (implemented `_pull_back`, `_best_response_values`, `_compute_exploitability`, and full replay batching using sparse helpers).
- [x] (2025-11-09 12:05Z) Add regression tests comparing sparse vs rebel outputs on deterministic seeds (added CPU-only parity tests for tree structure and initial policy/beliefs, covering Sparse vs Rebel alignment).
- [ ] (2025-11-08 17:40Z) Validate with pytest (targeted module) and document results in Outcomes section.
- [ ] (2025-11-10 19:45Z) Gap review re-opened policy/CFR parity (remaining gaps: fold/showdown bookkeeping, terminal value wiring + warm start, legal-weighted backups/regrets, and sampling/replay parity; see “Gap Review”).

## Surprises & Discoveries

- (2025-11-10) Comparing `SparseCFREvaluator` to `RebelCFREvaluator` shows that, despite model calls being wired up, we still need to persist fold/showdown payoffs, finish the warm-start/leaf-value flow, ensure legal-weighted backups/regrets match ReBeL, and complete sampling/replay parity.

## Gap Review (2025-11-10)

- **State bookkeeping still incomplete.** `SparseCFREvaluator.initialize_subgame` needs to capture showdown indices and immediate fold rewards during tree construction so that later passes can plug leaf payoffs without bespoke fold tracking (src/alphaholdem/search/sparse_cfr_evaluator.py:118-214). Rewards computed in the loop (`reward_levels`) are kept but never written back to nodes, so folds/showdowns retain zero EV.
- **Policy initialisation mostly aligned.** `_get_model_policy_probs` lives in `CFREvaluator` and is already invoked from `initialize_policy_and_beliefs`; remaining work is to guard against redundant feature encoding when we propagate policies/beliefs level by level, but no major rewrites are required.
- **Leaf handling, warm start, and model eval still partial.** `set_leaf_values` leaves terminal rewards unimplemented (lines 432-456) and always encodes `pre_chance_node=True`, ignoring `new_street_mask`. `warm_start` merely scales regrets by a constant (lines 419-428) instead of running the best-response sweep found in ReBeL (src/alphaholdem/search/rebel_cfr_evaluator.py:646-694).
- **Backups and regret updates need legal weighting.** `compute_expected_values`/`compute_instantaneous_regrets` operate on the flattened tensors without `_pull_back` reshaping or per-action legal masking (lines 458-543). ReBeL’s versions condition on legality when aggregating child values (src/alphaholdem/search/rebel_cfr_evaluator.py:760-857); sparse needs equivalent weighting even though every node is considered valid.
- **Sampling/replay parity absent.** `sample_leaf` is still a placeholder (lines 600-607) and `training_data` collects all non-leaf nodes regardless of whether they correspond to real decisions (lines 799-862). Compared to ReBeL’s traversal (src/alphaholdem/search/rebel_cfr_evaluator.py:892-1214), there is no PBS sampling flow or showdown-aware statistics in the sparse evaluator.

## Decision Log

- Decision: Keep policy, regrets, and related tensors canonically stored on destination nodes while porting helper logic, mirroring RebelCFREvaluator semantics without introducing sparse-only scatter paths.  
  Rationale: Destination alignment matches the dominant access pattern (belief propagation, reach weights, replay sampling) and avoids extra scatter/gather work compared to parent-aligned storage.  
  Date/Author: 2025-11-08 / Assistant.

## Outcomes & Retrospective

- Empty until milestones complete.

## Context and Orientation

`SparseCFREvaluator` in `src/alphaholdem/search/sparse_cfr_evaluator.py` is intended to store the game tree sparsely (each node has a parent pointer and action label). `RebelCFREvaluator` in `src/alphaholdem/search/rebel_cfr_evaluator.py` is the authoritative implementation of the CFR search loop, using dense arrays indexed by depth offsets. The sparse evaluator already defines helpers `_fan_out`, `_push_down`, and `_pull_back_sum` analogous to Rebel’s reshaping routines; these must be used consistently to keep sparse storage efficient.

Key modules:

- `alphaholdem.env.hunl_tensor_env.HUNLTensorEnv`: vectorised poker environment.
- `alphaholdem.models.mlp.rebel_ffn.RebelFFN` and `alphaholdem.models.mlp.better_ffn.BetterFFN`: neural models providing policy and value heads.
- `alphaholdem.rl.rebel_replay.RebelBatch`: replay container used by trainers.
- `alphaholdem.search.chance_node_helper.ChanceNodeHelper`: handles chance-node rollouts for pre-chance value estimation.

We must ensure the sparse evaluator reproduces features such as warm-start regrets, reach-weight normalisation, belief blocking by board cards, and replay/statistics generation.

## Plan of Work

First, finish state bookkeeping inside `initialize_subgame`: retain immediate fold rewards per node, track which nodes correspond to showdowns, and keep `new_street_mask` so later passes know when to query the pre-chance model.

Then, port the remaining CFR loop pieces—`set_leaf_values`, `warm_start`, `compute_expected_values`, `compute_instantaneous_regrets`, `update_policy`, `update_average_policy`, and `sample_leaf`—so they mirror Rebel’s semantics (terminal payoffs, CFR+ clamping, opponent-weighted regrets), while leaning on the assumption that every sparse node is valid and only gating on legality/leaf masks.

Finally, align the replay/sampling surface area by finishing `_compute_exploitability`, `training_data`, and PBS sampling so they reuse showdown metadata, reweight chance nodes correctly, and emit the same `RebelBatch` statistics as the dense evaluator, then extend parity tests and document a broader testing strategy.

## Concrete Steps

1. Activate the virtual environment from the repository root `/Users/phulin/Documents/Projects/poker2` with `source venv/bin/activate`.
2. Audit `SparseCFREvaluator` to catalogue methods needing parity updates and confirm helper coverage.
3. Flesh out `initialize_subgame` / tree construction with the missing bookkeeping (showdown indices, fold reward assignment, `new_street_mask`) and confirm `allowed_hands(_prob)` are propagated correctly to every node.
4. Verify `_get_model_policy_probs` + belief-propagation helpers by walking through `initialize_policy_and_beliefs`, ensuring we don’t re-encode features unnecessarily and that legal masks/allowed hands stay honoured.
5. Port the CFR loop: finalize `set_leaf_values` (fold/showdown payouts + model heads via `new_street_mask`), copy Rebel’s warm-start best-response sweep, and rewrite `compute_expected_values`/`compute_instantaneous_regrets`/`update_policy`/`update_average_policy` to use `_pull_back` plus leaf/legality gating (no `valid_mask` assumptions needed).
6. Finish `sample_leaf`, `_compute_exploitability`, and `training_data` so PBS sampling, exploit stats, and replay batches use showdown metadata and legality checks, matching Rebel’s statistics/signatures.
7. Extend/refresh parity tests under `tests/search/test_sparse_cfr_evaluator.py` to cover policy init, CFR iteration, sampling, and replay outputs, and document the broader test matrix below.
8. Run `pytest tests/search/test_sparse_cfr_evaluator.py` (and broader suites if needed) to document parity before concluding.
9. Update this plan’s Progress, Decision Log, Surprises, and Outcomes sections as milestones complete.

## Testing Strategy

1. **Parity checks (existing):** Continue running deterministic sparse vs. rebel comparisons on small search trees to ensure tree construction, initial policy/belief tensors, and CFR iteration outputs (`policy_probs`, `latest_values`, `cumulative_regrets`) match within tolerance.
2. **Leaf payout regression:** Create fixtures that force fold-only branches and pure showdowns; assert `initialize_subgame` seeds fold rewards correctly and `set_leaf_values` injects showdown EVs via `_showdown_value`.
3. **Belief/reach propagation:** Add tests that initialise asymmetric belief distributions, run `initialize_policy_and_beliefs`, and verify that reach weights and `beliefs`/`beliefs_avg` remain normalised and blocked according to `allowed_hands`.
4. **Warm-start behaviour:** Run a single warm-start iteration on a toy tree and compare sparse vs. rebel cumulative regrets/self-reach tensors to ensure the best-response sweep matches.
5. **Sampling path:** Exercise `sample_leaf` with deterministic RNG seeds, verifying the sampled PBS states/actions align with reach probabilities and that epsilon-greedy exploration respects `sample_epsilon`.
6. **Replay batches:** Compare sparse/rebel outputs of `training_data` (value, pre-value, policy batches) for identical trees, checking features, masks, and exploitability statistics match.

## Validation and Acceptance

Validation passes when `pytest tests/search/test_sparse_cfr_evaluator.py` succeeds, demonstrating that sparse and rebel evaluators agree on key tensors within tolerance. Manual inspection or additional asserts should confirm belief normalisation and reach weights match the dense implementation. Full `pytest` should continue to pass.

## Idempotence and Recovery

Tree building and tensor allocations depend only on inputs, so re-running initialisation is safe. Code edits are confined to `src/alphaholdem/search/` and new tests; if problems arise, revert files via version control. Re-activate the virtual environment and rerun pytest to recover from environment drift.

## Artifacts and Notes

Populate this section with relevant command outputs or tensor diagnostics as work progresses.

## Interfaces and Dependencies

Ensure `SparseCFREvaluator.initialize_subgame` maintains its public contract while populating tensors equivalent to Rebel’s (`leaf_mask`, `allowed_hands`, `prev_actor`, showdown metadata, etc.). Implement helper methods mirroring Rebel’s signatures, including `_calculate_reach_weights`, `_normalize_beliefs`, `_block_beliefs`, `_propagate_all_beliefs`, `_propagate_level_beliefs`, and `_get_mixing_weights`. CFR iteration methods must accept the same parameters as Rebel’s counterparts, notably `set_leaf_values(self, t: int)` and `compute_expected_values(self, policy=None, values=None)`. `training_data` must return three `RebelBatch` instances aligned with `RebelCFREvaluator.training_data`.

Revision history for this plan must be appended below with date and rationale whenever updates occur.
