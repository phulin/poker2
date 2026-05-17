# BetterFFN / ReBeL Correctness Audit

## Summary

The model path is wired as `BetterFFN`; "rebel" is the CFR/search training path. I did not see a basic tensor-shape mismatch or obvious value sign flip in the value/policy supervised losses. The bigger correctness risks are around chance-boundary beliefs/targets, checkpoint/config drift, and search-time precision.

## Findings

1. Checkpoint resume can silently change the training semantics.
   - Current YAML: `num_envs=512`, `batch_size=2048`, `search.depth=5`, `iterations=200 -> 500`.
   - Latest `rebel_latest.pt`: `num_envs=1024`, `batch_size=4096`, `search.depth=3`, `iterations=400 -> 1000`, step 99.
   - `load_checkpoint()` restores weights and optimizer, but does not restore or validate config. Resuming that checkpoint under the current YAML changes the search/data distribution.

2. Root beliefs from sampled street-boundary PBS are pre-chance, but root policy is computed before blocking to the current board.
   - `PublicBeliefState` explicitly allows street-end beliefs to be pre-chance.
   - `sample_leaves()` copies `self.beliefs[sampled_continue]`.
   - `initialize_subgame()` stores those beliefs directly in `self.beliefs[:N]`.
   - `initialize_policy_and_beliefs()` calls `_get_model_policy_probs()` before `_block_beliefs()` / `_normalize_beliefs()`.
   - For a root that is now on the flop/turn/river, the initial model policy can be conditioned on a range that still includes hands blocked by the newly dealt board.

3. The pre-chance chance-value target appears hand-conditionally wrong.
   - `flop_chance_values()` and `single_card_chance_values()` average model hand values over public cards, then divide by a root-level card/flop count.
   - A per-hand value before chance should average only chance outcomes compatible with that hand, and usually with opponent-range blocker weighting too.
   - As written, outcomes that collide with a specific hero hand are still included in that hand's denominator, and their blocked-hand model values can leak into the average.

4. Canonical flop averaging is not active after cleanup.
   - `flop_chance_values()` evaluates raw flop samples from the 22,100-card-combination list and blocks/normalizes beliefs against each raw board.
   - The unused canonical flop cache was removed, so the earlier suit-symmetry concern only applies if canonical flop evaluation is reintroduced later.

5. Search-time neural values are generated under bf16 autocast.
   - CFR leaf values and chance targets call the model inside `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`, then cast outputs back to fp32.
   - CFR accumulation itself is fp32 and `calculate_unblocked_mass()` uses fp64 internally, so this is not a global fp32 issue.
   - Still, if local exploitability/targets are sensitive at ~1e-3 scale, compare one short run with autocast disabled for value-target generation.

## Things That Look Internally Consistent

- BetterFFN zero-sum enforcement makes the belief-weighted sum of both players' values zero, not per-hand `v0[h] = -v1[h]`; that seems intentional.
- Policy/value loss weighting by opponent-compatible mass is conceptually consistent for counterfactual hand values.
- Suit permutation target mapping and the consistency loss appear directionally correct.
