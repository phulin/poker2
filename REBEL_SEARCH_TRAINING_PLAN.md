## Goal

Add lightweight training-time search (depth-limited CFR-D) during the model update step (not during data collection), using the current model’s policy head to initialize policies at interior nodes and the value head to evaluate leaves. Restrict the action set to 4 branches per node: fold, call, 1x pot, all-in. Only enable when using a non-quantile value head. Use the CFR-D algorithm as described in `rebel.txt` for depth-limited solving.

## High-level approach

- During the update step, for each sampled minibatch transition, treat the current state (PBS) as a root and run a small, batched, depth-limited CFR-D search over a simplified action set (arity 4) to compute an improved (near-equilibrium) policy at the root.
- Use the model’s policy head as initialization priors at interior nodes; use the model’s value head to evaluate depth-D leaves conditional on the iteration policy (CFR-D leaf semantics).
- Use the resulting CFR policy at the root as the training target for the policy head (distillation/CE or KL loss), instead of PPO ratio/clipping. Keep value loss (returns/quantile off) and entropy regularization configurable.
- Keep implementation vectorized over a "search env" that holds a replicated/cloned batch of the current states sized B * (1 + 4 + 4^2 + … + 4^D), where B is the minibatch size.

## Constraints and simplifications

- Action set: {fold, check/call, 1x pot raise, all-in} only. Map 1x pot raise via `legal_bins_amounts_and_mask([1.0])`; fold/call/all-in map to fixed bins.
- Value head must be non-quantile (scalar); enforce guard in trainer init.
- Depth D initial: 2; iterations K initial: 100. Budget O(B * 4^D) memory/time.

## Components to add

1) `alphaholdem/rl/cfr_manager.py`
- Coordinator that owns one large `HUNLTensorEnv` and runs batched CFR-D per minibatch.
- Key responsibilities:
  - Preallocate a big `HUNLTensorEnv` with capacity M = B * Σ_{i=0..D} 4^i to hold the full tree.
  - Seed the depth-0 slice from the sampled minibatch via vectorized state copy helpers on `HUNLTensorEnv`.
  - Provide vectorized `expand(parent_indices) -> child_indices` and `step(children_indices, collapsed_actions)` over the internal env (no Python loops); keep everything fully vectorized.
  - Maintain per-node buffers: regrets, current policies, average policies, node values, logits/masks used by `dcfr`.
  - Interface with the model to get logits/values for node sets; collapse policy to 4 actions (see below) and pass tensors to `dcfr`.
  - Return root CFR policy targets `[B, 4]` mapped to full-bin space for loss computation.

2) `alphaholdem/search/dcfr.py`
- Batched depth-limited search driver implementing CFR-D update rules.
- Key dataclasses / functions:
  - `@dataclass class SearchConfig`: `depth: int = 2`, `iterations: int = 100`, `branching: int = 4`, `use_avg: bool = False` (pure CFR-D for v1).
  - `def run_dcfr(model, search_env, tsb_like, indices_root: Tensor, cfg: SearchConfig) -> Tensor`
    - Returns improved root policy over 4 actions for each root index (shape [B, 4]) and optional node stats.
  - Internals (CFR-D):
    - On iteration t, leaf values use actual env rewards when `done` (terminal), else V(·) conditioned on β derived from π^t (per `rebel.txt`).
    - Update regrets at interior nodes from counterfactual values; compute π^{t+1} via regret matching.
    - Optionally track average policy π̄ if needed for diagnostics.

Division of responsibilities:
- `CFRManager`: env and batching only (node expansion/stepping/embedding/logit & value gathering), fully vectorized.
- `dcfr`: math-only (regret computation, policy updates, recording logits/values/regrets per node).

3) `SelfPlayTrainer` integration (search during update)
- Config gate: `cfg.search.enabled`, `cfg.search.depth`, `cfg.search.iterations`.
- Guard: assert `self.value_head_type != ValueHeadType.quantile` if search enabled.
- In `update_model` before computing policy loss:
  - Use `CFRManager` to run batched CFR-D over the current minibatch roots (seeded into the manager-owned env) and get root CFR policy targets over 4 actions; map to full-bin action space for loss.
  - Replace PPO policy loss with a new LossCalculator distillation loss to the CFR target (e.g., `KL(target||model)` or CE with soft targets). Keep value loss and entropy regularization as configured.
- Logging: Add counters for search time per update, CFR iterations actually run, and diagnostic KL between model and CFR policy.

## Mapping: 4 collapsed actions → bin indices

- fold → bin 0 (only legal when facing bet; else illegal and masked out)
- check/call → bin 1
- bet/raise (collapsed "bet pot") → aggregate across all preset raise bins (2..B-2); see collapsing below
- all-in → bin `B-1` (last index)

If a collapsed action is illegal, regret matching and sampling should assign zero probability; distributions renormalize over legal set.

## Value and policy extraction

- Policy collapsing (vectorized): obtain model logits over full `num_bet_bins`; apply legal mask; compute softmax over the full legal set to get probabilities; then collapse to 4 actions as
  - P_fold = P(bin 0)
  - P_call = P(bin 1)
  - P_betpot = Σ_{k=2..B-2} P(bin k)
  - P_allin = P(bin B-1)
  This preserves all raise-bin logit mass in the collapsed bet action. Illegal bins contribute 0 by masking.
- Values: use scalar value head. If PopArt is active, apply `denormalize_value` as done in training code.
- For opponent-acting nodes (player=1), flip perspective for value targets and invert action-actor features (use `TokenSequenceBuilder.encode_tensor_states(player=1, idxs)` which handles perspective flipping).

## CFR details (single-iteration baseline)

- For each root:
  1) Initialize π at every node from policy head (masked to 4 actions). Initialize regrets R=0, π̄=π.
  2) Do 1 forward sweep to depth D computing counterfactual values using π on opponent/our nodes and leaf V(s) at depth D.
  3) Compute instantaneous regrets at visited nodes: r(a) = Q(a) − Σ_b π(b)Q(b). Accumulate into R += r⁺.
  4) Update π_new by regret matching over R (with floor 0). Update π̄ = linear average if iterations > 1.
- Return root π (single-iter) or π̄ (multi-iter) for sampling.

## Data structures and shapes

- M = B * Σ_{i=0..D} 4^i total nodes.
- Single big `HUNLTensorEnv` owned by `CFRManager`, sized M.
- Per-depth slices tracked as `[depth_offsets[d] : depth_offsets[d+1])` and mapping parent→children via arange and reshape.
- Buffers per node: `policy[M, 4]`, `regrets[M, 4]`, `avg_policy[M, 4]`, `value[M]`, `legal_mask_full[M, B]`, `logits_full[M, B]` (as needed for `dcfr`).

### `HUNLTensorEnv` helpers needed

Add vectorized utilities to support manager operations:
- `copy_state_from(src_env, src_indices, dst_indices)` → copies all state tensors from `src_env` rows to `self` rows.
- `clone_states(dst_children, src_parents)` → copies parent rows into child rows using advanced indexing (used for tree expansion).
- `legal_mask_bins(indices)` → returns `[len(indices), B]` legal mask for those rows.

## Integration points and file changes

- New files:
  - `src/alphaholdem/search/cfr_manager.py`
  - `src/alphaholdem/search/dcfr.py`
- Edits:
  - `src/alphaholdem/rl/self_play.py`
    - Add config gate and guards in `__init__`.
    - In `update_model`, call `CFRManager` to compute CFR targets for the minibatch.
    - Replace PPO policy loss with supervised distillation to CFR target (configurable).
  - `src/alphaholdem/models/transformer/token_sequence_builder.py`
    - see below.
  - `src/alphaholdem/env/hunl_tensor_env.py`
    - Add `copy_state_from`, `clone_states`, and `legal_mask_bins(indices)` helpers used by `CFRManager`.
### Token sequences for search (transformer models)

- `CFRManager` will also own a `TokenSequenceBuilder` sized to the big env (capacity M). It must:
  - Seed root nodes with `add_cls`, `add_game`, and initial `add_context` like in rollout.
  - Provide `clone_tokens(dst_children, src_parents)` to copy token buffers (ids, streets, actors, masks, amounts, context, lengths) when expanding nodes.
  - Before evaluating a node set, call `add_context(indices)` to snapshot current legality and context, then after stepping, call `add_action(indices, actors, action_ids, legal_masks, action_amounts, token_streets)` to append action tokens.
  - For v1 (within-round only), do not append street/card tokens; future versions can add `add_street`/`add_card` when allowing round advancement.
- `dcfr` remains unaware of token sequences; it receives `StructuredEmbeddingData` from the manager via `TokenSequenceBuilder.encode_tensor_states(player, idxs)` for both our and opponent nodes.

## Config additions

Extend config schema (Hydra) with:

```yaml
search:
  enabled: false
  depth: 1
  iterations: 1
  policy_loss: "kl"   # or "ce"
  weight: 1.0         # weight for CFR distillation in total loss
```

Guard: `assert cfg.model.value_head_type != "quantile"` if `search.enabled`.

## Update-time loss and logging

- Policy loss: distill to CFR target: `L_policy = KL(target || model)` over the 4-action slice (mapped to full space), masked by legality. Optionally use CE with soft targets.
- Value loss: unchanged (returns vs predicted values), PopArt handling as-is.
- Entropy: keep as regularization if desired.
- Log:
  - `search/time_ms_per_update`
  - `search/iterations`
  - `search/root_entropy`
  - `search/model_vs_cfr_kl`

## Performance considerations

- Depth and branching controlled to keep compute bounded. Start with D=2, K=100.
- Use vectorized cloning and stepping; avoid Python loops over nodes.
- Re-use model with autocast and no_grad.
- If transformer KV-cache is used, search likely disables KV cache for now (short tree); or use a separate cache manager per depth if trivial.

## Testing strategy

- Unit-test legality and mapping for 4-action collapse on random states.
- Snapshot test: compare search policy vs base head on hand-crafted states; ensure distribution excludes illegal actions and sums to 1.
- End-to-end: run short rollout with `search.enabled=true, depth=1` and ensure update-time search runs and training proceeds without NaNs.

## Incremental rollout plan

1) Implement 4-action mapping utilities and legality checks (unit tests).
2) Extend `VectorizedReplayBuffer` to store `SearchStateSnapshot` and plumb through sampling.
3) Implement `HUNLTensorEnv` helpers and vectorized manager-side clone/expand and within-round `step`.
4) Implement `CFRManager` and `run_dcfr` single-iteration (K=1) with value leaves (CFR-D semantics).
5) Integrate into `update_model` with distillation loss; add config flags.
6) Add optional K>1 iterations and diagnostics; profiling and guardrails.

## Risks & mitigations

- Perspective/value sign errors at opponent nodes → use `TokenSequenceBuilder.encode_tensor_states(player=1, idxs)` and ensure leaf values are always from root player perspective; add asserts on symmetry small cases.
- All-in and terminal handling → propagate terminal rewards/values correctly; mask further expansion.
- Transformer sequence length for hypothetical nodes → for D≤2 the added tokens are small; still ensure builder length guard is respected.


