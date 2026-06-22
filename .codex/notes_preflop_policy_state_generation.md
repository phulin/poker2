# Notes: Preflop Policy State Generation

## Checkpoint
- Requested `v8yxyiya` has W&B metadata/logs but no local or W&B checkpoint file.
- Fallback `eroymcd2` checkpoint exists at `checkpoints-rebel-curriculum-preflop_2000_p6_lr0p01_backupcons_actor_lam01_rb32_from2p_norb/preflop/rebel_latest.pt`.
- `eroymcd2` is a 6-player compact 169-hand preflop transformer policy/value split model with hidden 192, ffn 256, policy layers 4, value layers 5.

## Rollout Semantics
- Use PBSEnv legal-bin mask and `step_bins`.
- Policy action sampling uses actor belief-weighted per-hand action probabilities.
- After an action is sampled, update only the actor's 169-class belief by multiplying by the selected action probability and normalizing.
- Reset rows once they leave preflop or terminate; generated rows are pre-action, nonterminal preflop public states.

## State Schema
- Save compact public state fields needed to reconstruct betting state: actor/button/action counts, pot/min-raise/last-aggressive amount, stacks/committed/chips placed, folded/all-in/acted masks, board placeholders, and scale.
- Do not save full solved CFR targets or replay batches.

## Generated Dataset
- Path: `outputs/preflop_policy_states/eroymcd2_policy_rollout_3m_20260621`
- Rows: 3,000,000 across 12 shards.
- Size: 496 MB.
- Source checkpoint: `eroymcd2`, step 1949.
- Validation: all rows have `street == 0` and `done == false`.
- `actions_this_round` histogram: 0:421636, 1:412658, 2:406235, 3:384874, 4:388817, 5:369193, 6:277263, 7:177148, 8:93152, 9:44947, 10:18460, 11:4506, 12:940, 13:157, 14:12, 15:2.
- Live-player histogram: 2:530684, 3:615370, 4:618021, 5:606827, 6:629098.

## Stratified Follow-up
- Add `--stratified` to generate per-bucket subdirectories for action counts 0-3, 4-7, 8-11, and 12-15.
- Internal frontier pools store beliefs at action counts 4, 8, and 12 so deeper buckets can be rolled from policy-reached states.
- Final saved shards still contain compact public state only.
- Stratified output: `outputs/preflop_policy_states/eroymcd2_policy_rollout_stratified_1m_buckets_20260621`
- Rows: 4,000,000 total, exactly 1,000,000 per bucket, 661 MB.
- Frontier counts used: action 4: 262,144, action 8: 75,918, action 12: 16,433.
- Per-action histogram:
  - 0:262144, 1:262144, 2:262144, 3:213568
  - 4:336857, 5:307066, 6:223858, 7:132219
  - 8:591871, 9:274920, 10:107474, 11:25735
  - 12:844106, 13:136990, 14:16842, 15:2062

## Unique Frontier Retry Probe
- Output: `outputs/preflop_policy_states/eroymcd2_unique_frontier_100k_n5_20260621`
- Semantics: 100,000 root states, up to 5 independent retry continuations per source frontier state, first success only, at most one saved successor per root at each frontier.
- Counts: frontier 0: 100,000; frontier 4: 100,000; frontier 8: 61,812; frontier 12: 1,802.
- 4->8 attempts: 26,182, 14,159, 9,363, 6,807, 5,301 successes.
- 8->12 attempts: 556, 409, 338, 282, 217 successes.
- Validation: all saved rows have exact frontier action count, `street == 0`, and `done == false`.
