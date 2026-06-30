# Notes: EOS Leaf Value Investigation

## Artifacts
- Corrected even-stack BTN tree: `outputs/preflop_open_tree_snapshots/btn_only_even20k_step1900_iter5000_seed42_uniform_eos/03_BTN_tree.pt`
- Corrected even-stack report: `outputs/preflop_open_tree_snapshots/btn_only_even20k_step1900_iter5000_seed42_uniform_eos/BTN_report.txt`
- EOS checkpoint: `outputs/epreflop_6p_live_pair/full_b16384_s5260_muon2e-2_adamw2e-3_warmup10_linear_shuffled_per_player_val/checkpoints/distilled_final.pt`

## Findings

### EOS Wiring
- Fused mixed-street preflop evaluation uses the active 4-7 value model for same-street cutoff positions and `closing_leaf_value_model` for `new_street_model_positions`.
- The corrected analyzer config now passes the checkpoint's embedded closing leaf checkpoint, so new-street leaves use the 6p EOS distilled model.
- The value feature encoder uses `pre_chance_node=True` for new-street leaves, selecting `last_board_indices` and setting the chance-phase feature appropriately.

### EOS Distillation Target
- EOS checkpoint metadata:
  - kind: `epreflop_6p_uniform_live_pair_distilled_model`
  - state dataset: `outputs/preflop_street_closed_states/live_pair_all_shuffled_seed20260628`
  - frozen target checkpoint: `checkpoints-epreflop-distill-100k-lr2e4-sample256-from-sflop2000-169/promoted/E_preflop.pt`
  - belief mode: `random`
  - target projection: `uniform unordered live pairs with stack-baseline nonparticipants`
- Target code enumerates every unordered pair of currently live players, projects each pair into a 2-player PBSEnv, evaluates the frozen 2p E_preflop with `pre_chance_node=True`, and averages only projections that include the player. Nonparticipants receive the stack baseline if they appear in no live pairs.
- This means a three-live-player leaf for BTN is trained to average BTN-vs-SB and BTN-vs-BB projections, not only the actual future pair produced by forced HU closure.

### Exact Leaf Inputs
- Primary nodes from even-20k BTN tree:
  - `limp_fold_check`: node 44, path `call/check -> fold -> call/check`
  - `r325_fold_call`: node 100, path `r325 -> fold -> call/check`
- Both nodes are preflop-closed street-boundary leaves:
  - `street=1`, `actions_this_round=0`, `to_act=5`, `button=3`
  - `has_folded=[True, True, True, False, True, False]`, so only BTN seat 3 and BB seat 5 are live.
  - Because only BTN and BB are live, the uniform live-pair target reduces to a single pair `3-5`; there is no averaging over BTN-vs-SB on these particular leaves.
- Leaf environment:
  - limp/check: pot 250, scale 20000, BTN/BB stacks 19900/19900.
  - r325/call: pot 700, scale 20000, BTN/BB stacks 19675/19675.

### Model/Target Probe Artifacts
- Raw and postprocessed probe JSON: `outputs/preflop_open_tree_snapshots/btn_only_even20k_step1900_iter5000_seed42_uniform_eos/eos_leaf_postprocess_probe.json`
- Raw pair-target probe JSON: `outputs/preflop_open_tree_snapshots/btn_only_even20k_step1900_iter5000_seed42_uniform_eos/eos_leaf_target_probe.json`
- In-domain belief probe JSON: `outputs/preflop_open_tree_snapshots/btn_only_even20k_step1900_iter5000_seed42_uniform_eos/eos_indomain_belief_probe.json`
- The evaluator postprocess subtracts a belief-weighted public-state correction from live players' raw model outputs before writing leaf values. The relevant correction is larger for `r325_fold_call` than for `limp_fold_check`.

### Underlying 2p Target Values
- After evaluator-style postprocess, the frozen 2p target still prefers limp/check over r325/call, but only modestly:
  - AA: `0.3491` vs `0.3297` (`+0.0194`)
  - KK: `0.2671` vs `0.1949` (`+0.0722`)
  - QQ: `0.2046` vs `0.1363` (`+0.0682`)
  - TT: `0.0718` vs `0.0406` (`+0.0311`)
  - AKs: `0.0093` vs `-0.0033` (`+0.0126`)
  - KQs: `-0.0342` vs `-0.0473` (`+0.0131`)
- The direct 6p student postprocessed values also prefer limp/check modestly:
  - AA delta about `+0.0240`
  - KK/QQ delta about `+0.0201`
  - AKs delta about `+0.0221`
- The saved tree `values_avg` gap is much larger (AA about `+0.1008`), so the large displayed gap is not explained by the frozen target alone.

### Belief Inputs
- For `values_avg`, the BB belief in `r325_fold_call` is extremely concentrated:
  - BB top class is AA at about `0.743`, KK at about `0.170`, entropy `1.06`, KL vs combo prior `4.31`.
- For `values_avg`, the BB belief in `limp_fold_check` is still strong but less degenerate:
  - KK `0.132`, AA `0.108`, QQ `0.105`, entropy `2.67`, KL `2.65`.
- The latest/current beliefs saved in the tree are near uniform for both leaves, but the saved `values_avg` reflects average-policy/average-belief dynamics.

### Dataset Distribution
- EOS training state dataset: `outputs/preflop_street_closed_states/live_pair_all_shuffled_seed20260628`, 8,609,903 rows.
- Live-count coverage: live-2 rows are common (`7,508,781`, about 87.2%).
- Scale distribution is much shallower than the even-20k probe:
  - median scale 3416
  - 95th percentile 10595
  - 99th percentile 14348
  - 99.9th percentile 18227
  - exact scale 20000 appears only once.
- Nearby even-stack low-pot coverage is sparse:
  - live-2, scale 19k-21k: 3002 rows.
  - live-2, scale 19k-21k, pot 200-300: 35 rows.
  - live-2, scale 19k-21k, pot 650-750: 65 rows.
- Pot/scale for the exact leaves (`0.0125` and `0.035`) is below the 0.1% overall pot/scale percentile (`0.0229`) or near the extreme low tail.
- Training beliefs for EOS distillation used exponential random beliefs weighted by combo prior. A 1M-sample Monte Carlo had:
  - max class 99.99th percentile about `0.114`, maximum observed `0.142`.
  - AA-mass 99.99th percentile about `0.041`, maximum observed `0.057`.
  - No sample had AA mass above `0.1`.
- Therefore the r325/call average BB belief with AA mass `0.743` is far outside the EOS student belief training distribution.

### In-Domain Belief Probe
- Added `scripts/probe_eos_in_domain_beliefs.py` to reconstruct the two saved BTN EOS leaf states, replace only the belief tensors, compare direct 6p EOS student values against the frozen 2p live-pair teacher, and save raw/postprocessed error summaries.
- On uniform combo-prior beliefs, the student and teacher are close:
  - `limp_fold_check`: BTN postprocessed weighted MAE `0.001105`; AA target/student `0.461603` / `0.465472` (`+0.003870`).
  - `r325_fold_call`: BTN postprocessed weighted MAE `0.000981`; AA target/student `0.467627` / `0.471452` (`+0.003825`).
- On 2048 training-style random exponential beliefs per leaf, they remain close:
  - `limp_fold_check`: BTN postprocessed weighted MAE `0.001989`, p95 `0.002666`; AA target/student `0.465388` / `0.465177` (`-0.000211`).
  - `r325_fold_call`: BTN postprocessed weighted MAE `0.001866`, p95 `0.002522`; AA target/student `0.469893` / `0.469715` (`-0.000178`).
- Random belief stats stayed in the training-like range:
  - BB max class mean about `0.045`, p95 about `0.066`, max about `0.108`.
  - BB AA mass mean about `0.0047`, max about `0.041`.
- Compared with the saved CFR average beliefs, the AA student-minus-teacher postprocessed error shrinks from about `-0.099` to `-0.0002` on `limp_fold_check`, and from about `-0.104` to `-0.0002` on `r325_fold_call`.
- This isolates the large student/teacher mismatch to OOD average-belief conditioning, not the public state alone. The even-20k low-pot public state is sparse in the dataset, but the student tracks the teacher well on that same public state when beliefs are training-like.

### ReBeL Policy/Value Target Consistency Check
- Source checked: Brown et al., "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games", arXiv:2007.13544.
- ReBeL treats the policy network target separately from value targets: after solving a subgame, it adds the final average subgame policy `pi_bar^T(beta)` for each PBS in the subgame to the policy dataset.
- Our policy extraction matches that shape:
  - `policy_targets` are pulled from `policy_probs_avg`.
  - `policy_features` are encoded from `beliefs_avg`.
  - With `cfr_avg=false`, fused preflop finalizes deferred `policy_probs_avg`, recomputes average-policy self-reach, and refreshes `beliefs_avg` before `training_data()`.
- This means policy training samples are internally consistent as "average-policy PBS -> average policy" targets.
- Value targets are a separate issue:
  - `training_data()` chooses `latest_values` only when `value_targets_from_final_policy=true`; otherwise it uses `values_avg`.
  - In the current runs, `value_targets_from_final_policy=false`, so value labels are the accumulated per-iteration backed-up values, not a direct final-average-policy evaluation.
  - This is compatible with CFR-D-style root value training, but those labels should not be interpreted as exact values of the final average policy.
- ReBeL's CFR-AVG appendix explicitly discusses that value-network input/output consistency can become subtle under average-policy leaf PBSs. Our current run is `cfr_avg=false`, so online leaf model queries use current beliefs for search, while extracted policy targets use average beliefs/policy after finalization.

### Interpretation
- Inputs are structurally consistent with the training pipeline: preflop-closed state, committed reset, `actions_this_round=0`, `actions_last_round` used under `pre_chance_node=True`, and only live BTN/BB pair for these leaves.
- The target construction's uniform live-pair averaging is not the issue for these particular leaves because only one live pair remains.
- The underlying frozen 2p target does mildly favor limp/check over r325/call under the branch beliefs, and that looks directionally sane if BB's raise-call range is much stronger than BB's check-behind range.
- The large tree `values_avg` gap appears amplified by average-belief dynamics and OOD branch beliefs, especially the near-pure-AA BB range on a very low-reach raise-call branch.
- The even-20k low-pot states are in the configured stack range but poorly represented in the EOS student state distribution. The in-domain belief probe suggests the branch beliefs, not the public state alone, are the dominant cause of the large EOS student/teacher mismatch observed under `values_avg`.
