# Notes: Preflop Value Sweep Follow-Ups

## Outputs
- 400-CFR presolved value dataset: `outputs/preflop_backward_induction/actions_12_15_value_presolve_lrbs_epoch1_400cfr_b8192_20260622/actions_12_15/solved`
- Fixed validation cache: `outputs/preflop_backward_induction/depth4_full_epoch_packed_shuffle_val10k_4x12_b8192_2048_train512_20260622/actions_12_15/validation/validation_n4096_cfr10000_b1d5684efe99ecee.pt`
- Prior LR/BS sweep: `outputs/preflop_backward_induction/actions_12_15_value_lrbs_sweep_20260622`

## Baseline Sweep Result
- Best one epoch on 400-CFR data: LR `0.01`, batch size `512`, validation value loss `0.013845053`.

## Findings
- Removed mistaken partial rollout output: `outputs/preflop_policy_states/eroymcd2_unique_buckets_20m_n5_cap5m_20260622_rerun`.
- `RebelCFRTrainer` supports `cosine`, `linear`, and `wsd`; "constant" is represented by setting `learning_rate_final == learning_rate`.
- AdamW LR is independently initialized by `train.adamw_learning_rate`, then scaled by the same schedule ratio as the main Muon LR.
- `train_value_batch` calls `_apply_schedules(step)`, so one-epoch sweeps exercise the actual scheduler path.
- Smoke test: LR `0.01`, batch `512`, linear schedule final ratio `0.1`, AdamW `0.005` gave validation value loss `0.0097839907` on 400-CFR data.
- Fixed-shuffle schedule sweep at LR `0.01`, batch `512`, AdamW `0.01`: constant `0.013572628`, cosine `0.010121048`, linear `0.010018783`, WSD-20 `0.010196411`, WSD-50 `0.009983364`.
- Fixed-shuffle AdamW/schedule sweep at LR `0.01`, batch `512`: best was WSD-50 with AdamW `0.005`, validation value loss `0.0096445081`. Linear best was AdamW `0.005`, loss `0.009893538`.
- Fixed-shuffle Muon LR sweep with WSD-50, AdamW `0.005`, batch `512`: LR `0.004` loss `0.010283897`, `0.006` loss `0.0099465819`, `0.008` loss `0.009797734`, `0.01` loss `0.0096445081`, `0.012` loss `0.009952401`, `0.016` loss `0.010279209`.
- Fixed-shuffle batch sweep with WSD-50, LR `0.01`, AdamW `0.005`: batch `256` loss `0.0095354964`, `512` loss `0.0096445081`, `768` loss `0.010371168`, `1024` loss `0.010546051`, `2048` loss `0.011933032`.
- 500-CFR presolve completed with `449,082` roots/value examples, `hands=169`, CFR batch `8192`, mean total nodes `59225.47`.
- Training on 500-CFR bfloat16 targets with best 400-CFR hyperparameters produced identical validation/train stats to 400-CFR: `0.0095354964`.
- 400-vs-500 target shard diff at bfloat16 precision: all 55 value shards exactly identical (`max_abs=0`).
- Huber-vs-MSE value-loss sweep under tuned settings: both produced `0.0095354964`.
