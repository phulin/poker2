# Notes: Policy Loss Divergence Debug

## Findings

- Latest local W&B run is `wandb/run-20260521_154852-pkf73gjr`, started 2026-05-21 15:48:52 UTC.
- Console log shows policy loss stable around 0.43 through step 671, then jump at step 672:
  - step 671: policy 0.43529, value 0.00396
  - step 672: policy 3.62332, value 0.00597
  - step 700: policy 2515.03958, value 0.00472
  - step 718: policy 3469.26745, value 0.00506
- W&B summary confirms this is model-vs-policy-target KL, not target entropy:
  - `policy_target_entropy`: 0.2263
  - `policy_target_model_kl`: 3469.0411
  - `fresh_policy_target_model_kl`: 4745.6748
- Summary gradient histograms show the largest raw gradient bin on `policy_hand_bias_action.linear_out.weight` at about 25k; `grad_norm_clipped` is still about 1.0, so global clipping happened after the large raw policy-gradient signal.
- The run config used `train.optimizer=muon`, `train.learning_rate=0.017`, `train.learning_rate_final=0.0017`, `train.episodes_per_step=60`, and `train.policy_head_muon_learning_rate=0.02`.
- `src/p2/rl/cfr_trainer.py` decayed ordinary param groups but kept `policy_head_muon` fixed at 0.02. W&B only logged the first param-group LR, so the fixed policy-head LR was hidden in the charts.
- At checkpoint step 699, the optimizer state still has `policy_head_muon` LR 0.02 while the ordinary Muon group is about 0.01283.
- Implemented a narrow guardrail: scale `policy_head_muon_learning_rate` by the main schedule ratio and log it separately as `policy_head_muon_learning_rate`.
- Verification: `uv run pytest tests/test_optimizers.py` passed.
