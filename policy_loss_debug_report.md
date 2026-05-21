# Policy Loss Divergence Debug Report

## Diagnosis

The latest local W&B run, `wandb/run-20260521_154852-pkf73gjr`, diverged in the policy head. Policy loss was stable around 0.43 through step 671, then jumped at step 672 and reached thousands by step 700, while value loss stayed around 0.004 to 0.006.

W&B summary shows the high policy loss is mostly model-target KL rather than target entropy:

- `policy_target_entropy`: 0.2263
- `policy_target_model_kl`: 3469.0411
- `fresh_policy_target_model_kl`: 4745.6748

The run used `train.optimizer=muon`, `train.learning_rate=0.017`, `train.learning_rate_final=0.0017`, and `train.policy_head_muon_learning_rate=0.02`. The trainer decayed ordinary optimizer groups but kept the `policy_head_muon` group fixed at 0.02. W&B only logged the first optimizer group as `learning_rate`, so the high constant policy-head LR was not visible in charts.

At `checkpoints-rebel/rebel_latest.pt` step 699, the ordinary Muon group LR had decayed to about 0.01283, but `policy_head_muon` was still 0.02.

## Patch

- Scaled `policy_head_muon_learning_rate` by the main LR schedule ratio.
- Logged the active policy-head LR as `policy_head_muon_learning_rate`.
- Updated the optimizer schedule test and config docs.

## Verification

`uv run pytest tests/test_optimizers.py` passed.
