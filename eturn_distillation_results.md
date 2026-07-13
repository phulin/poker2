# E-turn Distillation LR Results

## Setup
- Source checkpoint: `checkpoints-rebel-curriculum-sapcfr-80-40-300it-8000-val-ctx41-live-board96-belief128-canonical-k32-nobaseline-out0-lr001-random-wandb/promoted/S_river.pt`
- Student config: current `conf/config_rebel_curriculum_turn.yaml` TEB settings, `value_output_init_scale=0.1`
- Trial length: 1000 distillation steps
- Batch size: 1024
- Schedule: linear decay to 10% final LR unless noted
- W&B: disabled

## Validation Set
- Path: `outputs/rebel_postflop/eturn_val_16384_current_teb_sriver8000_20260708`
- Examples: 16,384 value-only end-of-turn rows
- Target source: `chance_expectation`
- Target construction: average over 48 legal river-card `S_river` outputs per example
- Builder: `scripts/build_eturn_validation_set.py`

## 8k No-Warmup Sweep

Command sweep: LR 0.04, 0.05, 0.06; plus follow-up LR 0.03. All runs used 8000 distillation steps, linear decay to 10% final LR, and no warmup.

Validation ranking:

| LR | Validation value loss | Element MSE | Trial |
|---:|---:|---:|---|
| 0.03 | 0.00024298 | 0.00018966 | `8k_lr0p03/t001_lr0p03_8000st_b1024` |
| 0.04 | 0.00024471 | 0.00019101 | `8k/t001_lr0p04_8000st_b1024` |
| 0.05 | 0.00024687 | 0.00019269 | `8k/t002_lr0p05_8000st_b1024` |
| 0.06 | 0.00025309 | 0.00019755 | `8k/t003_lr0p06_8000st_b1024` |

Training results:

| LR | Best live value loss | Best step | Final live value loss | Avg step time |
|---:|---:|---:|---:|---:|
| 0.03 | 0.00019 | 7456 | 0.00022 | 0.29121s |
| 0.04 | 0.00019 | 7456 | 0.00022 | 0.29154s |
| 0.05 | 0.00019 | 7456 | 0.00022 | 0.28991s |
| 0.06 | 0.00019 | 7456 | 0.00022 | 0.29120s |

Startup check:

| LR | Max value loss in first 25 steps | Step | Ratio vs step 0 |
|---:|---:|---:|---:|
| 0.03 | 0.00417 | 1 | 1.05x |
| 0.04 | 0.00506 | 1 | 1.27x |
| 0.05 | 0.00631 | 1 | 1.58x |
| 0.06 | 0.01213 | 4 | 3.04x |

## 100k Existing Checkpoint

Checkpoint: `checkpoints-rebel-curriculum-eturn-100k-turneq-posneg-noblockers-lr0p02-linear-b1024-from-3ytaa643-mlp-b96-belief128-wandb/promoted/E_turn.pt`

This run used LR 0.02, final LR 0.002, linear schedule, 100k steps. It was evaluated on the same fixed 16,384-item validation set. The checkpoint and validation set both use context width 41; the only compatibility issue was the old checkpoint config's action schedule metadata, so evaluation loaded the dataset with the validation manifest's action schedule.

| Steps | LR | Validation value loss | Element MSE | Notes |
|---:|---:|---:|---:|---|
| 100,000 | 0.02 | 0.00010909 | 0.00008515 | Existing promoted checkpoint |

Artifact: `outputs/experiments/eturn_100k_turneq_posneg_noblockers_current_teb_val_20260708.json`

## 300k Base Run

Started a W&B-enabled, no-warmup LR 0.01 run with 300,000 distillation steps, batch size 1024, and linear decay to final LR 0.001.

W&B:
- Run id: `owqc9mq4`
- URL: `https://wandb.ai/phulin-self/poker-rebel-postflop-curriculum/runs/owqc9mq4`

Paths:
- Checkpoints: `checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-wandb-20260708/t001_lr0p01_300000st_b1024`
- Train results: `outputs/experiments/eturn_distill_lr_current_teb_300k_lr0p01_wandb_20260708_results.jsonl`
- Logs: `outputs/experiments/eturn_distill_lr_current_teb_300k_lr0p01_wandb_logs/t001_lr0p01_300000st_b1024.log`

Initial startup check: stable; first 50 steps did not spike above step 0.

Prior attempt: a no-W&B run at the same LR/steps was stopped at about step 651 by request and should not be treated as the active 300k run.

Completed 2026-07-09 at step 300,000. Final W&B train loss was about `0.00007`; final LR was `0.001`.

## 300k + 50k Continuation

Started 2026-07-09 from the optimizer/RNG-bearing checkpoint:
`checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-wandb-20260708/t001_lr0p01_300000st_b1024/distill_E_turn/rebel_latest.pt`.

Continuation settings:
- Additional steps: 50,000, from global step 300,000 to target 350,000.
- LR: constant `0.001` (`learning_rate=learning_rate_final=adamw_learning_rate=0.001`).
- Batch size: 1024.
- W&B: resumed run id `owqc9mq4`, URL `https://wandb.ai/phulin-self/poker-rebel-postflop-curriculum/runs/owqc9mq4`.

Paths:
- Checkpoints: `checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-plus50k-lr0p001-wandb-20260709/distill_E_turn`
- Promoted: `checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-plus50k-lr0p001-wandb-20260709/promoted/E_turn.pt`
- Local W&B logs: `wandb/run-20260709_223137-owqc9mq4`

Startup check: resumed at step 300,000 and first continuation losses were stable around `0.00008` to `0.00011`.

## 4k No-Warmup Sweep

Command sweep: LR 0.04, 0.06, 0.08; 4000 distillation steps; linear decay to 10% final LR; no warmup.

Validation ranking:

| LR | Validation value loss | Element MSE | Trial |
|---:|---:|---:|---|
| 0.06 | 0.00030398 | 0.00023727 | `4k/t002_lr0p06_4000st_b1024` |
| 0.08 | 0.00030674 | 0.00023942 | `4k/t003_lr0p08_4000st_b1024` |
| 0.04 | 0.00031750 | 0.00024782 | `4k/t001_lr0p04_4000st_b1024` |

Training results:

| LR | Best live value loss | Best step | Final live value loss | Avg step time |
|---:|---:|---:|---:|---:|
| 0.04 | 0.00026 | 3719 | 0.00028 | 0.29286s |
| 0.06 | 0.00025 | 3661 | 0.00026 | 0.29239s |
| 0.08 | 0.00025 | 3719 | 0.00026 | 0.29292s |

Startup check:

| LR | Max value loss in first 25 steps | Step | Ratio vs step 0 |
|---:|---:|---:|---:|
| 0.04 | 0.00506 | 1 | 1.27x |
| 0.06 | 0.01215 | 4 | 3.05x |
| 0.08 | 0.01263 | 4 | 3.17x |

## LR Surface Fit

Fit source: completed no-warmup validation points from 1k, 4k, and 8k runs. The pending LR 0.03 8k run is not included.

Profiled model:

`log(value_loss) = b0 + b1*u + b2*u^2 + a*(log(lr) - (m0 + m1*u))^2`, where `u = log(steps) - mean_log_steps`.

Fit quality:

| Model | Log RMSE | Approx multiplicative error |
|---|---:|---:|
| Profiled convex LR surface | 0.04947 | 5.07% |
| Unconstrained full quadratic | 0.03486 | 3.55% |

The unconstrained full quadratic fit had no convex finite LR optimum in these extrapolated slices, so it is not useful for choosing an LR. The profiled model is usable, but it predicts a 1k optimum beyond the measured LR range, so treat long-step extrapolations as directional only.

Profiled optimum curve:

| Steps | Predicted optimal LR | Predicted val loss |
|---:|---:|---:|
| 1,000 | 0.30000 | 0.00055252 |
| 2,000 | 0.16699 | 0.00040261 |
| 4,000 | 0.09295 | 0.00030831 |
| 8,000 | 0.05174 | 0.00024812 |
| 12,000 | 0.03673 | 0.00022360 |
| 16,000 | 0.02880 | 0.00020984 |
| 32,000 | 0.01603 | 0.00018650 |
| 64,000 | 0.00892 | 0.00017419 |
| 100,000 | 0.00612 | 0.00017114 |

Two-stage sanity check:

Fitting each completed step-count slice independently in `log(lr)` gives convex optima at 4k and 8k only:

| Steps | Slice optimum LR | Predicted val loss | Observed best |
|---:|---:|---:|---|
| 1,000 | non-convex | n/a | 0.08 at 0.00051788 |
| 4,000 | 0.06405 | 0.00030372 | 0.06 at 0.00030398 |
| 8,000 | 0.04121 | 0.00024466 | 0.04 at 0.00024471 |

The two-stage trend from 4k and 8k predicts LR 0.00826 at 100k, with a loss trend around 0.00011127. This is a much more aggressive extrapolation than the profiled surface and should not be trusted without intermediate runs.

Fit artifact: `outputs/experiments/eturn_distill_lr_surface_fit_20260708.json`

## Earlier Fixed Validation Results

| LR | Warmup | Validation value loss | Element MSE | Trial |
|---:|---:|---:|---:|---|
| 0.08 | 0 | 0.00051788 | 0.00040422 | `t005_lr0p08_1000st_b1024` |
| 0.10 | 0 | 0.00052522 | 0.00040996 | `extra/t002_lr0p1_1000st_b1024` |
| 0.08 | 10 | 0.00056349 | 0.00043983 | `warmup10_r2/t002_lr0p08_1000st_b1024` |
| 0.06 | 0 | 0.00058775 | 0.00045877 | `extra/t001_lr0p06_1000st_b1024` |
| 0.01 | 0 | 0.00060266 | 0.00047040 | `t002_lr0p01_1000st_b1024` |
| 0.06 | 10 | 0.00063402 | 0.00049488 | `warmup10_r2/t001_lr0p06_1000st_b1024` |
| 0.02 | 0 | 0.00063435 | 0.00049514 | `t003_lr0p02_1000st_b1024` |
| 0.04 | 0 | 0.00064718 | 0.00050515 | `t004_lr0p04_1000st_b1024` |
| 0.005 | 0 | 0.00065149 | 0.00050852 | `t001_lr0p005_1000st_b1024` |

## Training-Loss Results

| LR | Warmup | Best live value loss | Best step | Final live value loss | Avg step time |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 0 | 0.00053 | 878 | 0.00060 | 0.29618s |
| 0.01 | 0 | 0.00050 | 878 | 0.00056 | 0.29743s |
| 0.02 | 0 | 0.00052 | 970 | 0.00059 | 0.29650s |
| 0.04 | 0 | 0.00053 | 970 | 0.00060 | 0.29642s |
| 0.06 | 0 | 0.00048 | 970 | 0.00056 | 0.29912s |
| 0.08 | 0 | 0.00042 | 989 | 0.00051 | 0.29644s |
| 0.10 | 0 | 0.00042 | 989 | 0.00050 | 0.29745s |
| 0.06 | 10 | 0.00052 | 970 | 0.00058 | 0.29637s |
| 0.08 | 10 | 0.00046 | 989 | 0.00052 | 0.29746s |

## Startup Spike Check

The first three LRs did not spike above step 0 in the first 25 steps. Higher LRs showed a clear startup spike. A 10-step warmup reduced the LR 0.06 and 0.08 startup peaks, but did not improve held-out validation.

| LR | Warmup | Max value loss in first 25 steps | Step | Ratio vs step 0 |
|---:|---:|---:|---:|---:|
| 0.005 | 0 | 0.00399 | 0 | 1.00x |
| 0.01 | 0 | 0.00399 | 0 | 1.00x |
| 0.02 | 0 | 0.00399 | 0 | 1.00x |
| 0.04 | 0 | 0.00506 | 1 | 1.27x |
| 0.06 | 0 | 0.01211 | 4 | 3.04x |
| 0.06 | 10 | 0.00606 | 8 | 1.52x |
| 0.08 | 0 | 0.01250 | 4 | 3.13x |
| 0.08 | 10 | 0.00849 | 9 | 2.13x |
| 0.10 | 0 | 0.01916 | 4 | 4.80x |

## Result Artifacts
- Initial sweep train results: `outputs/experiments/eturn_distill_lr_current_teb_20260708_results.jsonl`
- Initial sweep validation summary: `outputs/experiments/eturn_distill_lr_current_teb_20260708_val_summary.json`
- Extra LR train results: `outputs/experiments/eturn_distill_lr_current_teb_extra_20260708_results.jsonl`
- Extra LR validation summary: `outputs/experiments/eturn_distill_lr_current_teb_extra_20260708_val_summary.json`
- Warmup train results: `outputs/experiments/eturn_distill_lr_current_teb_warmup10_r2_20260708_results.jsonl`
- Warmup validation summary: `outputs/experiments/eturn_distill_lr_current_teb_warmup10_r2_20260708_val_summary.json`
- 4k train results: `outputs/experiments/eturn_distill_lr_current_teb_4k_20260708_results.jsonl`
- 4k validation summary: `outputs/experiments/eturn_distill_lr_current_teb_4k_20260708_val_summary.json`
- 8k train results: `outputs/experiments/eturn_distill_lr_current_teb_8k_20260708_results.jsonl`
- 8k validation summary: `outputs/experiments/eturn_distill_lr_current_teb_8k_20260708_val_summary.json`
- 8k LR 0.03 train results: `outputs/experiments/eturn_distill_lr_current_teb_8k_lr0p03_20260708_results.jsonl`
- 8k LR 0.03 validation summary: `outputs/experiments/eturn_distill_lr_current_teb_8k_lr0p03_20260708_val_summary.json`
- Existing 100k checkpoint validation: `outputs/experiments/eturn_100k_turneq_posneg_noblockers_current_teb_val_20260708.json`
- LR/steps surface fit: `outputs/experiments/eturn_distill_lr_surface_fit_20260708.json`

## Current Read
- Best held-out checkpoint in this set is the existing 100k LR 0.02 no-warmup run: validation value loss 0.00010909.
- Among the short new sweeps, the best is the 8k LR 0.03 no-warmup run: validation value loss 0.00024298.
- The 8k LR 0.04 run is very close: validation value loss 0.00024471.
- Extending from 4k to 8k steps improved validation again for overlapping tested LRs: LR 0.04 improved from 0.00031750 to 0.00024471; LR 0.06 improved from 0.00030398 to 0.00025309.
- The surface extrapolation under-predicted the benefit of going to 100k: it predicted about 0.00017378 at LR 0.02, while the measured fixed-set validation loss is 0.00010909.
- 10-step warmup reduced early spikes for LR 0.06 and 0.08, but worsened validation in the 1k setting.

## Errors Encountered
- `warmup_steps` override initially failed because the key is absent from `curriculum.substeps.distill_E_turn.train_overrides`; fixed `scripts/eturn_distill_lr_sweep.py` to use `+curriculum.substeps.distill_E_turn.train_overrides.warmup_steps=...`.
