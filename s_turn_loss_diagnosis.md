# S_turn Validation Loss Diagnosis

Most recent local W&B run inspected: `wandb/run-20260710_020213-j9k53hh3`.

Checkpoint: `checkpoints-rebel-curriculum-sturn-5k-turnbase-newposneg-initfix-val4096-eturn300k-fp32pair-v2-wandb/turn/rebel_final.pt`

Validation set: `outputs/rebel_postflop/turn_val_4096_5kit_eturn100k_allincutoff_fp32pair_v2_20260707`

Overall validation metrics reproduced locally:
- `value_loss`: 0.00181489297
- `pot_relative_mae`: 0.18778037
- `pot_relative_rmse`: 0.51229661
- examples: 4096

## Entropy Finding

High-entropy beliefs are not the high-loss slice. Loss is concentrated in low-entropy beliefs.

Normalized belief entropy quartiles:

| entropy quartile | examples | value_loss | pot_rel_rmse | pct value loss sum | pct pot-rel sq sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| q1 lowest, 0.022-0.539 | 1024 | 0.00378722 | 0.791395 | 52.17% | 59.66% |
| q2, 0.539-0.691 | 1024 | 0.00213136 | 0.515593 | 29.36% | 25.32% |
| q3, 0.691-0.864 | 1024 | 0.000845656 | 0.316057 | 11.65% | 9.52% |
| q4 highest, 0.864-0.978 | 1024 | 0.000495333 | 0.240327 | 6.82% | 5.50% |

Correlations with per-example loss:
- entropy vs value loss: -0.4043
- entropy vs per-example pot-relative RMSE: -0.3666
- pot size vs value loss: +0.1480
- pot size vs per-example pot-relative RMSE: -0.3251
- scale/pot vs per-example pot-relative RMSE: +0.5879

## Pot-Size Split

Pot sizes are in bb using `bb=100`.

| pot range | examples | value_loss | pot_rel_rmse | pct examples | pct value loss sum | pct pot-rel sq sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-5bb | 828 | 0.00113815 | 0.682230 | 20.21% | 12.68% | 35.85% |
| 5-10bb | 791 | 0.00135249 | 0.702191 | 19.31% | 14.39% | 36.28% |
| 10-20bb | 825 | 0.00172604 | 0.514912 | 20.14% | 19.16% | 20.35% |
| 20-50bb | 773 | 0.00203292 | 0.294386 | 18.87% | 21.14% | 6.23% |
| 50-100bb | 451 | 0.00264010 | 0.152377 | 11.01% | 16.02% | 0.97% |
| 100-200bb | 336 | 0.00272631 | 0.0910315 | 8.20% | 12.32% | 0.26% |
| 200bb+ | 92 | 0.00347220 | 0.0807971 | 2.25% | 4.30% | 0.06% |

Interpretation:
- Raw value loss rises with pot size.
- Pot-relative error is dominated by small pots because `scale/pot` is large.
- Pots below 10bb are 39.5% of examples and 72.1% of total pot-relative squared error.

## Cross Slice

Entropy half by coarse pot range:

| slice | examples | value_loss | pot_rel_rmse | pct pot-rel sq sum |
| --- | ---: | ---: | ---: | ---: |
| low entropy + pot <10bb | 830 | 0.00202302 | 0.888604 | 60.97% |
| low entropy + 10-50bb | 783 | 0.00315364 | 0.561890 | 23.00% |
| low entropy + >=50bb | 435 | 0.00439591 | 0.158746 | 1.02% |
| high entropy + pot <10bb | 789 | 0.000422180 | 0.390022 | 11.16% |
| high entropy + 10-50bb | 815 | 0.000645554 | 0.217395 | 3.58% |
| high entropy + >=50bb | 444 | 0.00115754 | 0.0807410 | 0.27% |

The single biggest pot-relative offender is low-entropy small-pot turn roots.
