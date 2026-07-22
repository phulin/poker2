# S_turn No-TEB Experiment Addition

Added seven fixed-data, three-epoch `S_turn` trials:

- Warm-started no-TEB cosine LR sweep at 0.5x, 1x, and 2x the production
  `0.004` Muon / `0.0004` AdamW rates. Both optimizer rates and the cosine
  final LR are scaled together.
- Cold-started no-TEB value-output initialization scales at `0.0`, `0.03`,
  `0.1`, and `0.3`.

The output-scale trials deliberately skip E_turn checkpoint initialization.
Loading that checkpoint copies compatible value-head weights after model
construction and would otherwise erase the requested initialization-scale
difference.

Run all trials with:

```bash
scripts/launch_sturn_no_teb_experiments_20260713.sh
```

Validation completed with shell syntax checking, Python bytecode compilation,
CLI choice inspection, config assertions for all seven trials, and
`git diff --check`.

## Production-Relative Results

All corrected trials completed successfully on the 32,768-example hard
holdout. Lower is better.

| Experiment | Muon / AdamW LR | Value loss | Pot-relative RMSE | Pot-relative MAE |
| --- | --- | ---: | ---: | ---: |
| `no_teb_prod_lr2_cosine` | `0.002 / 0.0002` | 0.00059841 | **0.17472** | 0.07756 |
| `no_teb_prod_lr4_cosine` | `0.004 / 0.0004` | **0.00057485** | 0.17685 | **0.07634** |
| `no_teb_prod_lr8_cosine` | `0.008 / 0.0008` | 0.00060023 | 0.18785 | 0.07946 |
| `no_teb_cold_out0p00` | `0.004 / 0.0004` | 0.00200733 | **0.35127** | 0.15960 |
| `no_teb_cold_out0p03` | `0.004 / 0.0004` | 0.00201693 | 0.35132 | 0.16057 |
| `no_teb_cold_out0p10` | `0.004 / 0.0004` | 0.00200366 | 0.35130 | **0.15935** |
| `no_teb_cold_out0p30` | `0.004 / 0.0004` | **0.00199350** | 0.35243 | 0.15987 |

The production 1x pair has the best value loss, about 3.9% below 0.5x and
4.2% below 2x. The 0.5x pair narrowly has the best pot-relative RMSE, while 2x
is worse on every holdout metric. Cold output initialization scale again has
little effect: `0.3` wins value loss, `0.0` wins RMSE, and total value-loss
spread is only about 1.2%. All cold starts remain far behind the E_turn
warm-started LR trials after three epochs.

Corrected artifacts and checkpoints are under
`outputs/sturn_3epoch_no_teb_production_lr_sweep_bs2048_20260713/`.

## Batch-4096 TEB Ablation

The production 1x no-TEB trial was repeated at batch 4096 for an exact
batch/step match to the existing TEB-on baseline: 750 steps, three passes over
the same 1,024,000-example cycle, seed 42, and the same E_turn initialization.

| TEB | Batch / steps | Value loss | Pot-relative RMSE | Pot-relative MAE |
| --- | --- | ---: | ---: | ---: |
| On | `4096 / 750` | **0.00046527** | **0.16794** | **0.07125** |
| Off | `4096 / 750` | 0.00060392 | 0.18191 | 0.07932 |

At matched batch size, disabling TEB increases value loss by 29.8%, RMSE by
8.3%, and MAE by 11.3%. The no-TEB batch-4096 run is also 5.1% worse in value
loss than the no-TEB batch-2048 run.

Batch-4096 no-TEB artifacts are under
`outputs/sturn_3epoch_no_teb_production_lr_sweep_bs4096_20260714/`.

## No-TEB LR × Batch-Size Sweep

The production-relative LR sweep was completed at 0.5x, 1x, and 2x batch
size, then extended to batch 1024. Every cell uses seed 42, the same E_turn
initialization and fixed dataset, and exactly 3,072,000 training examples
(three epochs).

### Holdout value loss

| Batch size | 0.5x LR | 1x LR | 2x LR |
| ---: | ---: | ---: | ---: |
| 1024 | 0.00057140 | **0.00056859** | 0.00061105 |
| 2048 | 0.00059841 | **0.00057485** | 0.00060023 |
| 4096 | 0.00066151 | **0.00060392** | 0.00060717 |
| 8192 | 0.00086520 | 0.00068487 | **0.00065249** |

### Pot-relative RMSE

| Batch size | 0.5x LR | 1x LR | 2x LR |
| ---: | ---: | ---: | ---: |
| 1024 | **0.17286** | 0.17600 | 0.18781 |
| 2048 | **0.17472** | 0.17685 | 0.18785 |
| 4096 | 0.18655 | **0.18191** | 0.18670 |
| 8192 | 0.22642 | **0.19337** | 0.19615 |

Batch 1024 is best overall: 1x LR minimizes value loss, while 0.5x LR
minimizes RMSE. The gain over batch 2048 is modest (`0.00056859` vs
`0.00057485` best value loss), while requiring twice as many optimizer steps.
Batch 4096 also prefers 1x. Batch 8192 benefits from a higher LR, with 2x
winning value loss and 1x narrowly winning RMSE, but every batch-8192 cell
remains worse than the smaller-batch cells.

The batch-size artifact roots are:

- `outputs/sturn_3epoch_no_teb_production_lr_sweep_bs2048_20260713/`
- `outputs/sturn_3epoch_no_teb_production_lr_sweep_bs4096_20260714/`
- `outputs/sturn_3epoch_no_teb_production_lr_sweep_bs8192_20260714/`
- `outputs/sturn_3epoch_no_teb_production_lr_sweep_bs1024_20260714/`

## Cold-Start Value-Depth Sweep

True cold-start no-TEB models with 6, 10, and 14 value layers were trained at
batch 1024 for three epochs. All use production 1x LR (`0.004/0.0004`), output
initialization scale `0.3`, seed 42, and no E_turn checkpoint initialization.

| Value layers | Value loss | Pot-relative RMSE | Pot-relative MAE | Train step | Inference 4096 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0.00162805 | 0.32233 | 0.14389 | 26.86ms | 1.723ms |
| 10 | 0.00154893 | 0.31386 | 0.13964 | 34.57ms | 1.987ms |
| 14 | **0.00149723** | **0.30708** | **0.13536** | 41.68ms | 2.307ms |

Accuracy improves monotonically with depth. Relative to 6 layers, 14 layers
reduces value loss by 8.0% and RMSE by 4.7%, but increases steady-state train
step time by 55.2% and inference time by 33.9%. Even the 14-layer cold start
remains far behind the 7-layer E_turn-warm-started no-TEB model at the same
batch/LR (`0.00056859`), reinforcing that initialization matters much more than
depth after three epochs.

Artifacts are under
`outputs/sturn_3epoch_no_teb_cold_depth_sweep_bs1024_20260714/`.

## Superseded Results (Wrong LR Anchor)

> These results used `0.04/0.04` as 1x instead of the production S_turn rates
> `0.004` Muon / `0.0004` AdamW. They are retained only as an audit trail and
> must not be used for the requested production-relative comparison.

All trials completed successfully on the 32,768-example hard holdout. Lower is
better.

| Experiment | Value loss | Pot-relative RMSE | Pot-relative MAE |
| --- | ---: | ---: | ---: |
| `no_teb_lr20_cosine` | **0.00071080** | **0.21756** | **0.08982** |
| `no_teb_lr40_cosine` | 0.00088945 | 0.24178 | 0.10033 |
| `no_teb_lr80_cosine` | 0.00098301 | 0.25291 | 0.10715 |
| `no_teb_cold_out0p00` | 0.00120129 | 0.28429 | 0.12183 |
| `no_teb_cold_out0p03` | **0.00118674** | 0.28417 | 0.12177 |
| `no_teb_cold_out0p10` | 0.00118716 | 0.28258 | **0.12034** |
| `no_teb_cold_out0p30` | 0.00120104 | **0.28224** | 0.12147 |

The warm-started 0.5x LR trial is the clear winner: its value loss is about
20.1% below 1x and 27.7% below 2x. Among cold starts, output scale has little
effect after three epochs. Scale `0.03` narrowly wins value loss, while `0.3`
has the best pot-relative RMSE; the spread in cold-start value loss is only
about 1.2%.

Superseded artifacts and checkpoints are under
`outputs/sturn_3epoch_no_teb_sweep_bs2048_20260713/`.
