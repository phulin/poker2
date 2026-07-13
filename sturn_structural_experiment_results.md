# S_turn Structural Experiment Results

## Controlled Setup

- Training data: 1,024,000 fixed random-turn roots solved at 300 CFR iterations with promoted 300k `E_turn`.
- Validation data: disjoint 32,768-root dataset with the same sampler, solve budget, and closing model; seed 777.
- Training runs: one shuffled epoch, 500 updates, batch size 2048.
- Replication seeds: 42, 43, and 44 for baseline, blockers, and second moment plus blockers.

## Replicated Results

| Experiment | Runs | Validation loss mean | Loss std | Pot-relative RMSE mean | RMSE std |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3 | 0.000885696 | 0.000001902 | 0.266092 | 0.002081 |
| turn blockers | 3 | **0.000805732** | 0.000001268 | 0.266210 | 0.002275 |
| second moment + blockers | 3 | 0.000820933 | 0.000015154 | **0.253841** | 0.003202 |
| blocker coefficients refit to CFR roots | 2 | 0.000812824 | 0.000001018 | 0.267505 | 0.001651 |
| second moment + blockers + refit | 2 | 0.000830015 | 0.000018980 | 0.255113 | 0.002650 |

Blocker correction lowers mean raw validation loss by 9.03% relative to baseline. The gain is about 42 times the baseline loss standard deviation, so it is not seed noise.

Second moment plus blockers lowers pot-relative RMSE by 4.60% relative to baseline, but its raw-loss improvement is smaller and more seed-sensitive than blockers alone.

Refitting the analytic baseline against CFR root targets does not improve either blocker variant. The original `E_turn`-fit coefficients should remain in use.

## Other Structural Runs

Single-seed disjoint-holdout results:

| Experiment | Validation loss | Pot-relative RMSE |
| --- | ---: | ---: |
| turn equity feature head | 0.000877201 | 0.266343 |
| rank-64 cross-range interaction | 0.000880598 | 0.266757 |
| direct pair-operator application | 0.000886074 | 0.263951 |

These changes do not materially improve raw validation loss.

## Learned Equity Input

The turn equity feature head replaces the fixed calibrated equity mapping with a trainable per-hand 6-to-16-to-1 MLP. Its inputs are showdown value, own belief mass, compatible opponent mass, blocked opponent fraction, pot, and SPR. The head is initialized to reproduce the fixed positive/negative baseline before training. Runs before the board-conditioned sweep used zero in the blocker channel.

Single-seed blocker-aware results:

| Experiment | Validation loss | Pot-relative RMSE |
| --- | ---: | ---: |
| fixed blockers | 0.000806842 | 0.264012 |
| learned equity input + blockers | **0.000803143** | 0.264279 |
| learned equity input + blockers + root-refit initialization | 0.000807937 | 0.267026 |
| learned equity input + blockers + second moment | 0.000881454 | 0.265435 |

Learned blocker-aware equity input provides a small 0.46% raw-loss improvement over fixed blockers at the same seed. This is much smaller than the blocker correction itself and has not been replicated across seeds. Root-refit initialization does not help, and combining the learned equity input with second-moment belief features is harmful at 500 steps.

## Board-Conditioned Equity Input

Single-seed, 500-step results on the same pregen data and disjoint 300-CFR holdout:

| Experiment | Validation loss | Pot-relative RMSE |
| --- | ---: | ---: |
| learned equity input + explicit blocked fraction | **0.000787884** | **0.265683** |
| blocked fraction + board-only FiLM | 0.000848465 | 0.275439 |
| blocked fraction + exact-hand/board FiLM | 0.000852979 | 0.276349 |

The explicit blocked-range statistic improves raw loss by 1.90% over the earlier learned-equity-input result and by 2.35% over the same-seed fixed-blocker result. Both learned board-conditioning schemes regress substantially at 500 steps. Their zero-initialized output starts from the same prediction, but the unrestricted FiLM updates disturb early optimization. Replicate the blocked-fraction result before treating it as the new production candidate; do not pursue these FiLM forms without stronger regularization or a lower conditioner learning rate.

Additional analytic and capacity ablations:

| Experiment | Validation loss | Pot-relative RMSE |
| --- | ---: | ---: |
| blocked fraction control | **0.000787884** | 0.265683 |
| unblocked/blocked equity decomposition | 0.000793514 | 0.264536 |
| river-runout equity standard deviation | 0.000792546 | 0.264513 |
| decomposition + runout standard deviation | 0.000788281 | 0.265501 |
| explicit blocker interactions, width 16 | 0.000790769 | 0.267877 |
| blocked fraction, width 32 | 0.000806645 | 0.266992 |
| blocker interactions, width 32 | 0.000788402 | **0.264096** |

None beats the blocked-fraction control on raw loss. Decomposition and runout volatility slightly improve pot-relative RMSE but add compute, while doubling head width is harmful by itself. The simple 16-unit blocked-fraction head remains the best 500-step candidate.

## Update Budget And Per-Hand Range Interaction

Per-hand board-conditioned opponent-range buckets did not materially improve the 500-step result:

| Experiment | Validation loss | Pot-relative RMSE |
| --- | ---: | ---: |
| blocked fraction control | **0.000787884** | 0.265683 |
| 16 relative strength buckets | 0.000789480 | **0.264451** |
| 32 relative strength buckets | 0.000799667 | 0.266002 |
| 16 coarse strength buckets | 0.000803435 | 0.265894 |

The original 500-step protocol is one pass over 1,024,000 unique examples, so its online training loss is not empirical loss on previously optimized rows. Repeating 32,768 rows for 500 updates drives online loss to approximately 0.00020, proving the model can fit 300-CFR targets but overfits the small subset.

Repeating the full pregen dataset produces the first large improvement:

| Full-dataset passes | Steps | Validation loss | Pot-relative RMSE |
| ---: | ---: | ---: | ---: |
| 1 | 500 | 0.000787884 | 0.265683 |
| 3 | 1,500 | 0.000644527 | 0.237048 |
| 5 | 2,500 | **0.000599902** | **0.226255** |

Five passes improve raw loss by 23.86% and pot-relative RMSE by 14.84% over one pass. The gain from three to five passes is still 6.92% raw loss and 4.55% RMSE. Update budget, rather than additional scalar equity features or range-interaction capacity, is the dominant bottleneck in these controlled 300-CFR experiments.

## Paired CFR Targets

Saved 4,096 identical root PBSs and solved each at 300, 1,000, and 5,000 CFR iterations with solve seed 9001. The 300-iteration solve was repeated with seed 9002.

Weighted target disagreement:

| Pair | MSE | RMSE | Pot-relative RMSE |
| --- | ---: | ---: | ---: |
| 300 seed 9001 vs 300 seed 9002 | 0.0000000274 | 0.000165 | 0.00123 |
| 300 vs 1,000 | 0.000268 | 0.01638 | 0.1825 |
| 300 vs 5,000 | 0.000406 | 0.02015 | 0.2447 |
| 1,000 vs 5,000 | 0.0000681 | 0.00825 | 0.1042 |

Solve-seed noise is negligible. CFR-budget bias is large and remains meaningful between 1,000 and 5,000 iterations.

The 300-to-5,000 target MSE is highest in the lowest-entropy quartile: 0.000618 versus 0.000323 in the highest-entropy quartile. By pot, it peaks around 50-200bb at approximately 0.00058. Thus the model's previously identified hard slices are also the slices where 300-iteration labels differ most from converged labels.

## Conclusion

Two distinct bottlenecks are now isolated:

1. Missing blocker correction is a real representation error and should be enabled for `S_turn`.
2. The remaining low-entropy/large-pot problem is substantially target quality: 300-iteration labels are biased relative to 5,000 iterations, not noisy across solve seeds.

The production candidate for raw value loss is blockers alone. If pot-relative error is the priority, second moment plus blockers is the better candidate. Further small belief-encoder ablations are lower priority than improving the CFR teacher budget or distilling a correction from paired higher-iteration targets.

Artifacts:

- `outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711`
- `outputs/rebel_postflop/paired_sturn_4096_300_1000_5000it_eturn300k_20260711`
- `outputs/sturn_pregen_500step_structural_sweep_20260711`
- `outputs/sturn_blocker_replication_seed43_20260712`
- `outputs/sturn_blocker_replication_seed44_20260712`
- `outputs/sturn_equity_board_conditioned_500step_sweep_20260712`
- `outputs/sturn_equity_analytic_features_500step_sweep_20260712`
- `outputs/sturn_equity_blocker_interactions_500step_sweep_20260712`
- `outputs/sturn_perhand_range_interaction_500step_sweep_20260712`
- `outputs/sturn_repeat32k_capacity_diagnostic_20260712`
- `outputs/sturn_full_dataset_3epoch_1500step_20260712`
- `outputs/sturn_full_dataset_5epoch_2500step_20260712`
- `outputs/experiments/sturn_root_turneq_blockers_fit_32768_20260712.json`
