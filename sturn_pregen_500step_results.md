# S_turn Pregenerated 500-Step Sweep

Dataset: `outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711`

All runs used one shuffled epoch of 1,024,000 examples, 500 updates at batch size 2048, seed 42, the `config_rebel_curriculum_turn.yaml` training settings, and compatible value initialization from the promoted 300k `E_turn` checkpoint.

Validation:

- Matched: first 32,768 examples from the 300-CFR pregenerated dataset.
- Hard: existing 4,096-example 5k-CFR turn validation set.

| Experiment | Matched loss | Matched pot-rel RMSE | Hard loss | Hard pot-rel RMSE |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.000918715 | 0.274418 | 0.002570695 | 0.524564 |
| belief second moment | 0.000918918 | 0.260211 | **0.002523063** | 0.517976 |
| board mass | 0.000923114 | 0.278436 | 0.002553214 | 0.530388 |
| belief rank 256 | 0.000935172 | 0.278305 | 0.002558614 | 0.526330 |
| board conditioned | **0.000918694** | 0.275345 | 0.002565173 | 0.527801 |
| capped pot-relative weighting | 0.000966658 | **0.259406** | 0.002576799 | 0.518775 |
| linear belief encoder | 0.000968113 | 0.285170 | 0.002578679 | 0.532264 |
| range statistics | 0.001000758 | 0.274484 | 0.002610398 | **0.517252** |
| low-entropy 3x weighting | 0.001050235 | 0.293673 | 0.002659588 | 0.540006 |

## Findings

- `belief_second_moment` is the best balanced change. It improves hard loss by 1.85% and hard pot-relative RMSE by 1.26% versus baseline.
- The lowest-entropy hard-validation quartile improves from 0.00466123 to 0.00454750 with second moment, a 2.44% reduction. All four entropy quartiles improve.
- Increasing `belief_low_rank_dim` from 128 to 256 is effectively neutral. Low-rank width alone is not the evident bottleneck.
- Simple 3x low-entropy weighting hurts every aggregate metric. The hard states need a better representation or target, not merely more gradient weight.
- Capped pot-relative weighting improves pot-relative RMSE while leaving hard raw loss roughly flat. It is useful only if that metric is the primary objective.
- Range statistics produce the best hard pot-relative RMSE but worsen raw loss. Board-mass, board-conditioned, and linear-encoder variants do not beat second moment.
- The large matched-to-hard gap remains: baseline 0.000919 matched versus 0.002571 hard. Fixed-data architecture changes reduce it only modestly, so target/distribution mismatch remains a leading bottleneck.

## Scope Limits

The pregenerated tensors contain features and 300-CFR targets, not replayable public roots. They therefore cannot produce same-root 5k-CFR labels or test alternate closing `E_turn` models without generating another paired dataset. Those are solver/target experiments rather than fixed-data training ablations.

Artifacts are under `outputs/sturn_pregen_500step_sweep_20260711`; `queue_summary.json` contains all run summaries.
