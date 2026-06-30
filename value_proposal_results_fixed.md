# Fixed-Config Value Proposal Results

All runs use the requested pregenerated dataset:
`outputs/rebel_postflop/river_value_100steps_102400_300it_20260630/manifest.json`.

These reruns use the fixed Hydra-composed overnight baseline:
`hidden_dim=384`, `ffn_dim=768`, `range_hidden_dim=192`,
`num_hidden_layers=0`, `num_value_layers=7` unless a preset explicitly changes
one of those fields.

Output root:
`outputs/value_arch_proposals_fixed_100step_20260630`.

Timing note: new runs write `summary.json` `step_timing` as a post-training
no-grad value-forward benchmark: one 4096-example warmup/compile batch followed
by two timed 4096-example batches. Older rows below used training step wall time.

| Proposal | Status | Final validation value loss | Timing |
| --- | --- | ---: | ---: |
| `baseline` | done | 0.0082654375 | 142.24s; mean step excl. first 1.330s |
| `flops_thin_value_tower` | done | 0.0118183662 | 134.03s; mean step excl. first 1.200s |
| `flops_value_layers5` | done | 0.0084605039 | 127.45s; mean step excl. first 1.203s |
| `flops_value_lr192` | done | 0.0089381545 | 126.59s; mean step excl. first 1.187s |
| `flops_value_lr256` | done | 0.0086728144 | 136.83s; mean step excl. first 1.292s |
| `flops_value_lr256_bet_film` | done | 0.0088067464 | 148.32s; mean step excl. first 1.403s |
| `bet_strat_film` | done | 0.0087088210 | 140.67s; mean step excl. first 1.331s |
| `flops_hidden512` | done | 0.0082405435 | 129.21s; mean step excl. first 1.216s |
| `flops_hidden320_ffn640` | done | 0.0083703313 | 127.30s; mean step excl. first 1.189s |
| `flops_hidden256_ffn512` | done | 0.0084257467 | 131.47s; mean step excl. first 1.243s |
| `flops_hidden256_ffn512_value4` | done | 0.0091969430 | 150.27s; mean step excl. first 1.420s |

## Fixed Checkpoint Inference Timing

Timing uses one 4096-example no-grad warmup/compile batch, then two timed
4096-example value-forward batches averaged with CUDA events. Full outputs:
`outputs/value_arch_proposals_fixed_100step_20260630/checkpoint_inference_timing.json`
and `outputs/value_arch_proposals_fixed_100step_20260630/checkpoint_inference_timing.md`.

| Proposal | Validation loss | Mean 4096 forward |
| --- | ---: | ---: |
| `baseline` | 0.0082654375 | 0.003696s |
| `bet_strat_film` | 0.0087088210 | 0.030474s |
| `flops_hidden256_ffn512` | 0.0084257467 | 0.003406s |
| `flops_hidden256_ffn512_value4` | 0.0091969430 | 0.003014s |
| `flops_hidden320_ffn640` | 0.0083703313 | 0.003543s |
| `flops_hidden512` | 0.0082405435 | 0.003596s |
| `flops_thin_value_tower` | 0.0118183662 | 0.002572s |
| `flops_value_layers5` | 0.0084605039 | 0.003168s |
| `flops_value_lr192` | 0.0089381545 | 0.003564s |
| `flops_value_lr256` | 0.0086728144 | 0.003637s |
| `flops_value_lr256_bet_film` | 0.0088067464 | 0.030342s |

## Width Follow-Up

Output root:
`outputs/value_arch_proposals_width_followup_20260630`.

| Proposal | Settings | Final validation value loss | Mean 4096 forward | Training mean excl. first |
| --- | --- | ---: | ---: | ---: |
| `flops_hidden512_ffn1024` | `hidden_dim=512`, `ffn_dim=1024`, no low-rank beliefs | 0.0081931020 | 0.003625s | 0.029169s |

Read:

- `flops_hidden512_ffn1024` improved loss versus fixed baseline
  (`0.0082654375`) and the earlier `flops_hidden512` (`0.0082405435`).
- Its recorded 4096-row forward time (`3.625ms`) is effectively tied with
  `flops_hidden512` (`3.596ms`) and a little faster than fixed baseline
  (`3.696ms`) in these two-batch timing samples.

## Hidden512/FFN1024 Value-Depth Sweep

Output root:
`outputs/value_arch_proposals_width_depth_20260630`.

All rows use `hidden_dim=512`, `ffn_dim=1024`, and no low-rank belief settings.
The `value7` reference is the prior `flops_hidden512_ffn1024` run from
`outputs/value_arch_proposals_width_followup_20260630`.

| Value layers | Proposal | Final validation value loss | Mean 4096 forward | Training mean excl. first |
| ---: | --- | ---: | ---: | ---: |
| 2 | `flops_hidden512_ffn1024_value2` | 0.0098696799 | 0.002699s | 0.021889s |
| 3 | `flops_hidden512_ffn1024_value3` | 0.0088662005 | 0.002923s | 0.021576s |
| 4 | `flops_hidden512_ffn1024_value4` | 0.0085050846 | 0.003023s | 0.023134s |
| 5 | `flops_hidden512_ffn1024_value5` | 0.0082535312 | 0.003324s | 0.024792s |
| 6 | `flops_hidden512_ffn1024_value6` | 0.0081680202 | 0.003574s | 0.027203s |
| 7 | `flops_hidden512_ffn1024` | 0.0081931020 | 0.003625s | 0.029169s |
| 8 | `flops_hidden512_ffn1024_value8` | 0.0079966874 | 0.003749s | 0.031246s |

Read:

- `value8` is the best loss seen in this local 100-step set so far:
  `0.0079966874`, beating fixed baseline (`0.0082654375`) and the previous
  best low-rank belief result (`0.0081398193`).
- `value6` looks like the best speed/quality tradeoff in this sweep:
  `0.0081680202` at `3.574ms`, slightly better loss than `value7` and a little
  faster in the two-batch timing sample.
- `value2` through `value4` are not competitive on loss despite the speed gain.
