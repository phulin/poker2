# 500-Step Value Proposal Results

All runs use the assembled pregenerated river value dataset:
`outputs/rebel_postflop/river_value_500steps_512000_300it_20260630/manifest.json`.

Output root:
`outputs/value_arch_proposals_500step_20260630`.

The runner now accepts multiple proposals in one invocation. For the queued
rerun, it loaded and shuffled the full GPU value epoch once, then reused it for
each proposal. The 500-step GPU epoch is 10.33 GiB and took 15.13s to load for
the shared queue.

Timing is the runner's post-training no-grad value-forward benchmark. Architecture
comparison timings use the post value head explicitly, with
`model.compile=reduce-overhead`, 3 warmup 4096-example batches, and 20 timed
4096-example batches averaged with CUDA events.

| Proposal | Settings | Final validation value loss | Corrected post-only mean 4096 forward | Corrected post-only median | Training mean excl. first |
| --- | --- | ---: | ---: | ---: | ---: |
| `baseline` | overnight baseline, `hidden_dim=384`, `ffn_dim=768`, `num_value_layers=7` | 0.0039288434 | 0.001600s | 0.001587s | 0.028803s |
| `flops_hidden384_ffn768_value6_board128` | `hidden_dim=384`, `ffn_dim=768`, `num_value_layers=6`, `board_interaction_dim=128` | 0.0038509251 | 0.001530s | 0.001524s | 0.026632s |
| `flops_hidden384_ffn768_value6_board128_belief96` | previous + `belief_low_rank_dim=96` | 0.0038332546 | 0.001519s | 0.001497s | 0.028213s |
| `flops_hidden384_ffn768_value6_board96_belief128` | `384/768 value6 board96`, `belief_low_rank_dim=128` | 0.0038462392 | 0.001424s | 0.001413s | 0.025829s |
| `flops_hidden384_ffn640_value6_board128_belief128` | `384/640 value6 board128`, `belief_low_rank_dim=128` | 0.0039461650 | 0.001471s | 0.001465s | 0.026476s |
| `flops_hidden384_ffn896_value6_board128_belief128` | `384/896 value6 board128`, `belief_low_rank_dim=128` | 0.0037827833 | 0.001835s | 0.001818s | 0.026778s |
| `flops_hidden384_ffn768_value6_board128_belief128_handbasis384` | current frontier + `value_hand_basis_rank=384` | 0.0085036309 | 0.001839s | 0.001828s | 0.029322s |
| `flops_hidden384_ffn768_value6_board128_belief128_handbasis512` | current frontier + `value_hand_basis_rank=512` | 0.0085454946 | 0.001887s | 0.001864s | 0.029642s |
| `flops_hidden384_ffn768_value6_board64gated_belief128` | `384/768 value6 board64`, scalar-gated board interaction, `belief_low_rank_dim=128` | 0.0040292552 | 0.001626s | 0.001622s | 0.026796s |
| `flops_hidden384_ffn768_value6_board128gated_belief128` | `384/768 value6 board128`, scalar-gated board interaction, `belief_low_rank_dim=128` | 0.0039253799 | 0.001657s | 0.001651s | 0.026796s |
| `flops_hidden384_ffn768_value6_board128_belief192` | `384/768 value6 board128`, `belief_low_rank_dim=192` | 0.0038506099 | 0.001511s | 0.001496s | 0.027297s |
| `flops_hidden384_ffn768_value6_board128_belief192_skip` | previous + matching-dim belief encoder skip | 0.0038386054 | 0.001580s | 0.001560s | 0.025614s |
| `flops_hidden384_ffn768_value6_board192skip_belief128` | `384/768 value6`, `board_interaction_dim=192`, direct board interaction skip, `belief_low_rank_dim=128` | 0.0045961309 | 0.001605s | 0.001588s | 0.026157s |
| `flops_hidden384_ffn768_value6_board192skip_belief192_skip` | direct board interaction skip + matching-dim belief encoder skip | 0.0042230190 | 0.001386s | 0.001377s | 0.024954s |
| `flops_hidden384_ffn768_value6_board128_belief64_boardcond` | `384/768 value6 board128`, fixed board-conditioned `belief_low_rank_dim=64` | 0.0039640092 | 0.003011s | 0.002998s | 0.034062s |
| `flops_hidden384_ffn768_value6_board128_belief96_boardcond` | `384/768 value6 board128`, fixed board-conditioned `belief_low_rank_dim=96` | 0.0039822406 | 0.003509s | 0.003504s | 0.036730s |
| `flops_hidden384_ffn768_value6_board128_belief128_boardcond` | `384/768 value6 board128`, fixed board-conditioned `belief_low_rank_dim=128` | 0.0040144579 | 0.004008s | 0.003993s | 0.038678s |
| `flops_hidden384_ffn576_value6_board128_belief96_boardcond` | `384/576 value6 board128`, fixed board-conditioned `belief_low_rank_dim=96` | 0.0041183198 | 0.003450s | 0.003444s | 0.036832s |
| `flops_hidden320_ffn640_value6_board128_belief96_boardcond` | `320/640 value6 board128`, fixed board-conditioned `belief_low_rank_dim=96` | 0.0041201025 | 0.003408s | 0.003396s | 0.037188s |
| `flops_hidden384_ffn768_value6_board128_belief96_boardcond_linear` | fixed board-conditioned rank96 + linear belief encoder | 0.0039854188 | 0.003943s | 0.003948s | 0.037276s |
| `flops_hidden384_ffn768_value5_board128` | `hidden_dim=384`, `ffn_dim=768`, `num_value_layers=5`, `board_interaction_dim=128` | 0.0039126018 | 0.001686s | 0.001673s | 0.026317s |
| `flops_hidden384_ffn768_value5_board128_belief96` | previous + `belief_low_rank_dim=96` | 0.0038665688 | 0.001636s | 0.001624s | 0.027077s |
| `flops_hidden384_ffn768_value4_board128` | `hidden_dim=384`, `ffn_dim=768`, `num_value_layers=4`, `board_interaction_dim=128` | 0.0040506543 | 0.001623s | 0.001597s | 0.024395s |
| `flops_hidden320_ffn640_value6_board128` | `hidden_dim=320`, `ffn_dim=640`, `num_value_layers=6`, `board_interaction_dim=128` | 0.0039709246 | 0.001506s | 0.001486s | 0.027739s |
| `flops_hidden320_ffn640_value5_board128` | `hidden_dim=320`, `ffn_dim=640`, `num_value_layers=5`, `board_interaction_dim=128` | 0.0040309760 | 0.001595s | 0.001585s | 0.026619s |
| `flops_hidden320_ffn640_value4_board128` | `hidden_dim=320`, `ffn_dim=640`, `num_value_layers=4`, `board_interaction_dim=128` | 0.0041812216 | 0.001303s | 0.001288s | 0.024435s |
| `flops_hidden256_ffn512_value6_board128` | `hidden_dim=256`, `ffn_dim=512`, `num_value_layers=6`, `board_interaction_dim=128` | 0.0041026263 | 0.001363s | 0.001354s | 0.027960s |
| `flops_hidden512_ffn1024_value6` | `hidden_dim=512`, `ffn_dim=1024`, `num_value_layers=6` | 0.0038314143 | 0.001891s | 0.001882s | 0.027582s |
| `flops_hidden512_ffn1024_value6_board16` | value6 + `board_interaction_dim=16` | 0.0041446253 | 0.001887s | 0.001880s | 0.027504s |
| `flops_hidden512_ffn1024_value6_board128` | value6 + `board_interaction_dim=128` | 0.0037000316 | 0.001918s | 0.001918s | 0.027047s |
| `flops_hidden512_ffn1024_value8` | `hidden_dim=512`, `ffn_dim=1024`, `num_value_layers=8` | 0.0038096527 | 0.002083s | 0.002073s | 0.030703s |
| `flops_belief_in96` | `belief_low_rank_dim=96` | 0.0039158596 | 0.001572s | 0.001488s | 0.029864s |

## Corrected Post-Only Retiming

Retimed final checkpoints with `model.compile=reduce-overhead`, `value_head=post`,
3 warmup 4096-example batches, and 20 timed 4096-example value-forward batches.
Full outputs:
`outputs/value_arch_proposals_500step_20260630/checkpoint_inference_timing_reduce_overhead_post_20.json`
and
`outputs/value_arch_proposals_500step_20260630/checkpoint_inference_timing_reduce_overhead_post_20.md`.

| Proposal | Validation value loss | Mean 4096 forward | Median 4096 forward | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 0.0039288434 | 0.001600s | 0.001587s | 0.001575s | 0.001731s |
| `flops_hidden512_ffn1024_value6` | 0.0038314143 | 0.001891s | 0.001882s | 0.001865s | 0.001948s |
| `flops_hidden512_ffn1024_value8` | 0.0038096527 | 0.002083s | 0.002073s | 0.002058s | 0.002135s |
| `flops_belief_in96` | 0.0039158596 | 0.001572s | 0.001488s | 0.001470s | 0.002765s |

## Prior Auto-Head Timing Note

The earlier 20-sample table used the compiled `auto` value path, which evaluates
both pre and post heads for `BetterStreetValueFFN` before selecting by phase.
Those timings are useful as a diagnostic but should not be used for architecture
comparison on homogeneous river batches.

## Value-Path Ablation Timing

Timing-only ablations monkey-patch checkpointed value-network components before
`reduce-overhead` compilation, then use the same 3 warmup and 20 timed 4096-row
no-grad value forwards. Full outputs:
`outputs/value_arch_proposals_500step_20260630/flops_hidden512_ffn1024_value6_value_forward_ablations.json`
and
`outputs/value_arch_proposals_500step_20260630/baseline_value_forward_ablations.json`.

### `flops_hidden512_ffn1024_value6`

| Ablation | Mean 4096 forward | Median | What remains |
| --- | ---: | ---: | --- |
| `full_auto` | 3.372ms | 3.310ms | Original compiled auto path |
| `post_only` | 2.877ms | 2.868ms | Force one post value head |
| `pre_only` | 2.920ms | 2.892ms | Force one pre value head |
| `base_only` | 1.700ms | 1.680ms | Base/range encoding, no value head |
| `zero_output` | 0.091ms | 0.089ms | Output allocation only |
| `post_tower_only` | 2.758ms | 2.748ms | Base + post residual tower, no final projection |
| `post_final_only` | 1.794ms | 1.789ms | Base + final H->2N projection only |
| `post_head_zero_base` | 1.395ms | 1.379ms | Full post head from zero trunk state |
| `post_final_zero_base` | 0.398ms | 0.388ms | Final projection from zero trunk state |
| `post_no_belief_moments` | 2.837ms | 2.829ms | Post path with belief moments zeroed |
| `post_no_static_context` | 2.480ms | 2.464ms | Post path with static context/board base zeroed |

### `baseline`

| Ablation | Mean 4096 forward | Median | What remains |
| --- | ---: | ---: | --- |
| `full_auto` | 3.570ms | 3.537ms | Original compiled auto path |
| `post_only` | 2.990ms | 2.953ms | Force one post value head |
| `base_only` | 1.690ms | 1.669ms | Base/range encoding, no value head |
| `post_tower_only` | 2.853ms | 2.850ms | Base + post residual tower, no final projection |
| `post_final_only` | 1.824ms | 1.814ms | Base + final H->2N projection only |
| `post_head_zero_base` | 1.548ms | 1.537ms | Full post head from zero trunk state |
| `post_final_zero_base` | 0.373ms | 0.365ms | Final projection from zero trunk state |
| `post_no_belief_moments` | 2.975ms | 2.969ms | Post path with belief moments zeroed |
| `post_no_static_context` | 2.633ms | 2.632ms | Post path with static context/board base zeroed |

## Read

- `flops_hidden512_ffn1024_value6_board128` is now the best validation-loss
  option in this subset on the larger dataset. It beats value6 by about 3.4%
  relative loss and value8 by about 2.9%, while landing between value6 and
  value8 on post-only forward timing.
- `flops_hidden384_ffn768_value6_board128` is the best near-1.5ms candidate so
  far: mean/median `1.530/1.524ms`, with validation loss `0.0038509251`. It is
  faster and lower-loss than the original 384/768 value7 baseline, but it does
  not match the wider 512/1024 board128 loss.
- Adding `belief_low_rank_dim=96` to `384/768 value6 board128` improves both
  quality and median timing in this sample: validation `0.0038332546`, median
  `1.497ms`. This is currently the best candidate at the requested `~1.5ms`
  frontier.
- Increasing unconditioned low-rank belief from 96 to 192 did not help. Rank192
  was about timing-neutral but worse on loss (`0.0038506099`), and the matching
  belief encoder skip removed matrices but was slower and still slightly worse
  than rank96/rank128.
- `board_interaction_dim=96` with rank128 is now the best measured sub-1.5ms
  mean-time point: validation `0.0038462392`, mean/median `1.424/1.413ms`.
  It is a little worse than board128 rank128 on loss, but materially faster.
- Shrinking the FFN to 640 did not help enough: it stayed under 1.5ms but loss
  worsened to `0.0039461650`. Widening to 896 improved loss to `0.0037827833`
  but moved timing to `1.835/1.818ms`, outside the target.
- The hand-basis final value projection is not viable in this form. Rank384 and
  rank512 both landed around `0.0085` validation loss and were slower than the
  dense frontier.
- Scalar-gated board interaction underperformed the plain projected board
  interaction. Both gated variants were slower than board96/board128 and worse
  on validation loss.
- The direct board-interaction skip variants are dominated. The fastest one
  reached `1.386/1.377ms`, but loss rose to `0.0042230190`; the rank128 skip
  variant was worse on both speed and loss. The naive direct residual appears
  too disruptive, and the run emitted an RMSNorm dtype fallback warning on this
  path.
- Fixed the street-value board-conditioned low-rank belief path so it actually
  receives `board_context`, then reran board-conditioned rank 64/96/128 variants.
  They are all dominated by unconditioned rank96: worse loss and much slower
  post-only timing (`~3.0-4.0ms`). The current per-board card-offset mechanism
  is too expensive for the 1.5ms target and did not improve validation loss.
- In this sweep, reducing value depth from 6 to 5 was not a useful speed win
  after compilation. Both 384/768 value5 variants were slower than value6 and
  worse on loss. Value4 only became clearly faster at 320/640, with a much
  larger loss hit.
- The lower-width options define the faster side of the frontier:
  `320/640 value6 board128` gives median `1.486ms` at loss `0.0039709246`,
  `256/512 value6 board128` gives median `1.354ms` at loss `0.0041026263`, and
  `320/640 value4 board128` gives median `1.288ms` at loss `0.0041812216`.
- `flops_hidden512_ffn1024_value6` is still the better speed/quality compromise
  among the 512/1024 width variants: it beats baseline loss by about 2.5%, but
  its post-only forward is about 0.29ms slower than baseline.
- `flops_hidden512_ffn1024_value6_board16` did not help. It is essentially tied
  on post-only timing versus value6 (`1.887ms` vs `1.891ms` mean) but worsens
  validation loss (`0.0041446253` vs `0.0038314143`).
- `board_interaction_dim=128` looks materially different from 16. It adds only
  about `0.027ms` versus value6 in the post-only timing sample (`1.918ms` vs
  `1.891ms`) but improves final validation loss to `0.0037000316`.
- `flops_belief_in96` is now the fastest post-only forward in this subset and is
  slightly better than baseline on validation loss, but the timing sample has a
  visible outlier; median is the more representative number.
- The compiled `auto` path costs about 0.5-0.6ms more than forcing a single
  post head on river data. That points to a direct optimization: use the known
  street/chance phase to avoid compiling both pre and post heads when the batch
  is homogeneous.
- The base/range path is large: about 1.7ms by itself. Zeroing belief moments
  barely helps, but zeroing static context/board base saves 0.36-0.51ms. The
  remaining base cost is mostly hand embedding, belief projection, phase shift,
  and associated tensor movement.
- The value head is also large. From zero trunk state, the full post head costs
  about 1.4-1.55ms, while the final H->2N projection alone costs about
  0.37-0.40ms. The residual tower therefore dominates the value-head compute,
  not just the final wide projection.

## RiverCanonicalValueHead (canonical-strength quantile mixer)

New module `RiverCanonicalValueHead` (see `src/p2/models/mlp/better_ffn.py`).
It builds a board-invariant canonical strength coordinate `u ∈ [0,1]` per hand
(combined-belief mass-midpoint of each rank group), quantile-bins it into `K`
tokens of equal combined mass, computes per-bin per-player features
(`mass`, mean-`u`, belief-weighted equity vs. opponent, and a `K`-dim
row-normalized blocked-mass matrix `B[p,k,:]`), runs a 2-block MLP-mixer over
the `K` tokens with broadcast globals (pot + per-player SPR), and emits per
player `K` nodal values (pot-scaled) that are linearly interpolated back to each
hand at coordinate `u_h`. No 52-card identity ever enters the token features, so
suit isomorphism is exact by construction (verified: residual is invariant to a
board+range suit permutation to 1e-5). The final linear is zero-initialized and
the residual is added into `hand_values` for river rows next to the FiLM
residual, so training starts exactly at the analytic
`river_range_equity_blockers_posneg_r96` baseline and only learns the residual it
misses (k32's step-50 validation `0.00238` matches the baseline's final
`0.00237`).

All runs: same `--steps 500` river value dataset and validation set as above,
`lr=0.04→0.004`, `value_output_init_scale=0.0`, analytic
posneg-blockers-r96 baseline enabled.

| Proposal | Settings | Final validation value loss | Pot-relative RMSE | Inference mean 4096 forward | Training mean excl. first |
| --- | --- | ---: | ---: | ---: | ---: |
| `river_range_equity_blockers_posneg_r96_lr0p04_out0p00` | analytic posneg-blockers-r96 baseline only (no canonical head) | 0.0023664166 | 0.5144 | — | — |
| `river_canonical_k32` | baseline + canonical head `K=32`, `d=64`, 2 layers | 0.0011141200 | 0.4389 | 0.011908s | 0.04352s |
| `river_canonical_k64` | baseline + canonical head `K=64`, `d=64`, 2 layers | 0.0011705970 | 0.4474 | 0.013158s | 0.04595s |
| `river_canonical_k32_no_blocker_rows` | `K=32` head with the `B`-matrix blocker rows dropped | 0.0011110898 | 0.4406 | 0.010597s | 0.04283s |

Findings:
- The canonical head roughly halves the analytic baseline's validation value
  loss (`0.00237 → 0.00111`, −53%). Training is monotone and stable (no NaNs).
- `K=32` slightly beats `K=64` and is cheaper, so 32 bins is the better default.
- Dropping the pairwise blocked-mass rows (the spec's "key move") costs
  essentially nothing on this dataset (`0.0011111` vs `0.0011141`) and is a bit
  faster. On this pregenerated river distribution the cross-range strength mixer
  alone (mass / mean-u / equity tokens) captures almost all of the gain; the
  second-order blocker structure adds negligible marginal value here.
- Cost: the head's per-hand scatter/gather/einsum makes the value forward
  markedly slower than the analytic-only baseline post head (~1.6ms → ~10–13ms
  at 4096 rows). The `no_blocker_rows` variant is the fastest of the three.

## Canonical head as sole river predictor

Follow-up live-arch ablation: on river rows, drop the trunk's per-hand value and
replace it with the canonical head output. Non-river rows still use the trunk.
When the analytic baseline is enabled it remains an exact per-hand additive
output anchor; `value_river_canonical_only` only removes the learned trunk value.

| Config | Train | Val | vs. residual val |
| --- | ---: | ---: | ---: |
| canonical nb -- residual (add to trunk) | 0.000839 | 0.001357 | -- |
| canonical-only nb (trunk dropped on river) | 0.000895 | 0.001373 | +1.2% |
| canonical-only nb K48 | 0.000903 | 0.001399 | +3.1% vs K32 nb |
| canonical-only nb K64 | 0.000931 | 0.001409 | +3.8% vs K32 nb |
| canonical bi -- residual (add to trunk) | 0.000613 | 0.001105 | -- |
| canonical-only bi (trunk dropped on river) | 0.000585 | 0.001097 | -0.7% |
| canonical-only bi K48 | 0.000632 | 0.001113 | +0.7% |
| canonical-only bi K64 | 0.000655 | 0.001105 | ~0.0% |
| canonical-only bi K128 no blocker rows | 0.000680 | 0.001171 | +6.7% |

Conclusion: the trunk's card-aware per-hand value contributes essentially
nothing on river for this setup. With the blocker-corrected analytic baseline as
an output anchor, replacing the learned trunk value with the canonical
rank-space computation is marginally better; without that baseline it costs only
about 1.2%. Both differences are within likely run-to-run noise.

Higher-bin follow-up:
- No-baseline higher-bin runs also failed to beat K32:
  `river_canonical_only_nb_k48` reached validation `0.0013989617`,
  pot-relative RMSE `0.45382`, 4096-row post-forward mean `0.00763s`, and
  training mean excluding first `0.03818s`; `river_canonical_only_nb_k64`
  reached validation `0.0014089439`, pot-relative RMSE `0.45082`, 4096-row
  post-forward mean `0.00809s`, and training mean excluding first `0.03786s`.
- `river_canonical_only_bi_k48` completed at validation `0.0011130023`,
  pot-relative RMSE `0.45054`, 4096-row post-forward mean `0.01294s`, and
  training mean excluding first `0.04100s`. It does not beat K32.
- `river_canonical_only_bi_k64` completed at validation `0.0011045693`,
  pot-relative RMSE `0.44937`, 4096-row post-forward mean `0.04309s`, and
  training mean excluding first `0.07857s`. It does not beat K32.
- `river_canonical_only_bi_k128` with blocker rows OOMed at the current
  1024-example value batch because the blocker row feature scales as `K x K`.
- `river_canonical_only_bi_k128_no_blocker_rows` completed at validation
  `0.0011706989`, pot-relative RMSE `0.45014`, 4096-row post-forward mean
  `0.01455s`, and training mean excluding first `0.04522s`; it is worse than
  K32/K64.
- Result: more bins are not the next lever for this 500-step setup.

## River hand-independent fidelity diagnostic

Added `scripts/diagnose_river_value_fidelity.py` to quantify the representation
ceiling on the validation set without training. It computes an oracle
per-board/per-player canonical strength-bin mean, then asks how much of the
within-bin residual is explained by hand-independent per-hand scalars. Full
outputs:
`task_notes/river_hand_independent_fidelity/river_value_fidelity_val8192.json`
and
`task_notes/river_hand_independent_fidelity/river_value_fidelity_val8192_noblock.json`.

Validation-set target scale:

| Metric | Value |
| --- | ---: |
| Weighted target variance MSE | 0.028490 |
| Weighted zero-prediction MSE | 0.028608 |

Blocker-corrected scalar diagnostic (`showdown_rank_bins=96`):

| K | Strength-bin oracle MSE | + posneg baseline scalar MSE | Baseline explains within-bin |
| ---: | ---: | ---: | ---: |
| 16 | 0.0005368 | 0.0002166 | 59.6% |
| 32 | 0.0004083 | 0.0001612 | 60.5% |
| 64 | 0.0003689 | 0.0001449 | 60.7% |
| 128 | 0.0003507 | 0.0001374 | 60.8% |

No-blocker scalar ablation:

| K | Strength-bin oracle MSE | + posneg baseline scalar MSE | Baseline explains within-bin |
| ---: | ---: | ---: | ---: |
| 32 | 0.0004083 | 0.0003906 | 4.3% |
| 128 | 0.0003507 | 0.0003502 | 0.1% |

Read:
- More bins help, but only modestly once the exact per-hand baseline is present
  (`K=32 -> 128` improves the baseline-augmented oracle from `0.000161` to
  `0.000137`, about 15% of the remaining oracle residual).
- The useful per-hand fidelity signal is overwhelmingly blocker correction.
  Without blocker correction, per-hand showdown/equity scalars explain almost
  none of the within-bin residual at high K.
- A generic per-hand scalar MLP is not the first thing to try. In this linear
  oracle, all scalar features barely improve over the single posneg baseline
  scalar (`K=32`: `0.0001605` vs `0.0001612`).
- Since `river_canonical_only_bi` already uses the exact per-hand analytic
  baseline as an output anchor, the larger gap between its 500-step validation
  (`~0.00110`) and the oracle ceiling (`~0.00016`) is more likely
  optimization/training horizon/canonical-head capacity than missing per-hand
  scalar information. The next clean experiments are a longer
  `river_canonical_only_bi` convergence run and, secondarily, `K=64/128`
  canonical-only-bi checks.
