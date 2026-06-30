# Input/Output FLOP Proposal Results

All runs use the fixed Hydra-composed overnight baseline and the requested
pregenerated river value dataset:
`outputs/rebel_postflop/river_value_100steps_102400_300it_20260630/manifest.json`.

Output root:
`outputs/value_arch_proposals_io_flops_20260630`.

Timing is the runner's post-training no-grad value-forward benchmark: one
4096-example warmup/compile batch followed by two timed 4096-example batches
averaged with CUDA events.

| Proposal | Settings | Final validation value loss | Mean 4096 forward | Training mean excl. first |
| --- | --- | ---: | ---: | ---: |
| `flops_belief_in128` | `belief_low_rank_dim=128` | 0.0081448591 | 0.003628s | 0.029543s |
| `flops_belief_in128_board` | `belief_low_rank_dim=128`, board-conditioned | 0.0081578707 | 0.003667s | 0.029548s |
| `flops_belief_in96` | `belief_low_rank_dim=96` | 0.0081398193 | 0.003919s | 0.029352s |
| `flops_belief_in96_board` | `belief_low_rank_dim=96`, board-conditioned | 0.0082854940 | 0.003564s | 0.030195s |
| `flops_value_lr256_residual` | `value_head_rank=256`, per-hand residual | 0.0093405668 | 0.007882s | 0.132018s |
| `flops_board_basis_r256` | `value_hand_basis_rank=256`, board-conditioned hand embedding dim 8 | 0.0161042103 | 0.085679s | 0.155656s |

## Card-Feature Board-Conditioned Rerun

After the initial board-conditioned belief runs, the board-conditioned low-rank
belief path was changed to generate each per-card offset from explicit
`[board_context, card_id, card_rank, card_suit]` features. Rerun output root:
`outputs/value_arch_proposals_io_flops_cardcond_20260630`.

| Proposal | Settings | Final validation value loss | Mean 4096 forward | Training mean excl. first |
| --- | --- | ---: | ---: | ---: |
| `flops_belief_in128_board` | `belief_low_rank_dim=128`, board + card id/rank/suit | 0.0082035647 | 0.004141s | 0.029737s |
| `flops_belief_in96_board` | `belief_low_rank_dim=96`, board + card id/rank/suit | 0.0082742298 | 0.003736s | 0.029152s |

## Read

- `flops_belief_in128` is the best result in this batch: it improves loss
  versus the previous fixed baseline (`0.0082654375`) with similar forward
  timing.
- Direct board-conditioned low-rank belief offsets did not help at 100 steps:
  rank 128 is slightly worse than unconditioned, and rank 96 board-conditioned
  is worse than both unconditioned belief ranks.
- Adding explicit card id/rank/suit to the board-conditioned low-rank offset did
  not improve the result. Rank 128 became slower and worse (`0.0082035647`,
  `4.141ms`); rank 96 remained worse than unconditioned rank 96.
- Low-rank output plus residual did not rescue the low-rank output head. It is
  slower than baseline and worse loss.
- Board-conditioned hand-basis value rank 256 is not viable in this form. It
  starts unstable, remains high loss, and is much slower in forward timing.

## Second-Moment Low-Rank Belief Rerun

Second-moment low-rank belief runs used output root:
`outputs/value_arch_proposals_second_moment_io_20260630`.

| Proposal | Settings | Final validation value loss | Mean 4096 forward | Training mean excl. first |
| --- | --- | ---: | ---: | ---: |
| `flops_belief_in128_second` | `belief_low_rank_dim=128`, second moment | 0.0083347373 | 0.003836s | 0.029483s |
| `flops_belief_in128_board_second` | `belief_low_rank_dim=128`, board + card id/rank/suit, second moment | 0.0082211304 | 0.003703s | 0.030331s |
| `flops_belief_in96_second` | `belief_low_rank_dim=96`, second moment | 0.0083160662 | 0.003704s | 0.031644s |
| `flops_belief_in96_second_skip_encoder` | `belief_low_rank_dim=96`, second moment, skip matching belief encoder | 0.0083800587 | 0.003825s | 0.028486s |
| `flops_belief_in96_board_second` | `belief_low_rank_dim=96`, board + card id/rank/suit, second moment | 0.0081640049 | 0.003796s | 0.030654s |

Read:

- Second moment did not beat the first-moment low-rank belief variants at 100
  steps. The best second-moment result was `flops_belief_in96_board_second`
  (`0.0081640049`), still behind `flops_belief_in128` (`0.0081448591`) and
  `flops_belief_in96` (`0.0081398193`).
- Board conditioning helped within the second-moment group, unlike the
  first-moment card-feature rerun, but the net result still was not better than
  the simpler unconditioned first-moment models.
- `belief_skip_matching_encoder` removed the square belief FFN matrices for the
  rank-96 second-moment shape, but it worsened loss and did not improve the
  recorded no-grad forward timing.
