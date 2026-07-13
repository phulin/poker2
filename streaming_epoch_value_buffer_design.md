# Streaming Epoch Value Buffer

## Behavior

`StreamingEpochValueBuffer` owns two fixed tensor blocks. Training reads a sealed block through a random permutation without replacement. At each epoch boundary it reshuffles; after the configured number of epochs it swaps to the full write block and reuses the old read storage for generation.

The trainer fills the complete first block before calling `prepare_step` or performing supervision. During steady-state training, live CFR generation fills the next block while the current block is consumed.

Whole CFR exports can overshoot a requested logical generation count. The buffer retains this bounded tail in memory and drains it into the next write block after swapping. Pending tails are included in replay checkpoints.

## Rate Balance

Live generation uses the exact rational rate:

```text
batch_size * episodes_per_step / value_replay_epochs
```

The source emits floor-difference counts, avoiding long-run rounding drift. Configuration validation requires:

```text
(value_replay_epochs * value_epoch_block_batches) % episodes_per_step == 0
```

For the S-turn/S-flop defaults, batch 4096, five updates per outer step, three epochs, and 20 batches per block produce an 81,920-example block. It is consumed as 60 minibatches over 12 outer steps while exactly one replacement block is generated.

## Configuration

```yaml
train_overrides:
  value_replay_mode: streaming_epoch
  value_replay_epochs: 3
  value_epoch_block_batches: 20
```

Random replay remains the global default and policy replay is unchanged. Streaming epoch replay is enabled for the S-turn and S-flop curriculum stages only. It supports live and hybrid data modes; bootstrap and pregenerated modes retain their existing paths.

## Metrics

- `value_epoch`
- `value_epoch_cursor_fraction`
- `value_epoch_write_fraction`
- `value_epoch_ready`

## Verification

Focused tests cover initial sealing, exact epoch coverage, block swaps, generation overflow, replay checkpoint restoration, rational generation rates, trainer construction, and a complete live training step.
