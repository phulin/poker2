# Notes: Streaming Epoch Value Buffer

## Current Replay
- `K_value = round(batch_size / value_reuse_goal)` controls fresh generation.
- Value capacity is `replay_buffer_batches * K_value`.
- Uniform value sampling uses replacement, so reuse is only an expectation.
- Policy and value buffers are both currently owned by buffered data sources.

## Target Behavior
- Two bounded blocks: sealed read block and active write block.
- Exact `value_epochs` shuffled passes over each sealed read block.
- Generate `batch_size / value_epochs` fresh examples per optimizer update.
- Fill the first complete block before any training update.
- Swap when the read block completes and the write block is full.

## Implemented Geometry
- `block_capacity = batch_size * value_epoch_block_batches`.
- Fresh generation rate is the exact rational `batch_size * episodes_per_step / value_replay_epochs`.
- Default street geometry is batch 4096, 5 updates/step, 3 epochs, 20 batches/block.
- One block therefore consumes 60 minibatches over 12 outer steps and generates exactly one replacement block over those 12 steps.
- Whole CFR export batches can overshoot requested generation counts; bounded tails are retained in memory and drained after block swaps.
