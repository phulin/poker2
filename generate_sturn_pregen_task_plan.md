# Task Plan: Generate S_turn Pregenerated Dataset

## Goal
Generate about 500 training steps worth of `S_turn` random-turn solved value examples using the 300k `E_turn` closing checkpoint and 300 CFR iterations.

- [x] Phase 1: Confirm checkpoint and training settings
- [x] Phase 2: Launch pregeneration
- [x] Phase 3: Monitor initial progress and record command/log path
- [x] Phase 4: Validate manifest once complete

## Key Questions
1. Use 300k `E_turn`, not continuation run? Yes.
2. How many examples is 500 steps worth? `500 * 2048 = 1,024,000` value examples.
3. Policy examples? Set to `0` for value-focused `S_turn` experiments.

## Decisions Made
- Output directory: `outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711`
- Log file: `outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711.log`
- Use `config_rebel_pregenerate_postflop` with overrides matching the completed `S_turn` run's turn/root/search/model settings.

## Errors Encountered
- Initial `nohup` launch inside the sandbox was killed when the wrapper exited; relaunched via detached `tmux`.
- Hydra struct config required `++` for `seed` and newer model/search fields.
- Triton could not find `libcuda.so.1` until `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu` was set.
- The first long run filled disk after `shard_001971.pt`; the partial `shard_001972.pt` was moved to `shard_001972.pt.corrupt`.
- After disk was cleared, a 28-shard top-up run completed and its shards were hardlinked into `shard_001972.pt` through `shard_001999.pt`.

## Status
**Complete** - Final manifest written at `outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711/manifest.json`.

Validated with `RebelSolvedDataset`: `1,024,000` value examples, `0` policy examples, `2,000` value shards. Loaded batches at the start, stitched boundary near `shard_001972.pt`, and final shard; all checked batches had finite `(batch, 2, 1326)` value targets.
