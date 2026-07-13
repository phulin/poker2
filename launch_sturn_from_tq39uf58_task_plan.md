# Task Plan: Launch S_turn from tq39uf58 Settings

## Goal
Start a new 5k-step `S_turn` curriculum run using the settings from W&B run `tq39uf58`, but with the newest distilled `E_turn` checkpoint as the closing checkpoint.

## Phases
- [x] Phase 1: Create isolated launch plan
- [x] Phase 2: Recover `tq39uf58` command/config
- [x] Phase 3: Identify the new distilled `E_turn` checkpoint
- [x] Phase 4: Construct and launch the new run command
- [x] Phase 5: Record run details and report status

## Key Questions
1. What exact overrides did `tq39uf58` use?
2. Which checkpoint is the new distilled `E_turn`?
3. What output/checkpoint directory and W&B name should the new run use?

## Decisions Made
- Use run-specific planning files to avoid overwriting existing root planning notes.
- Use the completed/promoted 300k `E_turn.pt` rather than the interrupted plus50k `rebel_latest.pt`.

## Errors Encountered
- `nvidia-smi` could not access the NVIDIA driver from the sandbox; launch must run outside sandbox for GPU/W&B.
- Failed W&B run `59k2vrnb`: Triton could not find `libcuda.so.1`; relaunched with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`.

## Status
**Complete** - Run is active in tmux session `sturn_eturn300k_v2`; step 0 completed.
