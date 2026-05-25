## Directory summary
Committed fixture for the `checkpoints-rebel/rebel_296_4000.pt` split BetterFFN checkpoint and its matching WebGPU export.

### Source files
- `checkpoint.pt`: Original PyTorch checkpoint used as the Python reference source.
- `model.json`: WebGPU BetterFFN manifest exported from `checkpoint.pt`, including all-in table metadata.
- `weights.bin.gz`: Gzipped fp16 WebGPU model weights referenced by `model.json`.

### Subdirectories
- `allin/`: Preflop all-in payoff asset referenced by the export manifest.
