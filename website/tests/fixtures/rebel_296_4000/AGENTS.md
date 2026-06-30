## Directory summary
Committed split BetterFFN checkpoint fixture and matching WebGPU export.

### Source files
- `checkpoint.pt`: Original PyTorch checkpoint used as the Python reference source.
- `model.json`: WebGPU BetterFFN manifest exported from `checkpoint.pt`, including all-in table metadata.
- `weights.bin.gz`: Gzipped fp16 WebGPU model weights referenced by `model.json`.

### Subdirectories
- `allin/`: Preflop all-in payoff assets used by the WebGPU export and Python sparse CFR reference.
