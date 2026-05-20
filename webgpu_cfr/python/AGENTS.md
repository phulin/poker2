## Directory summary
Python utilities used by the WebGPU CFR subproject to export PyTorch checkpoints and produce reference fixtures.

### Source files
- `export_model.py`: Exports a PyTorch BetterFFN checkpoint to `model.json` and `weights.bin`, including scheduled CFR search defaults from checkpoint config.
- `reference.py`: Builds Python CFR fixtures/results from a checkpoint and action-bin sequence for parity checks, including compatibility loading for older direct-output BetterFFN heads.

### Subdirectories
There are no child source directories.
