## Directory summary
Python utilities used by the WebGPU CFR subproject to export PyTorch checkpoints and produce reference fixtures.

### Source files
- `export_model.py`: Exports PyTorch BetterFFN checkpoints to `model.json` and `weights.bin`, including split BetterFFN checkpoint normalization that deploys the value model base with `pre_value_head`, omits the training-only `post_value_head`, and retains prefixed policy tensors for sparse-policy initialization.
- `reference.py`: Builds Python CFR fixtures/results from a checkpoint and action-bin sequence for parity checks, including compatibility loading for older direct-output BetterFFN heads.

### Subdirectories
There are no child source directories.
