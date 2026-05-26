## Directory summary
Python utilities used by the WebGPU CFR subproject to export PyTorch checkpoints and produce reference fixtures.

### Source files
- `export_model.py`: Exports PyTorch BetterFFN checkpoints to `model.json` and compressed fp16 `weights.bin.gz`, including split BetterFFN checkpoint normalization that deploys the value model base with `pre_value_head`, omits the training-only `post_value_head`, and retains prefixed policy tensors for sparse-policy initialization.
- `precompute_allin_assets.py`: Generates WebGPU all-in payoff assets, including canonical flop lookup files, combo suit permutations, int16 canonical flop table shards, and an embeddable `allin_manifest.json`.
- `reference.py`: Builds Python CFR fixtures/results from a checkpoint and action-bin sequence for parity checks, including compatibility loading for older direct-output BetterFFN heads.
- `split_cfr_reference.py`: Runs the Python sparse CFR loop for a split BetterFFN checkpoint and emits root policy/action probabilities for realistic WebGPU parity tests.
- `split_reference.py`: Loads split BetterPolicyFFN/BetterStreetValueFFN checkpoints and emits selected PyTorch policy/value outputs for exported-model parity tests.

### Subdirectories
There are no child source directories.
