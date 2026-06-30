## Directory summary
Python utilities used by the WebGPU CFR subproject to export PyTorch checkpoints and produce reference fixtures.

### Source files
- `export_model.py`: Exports PyTorch BetterFFN checkpoints to WebGPU artifacts.
- `extract_bench_spots.py`: Extracts benchmark spots for the WebGPU harness.
- `precompute_allin_assets.py`: Generates WebGPU all-in payoff assets.
- `reference.py`: Builds Python CFR fixtures/results for parity checks.
- `split_cfr_reference.py`: Runs the Python sparse CFR loop for a split BetterFFN checkpoint and emits root policy/action probabilities for realistic WebGPU parity tests.
- `split_reference.py`: Loads split BetterPolicyFFN/BetterStreetValueFFN checkpoints and emits selected PyTorch policy/value outputs for exported-model parity tests.

### Subdirectories
There are no child source directories.
