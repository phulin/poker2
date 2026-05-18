## Directory summary
Heads-up no-limit Texas Hold 'Em environment logic, tensorized simulation, hand evaluation, card-combo utilities, and analyzer helpers.

### Source files
- `__init__.py`: Package exports.
- `types.py`: Dataclasses for actions, player state, and game state.
- `hunl_env.py`: Scalar Python HUNL environment.
- `hunl_tensor_env.py`: Batched tensorized HUNL environment for high-throughput training.
- `env_gather_triton.py`: Triton row-gather kernels for CUDA environment expansion.
- `rules.py`: PyTorch hand-ranking and comparison utilities.
- `rules_triton.py`: Triton-accelerated hand-ranking wrappers and fallbacks.
- `card_utils.py`: 1326-combo lookup, board masks, suit permutations, blockers, and unblocked-mass helpers.
- `aggression_analyzer.py`: Preflop hand-group mapping and aggression analysis.
- `analyze_tensor_env.py`: Preflop and ReBeL analyzers for tensor environments and model-backed policies.

### Subdirectories
There are no child source directories.
