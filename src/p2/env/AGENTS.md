## Directory summary
Heads-up no-limit Texas Hold 'Em environment logic, tensorized simulation, hand evaluation, card-combo utilities, and analyzer helpers.

### Source files
- `__init__.py`: Package exports.
- `types.py`: Dataclasses for actions, player state, and game state.
- `hunl_env.py`: Scalar Python HUNL environment.
- `hunl_tensor_env.py`: Batched tensorized HUNL environment for high-throughput training.
- `nl_env.py`: Scalar reference multiway no-limit Hold 'Em environment with side-pot rewards and per-seat starting stacks.
- `pbs_env.py`: Batched multi-player public-belief-state environment with no private card deals, side-pot payouts, per-seat starting stacks, and marginal-belief showdown EV helpers.
- `triton_pbs_env.py`: CUDA/Triton kernels and persistent scratch buffers for `PBSEnv` legal masks, stepping, rewards, public dealing, reset, and row materialization.
- `env_gather_triton.py`: Triton row-gather kernels for CUDA environment expansion.
- `rules.py`: PyTorch hand-ranking and comparison utilities.
- `rules_triton.py`: Triton-accelerated hand-ranking wrappers, including fused board+combo score generation, and fallbacks.
- `card_utils.py`: 1326-combo lookup, 169-class preflop mapping/expand/collapse helpers, board masks, suit permutations, canonical flop representatives with orbit weights, blockers, and unblocked-mass helpers.
- `aggression_analyzer.py`: Preflop hand-group mapping and aggression analysis.
- `analyze_tensor_env.py`: Preflop and ReBeL analyzers for tensor environments and model-backed policies.

### Subdirectories
There are no child source directories.
