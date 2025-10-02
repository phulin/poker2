# Counterfactual Regret Minimization Solver Plan

## Objectives
- Build a reusable, fully vectorized CFR implementation tailored to the existing `hunl_tensor_env` for heads-up no-limit poker.
- Ensure `GameNode`/game-tree representations operate on batched tensors so traversal and updates run efficiently on GPU (PyTorch-first design).
- Support both vanilla CFR and CFR+ variants with pluggable abstractions for betting trees and state representations.
- Provide integration hooks with existing RL utilities (experience buffers, logging) without coupling tightly to reinforcement learning training loops.

## Background Alignment
- `hunl_tensor_env` already encapsulates hand dealing, betting rounds, rewards, and tensorized observations. The CFR solver will wrap this environment to generate game trajectories and counterfactual values in batched form.
- The solver must respect environment interfaces (reset, step, tensor outputs) while exposing deterministic traversal required by CFR. Expect to extend the environment with helper methods for deterministic action application and vectorized stepping (e.g., synchronizing deck samples across batches).

## High-Level Architecture
1. **Vectorized Game Tree Interface**
   - Define a batched `GameNode` protocol describing players to act, legal actions mask, resulting successor indices, terminal flags, and payoff tensors; all tensors must be shape-consistent for GPU execution.
   - Implement adapters translating `hunl_tensor_env` states into batched `GameNode` tensors; cache tensor observations to avoid redundant env resets and enable gather/scatter operations.
2. **Regret and Strategy Storage**
   - Maintain per-information-set tensors keyed by serialized observation features produced by the environment (e.g., betting history + hole cards abstraction) with support for batched indexing.
   - Support configurable storage backends: start with PyTorch tensors on CPU/GPU; optionally wrap with memory-mapped or sharded stores for large trees while keeping vectorized access semantics.
3. **Traversal Engine**
   - Implement recursive CFR traversal using batched tensor operations (no Python loops over individual nodes) to compute counterfactual reach probabilities and regrets.
   - Provide utilities for stacking parallel traversals (e.g., different deck permutations) and ensure mapping between environment player indices and CFR roles remains vector-friendly.
4. **Variants**
   - Support toggling between vanilla CFR and CFR+ by swapping regret update kernels and strategy averaging routines, both operating on batched tensors.
   - Provide hooks for external sampling CFR and mini-batch updates that exploit the vectorized node representation.
5. **Integration & Tooling**
   - Expose a clean API (`solve(num_iterations, batch_size, device, callback=None)`) returning averaged strategies as tensors.
   - Integrate with existing logging utilities (`training_utils`) for iteration stats, exploitability estimation, and checkpointing, ensuring metrics can be computed from batched tensors.

## Implementation Milestones
1. **Scaffolding (Week 1)**
   - Sketch interfaces: `pokercfr.game_tree`, `pokercfr.regret`, `pokercfr.solver` modules with explicit tensor shapes and device handling.
   - Write adapter to step `hunl_tensor_env` deterministically given an action index and batch dimension; confirm environment can clone/reset states in bulk.
2. **Core CFR Loop (Week 2)**
   - Implement vanilla CFR recursion using batched tensor math for information set probabilities, regret updates, and strategy averaging.
   - Add unit tests using small toy games (e.g., Kuhn poker mock) that validate correctness on CPU and GPU.
3. **Sampling & Performance (Week 3-4)**
   - Introduce chance-sampled CFR leveraging batched sampling of chance events to maintain GPU throughput.
   - Optimize storage (PyTorch tensors with fused kernels) and implement batching of node traversals using `torch.vmap`/custom kernels where beneficial.
4. **Extensions (Week 5+)**
   - CFR+ variant with regret-matching+ and warm-start capabilities, ensuring updates remain vectorized.
   - Optional abstractions for card bucketing and bet sizing using existing transformer configs, keeping batch dimension consistent.
   - Export averaged strategies compatible with RL training loops for benchmarking.

## Dependencies & Open Questions
- Confirm whether `hunl_tensor_env` can produce deterministic, batched transitions (may require new helper to synchronize deck order across batch items). If not, plan to extend the env with a deterministic, vectorized mode.
- Determine serialization format for information sets that preserves game symmetry, supports batching, and keeps dictionary/tensor keys compact.
- Decide how to balance PyTorch tensor usage vs. potential NumPy fallbacks; PyTorch is preferred for GPU execution but may need CPU mirrors for logging.
- Need a clear interface for exploitability calculation—potentially reuse evaluation loops from `train_kbest.py` or create a dedicated evaluator that accepts batched strategies.

## Validation Strategy
- Write unit tests for regret updates and strategy averaging on toy games, verifying both scalar and batched execution paths.
- Add integration test running a small number of CFR iterations on a simplified `hunl_tensor_env` configuration with batch size > 1 to ensure vectorization works.
- Track iteration metrics (exploitability estimate, average strategy stability) via logging; compare against baseline results if available.

## Deliverables
- `pokercfr` package containing modules: `game_adapter.py`, `information_set.py`, `regret_store.py`, `solver.py`, `sampling.py`, each supporting batched tensor operations.
- Documentation: API docstrings, README snippet describing usage with `hunl_tensor_env`, including device management notes.
- Example script demonstrating a short CFR run, configurable batch size, and logging output on GPU.

