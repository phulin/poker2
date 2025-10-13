# ReBeL CFR Evaluator Design Document

## Overview

This document describes the design for a new `RebelCFREvaluator` class that implements the ReBeL (Recursive Belief-based Learning) algorithm for imperfect-information games, specifically designed for heads-up no-limit Texas hold'em poker. The class will enable efficient search procedures that precisely match the algorithm specification in the ReBeL paper, with particular attention to strategy-conditioned belief updates and proper leaf value recomputation.

## Core Architecture

### Environment Management
The `RebelCFREvaluator` maintains a single `HUNLTensorEnv` instance for the current subgame being solved. This approach provides:
- **Memory efficiency**: Single environment per subgame rather than per depth level
- **Simplicity**: Easier state management and belief propagation
- **Performance**: Reduced memory overhead for subgame construction

### Key Data Structures
- `env: HUNLTensorEnv` - Environment for current subgame
- `current_pbs: torch.Tensor` - Current public belief state
- `policy_history: List[torch.Tensor]` - CFR policy iterations for belief updates
- `avg_policy: torch.Tensor` - Running average policy
- `value_history: List[torch.Tensor]` - Expected values per iteration
- `sampled_iteration: int` - Randomly sampled iteration for safe search

## Method Specifications

### 1. `initialize_search(src_env: HUNLTensorEnv, src_indices: torch.Tensor) -> None`

**Purpose**: Set up the subgame for the current public belief state.

**Algorithm Correspondence**: Corresponds to `ConstructSubgame(β_r)` in the SELFPLAY function.

**Behavior**:
- Copies the current environment state to create the subgame
- Initializes the subgame environment with the root public belief state
- Sets up data structures for policy and value tracking

**Parameters**:
- `src_env`: Source environment containing current game state
- `src_indices`: Indices of states to use as subgame roots

**Implementation Notes**:
- Creates a depth-limited subgame from the current PBS
- Initializes all necessary data structures for CFR search

### 2. `run_cfr_iterations(num_iterations: int, value_network: Callable, policy_network: Optional[Callable] = None) -> CFRResult`

**Purpose**: Execute the main ReBeL CFR search algorithm with leaf value recomputation.

**Algorithm Correspondence**: Implements the core CFR loop from the SELFPLAY function.

**Behavior**:
- Initializes policies (average π̄ and warm-start π^{t_warm})
- Sets initial leaf values using both average and warm-start policies
- Computes initial expected value v(β_r) using warm-start policy
- Samples a random iteration t_sample for safe search
- For each iteration t from t_warm+1 to T:
  - If t == t_sample, samples leaf PBS(s) using π^{t-1}
  - Updates policy π^t using CFR regret minimization
  - Updates average policy π̄ = (t/(t+1))π̄ + (1/(t+1))π^t
  - Sets leaf values using both π̄ and π^t
  - Computes new expected value and updates running average
  - Adds {β_r, v(β_r)} to value network training data
  - Adds {β, π̄(β)} to policy network training data for all β in subgame

**Parameters**:
- `num_iterations`: Number of CFR iterations to perform (T)
- `value_network`: Function that computes infostate values given belief states
- `policy_network`: Optional policy network for warm-start initialization

**Returns**:
- `CFRResult` containing final policies, values, and next PBS for continued search

**Implementation Notes**:
- Uses both average and current policies for leaf value setting
- Maintains running averages for both policy and value
- Implements safe search by sampling random iteration for leaf PBS selection

### 3. `set_leaf_values(avg_policy: torch.Tensor, current_policy: torch.Tensor, value_network: Callable) -> None`

**Purpose**: Set leaf node values using strategy-conditioned belief states from both policies.

**Algorithm Correspondence**: Implements `SetLeafValues(G, π̄, π^t, θ^v)` from the algorithm.

**Behavior**:
- Computes belief states conditioned on average policy π̄
- Computes belief states conditioned on current policy π^t
- Uses value network to compute leaf values for both belief state sets
- Sets leaf node values in the subgame environment

**Parameters**:
- `avg_policy`: Average policy π̄ for belief state computation
- `current_policy`: Current policy π^t for belief state computation
- `value_network`: Network for computing infostate values

**Implementation Notes**:
- Critical for strategy-conditioned belief updates
- Must handle both player perspectives correctly
- Integrates with CFR algorithm's leaf value callback mechanism

### 4. `sample_leaf_pbs(policy: torch.Tensor) -> torch.Tensor`

**Purpose**: Sample leaf public belief state(s) using the given policy.

**Algorithm Correspondence**: Implements `SampleLeaf(G, π^{t-1})` from the algorithm.

**Behavior**:
- Traverses the subgame using the provided policy
- Samples a path through the subgame to reach a leaf node
- Returns the public belief state at the sampled leaf
- May return multiple leaf PBSs for parallel search continuation

**Parameters**:
- `policy`: Policy tensor for traversal

**Returns**:
- Tensor containing the sampled leaf public belief state(s)

**Implementation Notes**:
- Uses the policy to weight the probability of different paths
- Essential for continuing search from promising subgames

### 5. `compute_expected_value(policy: torch.Tensor, value_network: Callable) -> torch.Tensor`

**Purpose**: Compute expected value of the current subgame using the given policy.

**Algorithm Correspondence**: Implements `ComputeEV(G, π)` from the algorithm.

**Behavior**:
- Evaluates the current subgame using the provided policy
- Computes the expected value from the root public belief state
- Uses value network for leaf node evaluation

**Parameters**:
- `policy`: Policy for subgame evaluation
- `value_network`: Network for leaf value computation

**Returns**:
- Expected value tensor for the current subgame

**Implementation Notes**:
- Bottom-up computation through the subgame tree
- Integrates with belief state propagation

### 6. `extract_training_data() -> Tuple[Dict, Dict]`

**Purpose**: Extract training data from completed CFR iterations.

**Algorithm Correspondence**: Implements data collection from the CFR loop.

**Behavior**:
- Collects all {β_r, v(β_r)} pairs for value network training
- Collects all {β, π̄(β)} pairs for policy network training
- Formats data appropriately for the training pipelines

**Returns**:
- Tuple of (value_training_data, policy_training_data) dictionaries

**Implementation Notes**:
- Maintains proper weighting for different iterations
- Integrates with existing replay buffer systems

### 7. `get_safe_search_policy() -> torch.Tensor`

**Purpose**: Get the policy for safe search using randomly sampled iteration.

**Algorithm Correspondence**: Implements safe search mechanism from Section 6.

**Behavior**:
- Returns the policy from the randomly sampled iteration t_sample
- Ensures the policy is a Nash equilibrium in expectation
- Used for action selection during test-time play

**Returns**:
- Policy tensor from the sampled iteration

**Implementation Notes**:
- Critical for theoretical correctness at test time
- Must use random iteration rather than final iteration

### Training Integration
Provides data in format compatible with existing `RebelSupervisedLoss` and `RebelReplayBuffer`.

## Performance Considerations

### Memory Management
- Single environment per subgame reduces memory overhead
- Efficient caching of policy and value histories
- Minimal tensor allocations with reuse across iterations

### Computational Efficiency
- Subgame-based approach reduces computation per iteration
- Vectorized CFR operations within each subgame
- Batched value network calls for leaf evaluation
- Running averages avoid full recomputation

### Scalability
- Depth-limited subgames enable scaling to large game trees
- Belief-conditioned leaf values improve value function learning
- Safe search mechanism ensures theoretical correctness at scale

## Algorithm Fidelity

This design precisely implements the ReBeL SELFPLAY algorithm as specified in the paper:

1. **Subgame Construction**: Creates depth-limited subgames from current PBS
2. **Dual Policy Tracking**: Maintains both average (π̄) and current (π^t) policies
3. **Strategy-Conditioned Updates**: Uses both policies for belief state computation and leaf value setting
4. **Safe Search**: Samples random iterations for test-time action selection
5. **Training Data Collection**: Collects {β, v(β)} and {β, π̄(β)} pairs during search
6. **Recursive Search**: Continues search from sampled leaf PBSs

This ensures theoretical convergence to Nash equilibrium while providing practical efficiency for large-scale imperfect-information games.
