# DeepStack Integration Plan

## Overview
This plan outlines the complete transition from PPO-based training to DeepStack-style hybrid CFR using existing poker2 infrastructure. The approach leverages the vectorized game state engine (`hunl_tensor_env.py`) and training infrastructure (`self_play.py`) while replacing PPO policy optimization with CFR continual re-solving.

## Current Infrastructure Analysis

### ✅ **Assets to Leverage**
- **Vectorized Environment** (`hunl_tensor_env.py`): Perfect for batched CFR operations
- **Training Loop** (`self_play.py`): Can be adapted for CFR-based updates
- **Transformer Model** (`poker_transformer.py`): Can serve as value function approximator
- **Opponent Pool System**: Can store CFR-based agent snapshots
- **Existing CFR Foundation** (`src/pokercfr/`): Already has vectorized CFR implementation

### ⚠️ **Components to Adapt**
- **Value Head**: Need counterfactual value head (1000 hand buckets)
- **Training Updates**: Replace PPO loss with CFR strategy computation
- **State Tracking**: Add range and counterfactual value tracking
- **Decision Making**: Replace PPO sampling with CFR strategy

## Phase 1: Foundation Setup (Week 1-2)

### 1.1 Counterfactual Value Head Implementation
**Files to modify:**
- `src/alphaholdem/models/transformer/heads.py`
- `src/alphaholdem/models/transformer/poker_transformer.py`
- `src/alphaholdem/core/structured_config.py`

**Changes:**
- Add `TransformerCounterfactualValueHead` class with 1326 hand combinations (all possible two-card hands)
- Implement zero-sum constraint layer (as in DeepStack paper)
- Add `counterfactual` to `ValueHeadType` enum
- Remove `num_hand_buckets` parameter (use all 1326 hands)
- Update transformer forward pass to handle counterfactual values

**Note:** Using all 1326 possible two-card combinations instead of abstracted buckets for maximum precision.

**Configuration:**
```yaml
model:
  value_head_type: "counterfactual"
```

### 1.2 Range Tracking System
**Files to modify:**
- `src/alphaholdem/rl/self_play.py`

**Changes:**
- Create `RangeTracker` class to track self ranges for each environment (1326 hand combinations)
- Integrate range tracker into `SelfPlayTrainer`
- Add range reset functionality for new games
- Update ranges whenever new policy logits are computed
- Implement range update logic for actions and chance events (TODO for later)

**Key Methods:**
- `update_from_logits()`: Update ranges based on policy logits from current model
- `update_after_action()`: Update ranges using Bayes' rule (TODO)
- `update_after_chance()`: Filter impossible hands after cards dealt (TODO)
- `reset_for_new_game()`: Initialize uniform range over all 1326 hands

## Phase 2: CFR Integration (Week 3-4)

### 2.1 Extend Existing CFR Solver
**Files to modify:**
- `src/pokercfr/solver.py`
- `src/pokercfr/game_adapter.py`

**Changes:**
- Add sparse lookahead trees (fold, call, ½pot, pot, all-in)
- Implement depth-limited traversal (4 actions as in DeepStack)
- Add integration with tensor environment for vectorized operations
- Implement CFR-D gadget for re-solving (as described in DeepStack)

**Action Sets:**
```python
SPARSE_ACTIONS = [0, 1, 2, 3, 7]  # fold, call, 0.5x pot, 1x pot, all-in
```

### 2.2 Continual Re-solving Engine
**Files to create:**
- `src/alphaholdem/rl/continual_resolver.py`

**Components:**
- `ContinualResolver` class using existing CFR solver
- Integration with `StateTracker` for range management
- Integration with transformer for value function estimates
- Caching system for frequently-seen pre-flop situations

**Algorithm Flow:**
1. Extract current public state from tensor environment
2. Run depth-limited CFR with current ranges
3. Sample action from computed strategy
4. Update state tracker with new ranges/counterfactual values

## Phase 3: DeepStack Agent (Week 5-6)

### 3.1 DeepStack Agent Class
**Files to create:**
- `src/alphaholdem/rl/deepstack_agent.py`

**Components:**
- `DeepStackAgent` wrapping transformer model + continual resolver
- Integration with existing training infrastructure
- Decision-making using CFR instead of PPO policy sampling

**Key Methods:**
- `act(observation)`: Use continual resolver for decisions
- `get_counterfactual_values()`: Extract values for different hand buckets

### 3.2 Training Loop Adaptation
**Files to modify:**
- `src/alphaholdem/rl/self_play.py`

**Changes:**
- Replace PPO policy sampling with CFR strategy computation
- Adapt opponent pool to store CFR-based snapshots
- Modify loss computation to work with CFR updates
- Keep existing evaluation and checkpointing infrastructure

## Phase 4: Value Network Training (Week 7-8)

### 4.1 Training Data Generation
**Approach:**
- Generate 10M random turn situations (as in DeepStack)
- Generate 1M random flop situations
- Solve situations with CFR to get target counterfactual values
- Use existing tensor environment for efficient data generation

### 4.2 Network Training
**Files to create/modify:**
- `scripts/train_value_network.py`

**Training Process:**
- Use Huber loss (as in DeepStack paper)
- Train with Adam optimizer
- Validate on held-out poker situations
- Target: Achieve reasonable accuracy for CFR convergence

## Phase 5: Integration & Optimization (Week 9-10)

### 5.1 Performance Optimization
- **Decision Time**: Target <5 seconds per decision (like DeepStack)
- **Memory Usage**: Efficient tensor storage for CFR trees
- **Caching**: Pre-compute frequent pre-flop situations
- **Batch Operations**: Leverage vectorized environment

### 5.2 Testing & Validation
- **Exploitability Testing**: Use local best response (LBR) to measure exploitability
- **Performance Benchmarking**: Compare against existing PPO agents
- **Integration Testing**: Ensure compatibility with existing infrastructure

## Success Metrics

### Primary Goals
- **Exploitability**: Achieve <100 mbb/g (better than abstraction bots)
- **Decision Speed**: <5 seconds per decision on GTX 1080
- **Training Stability**: CFR converges reliably
- **Infrastructure Reuse**: >80% of existing code maintained

### Secondary Goals
- **Memory Efficiency**: <8GB GPU memory usage
- **Training Speed**: Value network trains in <24 hours
- **Scalability**: Handle multiple environments efficiently

## Risk Mitigation

### Technical Risks
- **CFR Convergence**: May require parameter tuning for poker domain
- **Memory Usage**: CFR trees could be large → Use efficient tensor storage
- **Decision Time**: CFR slower than PPO → Implement depth limiting and caching
- **Value Network Accuracy**: Critical for theoretical guarantees → Extensive validation

### Implementation Risks
- **Integration Complexity**: Many moving parts → Phased rollout with testing
- **Debugging Difficulty**: CFR harder to debug than PPO → Add comprehensive logging
- **Training Instability**: Different convergence behavior → Gradual parameter adjustment

## Configuration Example

```yaml
# DeepStack configuration
model:
  value_head_type: "counterfactual"

cfr:
  max_depth: 4
  sparse_actions: [0, 1, 2, 3, 7]  # fold, call, 0.5x, 1x, all-in
  num_iterations: 1000
  cache_preflop: true

training:
  use_cfr: true  # Enable CFR instead of PPO
  value_network_training_steps: 10000
```

## Testing Strategy

### Unit Tests
- CFR solver correctness on small games (Kuhn poker)
- Value network accuracy on synthetic situations
- State tracker range update logic

### Integration Tests
- DeepStack agent decision-making
- Training loop compatibility
- Checkpointing and loading

### Performance Tests
- Decision time benchmarking
- Memory usage profiling
- Exploitability measurement

## Rollout Plan

### Week 1-2: Foundation
- ✅ Counterfactual value head implementation (1326 hands)
- ✅ Range tracking system (RangeTracker class)
- ✅ Configuration updates (removed num_hand_buckets)
- ✅ Integration with SelfPlayTrainer

### Week 3-4: CFR Integration
- Extend existing CFR solver
- Create continual re-solving engine
- Integration testing

### Week 5-6: DeepStack Agent
- Create DeepStack agent class
- Modify training loop
- Basic functionality testing

### Week 7-8: Value Network Training
- Generate training data
- Train value network
- Validation and optimization

### Week 9-10: Integration & Optimization
- Performance tuning
- Comprehensive testing
- Documentation and examples

This plan provides a structured approach to implementing DeepStack while maximizing reuse of existing infrastructure. The key insight is treating CFR as the "policy" and the transformer as the "value function" within the existing training framework.
