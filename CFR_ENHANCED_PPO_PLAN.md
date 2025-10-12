# CFR-Enhanced PPO Implementation Plan

## Overview

This plan outlines an incremental approach to enhance your existing PPO training with CFR-based value estimation. Instead of completely replacing PPO with DeepStack, we'll add CFR computations during rollouts to provide better value targets for training.

## 🎯 Core Idea

**During rollouts**: Run small CFR computations to get better value estimates
**During training**: Use CFR values as targets instead of just terminal rewards

This gives you the benefits of CFR's theoretical soundness while keeping your existing PPO infrastructure.

## ✅ Current Assets

- **Existing CFR solver** (`src/pokercfr/`) - ready for integration
- **Vectorized tensor environment** - perfect for batched CFR
- **PPO training loop** - minimal changes needed
- **Self-play infrastructure** - generates realistic situations

## Phase 1: CFR Value Computation During Rollouts

### 1.1 CFR Integration Points

**Where to run CFR:**
- **Decision points**: When players make moves (fold/call/bet)
- **Chance events**: After cards are dealt
- **Key states**: Every N steps or at street boundaries

**Integration approach:**
```python
# In rollout generation:
def get_cfr_value_estimate(state, ranges):
    # Extract public state from tensor_env
    # Run small CFR computation (10-50 iterations)
    # Return value estimate for current player
    return cfr_value
```

### 1.2 CFR Configuration for Training

```python
CFR_CONFIG = {
    "max_depth": 2,           # Very limited depth for speed
    "num_iterations": 20,     # Few iterations for quick estimates
    "action_set": ["fold", "call", "min_bet", "pot_bet"],  # Limited actions
    "use_value_network": False,  # Don't use network during CFR (avoid cycles)
}
```

### 1.3 State Extraction for CFR

**From tensor environment to CFR:**
```python
def extract_cfr_state(tensor_env_state):
    return {
        "board_cards": tensor_env_state.board_indices,
        "pot_size": tensor_env_state.pot,
        "player_stacks": tensor_env_state.stacks,
        "current_player": tensor_env_state.to_act,
        "betting_history": extract_betting_history(tensor_env_state),
    }
```

## Phase 2: Enhanced Value Targets

### 2.1 Hybrid Value Targets

**Target computation:**
```python
def compute_hybrid_targets(cfr_values, actual_rewards, terminal_mask):
    targets = torch.zeros_like(cfr_values)

    # For terminal states: use actual rewards (ground truth)
    targets[terminal_mask] = actual_rewards[terminal_mask]

    # For non-terminal states: use CFR values (better estimates)
    targets[~terminal_mask] = cfr_values[~terminal_mask]

    return targets
```

**Weighting approach:**
```python
# Optional: weight by confidence or recency
def compute_weighted_targets(cfr_values, actual_rewards, steps_from_terminal):
    weights = 1.0 / (steps_from_terminal + 1)  # Closer to terminal = higher weight
    return weights * cfr_values + (1 - weights) * actual_rewards
```

### 2.2 Integration with PPO Loss

**Modified loss calculation:**
```python
# In PPO training loop:
cfr_targets = compute_cfr_targets_for_batch(batch)
hybrid_targets = compute_hybrid_targets(cfr_targets, batch.returns, batch.terminal)

# Use hybrid targets instead of just returns
loss = compute_ppo_loss(logits, values, hybrid_targets, advantages)
```

## Phase 3: Computational Efficiency

### 3.1 CFR Depth Limiting

**Aggressive depth limiting:**
- **River**: Depth 1 (just showdown evaluation)
- **Turn**: Depth 2 (one betting round)
- **Flop**: Depth 3 (two betting rounds max)
- **Pre-flop**: Skip CFR (too expensive)

### 3.2 Selective CFR Computation

**When to run CFR:**
```python
def should_run_cfr(state):
    return (
        state.street >= 1 and  # Post-flop only
        state.pot > state.bb * 2 and  # Significant pots only
        state.actions_this_round < 3  # Early in street
    )
```

### 3.3 Caching and Optimization

**CFR result caching:**
```python
cfr_cache = {}  # Cache by (board_cards, pot_size, ranges_hash)

def get_cached_cfr_result(state_key):
    return cfr_cache.get(state_key)
```

## Phase 4: Training Integration

### 4.1 Rollout Enhancement

**Modified rollout collection:**
```python
def collect_enhanced_rollout():
    # ... existing rollout logic ...

    cfr_values = []
    for step in rollout:
        if should_run_cfr(step.state):
            cfr_value = run_quick_cfr(step.state, step.ranges)
            cfr_values.append(cfr_value)
        else:
            cfr_values.append(None)

    return rollout, cfr_values
```

### 4.2 Batch Processing

**Vectorized CFR for efficiency:**
```python
def compute_cfr_batch(states, ranges):
    # Run CFR on batch of states
    # Return value estimates for each state
    return cfr_values_batch
```

## Phase 5: Benchmarking and Validation

### 5.1 Performance Comparison

**Metrics to track:**
- **Exploitability**: Measure against known opponents
- **Convergence speed**: How quickly does value error decrease?
- **Sample efficiency**: Performance per training sample
- **Training stability**: Does CFR help or hurt stability?

**A/B testing setup:**
```python
# Run parallel experiments
ppo_standard = train_ppo_standard()
ppo_cfr_enhanced = train_ppo_cfr_enhanced()

compare_performance(ppo_standard, ppo_cfr_enhanced)
```

### 5.2 Hyperparameter Tuning

**Key parameters to tune:**
- **CFR frequency**: How often to run CFR during rollouts
- **CFR depth**: How deep to search
- **CFR iterations**: How many iterations per computation
- **Target weighting**: Balance between CFR values and actual rewards

### 5.3 Validation

**Correctness checks:**
- **Value accuracy**: Do CFR values correlate with actual outcomes?
- **Training stability**: Does enhanced training converge reliably?
- **Exploitability improvement**: Does CFR help against strong opponents?

## Implementation Timeline

### Week 1: Foundation (CFR Integration)
- ✅ Set up CFR calls during rollouts
- ✅ Extract states for CFR computation
- ✅ Basic value target computation

### Week 2: Enhancement (Hybrid Targets)
- ✅ Implement hybrid target computation
- ✅ Add CFR result caching
- ✅ Integrate with PPO loss calculation

### Week 3: Optimization (Efficiency)
- ✅ Add depth and frequency limiting
- ✅ Implement selective CFR computation
- ✅ Profile and optimize performance

### Week 4: Validation (Testing)
- ✅ Set up A/B comparison framework
- ✅ Measure exploitability improvements
- ✅ Tune hyperparameters for best performance

## Configuration

**Add to your config:**
```yaml
cfr_enhanced_ppo:
  enabled: true
  cfr_frequency: 0.1  # Run CFR on 10% of steps
  cfr_max_depth: 2
  cfr_iterations: 20
  use_hybrid_targets: true
  cfr_cache_size: 10000
```

## Expected Outcomes

### Primary Benefits
- **Better value estimation** than standard PPO
- **Faster convergence** to good strategies
- **More robust training** with better intermediate targets

### Potential Trade-offs
- **Slightly slower rollouts** due to CFR computation
- **Increased memory usage** for CFR caching
- **Hyperparameter tuning** required for optimal performance

## Risk Mitigation

- **Start conservative**: Few CFR iterations, limited depth
- **Progressive rollout**: Enable for subset of training, expand gradually
- **Fallback mechanism**: Fall back to standard rewards if CFR fails
- **Performance monitoring**: Track both training speed and final performance

## Success Metrics

- **Exploitability reduction**: 10-20% improvement vs standard PPO
- **Training stability**: No degradation in convergence reliability
- **Computational overhead**: <50% increase in training time
- **Value accuracy**: CFR values correlate better with actual outcomes

This approach gives you the theoretical benefits of CFR-based value learning while maintaining the practical advantages of your existing PPO infrastructure!
