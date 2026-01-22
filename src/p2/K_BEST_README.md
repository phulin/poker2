# AlphaHoldem K-Best Self-Play Implementation

This implementation adds K-Best self-play functionality to the AlphaHoldem poker agent, as described in the original AlphaHoldem paper. K-Best self-play addresses the problem of agents getting trapped in local minima during self-play training by maintaining a pool of K best historical versions of the agent.

## Overview

K-Best self-play works by:
1. **Maintaining an opponent pool**: Keeps K best historical versions of the agent
2. **ELO rating system**: Tracks the strength of each version
3. **Diverse sampling**: Trains against different opponents to avoid overfitting
4. **Dynamic updates**: Adds new snapshots when significant improvement occurs

## Key Components

### KBestOpponentPool (`p2/rl/k_best_pool.py`)

The core class that manages the opponent pool:

```python
from p2.rl.k_best_pool import KBestOpponentPool

# Initialize pool
pool = KBestOpponentPool(k=5, min_elo_diff=50.0)

# Add snapshots
pool.add_snapshot(agent, elo_rating)

# Sample opponents
opponents = pool.sample(k=3)

# Update ELO after games
pool.update_elo_after_game(opponent, 'win')  # or 'loss', 'draw'
```

### Enhanced SelfPlayTrainer (`p2/rl/self_play.py`)

The training class now includes K-Best functionality:

```python
from p2.rl.self_play import SelfPlayTrainer

# Initialize with K-Best pool
trainer = SelfPlayTrainer(
    k_best_pool_size=5,
    min_elo_diff=50.0,
    # ... other parameters
)

# Training step now uses K-Best opponents
stats = trainer.train_step(num_trajectories=4)

# Evaluate against pool
eval_results = trainer.evaluate_against_pool(num_games=100)
```

## Usage Examples

### Basic Training

```python
from p2.rl.self_play import SelfPlayTrainer

# Initialize trainer
trainer = SelfPlayTrainer(
    num_bet_bins=9,
    learning_rate=1e-3,
    batch_size=256,
    num_epochs=4,
    gamma=0.999,
    gae_lambda=0.95,
    epsilon=0.2,
    delta1=3.0,
    value_coef=0.1,
    entropy_coef=0.01,
    grad_clip=1.0,
    k_best_pool_size=5,      # Maintain 5 best opponents
    min_elo_diff=50.0,       # Add new snapshot if ELO differs by 50+
)

# Training loop
for step in range(1000):
    stats = trainer.train_step(num_trajectories=4)

    print(f"Step {step}: ELO={stats['current_elo']:.1f}, "
          f"Pool size={stats['pool_stats']['pool_size']}")

    # Evaluate periodically
    if step % 50 == 0:
        eval_results = trainer.evaluate_against_pool(num_games=50)
        print(f"Win rate: {eval_results['overall_win_rate']:.3f}")
```

### Command Line Training

Use the provided training script:

```bash
# Basic training
python src/p2/cli/train_kbest.py --num-steps 1000

# Custom parameters
python src/p2/cli/train_kbest.py \
    --num-steps 2000 \
    --k-best-pool-size 10 \
    --min-elo-diff 30.0 \
    --trajectories-per-step 6 \
    --checkpoint-interval 100 \
    --eval-interval 50
```

### Demonstration

Run the demonstration script to see K-Best self-play in action:

```bash
python src/p2/cli/demo_kbest.py
```

## Configuration

### K-Best Pool Parameters

- `k_best_pool_size`: Number of opponents to maintain in pool (default: 5)
- `min_elo_diff`: Minimum ELO difference to add new snapshot (default: 50.0)

### Training Parameters

- `num_trajectories`: Number of trajectories per training step
- `checkpoint_interval`: How often to save checkpoints
- `eval_interval`: How often to evaluate against pool

## Benefits

1. **Prevents local minima**: Diverse opponents prevent strategy cycling
2. **Efficient training**: Only one main agent, but diverse training data
3. **Adaptive difficulty**: Automatically adjusts to agent's current strength
4. **Better exploration**: Forces agent to explore different strategies

## Implementation Details

### ELO Rating System

The implementation uses a standard ELO rating system:
- Starting ELO: 1200
- K-factor: 32 (standard for chess-like games)
- Updates after each game based on win/loss/draw

### Opponent Sampling

Opponents are sampled with replacement, weighted by ELO rating:
- Higher ELO opponents are more likely to be selected
- Ensures challenging training opponents
- Prevents overfitting to weak opponents

### Snapshot Management

- Snapshots are sorted by ELO rating (descending)
- Only top K snapshots are kept
- New snapshots added when ELO differs significantly
- Old snapshots can be cleaned up based on age

### Checkpointing

Both the main model and opponent pool are saved/loaded:
```python
# Save
trainer.save_checkpoint("checkpoint.pt", step)

# Load
trainer.load_checkpoint("checkpoint.pt")
```

## Testing

Run the test suite to verify functionality:

```bash
python tests/test_kbest.py
```

## Comparison with Paper

This implementation follows the AlphaHoldem paper's K-Best approach:

| Feature | Paper Description | Implementation |
|---------|------------------|----------------|
| Pool Size | K best opponents | Configurable K |
| Sampling | ELO-weighted | ELO-weighted with replacement |
| Updates | Based on performance | ELO difference threshold |
| Efficiency | Single agent training | Single agent + pool management |

## Future Extensions

The K-Best framework can be extended for:
- Multi-player poker (league-based approach)
- Different sampling strategies
- Adaptive pool sizes
- Meta-learning over opponent strategies

## References

- AlphaHoldem paper: "High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning"
- ELO rating system: Standard chess rating system
- Self-play improvements: Addresses cycling and local minima issues
