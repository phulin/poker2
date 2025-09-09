# AlphaHoldem Configuration Files

This directory contains Hydra configuration files for AlphaHoldem training. Each file is a complete, self-contained configuration that can be used independently.

## Available Configurations

### `config.yaml` - Default Configuration
- **Purpose**: Balanced configuration for general training
- **Steps**: 2000
- **Batch Size**: 1024
- **Learning Rate**: 1e-4
- **Device**: CUDA (auto-detects)
- **Environments**: 512 (tensorized)
- **K-Best Pool**: 10 opponents

### `config_high_perf.yaml` - High-Performance Configuration
- **Purpose**: Optimized for vast.ai GPUs (RTX 4090/4080)
- **Steps**: 5000
- **Batch Size**: 2048
- **Learning Rate**: 5e-5
- **Device**: CUDA
- **Environments**: 1024 (tensorized)
- **K-Best Pool**: 20 opponents
- **Wandb**: Enabled with vast.ai tags

### `config_fast.yaml` - Fast Testing Configuration
- **Purpose**: Fast testing and development
- **Steps**: 500
- **Batch Size**: 256
- **Learning Rate**: 1e-3
- **Device**: CPU
- **Environments**: 64 (non-tensorized)
- **K-Best Pool**: 5 opponents
- **Wandb**: Disabled
- **Model**: Smaller architecture for speed

## Usage

### Basic Usage
```bash
# Use default configuration
python alphaholdem/cli/train_kbest.py --config-name=config

# Use high-performance configuration
python alphaholdem/cli/train_kbest.py --config-name=config_high_perf

# Use fast configuration
python alphaholdem/cli/train_kbest.py --config-name=config_fast
```

### Parameter Overrides
```bash
# Override specific parameters
python alphaholdem/cli/train_kbest.py \
    --config-name=config_high_perf \
    num_steps=1000 \
    train.batch_size=1536 \
    train.learning_rate=1e-4
```

### Resume Training
```bash
python alphaholdem/cli/train_kbest.py \
    --config-name=config_high_perf \
    resume_from=checkpoints/checkpoint_step_1000.pt
```

## Configuration Structure

Each configuration file contains:

- **Training Parameters**: Learning rate, batch size, epochs, PPO settings
- **Model Configuration**: Architecture, channels, hidden layers, policy
- **Environment Settings**: Game rules, betting bins, encoders
- **RL Parameters**: K-Best pool size, evaluation intervals
- **Device Settings**: GPU/CPU selection, tensorized environments
- **Logging**: Wandb project, tags, checkpoint settings

## Creating Custom Configurations

To create a custom configuration:

1. Copy an existing config file
2. Modify the parameters as needed
3. Use it with `--config-name=your_config`

Example:
```bash
cp config.yaml config_custom.yaml
# Edit config_custom.yaml
python alphaholdem/cli/train_kbest.py --config-name=config_custom
```

## Parameter Reference

### Training Parameters (`train:` section)
- `learning_rate`: Learning rate for optimizer
- `batch_size`: Batch size for training
- `num_epochs`: Number of epochs per training step
- `ppo_eps`: PPO clipping parameter
- `ppo_delta1`: Trinal-clip PPO parameter
- `gae_lambda`: GAE lambda parameter
- `gamma`: Discount factor
- `entropy_coef`: Entropy coefficient
- `value_coef`: Value loss coefficient
- `grad_clip`: Gradient clipping threshold

### Model Parameters (`model:` section)
- `name`: Model architecture name
- `kwargs.cards_channels`: Number of card channels
- `kwargs.actions_channels`: Number of action channels
- `kwargs.fusion_hidden`: Hidden layer sizes
- `kwargs.num_actions`: Number of possible actions

### Environment Parameters (`env:` section)
- `stack`: Starting stack size
- `sb`: Small blind
- `bb`: Big blind
- `bet_bins`: Betting size multipliers
- `card_encoder`: Card encoding configuration
- `action_encoder`: Action encoding configuration

### RL Parameters (top level)
- `num_steps`: Total training steps
- `k_best_pool_size`: Size of K-Best opponent pool
- `min_elo_diff`: Minimum ELO difference for pool updates
- `checkpoint_interval`: Checkpoint save frequency
- `eval_interval`: Evaluation frequency
- `use_tensor_env`: Enable tensorized environments
- `num_envs`: Number of parallel environments
