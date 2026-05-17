## Directory summary
Shared training and model utilities used across CLI, RL, models, and tests.

### Source files
- `__init__.py`: Package marker.
- `config_loader.py`: Loads and merges structured configs from checkpoints.
- `training_utils.py`: Console training stats, preflop grids, checkpoint summaries, and evaluation output helpers.
- `model_utils.py`: Masked logits, probabilities, log-probs, values, and best-action helpers.
- `model_context.py`: Context managers for temporary train/eval mode changes.
- `ema.py`: Lightweight exponential moving average helper.
- `ema_helper.py`: EMA helper for model parameters.
- `kl_divergence.py`: KL divergence utilities for policies.
- `quantile_calculator.py`: Streaming quantile calculator.
- `profiling.py`: Profiling helper utilities.

### Subdirectories
There are no child source directories.
