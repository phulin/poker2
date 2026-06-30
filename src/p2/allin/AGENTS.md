## Directory summary
Preflop all-in equity modeling, random training-batch generation, Monte Carlo target estimation, and standalone training entry points.

### Source files
- `__init__.py`: Public package exports.
- `data.py`: Random terminal preflop all-in batch generation.
- `kernels.py`: Triton kernels for all-in target sampling.
- `model.py`: Preflop all-in equity model definitions.
- `oracle.py`: Native 169-class preflop all-in oracle.
- `precompute.py`: Precomputation utilities for all-in share tensors.
- `pregenerate.py`: CLI for pregenerating all-in training datasets.
- `sampler.py`: Monte Carlo and cached-board all-in target estimation.
- `train.py`: Hydra-driven standalone all-in training script.
- `training_data.py`: Online and pregenerated all-in training-data helpers.

### Subdirectories
There are no child source directories.
