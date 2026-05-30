## Directory summary
Preflop all-in equity modeling, random training-batch generation, Monte Carlo target estimation, and standalone training entry points.

### Source files
- `__init__.py`: Public package exports for the preflop all-in model, batch dataclass, random generator, and MC target sampler.
- `model.py`: Browser-friendly LeakyReLU/RMSNorm preflop all-in equity model with combo-moment, card-mass, rank/suit bucket, hand-conditioned blocker, optional dense-belief residual, and configurable low-rank FiLM branches that predict `[batch, players, 1326]` terminal values.
- `data.py`: Random terminal preflop all-in batch generation with per-player weighted-uniform stack sampling, folded masks, all-in masks, and one live covering caller.
- `training_data.py`: Reusable online/offline all-in training-data generation, sharded dataset writing, manifest loading, sequential/wrapped pregenerated batch reads, and suit-permutation remapping for beliefs and targets.
- `sampler.py`: Preflop full-board Monte Carlo all-in value target estimator with tuple-reject opponent sampling, side-pot layer accounting, CUDA fast-path dispatch, and CPU reference fallback.
- `kernels.py`: All-in-specialized Triton CDF tuple-reject sampler that scores by-hand side-pot payouts over sampled full-board rows.
- `train.py`: Hydra-driven standalone training script with a separate Weights & Biases project, Muon/AdamW split optimization for linear matrices vs. scalar parameters, optional cosine LR decay, opt-in torch.compile support, checkpoint resume support, batch-size phase scheduling, multi-epoch pregenerated train/validation data support with epoch-aware suit permutation augmentation, and detailed throughput/loss/target logging.
- `pregenerate.py`: CLI for pregenerating sharded all-in training datasets containing random features and `allin_values` targets.

### Subdirectories
There are no child source directories.
