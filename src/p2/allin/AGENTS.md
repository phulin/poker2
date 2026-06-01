## Directory summary
Preflop all-in equity modeling, random training-batch generation, Monte Carlo target estimation, and standalone training entry points.

### Source files
- `__init__.py`: Public package exports for the preflop all-in model, batch dataclass, random generator, and MC target sampler.
- `model.py`: Browser-friendly LeakyReLU/RMSNorm preflop all-in equity models with combo-moment, card-mass, rank/suit bucket, hand-conditioned blocker, optional dense-belief/output residuals, configurable low-rank FiLM branches, and a player-token transformer variant that mixes one scalar context token, per-player range tokens, and masked side-pot layer tokens to predict `[batch, players, 1326]` terminal values with optional folded-player key masking.
- `data.py`: Random terminal preflop all-in batch generation with per-player weighted-uniform stack sampling, folded masks, all-in masks, and one live covering caller.
- `training_data.py`: Reusable online/offline all-in training-data generation, optional online target workspace reuse, sharded dataset writing, manifest loading, pinned/async pregenerated shard reads, sequential/wrapped batch reads, and suit/player permutation remapping for beliefs, per-seat features, and targets.
- `sampler.py`: Preflop full-board Monte Carlo all-in value target estimator with tuple-reject opponent sampling, side-pot layer accounting, optional reusable CUDA workspaces, compact board-mask CUDA dispatch, and CPU reference fallback.
- `kernels.py`: All-in-specialized Triton CDF tuple-reject sampler that scores by-hand side-pot payouts over sampled full-board rows, including compact board-mask accumulation and folded-hero skip paths.
- `train.py`: Hydra-driven standalone training script with a separate Weights & Biases project, Muon or NorMuon/AdamW split optimization for linear matrices vs. scalar parameters, optional cosine/linear LR decay, stable warmdown, and cosine-then-linear-to-zero schedules, opt-in torch.compile support, checkpoint resume support, batch-size phase scheduling, online target workspace reuse, multi-epoch pregenerated train/validation data support with pinned CUDA prefetch, epoch-aware suit and player permutation augmentation, opt-in eligible-pot-share targets, and detailed throughput/loss/target/eval logging including validation MSE/MAE by live player count.
- `pregenerate.py`: CLI for pregenerating sharded all-in training datasets containing random features and `allin_values` targets, including an exhaustive all-boards target mode with fixed tuple samples per full board.

### Subdirectories
There are no child source directories.
