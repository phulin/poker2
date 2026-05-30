## Directory summary
Preflop all-in equity modeling, random training-batch generation, Monte Carlo target estimation, and standalone training entry points.

### Source files
- `__init__.py`: Public package exports for the preflop all-in model, batch dataclass, random generator, and MC target sampler.
- `model.py`: Browser-friendly LeakyReLU/RMSNorm preflop all-in equity model that predicts `[batch, players, 1326]` terminal values.
- `data.py`: Random preflop all-in batch generation with per-player weighted-uniform stack sampling and folded/all-in masks.
- `sampler.py`: Preflop full-board Monte Carlo all-in value target estimator with tuple-reject opponent sampling, side-pot layer accounting, CUDA fast-path dispatch, and CPU reference fallback.
- `kernels.py`: All-in-specialized Triton alias tuple-reject sampler copied from the showdown ultrafast path and modified to score by-hand side-pot payouts over sampled full-board rows.
- `train.py`: Standalone training script with a separate Weights & Biases project and detailed throughput/loss/target logging.

### Subdirectories
There are no child source directories.
