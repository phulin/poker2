## Directory summary
CLI entry points for training, search tuning, model inspection, sampling public belief states, and demos. Most commands are intended to run through `uv run`.

### Source files
- `demo_kbest.py`: Demonstrates K-best training behavior.
- `modal_train_rebel.py`: Modal wrapper for launching ReBeL training remotely through the shared ReBeL config loader.
- `param_count.py`: Prints model parameter counts for a Hydra config.
- `pregenerate_postflop_rebel.py`: CLI for bounded postflop ReBeL dataset pregeneration.
- `sample_spots.py`: Samples and serializes public belief states from ReBeL data generation using the shared ReBeL config loader.
- `train_kbest.py`: PPO self-play trainer using K-best/DReD-style opponent pools.
- `train_rebel.py`: Hydra entry point for ReBeL-style CFR training.
- `train_rebel_curriculum.py`: Hydra entry point for staged postflop ReBeL curriculum training.
- `train_rebel_preflop_buckets.py`: Hydra entry point for preflop bucket training and distillation.
- `tune_cfr.py`: Local parameter sweep utility for CFR evaluator settings using current ReBeL Hydra config plus checkpoint weights.
- `tune_cfr_search.py`: Search-oriented CFR tuning runner with parameter grids using the shared current-config tuning setup.

### Subdirectories
There are no child source directories.
