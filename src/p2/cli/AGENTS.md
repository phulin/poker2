## Directory summary
CLI entry points for training, search tuning, model inspection, sampling public belief states, and demos. Most commands are intended to run through `uv run`.

### Source files
- `train_rebel.py`: Main ReBeL-style CFR training loop with Hydra config, checkpointing, WandB logging, and training stats.
- `train_kbest.py`: PPO self-play trainer using K-best/DReD-style opponent pools and tensorized environments.
- `modal_train_rebel.py`: Modal wrapper for launching ReBeL training remotely.
- `sample_spots.py`: Samples and serializes public belief states from ReBeL data generation.
- `tune_cfr.py`: Local parameter sweep utility for CFR evaluator settings.
- `tune_cfr_search.py`: Search-oriented CFR tuning runner with parameter grids.
- `demo_kbest.py`: Demonstrates K-best concepts and configured training behavior.
- `param_count.py`: Prints model parameter counts for a Hydra config.

### Subdirectories
There are no child source directories.
