## Directory summary
CLI entry points for training, search tuning, model inspection, sampling public belief states, and demos. Most commands are intended to run through `uv run`.

### Source files
- `train_rebel.py`: Hydra entry point for ReBeL-style CFR training; builds config/WandB/trainer setup and delegates the shared step loop to `p2.rl.rebel_loop`.
- `train_rebel_curriculum.py`: Staged postflop ReBeL curriculum orchestrator that runs configured train substeps through the shared loop with per-substep checkpoints and metadata.
- `train_kbest.py`: PPO self-play trainer using K-best/DReD-style opponent pools and tensorized environments.
- `modal_train_rebel.py`: Modal wrapper for launching ReBeL training remotely.
- `sample_spots.py`: Samples and serializes public belief states from ReBeL data generation.
- `tune_cfr.py`: Local parameter sweep utility for CFR evaluator settings.
- `tune_cfr_search.py`: Search-oriented CFR tuning runner with parameter grids.
- `demo_kbest.py`: Demonstrates K-best concepts and configured training behavior.
- `param_count.py`: Prints model parameter counts for a Hydra config.

### Subdirectories
There are no child source directories.
