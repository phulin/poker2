## Directory summary
ReBeL-specific configuration boundaries layered over the shared trainer config while the repository separates current ReBeL experiments from legacy PPO/K-best configuration.

### Source files
- `__init__.py`: Package marker.
- `rebel_load.py`: Hydra/OmegaConf loader for ReBeL entry points, including ReBeL defaults and rejection of explicit PPO/K-best top-level fields.
- `rebel_schema.py`: Typed ReBeL experiment wrapper that groups run, checkpoint, logging, trainer, model, environment, search, data, validation, curriculum, pregeneration, and preflop bucket settings while round-tripping to the trainer `Config`.

### Subdirectories
There are no child source directories.
