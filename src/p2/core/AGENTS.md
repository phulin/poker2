## Directory summary
Shared package contracts and typed configuration used across environments, models, training, and search.

### Source files
- `__init__.py`: Package marker.
- `action_schedule.py`: Shared derivation of global CFR action spaces and per-depth legal action masks from search bet-bin schedules.
- `interfaces.py`: Abstract interfaces for environments, encoders, models, policies, leagues, and build context.
- `structured_config.py`: Dataclass-based Hydra config schema and enums for training, model, environment, search, data-source, curriculum, preflop validation, pregeneration, and rating options, including per-curriculum-substep model overrides and native 169 preflop all-in oracle paths.

### Subdirectories
There are no child source directories.
