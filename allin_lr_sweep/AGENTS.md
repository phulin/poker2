## Directory summary
Ad hoc all-in learning-rate sweep harnesses, notes, reports, and compact result summaries for 6-player pregenerated all-in training experiments.

### Source files
- `run_sweep.py`: Baseline 1,000-step cosine learning-rate sweep over 6-player pregenerated all-in train/validation manifests.
- `run_linear_sweep.py`: Linear decay comparison using the shared baseline sweep parser and dataset constants.
- `run_warmdown_sweep.py`: Stable warmdown comparison with flat initial LR and late cosine decay.
- `run_cosine_2k_sweep.py`: 2,000-step cosine decay comparison for the strongest 1,000-step candidate.
- `run_8k_hp_sweep.py`: 8,000-step MLP-vs-player-transformer comparison on the regenerated high-quality all-in validation set across cosine and linear LR schedules.
- `allin_lr_sweep_report.md`: Human-readable sweep setup, results, and recommendation.
- `notes.md`: Working notes on manifests, smoke tests, schedule variants, and observed results.
- `task_plan.md`: Completed checklist and decisions for the sweep task.

### Subdirectories
- `logs/`, `logs_linear/`, `logs_warmdown/`, `logs_cosine_2k/`: Compact `results.json` summaries for each sweep family. Raw `.log` files are ignored.
- `checkpoints*/`: Local model checkpoint artifacts from sweep runs. These are ignored and should not be committed.
