## Directory summary
Real-hand, real-chip agent evaluation. Plays duplicate (mirrored) heads-up
matches with actually dealt hole cards and env chip payoffs, and records every
game so rating models can be refit offline without replaying poker. This
replaces the public-belief-space scoring in `p2/rl/pbs_games.py`, which paid
out range-EVs computed from the models' own beliefs and was therefore not
comparable across models with different belief calibration.

### Source files
- `__init__.py`: Package exports (agents, match player, records).
- `agents.py`: `MatchAgent` interface plus `SearchAgent` (CFR evaluator playing
  its real dealt hand; beliefs are search input only) and the model-free
  scripted bots `FoldAgent`, `CallAgent`, `RandomAgent`.
- `duplicate_match.py`: `play_duplicate_match`, the batched duplicate-pair match
  player. Mirrored decks/buttons/stacks, common-random-number inverse-CDF action
  sampling, per-pair scoring in bb/100.
- `checkpoints.py`: `load_search_agent` — rebuilds a model + CFR evaluator from
  a checkpoint and a run's `resolved_config.json` and wraps it in a
  `SearchAgent`. CFR fidelity is pinned via `SearchFidelity` (never inherited
  from a training schedule) so evals stay comparable. Checkpoints are read-only.
  Deliberately not re-exported from `__init__.py`: it pulls in the trainer, and
  the scripted-agent path must stay importable without CUDA.
- `records.py`: Append-only JSONL per-game records (`GameRecord`,
  `GameBatchTensors`, `RecordWriter`, `load_records`, `pair_differences`) with a
  sidecar run manifest and a single device->host transfer per batch.

### Calibration
`scripts/eval_calibration_gates.py` is the harness's known-answer check: a
fold-bot in the small blind must score exactly -0.5 bb, identical agents must
score zero within 3 SE, and the duplicate-coupling benefit is measured rather
than assumed. Run it after any change to this directory.

### Subdirectories
There are no child source directories.
