## Directory summary
Reusable staged-training helpers for ReBeL workflows that are shared by CLI
entry points and scripts.

### Source files
- `__init__.py`: Package marker.
- `curriculum.py`: Staged postflop ReBeL curriculum implementation for configured train and end-of-street value-only distill substeps, including per-substep checkpoints, promotion state, metadata, warm-starting, value initialization, configured root-belief sampling, compact `E_preflop` validation, and substep-aware resume.
- `preflop_buckets.py`: Single typed execution-config boundary plus run-config build helpers for preflop backward-induction bucket stages, including compact preflop-safe model overrides, configured belief sampler fields, and checkpoint resume wiring.
- `preflop_backward_induction.py`: Packaged implementation of preflop depth-bucket specialist training, value-only presolve export, and distillation, including state-bucket reading, bucket-aware root-belief sampling, shared validation caches, cutoff-checkpoint-shaped CFR solvers, optional cross-architecture bootstrap distillation, separate supervised student configs, solve/train loops, in-progress specialist resume, solved-dataset manifests, and checkpoint summaries.

### Subdirectories
There are no child source directories.
