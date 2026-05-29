## Directory summary
Reusable multiway showdown equity evaluators and sampling kernels copied from benchmark prototypes.

### Source files
- `__init__.py`: Public package exports for exact showdown evaluators, by-hand result types, and triangle-weight checks.
- `results.py`: Shared per-hand equity result dataclass, safe division, aggregation, and active-to-full scatter helpers.
- `multiway_showdown_estimators.py`: Copied benchmark implementation containing exact IE, SIS/rejection samplers, Triton kernels, and support dataclasses, plus exact by-hand oracle output.
- `compare_multiway_showdown_tiers.py`: Copied tier-comparison benchmark implementation with package-relative imports, tier 1-4 by-hand outputs, a rank-prefix/Triton factor path for tier 2, tier 3 reuse of rank-prefix factors, and a memory-aware streaming Triton wedge path for large four-player CUDA batches.
- `exact.py`: Public exact-evaluator API, A+xB exact by-hand path for up to 4 players, and explicit `tri` weight helpers.
- `approximate.py`: Public exports for aggregate and by-hand tier 1-4 approximate equity calculators.
- `monte_carlo.py`: Public exports for the alias tuple-reject MC kernel, related workspaces, and a conditional per-hand PyTorch MC reference estimator.

### Subdirectories
There are no child source directories.
