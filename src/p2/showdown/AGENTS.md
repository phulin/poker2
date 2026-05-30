## Directory summary
Reusable multiway showdown equity evaluators and sampling kernels copied from benchmark prototypes.

### Source files
- `__init__.py`: Public package exports for exact showdown evaluators, by-hand result types, and triangle-weight checks.
- `results.py`: Shared per-hand equity result dataclass, safe division, aggregation, and active-to-full scatter helpers.
- `multiway_showdown_estimators.py`: Copied benchmark implementation containing exact IE, SIS/rejection samplers, Triton kernels, and support dataclasses, plus exact by-hand oracle output.
- `compare_multiway_showdown_tiers.py`: Copied tier-comparison benchmark implementation with package-relative imports, tier 1-4 by-hand outputs, cached board/rank-group/card-slot context with packed sorted card positions and compact direct-finish board LUTs, grouped p4 Triton prefix accumulation with an optimized five-event rank-prefix pair kernel, an alignment-stable default sparse by-card CUDA tier 2 direct-finish path that avoids compact pair-event materialization, tier 3 reuse of rank-group factors, and a memory-aware streaming Triton wedge path with quadrature-compressed tie-share numerators for large four-player CUDA batches.
- `exact.py`: Public exact-evaluator API, A+xB exact by-hand path for up to 4 players, and explicit `tri` weight helpers.
- `approximate.py`: Public exports for aggregate and by-hand tier 1-4 approximate equity calculators.
- `monte_carlo.py`: Public exports for the alias tuple-reject MC kernel, related workspaces, a conditional single-board PyTorch MC reference estimator, and a batched SIS-style per-hand equity-vector estimator.

### Subdirectories
There are no child source directories.
