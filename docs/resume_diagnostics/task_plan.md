# Task Plan: ReBeL Resume Equivalence Diagnosis

## Goal
Determine why raw fresh value loss changes after checkpoint resume and identify every state component that prevents equivalent continuation.

## Phases
- [x] Phase 1: Stop active training and preserve artifacts
- [x] Phase 2: Audit checkpoint and runtime state coverage
- [x] Phase 3: Build and run controlled equivalence probes
- [x] Phase 4: Isolate causal state differences
- [x] Phase 5: Verify the existing initialized-child control path
- [x] Phase 6: Write final diagnostic report

## Key Questions
1. Are model, optimizer, value replay, policy replay, and current PBS restored exactly?
2. Which RNG streams or mutable evaluator/environment state are omitted?
3. Does a repeated load produce identical fresh data, replay samples, updates, and metrics?
4. Is the raw-loss discontinuity entirely pot-distribution weighting, or is model behavior different?

## Decisions Made
- Preserve the existing step-2900 checkpoint pair and replay sidecar as the experiment source.
- Use read-only or isolated checkpoint copies so probes cannot overwrite production artifacts.

## Errors Encountered
- `uv run` briefly failed because its default cache was read-only; use `UV_CACHE_DIR=/tmp/uv-cache` for probes.
- Initial probe called `setup_torch_runtime` without its device argument; corrected before running experiments.
- Initial probe reconstructed grouped `resolved_config.json` as a flat `Config`, defaulting top-level `num_steps` to 2000. Switched to the exact flat config embedded in the full checkpoint and repeated causal experiments.

## Status
**Complete** - The current HU resume gap is omitted `last_aggressive_amount`
state; the original `has_folded` finding remains relevant only to multiway runs.
Ordinary GPU nondeterminism is a smaller secondary source of divergence.
