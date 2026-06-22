# Task Plan: Preflop Graph Evaluate

## Goal
Add a CUDA-graph CFR outer loop for the multiway preflop fused sparse evaluator without inheriting 2p-only finalization behavior.

## Phases
- [x] Phase 1: Inspect evaluator boundaries and graph helper assumptions
- [x] Phase 2: Implement preflop-specific graph-enabled evaluate_cfr
- [x] Phase 3: Run targeted tests and CUDA smoke
- [x] Phase 4: Summarize behavior, risks, and next steps

## Key Questions
1. Which parts of the 2p graph loop are scheduler-only and safe to reuse?
2. Which finalization/sample paths must remain preflop-specific?

## Decisions Made
- Keep graph orchestration in `FusedPreflopSparseCFREvaluator.evaluate_cfr` instead of inheriting `FusedSparseCFREvaluator.evaluate_cfr`.
- Capture only the multiway preflop `cfr_iteration`; preserve the generic preflop finalization and sampling behavior.
- Use `TScalars.t_tensor` for `t_sample` comparison so graph replay does not freeze the capture iteration.
- Honor `_skip_t_scalars_update` in preflop scalar refresh so `GraphedCFRIteration.prepare_replay()` supplies all replay-time scalars.
- Keep a debug/benchmark escape hatch: `P2_PREFLOP_CUDA_GRAPH_EVALUATE=0` forces the generic non-graph loop.

## Errors Encountered
- Ran `ruff` against `src/p2/search/AGENTS.md` by mistake; ruff treated Markdown as Python. Reran ruff against the Python file only and it passed.
- Initial CUDA smoke attempts used stale/wrong Hydra overrides (`preflop_buckets.use_wandb`, missing six-player compact model overrides). Corrected to top-level `use_wandb=false`, `+env.num_players=6`, and `+model.preflop_hand_dim=169`.

## Validation
- `uv run ruff check src/p2/search/fused_preflop_sparse_cfr_evaluator.py`
- `uv run pytest tests/test_sparse_cfr_evaluator.py -q -k 'preflop or FusedPreflop or fused_preflop'`
- CUDA smoke, graph enabled: 512 roots, 20 CFR iters, 4,010 nodes, 23.2s cold / 22.6s warm.
- CUDA smoke, graph disabled: 512 roots, 20 CFR iters, 4,010 nodes, 21.9s warm.
- CUDA microbenchmark, graph enabled: 512 roots, 400 CFR iters, 4,010 nodes, 23.5s.
- CUDA microbenchmark, graph disabled: 512 roots, 400 CFR iters, 4,010 nodes, 26.1s.

## Status
**Complete** - Source changes committed as `22d64f5`.
