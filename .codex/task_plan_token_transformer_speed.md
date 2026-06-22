# Task Plan: Preflop Token Transformer Speed

## Goal
Get the S_preflop token-stack training run back to roughly 30-40 seconds per step while preserving the intended value/policy depth semantics.

## Phases
- [ ] Phase 1: Stop the slow run and capture baseline evidence
- [ ] Phase 2: Benchmark attention/block/model variants under the live config
- [ ] Phase 3: Implement the fastest reasonable model change
- [ ] Phase 4: Validate tests and CUDA smoke benchmarks
- [ ] Phase 5: Restart training and verify step timing

## Key Questions
1. Is the slowdown from explicit attention, excessive FFN/model width, compile behavior, or overall layer count?
2. Does padding the 7-token stream to 8 tokens unlock a stable/faster attention path?
3. Can we reduce parameters or compute without undoing the requested `num_hidden_layers + num_value_layers` / `num_hidden_layers + num_policy_layers` attention depth?

## Decisions Made
- Slow `tokenstacks_noflash` run was stopped before benchmarking further; steady-state timing was still far above target.
- Padding to 8 tokens was not adopted; SDPA/padded-SDPA did not improve compiled timing and padding was worse in eager large-batch value eval.
- `ffn_dim` reduction helps parameters but not enough to solve wall-clock timing by itself.
- Token block count is the primary model-side lever; `0/7/6` after the semantic fix is too expensive.
- One-layer and zero-layer token configs were tested in live runs. One-layer still averaged around 50s; zero-layer at 512 envs was the best observed live run. Reducing to 384 envs worsened timing, likely from GPU underfill.
- Current active run is 512 envs, zero token blocks, `ffn_dim=256`, preserving 400-iteration SAPDCFR but prioritizing wall-clock recovery.

## Errors Encountered
- Initial `/tmp/bench_preflop_token_transformer.py` run failed because `MLPFeatures` defaults to 1326 hands; fixed script to pass `hand_dim=169`.
- Second benchmark attempt initialized modules after moving them to CUDA with a CPU generator; fixed script to initialize on CPU before `.to(device)`.
- Zero-layer lower-bound benchmark hit compact-MLP constructor divide-by-zero; made benchmark robust to constructor failures.

## Status
**Currently in Phase 5** - restarted the best observed run and waiting for later step timings to prove whether it reaches the 30-40s target.
