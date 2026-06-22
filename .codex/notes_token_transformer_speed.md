# Notes: Preflop Token Transformer Speed

## Baseline Evidence
- Run `preflop_2000_p6_lr0p01_depth_filtered_tokenstacks_noflash_h256_rh256`, W&B `zkk6etaq`, commit `56a9991`.
- Model parameters: `9,632,065`.
- Config used `hidden_dim=256`, `range_hidden_dim=256`, `ffn_dim=768`, `num_hidden_layers=0`, `num_policy_layers=6`, `num_value_layers=7`, `transformer_heads=8`.
- Recent step timings before stopping:
  - Step 50: 115.47s
  - Step 55: 144.00s
  - Step 60: 138.18s
  - Step 63: 89.31s
- Qualitative baseline: steady state is still roughly 110-120s/step with variance, much slower than the requested 30-40s/step.

## Benchmarks
- Eager model-path benchmark (`/tmp/bench_preflop_token_transformer.py`, bf16 autocast, 6p, hidden=256, range_hidden=256):
  - Current `0/7/6`, `ffn_dim=768`, explicit attention: value 6.45ms @ B=2048, policy 6.45ms @ B=2048, value_static 79.71ms @ B=32768.
  - `0/7/6`, `ffn_dim=768`, SDPA: value 6.63ms, policy 6.01ms, value_static 101.33ms.
  - `0/7/6`, `ffn_dim=768`, padded SDPA: value 6.95ms, policy 6.20ms, value_static 106.12ms.
  - Padding the 7-token stream to 8 did not improve eager timing; it made large value-static slower.
- Compiled model-path benchmark:
  - Current `0/7/6`, `ffn_dim=768`: value 3.68ms, policy 3.52ms, value_static 32.14ms, value params 5.07M, policy params 4.56M.
  - Current `0/7/6`, `ffn_dim=256`: value 3.66ms, policy 3.56ms, value_static 26.65ms, value params 3.23M, policy params 2.99M.
  - SDPA and padded SDPA were essentially tied with explicit attention once compiled; padding did not help.
  - Zero-block lower bound `0/0/0`, `ffn_dim=256`: value 1.30ms, policy 1.37ms, value_static 4.17ms.
  - Two-layer corrected `1/1/1`, `ffn_dim=256`: value 1.90ms, policy 2.07ms, value_static 10.17ms.
  - Three-layer corrected `2/1/1`, `ffn_dim=256`: value 2.19ms, policy 2.43ms, value_static 13.44ms.
  - Three-layer corrected `2/1/1`, `ffn_dim=512`: value 2.25ms, policy 2.40ms, value_static 14.57ms.
  - One-layer corrected `1/0/0`, `ffn_dim=256`: value 1.65ms, policy 1.76ms, value_static 7.24ms.
  - One-layer corrected `1/0/0`, `ffn_dim=512`: value 1.65ms, policy 1.93ms, value_static 8.18ms.
- Interpretation:
  - Flash/SDPA is not the main lever for this tiny 7-token stream.
  - `ffn_dim` changes parameter count substantially, but compiled runtime is dominated by token block count.
  - The prior 30-40s run had 3.46M total params because the old transformer implementation ignored branch token layers. After the semantic fix, `0/7/6` creates 13 real token blocks across split models and 9.63M params.

## Candidate Changes
- Padding 7 tokens to 8 for SDPA/flash eligibility.
- Explicit attention with smaller FFN ratio.
- Replacing full FFN width with a smaller default while preserving attention layer counts.
- First launch candidate: corrected transformer with `num_hidden_layers=2`, `num_value_layers=1`, `num_policy_layers=1`, `ffn_dim=512`. It reduced params to 4.25M but still produced post-startup step timings of 54-71s, above target.
- Second launch candidate: corrected transformer with `num_hidden_layers=1`, `num_value_layers=0`, `num_policy_layers=0`, `ffn_dim=256`; this gives one real shared/base token block per split value/policy model and is the closest nonzero-attention config to the old 34s timing envelope.

## Restart Evidence
- `token3_ffn512` run `e1q6sqcg` reached step timings 320.10s, 77.62s, 54.24s, 70.65s, 67.71s, so it was stopped.
- `token1_ffn256` run `4kj2gucj` used 1.89M params and reached step timings 176.02s, then mostly 46-63s through step 15; it was stopped because the recent median was still above the 30-40s target.
- GPU contention check with `nvidia-smi` showed only the training process using the A100, so the remaining gap was not another CUDA job.
- `token0_ffn256` at `num_envs=512` used 1.10M params and still ran mostly 41-52s after compile, so the remaining cost is CFR/data generation. It was stopped to relaunch at `num_envs=384` while preserving 400-iteration SAPDCFR per root.
