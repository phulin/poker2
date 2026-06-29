# Gated Token Mixer Megakernel Results

## Scope

Implemented an experimental inference-only Triton helper for the compact preflop gated token mixer that fuses:

- RMSNorm over each token embedding
- 7 -> 28 -> 7 token mixer
- dim -> dim gate projection
- sigmoid gate application and residual add

The production model forward path is unchanged. The new helper is benchmarked directly against eager PyTorch and the existing Triton mixer/gate residual helper.

## Benchmark Setup

- Command: `uv run python benchmarks/bench_preflop_gated_token_mixer_megakernel.py --iters 30 --warmup 8 --json`
- Device: NVIDIA A100-SXM4-80GB
- Dtype: bfloat16
- Config: `conf/config_rebel_preflop_buckets.yaml`
- Shape: hidden dim 192, FFN dim 256, 7 tokens
- Batch sizes: 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536

The live training process group was stopped during the timed run and resumed afterward. The inner training Python process shared process group 27817 and was observed in `Tl` state while stopped; WandB ran in a separate process group.

## Results

| Batch | Eager ms | Current Triton ms | Megakernel ms | Current speedup | Megakernel speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.228557 | 0.126805 | 0.049698 | 1.80x | 4.60x |
| 512 | 0.229342 | 0.129297 | 0.107247 | 1.77x | 2.14x |
| 1024 | 0.225348 | 0.122231 | 0.179678 | 1.84x | 1.25x |
| 2048 | 0.309180 | 0.128649 | 0.342938 | 2.40x | 0.90x |
| 4096 | 0.567467 | 0.165239 | 0.685295 | 3.43x | 0.83x |
| 8192 | 1.103940 | 0.313924 | 1.288260 | 3.52x | 0.86x |
| 16384 | 2.165112 | 0.599893 | 2.450534 | 3.61x | 0.88x |
| 32768 | 4.247962 | 1.141248 | 4.420369 | 3.72x | 0.96x |
| 65536 | 8.117214 | 1.881463 | 8.417109 | 4.31x | 0.96x |

Max absolute error versus eager was 0.015625-0.03125 for both Triton variants.

## Conclusion

The fused megakernel wins only at small batch sizes where kernel launch count dominates. At the actual buckets-scale batches, the existing path is better because it keeps the gate projection in PyTorch's optimized linear/GEMM path. This candidate should remain experimental until the gate projection is redesigned rather than hand-rolled inside the larger kernel.

## Follow-Up: No-Bubbles And Small-K Dot

The Hazy Research no-bubbles post changes the interpretation of the first result: our current candidate is only local fusion, not a true scheduled megakernel. Their design removes bubbles through an on-GPU instruction interpreter, shared-memory page management, and explicit dependency counters, including chunked producer/consumer scheduling. That suggests the next useful variants should focus on tile scheduling and dataflow, not just adding more math to one Triton function.

I also added `benchmarks/bench_triton_small_k_dot.py` to test whether `tl.dot` is the wrong primitive for our small matrix products. On A100 with `rows=57344`, `BLOCK_N=32`, `BLOCK_K=32`:

| Tile | BLOCK_M=8 best | BLOCK_M=16 best | Takeaway |
|---:|---|---|---|
| K=7, N=28 | manual fp32, 0.019876 ms | fp32/TF32 `tl.dot`, 0.021484 ms | Manual is only marginally best for the smallest current tile. |
| K=28, N=7 | fp32/TF32 `tl.dot`, 0.021084 ms | fp32/TF32 `tl.dot`, 0.019364 ms | `tl.dot` is better. |
| K=32, N=32 | bf16 `tl.dot`, 0.020326 ms | fp32/IEEE `tl.dot`, 0.021115 ms | `tl.dot` is not the obvious problem for gate tiles. |
| K=64, N=32 | fp32/TF32 `tl.dot`, 0.034959 ms | fp32/TF32 `tl.dot`, 0.022630 ms | Larger M tile helps substantially. |

Next direction: keep `tl.dot` for gate-like K=32 work, test a bf16 gate-dot variant inside the full kernel, and prototype a token-combined gate projection that treats the 7 tokens as part of the M dimension instead of issuing seven separate gate `tl.dot` calls per K tile.

## Follow-Up Variant Results

I added a bf16 gate-dot full-kernel variant and an isolated gate-projection layout benchmark.

Full kernel result: bf16 gate-dot is slower at realistic sizes. At batch 8192, current Triton is `0.315051 ms`, fp32-gate megakernel is `1.286042 ms`, and bf16-gate megakernel is `1.458927 ms`. At batch 65536, current Triton is `1.873886 ms`, fp32-gate megakernel is `8.382293 ms`, and bf16-gate megakernel is `10.618675 ms`.

Gate projection result: token-combined layout improves over seven separate token dots only with a large M tile, but still loses to PyTorch linear. At batch 8192, PyTorch linear is `0.061891 ms`, seven separate bf16 token dots are `0.139428 ms`, and token-combined bf16 with `M=64` is `0.092989 ms`. At batch 65536, PyTorch linear is `0.341135 ms`, seven separate bf16 token dots are `1.041879 ms`, and token-combined bf16 with `M=64` is `0.914770 ms`.

Updated next direction: do not pursue the current hand-rolled gate GEMM as the main path. The stronger candidate is a staged design that keeps the gate projection in the optimized PyTorch/cuBLAS path, or a more Hazy-style schedule that only embeds custom gate work if it avoids global traffic or overlaps with other work. The token mixer itself remains a good custom-kernel target because its 7-token structure is too small and position-specific for a regular GEMM to dominate.

## Persistent/Staged Prototype

I implemented a staged persistent design that keeps `RMSNorm + token_gate` in the optimized PyTorch path, then runs the token mixer, sigmoid gate application, and residual add in a persistent Triton kernel. The persistent kernel launches a bounded pool of programs and each program loops over `(batch_tile, dim_tile)` work tiles with `tl.range(..., flatten=True)`, matching the scheduling pattern from Triton's persistent matmul tutorial while keeping the stage boundary that our gate-projection measurements justified.

The first one-program-per-SM version was under-occupied for this tiny token-mixer tile. The wrapper now supports `programs_per_sm`, and the benchmark compares `x1`, `x2`, `x4`, and `x8`. The broad best setting from this run is `x8`.

| Batch | Current Triton ms | Persistent x1 ms | Persistent x2 ms | Persistent x4 ms | Persistent x8 ms |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.129604 | 0.141961 | 0.142404 | 0.136465 | 0.140254 |
| 512 | 0.138957 | 0.150084 | 0.146569 | 0.144862 | 0.139708 |
| 1024 | 0.131413 | 0.135919 | 0.140971 | 0.148275 | 0.139708 |
| 2048 | 0.129843 | 0.144828 | 0.142814 | 0.147763 | 0.139093 |
| 4096 | 0.165581 | 0.274500 | 0.182955 | 0.169984 | 0.161212 |
| 8192 | 0.314061 | 0.529101 | 0.344303 | 0.319625 | 0.325871 |
| 16384 | 0.600610 | 0.976111 | 0.550298 | 0.505719 | 0.509201 |
| 32768 | 0.954129 | 1.688201 | 1.064380 | 0.975326 | 0.942558 |
| 65536 | 1.877402 | 3.354760 | 2.098859 | 1.927851 | 1.850095 |

Result: the staged persistent design is a more promising direction than the full hand-rolled gate megakernel, but it should remain experimental for now. It is worse at small batches, approximately tied around 4096-8192, and modestly faster at some large bucket-scale sizes, especially batch 16384 in this run. Production wiring should use a conservative large-batch threshold or wait for repeated measurements across the live evaluation call shapes.

## Broader Design Follow-Up: Independent Producer Overlap

I also tested a larger design change instead of more local parameter tuning: split the block after RMSNorm, compute `token_mixer(y)` and `token_gate(y)` as independent producers, and fuse only the final consumer `x + mixed * sigmoid(gate) / sqrt(2)`. The benchmark includes a two-stream overlap version so the token mixer and gate projection can run concurrently.

Result on A100 bf16:

| Batch | Current serial ms | Split serial ms | Split overlap ms |
|---:|---:|---:|---:|
| 4096 | 0.163482 | 0.179569 | 0.309770 |
| 8192 | 0.313559 | 0.339384 | 0.340152 |
| 16384 | 0.496865 | 0.550953 | 0.554936 |
| 32768 | 0.951183 | 1.061550 | 1.059686 |
| 65536 | 1.886126 | 2.122650 | 2.118287 |

Conclusion: this producer-overlap design is slower. The extra materialized mixer output, extra combine launch, and stream synchronization cost more than any overlap recovered. The current serial path remains the best measured design for this architecture; a further speedup likely needs a more structural change than local scheduling, such as changing the gating architecture, using a library-supported fused gate epilogue, or moving a larger surrounding region of the model into one scheduled runtime.

## Measurement Mode: CUDA Graph Replay

I tested a broader scheduling approach: capture the current best token path (`RMSNorm -> PyTorch gate projection -> Triton mixer/gate/residual`) into a fixed-shape CUDA Graph and replay it.

| Batch | Current ms | Graph replay static ms | Copy + replay ms |
|---:|---:|---:|---:|
| 256 | 0.121559 | 0.023869 | 0.028600 |
| 512 | 0.126679 | 0.033690 | 0.038216 |
| 1024 | 0.130191 | 0.051692 | 0.057068 |
| 2048 | 0.132342 | 0.086170 | 0.099011 |
| 4096 | 0.138199 | 0.135895 | 0.148060 |
| 8192 | 0.261345 | 0.258314 | 0.282921 |
| 16384 | 0.499292 | 0.494090 | 0.543427 |
| 32768 | 0.957051 | 0.957891 | 1.055293 |
| 65536 | 1.892168 | 1.891277 | 2.090803 |

This is a real measurement result, but not a production model change for the live bucket trainer: the live evaluator already captures the surrounding CFR/model path with CUDA graphs. The useful takeaway is that future model-kernel comparisons should use graph-replay timing when estimating live impact, otherwise Python launch gaps overstate wins from reducing kernel count.

## Conditional Production Result: Fuse Next FFN RMSNorm

I implemented a staged path that keeps `token_norm(x)` and the optimized PyTorch/cuBLAS `token_gate(y)` projection, but fuses the custom token-mixer residual with the following FFN `RMSNorm`. The remaining FFN linear/activation/linear sequence stays on the normal PyTorch path. This removes a separate RMSNorm pass over the token output without hand-rolling the FFN GEMMs.

I also added a compiled otherwise-naive baseline: the same block math in ordinary PyTorch ops, wrapped in `torch.compile`, with compile time excluded from timing. Because the live evaluator already uses CUDA graphs, the decisive comparison is graph-replay timing, not normal launch timing.

Command:

`uv run python benchmarks/bench_preflop_token_mixer_next_norm.py --batch-sizes 512,8192,65536 --iters 120 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --include-compiled-naive --timing-mode cuda_graph --json`

| Batch | Old Triton ms | Fused next-norm b2 ms | Compiled naive ms | Fused vs old | Compiled naive vs old |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.094404 | 0.092058 | 0.116804 | 1.03x | 0.81x |
| 8192 | 0.707507 | 0.675234 | 1.003324 | 1.05x | 0.71x |
| 65536 | 4.766592 | 4.847326 | 7.307332 | 0.98x | 0.65x |

Cutoff check:

`uv run python benchmarks/bench_preflop_token_mixer_next_norm.py --batch-sizes 16384,32768 --iters 120 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --json`

| Batch | Old Triton ms | Fused next-norm b2 ms | Fused vs old |
|---:|---:|---:|---:|
| 16384 | 1.319979 | 1.285990 | 1.03x |
| 32768 | 2.434978 | 2.476979 | 0.98x |

Conclusion: compiled naive is not a replacement for the custom Triton residual path. The fused next-norm path is a small but real live-regime win through batch 16384 and regresses at 32768+. Production wiring is therefore conditional: eval/no-grad CUDA gated-token-mixer blocks use fused next-norm only for batches `<= 16384`; larger batches fall back to the old Triton residual path.

## Stack-Level Result: Cross-Block Next Token RMSNorm

The next larger boundary is between adjacent gated-token-mixer blocks. After a block computes its FFN output, the old path stores `out = token_out + ffn_out / sqrt(2)` and the next block immediately reads `out` to run `token_norm(out)`. I added a stack runner that fuses that residual add with the next block's token RMSNorm, threads the precomputed normalized tokens into the next block, and otherwise keeps the existing per-block math. Training and TorchDynamo paths still use the normal per-block loop; the stack runner is eval/no-grad CUDA only.

Command:

`uv run python benchmarks/bench_preflop_token_mixer_cross_norm.py --batch-sizes 512,8192,16384,32768,65536 --depth 4 --iters 120 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --json`

| Batch | Old per-block loop ms | Wired stack runner ms | Speedup |
|---:|---:|---:|---:|
| 512 | 0.372386 | 0.338304 | 1.10x |
| 8192 | 2.684177 | 2.410155 | 1.11x |
| 16384 | 5.137322 | 4.605687 | 1.12x |
| 32768 | 9.733854 | 8.732527 | 1.11x |
| 65536 | 19.158929 | 17.206366 | 1.11x |

Depth-5 check for the value stack:

`uv run python benchmarks/bench_preflop_token_mixer_cross_norm.py --batch-sizes 512,8192,65536 --depth 5 --iters 100 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --json`

| Batch | Old per-block loop ms | Wired stack runner ms | Speedup |
|---:|---:|---:|---:|
| 512 | 0.464978 | 0.420557 | 1.11x |
| 8192 | 3.352719 | 2.989926 | 1.12x |
| 65536 | 23.851858 | 21.284587 | 1.12x |

Compiled-naive stack comparison:

`uv run python benchmarks/bench_preflop_token_mixer_cross_norm.py --batch-sizes 512,8192,65536 --depth 4 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled-naive --json`

| Batch | Wired stack runner ms | Compiled naive stack ms | Wired vs compiled |
|---:|---:|---:|---:|
| 512 | 0.336896 | 0.465190 | 1.38x |
| 8192 | 2.412634 | 3.944423 | 1.63x |
| 65536 | 17.110963 | 29.129535 | 1.70x |

The compiled-naive stack also shows a larger numerical delta at small batch in this setup (`0.082` max abs at batch 512 versus the old custom path), while the wired stack runner stays around `0.009` to `0.011`.

Conclusion: this is the strongest measured production change so far. The live-regime graph replay win is roughly 10-12% across policy-depth and value-depth stacks, and the compiled-naive alternative is substantially slower.

## FFN Output Epilogue Probe

The next candidate boundary was the FFN output projection. In the stack runner, each intermediate block still computes `ffn_out = linear_out(h)` with cuBLAS/PyTorch, then a Triton epilogue reads `ffn_out` and `token_out` to compute the residual plus next token RMSNorm. I tested a full-width Triton GEMM epilogue that computes `linear_out + residual + next_token_norm` in one kernel.

Command:

`uv run python benchmarks/bench_preflop_ffn_epilogue.py --batch-sizes 512,8192,16384,32768,65536 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled --json`

| Batch | cuBLAS linear + Triton epilogue ms | Compiled torch boundary ms | Triton full-N BM16 ms |
|---:|---:|---:|---:|
| 512 | 0.022502 | 0.021837 | 0.020224 |
| 8192 | 0.170842 | 0.164531 | 0.172198 |
| 16384 | 0.300211 | 0.289536 | 0.333530 |
| 32768 | 0.577434 | 0.552845 | 0.653018 |
| 65536 | 1.136768 | 1.085043 | 1.296973 |

Conclusion: the custom full-width Triton GEMM epilogue is not worth wiring. It wins only at batch 512 and loses at larger row counts because giving up cuBLAS costs more than removing the materialized `ffn_out` and separate epilogue launch. The compiled torch boundary is a small isolated win, but the safer larger win is making the existing Triton stack path visible to the already-compiled model forward.

## Compile-Visible Stack Result

The live bucket evaluator uses `preflop_buckets.compile=default`, which maps to `model.compile=default`; `FusedSparseCFREvaluator` then calls `compile_forward_modes(dynamic=True)`. The previous stack runner was guarded out during Dynamo tracing, so a compiled model forward traced the slow naive stack instead of the custom Triton stack. I removed those compile guards so Dynamo can trace the Triton stack path directly.

Post-patch depth-4 command:

`uv run python benchmarks/bench_preflop_token_mixer_cross_norm.py --batch-sizes 512,8192,65536 --depth 4 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled-wired --compile-dynamic --json`

| Batch | Old per-block loop ms | Eager wired stack ms | Dynamic compiled wired stack ms | Compiled wired vs old |
|---:|---:|---:|---:|---:|
| 512 | 0.372301 | 0.337779 | 0.311782 | 1.19x |
| 8192 | 2.689869 | 2.408870 | 2.314394 | 1.16x |
| 65536 | 19.053503 | 17.118567 | 13.884224 | 1.37x |

Depth-5 value-stack command:

`uv run python benchmarks/bench_preflop_token_mixer_cross_norm.py --batch-sizes 512,8192,65536 --depth 5 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled-wired --compile-dynamic --json`

| Batch | Old per-block loop ms | Eager wired stack ms | Dynamic compiled wired stack ms | Compiled wired vs old |
|---:|---:|---:|---:|---:|
| 512 | 0.464627 | 0.419840 | 0.316723 | 1.47x |
| 8192 | 3.490854 | 2.991616 | 2.893056 | 1.21x |
| 65536 | 23.811418 | 21.262605 | 17.345395 | 1.37x |

Conclusion: this is a production-relevant win because it matches the live evaluator's compiled-forward mode. The as-committed stack fusion helped eager eval; making it compile-visible prevents `compile=default` from regressing to the naive stack and produces the largest measured stack speedups so far.

## Belief Projection Probe

The compact token model forms player tokens from two projections over the same `[B, P, 169]` beliefs:

- `player_beliefs @ range_projection` -> hidden range features
- `player_beliefs @ bucket_projection` -> 16 bucket masses

I tested replacing those with one concatenated projection `[169, hidden_dim + 16]`.

Command:

`uv run python benchmarks/bench_preflop_belief_projection.py --batch-sizes 512,8192,16384,32768,65536 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled --compile-dynamic --json`

| Batch | Current two matmuls ms | Fused one matmul ms | Dynamic compiled current ms | Dynamic compiled fused ms |
|---:|---:|---:|---:|---:|
| 512 | 0.097050 | 0.099443 | 0.072538 | 0.069683 |
| 8192 | 0.581837 | 0.576000 | 0.286067 | 0.304614 |
| 16384 | 0.997107 | 0.965069 | 0.455053 | 0.480883 |
| 32768 | 1.883418 | 1.831014 | 0.814170 | 0.912525 |
| 65536 | 3.624218 | 3.533632 | 1.525184 | 1.762662 |

Conclusion: do not wire the fused belief projection. It is a small eager win at larger batches, but the live-style dynamic compiled current path is faster than compiled fused from batch 8192 upward.

## Compile-Specific Inner FFN Norm Cutoff

The within-block fusion of token-mixer residual plus FFN RMSNorm was selected before the stack path was compile-visible. Rechecking under dynamic `torch.compile` showed the compiled graph is faster when that inner FFN RMSNorm fusion is disabled through the 16k range, while eager behavior still slightly prefers the old small-batch path. I kept eager behavior unchanged and disabled only the inner FFN-norm fusion during Dynamo tracing.

Pre-patch mode sweep:

`uv run python benchmarks/bench_preflop_token_mixer_cross_norm.py --batch-sizes 512,8192,16384,32768,65536 --depth 4 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled-wired --compile-dynamic --include-inner-norm-modes --json`

| Batch | Dynamic compiled current ms | Dynamic compiled inner-never ms | Faster mode |
|---:|---:|---:|---|
| 512 | 0.310208 | 0.233216 | inner-never |
| 8192 | 2.314445 | 1.944909 | inner-never |
| 16384 | 4.414797 | 3.691418 | inner-never |
| 32768 | 7.049869 | 7.054221 | current |
| 65536 | 13.854950 | 13.884300 | current |

Post-patch focused check:

`uv run python benchmarks/bench_preflop_token_mixer_cross_norm.py --batch-sizes 512,8192,16384 --depth 4 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled-wired --compile-dynamic --include-inner-norm-modes --json`

| Batch | Old loop ms | Eager wired stack ms | Dynamic compiled wired stack ms | Compiled wired vs old |
|---:|---:|---:|---:|---:|
| 512 | 0.372275 | 0.336986 | 0.289677 | 1.29x |
| 8192 | 2.675187 | 2.409792 | 1.947597 | 1.37x |
| 16384 | 5.134451 | 4.605133 | 3.684288 | 1.39x |

Conclusion: for live `compile=default`, do not trace the inner FFN-norm fused kernel. Let Inductor compile the token residual followed by FFN RMSNorm while keeping the cross-block next-token-RMSNorm fusion.

## Eval Projection Cache And Overall CFR Loop

The compiled static value path still had small constant work inside every eval
call: compact hand embedding, range projection, bucket projection dtype cast,
and value-head hand projection/fused weight construction. I added an eval/no-grad
cache for those tensors, refreshed inside the active autocast context before
compiled eval tracing. Training and grad-enabled paths bypass the cache.

Projection-only command:

`uv run python benchmarks/bench_preflop_belief_projection.py --batch-sizes 512,8192,16384,32768,65536 --iters 80 --warmup 20 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --include-compiled --compile-dynamic --json`

| Batch | Dynamic compiled current ms | Dynamic compiled cached current ms | Speedup |
|---:|---:|---:|---:|
| 512 | 0.072435 | 0.037542 | 1.93x |
| 8192 | 0.247219 | 0.215168 | 1.15x |
| 16384 | 0.455514 | 0.425754 | 1.07x |
| 32768 | 0.814950 | 0.784666 | 1.04x |
| 65536 | 1.527142 | 1.496435 | 1.02x |

The cached fused one-matmul variant still regresses at larger batches, so the
wired path keeps separate range and bucket matmuls.

Production-shaped compiled value-forward command:

`uv run python benchmarks/bench_preflop_eval_projection_cache.py --batch-sizes 512,8192,16384,32768,65536 --iters 50 --warmup 12 --dtype float32 --weight-dtype float32 --autocast --timing-mode cuda_graph --compile-dynamic --json`

| Batch | Cold live constants ms | Warm eval cache ms | Speedup |
|---:|---:|---:|---:|
| 512 | 0.483881 | 0.415334 | 1.17x |
| 8192 | 2.766725 | 2.448445 | 1.13x |
| 16384 | 4.765389 | 4.714865 | 1.01x |
| 32768 | 9.020621 | 8.968540 | 1.01x |
| 65536 | 17.589698 | 17.631150 | 1.00x |

Overall `evaluate_cfr` loop command shape:

`uv run python scripts/bench_preflop_evaluate_cfr_loop.py --cfr-batch-size 512 --cfr-iterations 300 --warmup-solves 1 --out /tmp/preflop_eval_cache_on_cfr_loop.json`

The pre-session baseline was measured from a temporary worktree at
`07ee2ab7` with the same benchmark script override fixes and current artifact
paths.

| Variant | Wall s | ms/iter | Speedup vs pre-session baseline |
|---|---:|---:|---:|
| Pre-session baseline `07ee2ab7` | 8.850995 | 29.503315 | 1.00x |
| Current stack changes, eval cache disabled | 8.596712 | 28.655707 | 1.03x |
| Current stack changes, eval cache enabled | 8.473718 | 28.245726 | 1.04x |

Conclusion: the accumulated session changes produce an end-to-end
`evaluate_cfr` improvement of about 4.45% on the current actions_4_7
512-root/300-iteration benchmark. The eval projection/value-head cache accounts
for about 1.45% of that in this overall loop; most model-forward microbenchmark
wins are diluted by non-model CFR work at this solve shape.

## Production Compile Wiring Check

After inspecting the trainer wiring, the prior overall-loop measurements above
turned out to match the old production behavior: the Hydra config said
`compile=default`, but `RebelCFRTrainer` created the inference twin,
`cfr_target_model`, and fused evaluator model with compile disabled. I changed
that production path so the fused evaluator and inference/target eval models use
the configured compile mode, and disabled TrueSkill for bucketed preflop runs.

The first version of this check had two benchmark problems: `--compile off`
only changed `model.compile`, while `build_run_config()` then overwrote it from
`preflop_buckets.compile=default`; and `--warmup-solves` warmed a discarded
evaluator/model, so compile-on runs still paid first-use Inductor compile inside
the timed evaluator. The benchmark now sends `--compile` through
`preflop_buckets.compile` and warms the same evaluator/model before
reinitializing the subgame for timing.

Corrected current live actions_4_7 command shape:

`uv run python scripts/bench_preflop_evaluate_cfr_loop.py --state-dataset /home/user/poker2/outputs/preflop_policy_states/eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622 --base-checkpoint /home/user/poker2/outputs/preflop_backward_induction/gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5/actions_4_7/checkpoints/specialist_inprogress.pt --closing-checkpoint /home/user/poker2/outputs/preflop_backward_induction/gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5/actions_8_11/checkpoints/specialist_final.pt --run-output-dir /home/user/poker2/outputs/preflop_backward_induction/gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5 --cfr-batch-size 512 --cfr-iterations 300 --warmup-solves 1`

| Variant | Wall s | ms/iter | Speedup vs dynamic compile |
|---|---:|---:|---:|
| Compile default, dynamic | 15.101322 | 50.337741 | 1.00x |
| Compile static | 9.391587 | 31.305291 | 1.61x |
| Compile static repeat | 12.456134 | 41.520446 | 1.21x |
| Compile off | 9.815967 | 32.719890 | 1.54x |

These absolute times are not directly comparable to the earlier 8.47s table
because this check used the restarted live v5 bucket artifacts and current
actions_8_11 closing checkpoint, but the A/B is same-shape and same-artifact:
512 roots, 125,551 total nodes, 65,536 model leaves, and 300 CFR iterations.
Dynamic compile-on is consistently worse for this production path once measured
correctly. Static compile-on removes the dynamic-shape overhead and is the right
compiled default for bucketed preflop; it is much faster than dynamic compile
and roughly competitive with eager compile-off, with a noisy small edge in one
run and a small regression in the repeat.

The static profile still showed two partitioned value-model evals dominating
`set_leaf_values`: 44,300 same-street cutoff leaves and 21,236 closing-model
leaves. I added a side-stream path that evaluates the smaller closing partition
while the main stream evaluates the cutoff partition, then synchronizes before
the existing writeback kernels. It is CUDA-graph compatible in the production
`evaluate_cfr` loop and can be disabled with
`P2_DISABLE_PREFLOP_PARALLEL_PARTITION_EVAL=1`.

| Variant | Wall s | ms/iter |
|---|---:|---:|
| Static compile baseline | 9.391587 | 31.305291 |
| Static compile baseline repeat | 12.456134 | 41.520446 |
| Static + parallel partition eval | 8.722004 | 29.073348 |
| Static + parallel partition eval repeat | 8.706321 | 29.021070 |
| Static + parallel partition eval default-on check | 8.702062 | 29.006874 |

Conclusion: parallel partition eval is the strongest corrected production-loop
number so far on this restarted v5 actions_4_7 shape, bringing the warmed
512-root/300-iteration solve to about 29.0 ms/iter.

## Compile Boundary Result: FFN Residual Plus Next Token RMSNorm

The static/parallel profile still showed the cross-block
`_preflop_ffn_residual_next_token_norm` Triton boundary kernels in the value
stack. The isolated FFN epilogue probe had already shown that, under
`torch.compile`, a normal Torch expression for
`token_out + ffn_out / sqrt(2)` followed by `next_token_norm(out)` can be a
little faster than the custom Triton epilogue. I wired that boundary only while
Dynamo is tracing; eager/no-compile eval still uses the existing Triton path.
The default can be disabled with
`P2_DISABLE_PREFLOP_COMPILED_FFN_BOUNDARY=1`, and the older explicit
`P2_PREFLOP_COMPILED_FFN_BOUNDARY=0/1` override is also honored.

Same corrected production-loop command shape as above, with static compile,
parallel partition eval default-on, 512 roots, 125,551 total nodes, 65,536 model
leaves, and 300 CFR iterations:

| Variant | Wall s | ms/iter |
|---|---:|---:|
| Static + parallel partition eval fresh baseline | 9.786 | 32.619 |
| Compiled FFN boundary env-gated | 8.549 | 28.498 |
| Compiled FFN boundary env-gated repeat | 8.546 | 28.485 |
| Compiled FFN boundary default-on check | 8.553 | 28.509 |

Conclusion: the compiled boundary is the new best corrected production-loop
result on the restarted v5 actions_4_7 shape, improving the warmed
512-root/300-iteration solve from the previous best `29.006874 ms/iter` to
about `28.5 ms/iter`.

## CFR Root Batch Size Result

The next production-loop lever was root batch size for the current
`actions_4_7` bucket. These runs used the same restarted v5 artifacts and the
current static-compiled evaluator path with parallel partition eval and the
compiled FFN boundary. The benchmark was the warmed 300-iteration
`evaluate_cfr` loop; the live trainer was paused only while timing.

| CFR roots | Wall s | ms/iter | Roots/s | Outcome |
|---:|---:|---:|---:|---|
| 512 | 8.542 | 28.474 | 59.94 | Current baseline |
| 1024 | 18.053 | 60.176 | 56.72 | Worse |
| 2048 | 32.672 | 108.907 | 62.68 | Best measured throughput |
| 4096 | n/a | n/a | n/a | OOM during CUDA graph capture |

Conclusion: wire a dedicated `actions_4_7_cfr_batch_size: 2048` override while
leaving the default `cfr_batch_size: 512` for `actions_0_3`. The 4096-root
attempt failed before timed pause/capture completion: graph capture needed
another `442 MiB` when only about `409 MiB` was free with the live trainer
resident.
