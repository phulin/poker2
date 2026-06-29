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
