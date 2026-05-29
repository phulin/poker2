## Directory summary
TypeScript source for WebGPU CFR solving, BetterFFN inference, browser/Node loading, WGSL kernels, HUNL public-state replay, shared browser helpers, and the Vite/Solid spot solver UI.

### Source files
- `types.ts`: Shared request, fixture, exported manifest, CFR config, solve-progress, and result types.
- `gpu.ts`: Barrel exports for GPU helpers.
- `gpuBuffers.ts`: WebGPU storage/uniform buffer creation and readback helpers.
- `gpuPipeline.ts`: Shared compute-pipeline creation and single-dispatch helpers.
- `kernels.ts`: WGSL kernels for regret matching, belief updates, and action probability reductions.
- `gpuPokerState.ts`: TypeScript wrapper for packed GPU poker state buffers and the poker-state kernels used by browser solving.
- `modelFormat.ts`: Manifest parsing, CFR default resolution, tensor loading, and action-label helpers.
- `cards.ts`: Standard card notation parsing/formatting, duplicate validation, and 1326 hand-combo lookup helpers.
- `beliefs.ts`: Initial belief normalization, public-card beliefs, and blocked-hand mask helpers.
- `modelCache.ts`: Browser IndexedDB model cache, streamed weights download progress, and cache invalidation helpers.
- `allInTableCache.ts`: Browser IndexedDB cache for the preflop all-in table and the most recent flop all-in table shards.
- `allInTables.ts`: Street-local all-in payoff table metadata loading, canonical flop lookup helpers, int16 table packing, and CPU reference value computation for sparse resolver all-in leaves.
- `allInTableGenerator.ts`: WebGPU fallback generation for exact flop/turn all-in payoff tables, with rank-code and payoff-table kernels.
- `betterFfnWebGpuModel.ts`: WebGPU BetterFFN inference implementation, including batched inference with shared, per-sample, or GPU-buffer belief vectors, split policy/value checkpoint loading, and CPU sparse-policy initialization for the newer factorized policy head.
- `hunlEnv.ts`: Browser-safe public HUNL environment, terminal showdown value/rank helpers, and legacy/new BetterFFN feature encoders.
- `evaluator.ts`: Local GPU CFR evaluator for fixtures.
- `browserEvaluator.ts`: Browser-facing evaluator that replays prefixes through the sparse public-tree CFR resolver, aggregates solve-progress callbacks, and returns beliefs/action probabilities for browser, CLI, and benchmark callers.
- `solverWorker.ts`: Browser module-worker entry point that owns the WebGPU device/model/evaluator, loads cached model bytes, runs solves off the UI thread, and posts progress/results back to the app.
- `solverWorkerMessages.ts`: Typed message contract shared between the Solid app and solver worker.
- `sparseResolver.ts`: Arbitrary-depth sparse public-tree CFR resolver that batches nonterminal leaf model evaluation, optionally evaluates regrets/leaves on CFR-average beliefs, uses the WebGPU BetterFFN runtime for policy/value inference, and keeps mutable CFR tensors GPU-resident across WGSL sparse-kernel iterations when a GPU device is available.
- `browser.ts`: Browser device/model loading exports.
- `main.tsx`: Vite/Solid application mount entry point.
- `App.tsx`: Guided browser spot solver UI and result rendering.
- `styles.css`: Solver UI styling.
- `vite-env.d.ts`: Vite client ambient types.
- `nodeGpu.ts`: Node/Dawn GPU device creation.
- `nodeModel.ts`: Node model loading from manifest and weights.
- `pythonBridge.ts`: Python fixture bridge for Node parity tests.
- `cli.ts`: Command-line evaluator entry point.
- `bench.ts`: Command-line benchmark entry point.
- `benchModelKernels.ts`: Dawn/Node microbenchmark harness for individual BetterFFN model WGSL kernels, including shape compatibility checks, generic-reference validation, and repeated-dispatch timing.
- `benchSparseCfrKernels.ts`: Dawn/Node microbenchmark harness for non-model sparse CFR WGSL operations, including synthetic sparse-tree buffers, aggregate-vs-direct overlap validation, and repeated-dispatch timing.
- `benchSpotsInterleaved.ts`: Dawn/Node spot benchmark that alternates baseline and candidate runtime variants in one process to reduce timing noise, with optional baseline-vs-candidate output diff checks.
- `webgpu.d.ts`: WebGPU ambient type declarations.

### Subdirectories
- `modelKernels/`: BetterFFN WebGPU WGSL kernels split into production matvec, generated/specialized variants, normalization, pointwise, belief-feature, and benchmark-only modules.
- `sparseCfr/`: Sparse public-tree CFR WebGPU runtime, tree buffer types, dispatch helpers, runtime flags, and grouped WGSL shader modules.
- `pokerStateKernels/`: GPU HUNL public-state layout constants and WGSL kernels for state transitions, terminal values, and model feature encoding.
