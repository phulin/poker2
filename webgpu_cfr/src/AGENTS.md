## Directory summary
TypeScript source for WebGPU CFR solving, BetterFFN inference, browser/Node loading, WGSL kernels, HUNL public-state replay, shared browser helpers, and the Vite/Solid spot solver UI.

### Source files
- `types.ts`: Shared request, fixture, exported manifest, CFR config, solve-progress, and result types.
- `gpu.ts`: Barrel exports for GPU helpers.
- `gpuBuffers.ts`: WebGPU storage/uniform buffer creation and readback helpers.
- `kernels.ts`: WGSL kernels for regret matching, belief updates, and action probability reductions.
- `sparseCfrKernels.ts`: WGSL sparse-tree CFR kernels and buffer helpers for regret matching, per-depth belief/reach propagation, opponent-conditioned policy and regret-weight construction, node belief/value gather-scatter, terminal showdown values, value backup, average-policy accumulation, and regret accumulation.
- `modelKernels.ts`: WGSL kernels for BetterFFN matvec, activations, normalization, residuals, and batching.
- `pokerStateKernels.ts`: WGSL kernels and state-layout constants for GPU-resident HUNL public state, legal masks, stepping, child-state construction, terminal values, and state feature encoding.
- `gpuPokerState.ts`: TypeScript wrapper for packed GPU poker state buffers and the poker-state kernels used by browser solving.
- `modelFormat.ts`: Manifest parsing, CFR default resolution, tensor loading, and action-label helpers.
- `cards.ts`: Standard card notation parsing/formatting, duplicate validation, and 1326 hand-combo lookup helpers.
- `beliefs.ts`: Initial belief normalization, hero-only exact-hand beliefs, and blocked-hand mask helpers.
- `modelCache.ts`: Browser IndexedDB model cache, streamed weights download progress, and cache invalidation helpers.
- `allInTables.ts`: Street-local all-in payoff table metadata loading, canonical flop lookup helpers, int16 table packing, and CPU reference value computation for sparse resolver all-in leaves.
- `betterFfnWebGpuModel.ts`: WebGPU BetterFFN inference implementation, including batched inference with shared, per-sample, or GPU-buffer belief vectors, split policy/value checkpoint loading, and CPU sparse-policy initialization for the newer factorized policy head.
- `hunlEnv.ts`: Browser-safe public HUNL environment, terminal showdown value/rank helpers, and legacy/new BetterFFN feature encoders.
- `evaluator.ts`: Local GPU CFR evaluator for fixtures.
- `browserEvaluator.ts`: Browser-facing evaluator that replays prefixes through the sparse public-tree CFR resolver, aggregates solve-progress callbacks, and returns beliefs/action probabilities for browser, CLI, and benchmark callers.
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
- `benchSpotsInterleaved.ts`: Dawn/Node spot benchmark that alternates baseline and candidate runtime variants in one process to reduce timing noise.
- `webgpu.d.ts`: WebGPU ambient type declarations.

### Subdirectories
There are no child source directories.
