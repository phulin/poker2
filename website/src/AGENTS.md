## Directory summary
TypeScript source for WebGPU CFR solving, BetterFFN inference, browser/Node loading, WGSL kernels, and the Vite/Solid spot solver UI.

### Source files
- `types.ts`: Shared request, manifest, CFR config, progress, and result types.
- `gpu.ts`: Barrel exports for GPU helpers.
- `gpuBuffers.ts`: WebGPU storage/uniform buffer creation and readback helpers.
- `gpuPipeline.ts`: Shared compute-pipeline creation and single-dispatch helpers.
- `kernels.ts`: WGSL kernels for regret matching, belief updates, and action probability reductions.
- `gpuPokerState.ts`: TypeScript wrapper for packed GPU poker state buffers and the poker-state kernels used by browser solving.
- `modelFormat.ts`: Manifest parsing, tensor loading, and action-label helpers.
- `modelRegistry.ts`: Curriculum street-model registry.
- `cards.ts`: Card notation and hand-combo helpers.
- `beliefs.ts`: Belief normalization and blocked-hand helpers.
- `modelCache.ts`: Browser IndexedDB model cache helpers.
- `allInTableCache.ts`: Browser IndexedDB cache for the preflop all-in table and the most recent flop all-in table shards.
- `allInTables.ts`: All-in payoff table loading and CPU reference helpers.
- `allInTableGenerator.ts`: WebGPU fallback all-in table generation.
- `betterFfnWebGpuModel.ts`: WebGPU BetterFFN inference implementation.
- `hunlEnv.ts`: Browser-safe public HUNL environment and feature encoders.
- `evaluator.ts`: Local GPU CFR evaluator for fixtures.
- `browserEvaluator.ts`: Browser-facing sparse CFR evaluator wrapper.
- `solverWorker.ts`: Browser worker entry point for off-thread solves.
- `solverWorkerMessages.ts`: Typed message contract shared between the Solid app and solver worker.
- `sparseResolver.ts`: Sparse public-tree CFR resolver.
- `browser.ts`: Browser device/model/runtime loading exports for either one BetterFFN manifest or a curriculum model-set manifest.
- `main.tsx`: Vite/Solid application mount entry point.
- `App.tsx`: Guided browser spot solver UI and result rendering.
- `styles.css`: Solver UI styling.
- `vite-env.d.ts`: Vite client ambient types.
- `nodeGpu.ts`: Node/Dawn GPU device creation.
- `nodeModel.ts`: Node model/runtime loading from single manifests or curriculum model-set manifests.
- `pythonBridge.ts`: Python fixture bridge for Node parity tests.
- `cli.ts`: Command-line evaluator entry point.
- `bench.ts`: Command-line benchmark entry point.
- `benchModelKernels.ts`: Dawn/Node benchmark harness for BetterFFN WGSL kernels.
- `benchSparseCfrKernels.ts`: Dawn/Node benchmark harness for sparse CFR WGSL kernels.
- `benchSpots.ts`: Dawn/Node spot benchmark harness.
- `benchSpotsInterleaved.ts`: Interleaved Dawn/Node spot benchmark harness.
- `webgpu.d.ts`: WebGPU ambient type declarations.

### Subdirectories
- `modelKernels/`: BetterFFN WebGPU WGSL kernels.
- `sparseCfr/`: Sparse public-tree CFR WebGPU runtime and shaders.
- `pokerStateKernels/`: GPU HUNL public-state layout and kernels.
