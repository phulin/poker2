## Directory summary
TypeScript source for WebGPU CFR solving, BetterFFN inference, browser/Node loading, WGSL kernels, and HUNL public-state replay.

### Source files
- `types.ts`: Shared request, fixture, manifest, and result types.
- `gpu.ts`: Barrel exports for GPU helpers.
- `gpuBuffers.ts`: WebGPU storage/uniform buffer creation and readback helpers.
- `kernels.ts`: WGSL kernels for regret matching, belief updates, and action probability reductions.
- `modelKernels.ts`: WGSL kernels for BetterFFN matvec, activations, normalization, residuals, and batching.
- `modelFormat.ts`: Manifest parsing, tensor loading, and action-label helpers.
- `betterFfnWebGpuModel.ts`: WebGPU BetterFFN inference implementation.
- `hunlEnv.ts`: Browser-safe public HUNL environment and feature encoder.
- `evaluator.ts`: Local GPU CFR evaluator for fixtures.
- `browserEvaluator.ts`: Browser-facing evaluator that replays prefixes, builds child values, and solves local CFR.
- `browser.ts`: Browser device/model loading exports.
- `nodeGpu.ts`: Node/Dawn GPU device creation.
- `nodeModel.ts`: Node model loading from manifest and weights.
- `pythonBridge.ts`: Python fixture bridge for Node parity tests.
- `cli.ts`: Command-line evaluator entry point.
- `bench.ts`: Command-line benchmark entry point.
- `webgpu.d.ts`: WebGPU ambient type declarations.

### Subdirectories
There are no child source directories.
