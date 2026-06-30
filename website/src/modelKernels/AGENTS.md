## Directory summary
WGSL source modules for BetterFFN WebGPU inference.

### Source files
- `matVec.ts`: Base dense matvec and leaky-ReLU matvec WGSL kernels used by the runtime.
- `matVecGenerated.ts`: Generated matvec shader variants.
- `norm.ts`: RMS normalization WGSL kernels.
- `pointwise.ts`: Small pointwise utility kernels.
- `beliefFeatures.ts`: Belief and board-interaction feature kernels.
- `benchVariants.ts`: Benchmark-only tiled GEMM experiment kernels.
- `reductions.ts`: Shared WGSL reduction snippets used by generated kernels.
