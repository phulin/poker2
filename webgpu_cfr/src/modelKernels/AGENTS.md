## Directory summary
WGSL source modules for BetterFFN WebGPU inference. Production matvec, normalization, pointwise, and belief-feature kernels are separated from benchmark-only variants and generated/specialized shader builders.

### Source files
- `matVec.ts`: Base dense matvec and leaky-ReLU matvec WGSL kernels used by the runtime.
- `matVecGenerated.ts`: Specialized exact-row, fixed-column, subgroup, and residual matvec shader variants generated from the base kernels.
- `norm.ts`: RMS normalization WGSL kernels.
- `pointwise.ts`: Small pointwise utility kernels such as residual add, add3, repeat rows, and zero-sum.
- `beliefFeatures.ts`: Belief and board-interaction feature kernels.
- `benchVariants.ts`: Benchmark-only tiled GEMM experiment kernels.
- `reductions.ts`: Shared WGSL reduction snippets used by generated kernels.
