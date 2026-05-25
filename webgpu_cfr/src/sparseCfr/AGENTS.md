## Directory summary
Sparse public-tree CFR WebGPU runtime. Shader source is grouped by CFR phase, while runtime code owns tree buffers, pipeline construction, dispatch chunking, and environment-controlled production variant selection.

### Source files
- `SparseCfrGpuKernels.ts`: Public runtime class that owns sparse CFR compute pipelines and encodes dispatches.
- `treeBuffers.ts`: Sparse tree input and GPU buffer interfaces.
- `dispatch.ts`: Dispatch-limit and chunk-size helpers for sparse kernels.
- `flags.ts`: Runtime environment toggles for production kernel variant selection.

### Subdirectories
- `shaders/`: WGSL source grouped into core CFR, opponent/reach weighting, terminal/showdown, and all-in table kernels.
