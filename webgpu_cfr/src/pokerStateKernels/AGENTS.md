## Directory summary
WGSL modules and layout constants for GPU-resident HUNL public-state replay, legal-action generation, terminal values, and BetterFFN feature encoding.

### Source files
- `layout.ts`: Packed state offsets and shared WGSL state helper functions.
- `stateMachine.ts`: Legal-mask, step, child-state, compaction, and value-scatter kernels.
- `terminal.ts`: Terminal rank and payoff-value kernels.
- `modelFeatures.ts`: Public-state to BetterFFN feature encoding kernel.
