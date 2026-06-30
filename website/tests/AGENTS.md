## Directory summary
Node test suite for WebGPU CFR parity, full exported-model evaluator behavior, and shared card/belief/cache helpers.

### Source files
- `all_in_tables.test.ts`: Checks manifest-backed all-in table provider behavior for partial street asset metadata.
- `cfr_parity.test.ts`: Compares low-level GPU CFR fixture output against Python reference data.
- `full_evaluator.test.ts`: Exports/loads a model and checks full evaluator results against Python references.
- `cards_beliefs_cache.test.ts`: Checks card, belief, and cache helpers.
- `sparse_cfr_kernels.test.ts`: Checks WGSL sparse-tree kernels.
- `sparse_resolver.test.ts`: Checks the WGSL-backed sparse resolver path.
- `split_checkpoint_fixture.test.ts`: Checks the committed split-checkpoint WebGPU fixture.

### Subdirectories
- `fixtures/`: Large model/checkpoint fixtures used by focused parity tests.
