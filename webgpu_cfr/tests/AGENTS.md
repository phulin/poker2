## Directory summary
Node test suite for WebGPU CFR parity, full exported-model evaluator behavior, and shared card/belief/cache helpers.

### Source files
- `cfr_parity.test.ts`: Compares low-level GPU CFR fixture output against Python reference data.
- `full_evaluator.test.ts`: Exports/loads a model and checks full evaluator results against Python references.
- `split_checkpoint_fixture.test.ts`: Loads the committed `rebel_296_4000` split checkpoint/export fixture, checks selected WebGPU policy/value outputs against PyTorch, and guards the root `AsKd` sparse-solve policy.
- `cards_beliefs_cache.test.ts`: Checks card parsing, hand-combo lookup, public-card belief masking, and model-cache invalidation logic.
- `sparse_cfr_kernels.test.ts`: Checks WGSL sparse-tree kernels for regret matching, belief/reach propagation, average-policy updates, value backup, all-in table values, and regret accumulation.
- `sparse_resolver.test.ts`: Checks arbitrary-depth sparse resolving, all-in table leaf handling, and CPU/GPU parity for the WGSL-backed sparse resolver path.

### Subdirectories
- `fixtures/`: Large model/checkpoint fixtures used by focused parity tests.
