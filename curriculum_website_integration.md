# Curriculum Website Integration

## Implemented
- Added `model_set.json` schema support for postflop curriculum registries.
- Added browser and Node runtime loaders that preserve single-model APIs and add model-set loading.
- Added a validated `BetterFfnModelRegistry` for flop/turn/river model dispatch.
- Updated sparse CFR CPU and GPU leaf evaluation to batch leaves by selected street model and scatter values back to existing node buffers.
- Added Python exporter `--curriculum` mode for promoted `S_flop`, `S_turn`, and `S_river` checkpoints.
- Updated R2 upload tooling to detect/stage curriculum model sets and publish `models/curriculum_latest/model_set.json` with `--latest`.
- Updated app status metadata, docs, AGENTS notes, and a tracked local curriculum descriptor.

## Verification
- `npm run typecheck`
- `node --test --test-concurrency=1 --import tsx tests/cards_beliefs_cache.test.ts`
- `node --test --test-concurrency=1 --import tsx tests/sparse_resolver.test.ts`

## Notes
- Full `npm test` needs a writable `UV_CACHE_DIR` in this sandbox because Python fixture tests invoke `uv`.
