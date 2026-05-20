# WebGPU CFR Evaluator

This folder contains an isolated TypeScript/WebGPU harness for evaluating a
BetterFFN-backed local CFR problem. It supports two paths:

- A Node/Dawn parity path that can still ask Python to produce CFR fixtures.
- A browser-safe exported-model path that loads `model.json` and `weights.bin`,
  builds child values with BetterFFN inference in WebGPU, and runs CFR without
  Python at runtime.
- A Vite/Solid spot-solver UI that caches the exported model in IndexedDB,
  guides hero-card/board/action entry, and runs local WebGPU CFR solves.

## Shape

- `python/export_model.py` converts a PyTorch `BetterFFN` checkpoint into
  `model.json` plus row-major float32 `weights.bin`.
- `python/reference.py` loads a PyTorch `BetterFFN` checkpoint, replays a
  heads-up action-bin sequence, emits model-derived local CFR fixtures, and
  computes the Python reference result.
- `src/betterFfnWebGpuModel.ts` runs the supported `leaky_relu` BetterFFN family
  with WebGPU dense-vector kernels.
- `src/hunlEnv.ts` ports the single-state public HUNL action-bin environment.
- `src/browserEvaluator.ts` replays prefixes, builds child values, solves local
  CFR, and returns beliefs/action probabilities. It also has a sparse
  arbitrary-depth resolver path enabled by `depth > 1`.
- `src/cards.ts`, `src/beliefs.ts`, and `src/modelCache.ts` provide browser-safe
  card parsing, hero-only belief initialization, blocked combo masks, streamed
  model download progress, and IndexedDB cache invalidation.
- `src/App.tsx` and `src/main.tsx` implement the Vite/Solid spot-solver UI.
- `src/kernels.ts` contains WGSL kernels for regret matching, regret
  accumulation, average-policy finalization, belief updates, and action
  probability reduction.
- `tests/cfr_parity.test.ts` and `tests/full_evaluator.test.ts` compare Dawn
  output against Python references from `checkpoints-rebel/rebel_latest.pt`.
- `tests/cards_beliefs_cache.test.ts` covers card parsing, combo lookup,
  hero-only beliefs, and model cache invalidation.

## Commands

Install dependencies:

```bash
npm install
```

Run checks from this folder:

```bash
npm run build
npm run typecheck
npm test
```

Run the browser spot solver in development:

```bash
npm run dev
```

Export a checkpoint for browser or Node runtime use:

```bash
npm run export:model
```

This writes:

```text
webgpu_cfr/public/models/rebel_latest/model.json
webgpu_cfr/public/models/rebel_latest/weights.bin
```

For custom snapshots or output paths, call the exporter directly from the repo
root:

```bash
uv run python webgpu_cfr/python/export_model.py --snapshot checkpoints-rebel/rebel_latest.pt --out webgpu_cfr/public/models/rebel_latest
```

Run the exported-model evaluator directly:

```bash
npm run eval -- --manifest public/models/rebel_latest/model.json --weights public/models/rebel_latest/weights.bin --spot 1 --iterations 8 --depth 2
```

Benchmark the exported-model evaluator on Node/Dawn:

```bash
npm run bench -- --manifest public/models/rebel_latest/model.json --weights public/models/rebel_latest/weights.bin --spot 1 --iterations 8 --depth 2 --warmups 1 --runs 5
```

Run the legacy Python-fixture evaluator directly:

```bash
npm run eval -- --snapshot checkpoints-rebel/rebel_latest.pt --spot 1 --iterations 8
```

`WEBGPU_BACKEND` defaults to `vulkan`. Set it if you need another Dawn backend.

For the browser spot solver, export the model into
`webgpu_cfr/public/models/rebel_latest` and run `npm run dev` or
`npm run build && npm run preview`. The app loads `/models/rebel_latest/model.json`
and `/models/rebel_latest/weights.bin`, caching the weights in IndexedDB until the
manifest weight hash or byte length changes.

The same build/export flow also enables `webgpu_cfr/public/benchmark.html`,
which times repeated browser evaluations with configurable spots, iterations,
warmups, and run counts.
