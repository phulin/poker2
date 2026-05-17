# WebGPU CFR Evaluator

This folder contains an isolated TypeScript/WebGPU harness for evaluating a
BetterFFN-backed local CFR problem. It supports two paths:

- A Node/Dawn parity path that can still ask Python to produce CFR fixtures.
- A browser-safe exported-model path that loads `model.json` and `weights.bin`,
  builds child values with BetterFFN inference in WebGPU, and runs CFR without
  Python at runtime.

## Shape

- `python/export_model.py` converts a PyTorch `BetterFFN` checkpoint into
  `model.json` plus row-major float32 `weights.bin`.
- `python/reference.py` loads a PyTorch `BetterFFN` checkpoint, replays a
  heads-up action-bin sequence, emits model-derived local CFR fixtures, and
  computes the Python reference result.
- `src/betterFfnWebGpuModel.ts` runs the supported `swiglu` BetterFFN family
  with WebGPU dense-vector kernels.
- `src/hunlEnv.ts` ports the single-state public HUNL action-bin environment.
- `src/browserEvaluator.ts` replays prefixes, builds child values, solves local
  CFR, and returns beliefs/action probabilities.
- `src/kernels.ts` contains WGSL kernels for regret matching, regret
  accumulation, average-policy finalization, belief updates, and action
  probability reduction.
- `tests/cfr_parity.test.ts` and `tests/full_evaluator.test.ts` compare Dawn
  output against Python references from `checkpoints-rebel/rebel_latest.pt`.

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

Export a checkpoint for browser or Node runtime use:

```bash
npm run export:model -- --snapshot checkpoints-rebel/rebel_latest.pt --out webgpu_cfr/public/rebel_latest
```

Run the exported-model evaluator directly:

```bash
npm run eval -- --manifest public/rebel_latest/model.json --weights public/rebel_latest/weights.bin --spot 1 --iterations 8
```

Run the legacy Python-fixture evaluator directly:

```bash
npm run eval -- --snapshot checkpoints-rebel/rebel_latest.pt --spot 1 --iterations 8
```

`WEBGPU_BACKEND` defaults to `vulkan`. Set it if you need another Dawn backend.

For the browser demo, run `npm run build`, export the model into
`webgpu_cfr/public/rebel_latest`, and serve `webgpu_cfr/public/index.html` from a
local static server.
