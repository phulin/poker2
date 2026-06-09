# WebGPU CFR Evaluator

This folder contains an isolated TypeScript/WebGPU harness for evaluating a
BetterFFN-backed local CFR problem. It supports:

- A Node/Dawn parity path that can still ask Python to produce CFR fixtures.
- A browser-safe exported-model path that loads either one `model.json` or a
  curriculum `model_set.json` with flop/turn/river BetterFFN exports, evaluates
  sparse public-tree leaves with the selected WebGPU model, and runs CFR without
  Python at runtime.
- A Vite/Solid spot-solver UI that caches the exported model in IndexedDB,
  guides hero-card/board/action entry, and runs local WebGPU CFR solves.

## Shape

- `python/export_model.py` converts a PyTorch `BetterFFN` checkpoint into
  `model.json` plus gzip-compressed row-major fp16 `weights.bin.gz`; its
  `--curriculum` mode writes `S_flop/`, `S_turn/`, `S_river/`, and a
  `model_set.json` street registry.
- `python/reference.py` loads a PyTorch `BetterFFN` checkpoint, replays a
  heads-up action-bin sequence, emits model-derived local CFR fixtures, and
  computes the Python reference result.
- `src/betterFfnWebGpuModel.ts` runs the supported `leaky_relu` BetterFFN family
  with WebGPU dense-vector kernels.
- `src/hunlEnv.ts` ports the single-state public HUNL action-bin environment.
- `src/browserEvaluator.ts` replays prefixes through the sparse public-tree CFR
  resolver and returns beliefs/action probabilities. Sparse CFR tensor updates
  run through WGSL kernels while model leaf evaluation is routed through either
  the single BetterFFN runtime or the curriculum street registry.
- `src/cards.ts`, `src/beliefs.ts`, and `src/modelCache.ts` provide browser-safe
  card parsing, public-card belief initialization, blocked combo masks, streamed
  model download progress, and IndexedDB cache invalidation.
- `src/App.tsx` and `src/main.tsx` implement the Vite/Solid spot-solver UI.
- `src/modelKernels/`, `src/sparseCfr/`, and `src/pokerStateKernels/` contain
  grouped WGSL modules for BetterFFN inference, sparse public-tree CFR, and
  GPU-resident HUNL public-state replay.
- `tests/cfr_parity.test.ts` and `tests/full_evaluator.test.ts` compare Dawn
  output against Python references from `checkpoints-rebel/rebel_latest.pt`.
- `tests/cards_beliefs_cache.test.ts` covers card parsing, combo lookup,
  public-card beliefs, and model cache invalidation.

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
website/public/models/rebel_latest/model.json
website/public/models/rebel_latest/weights.bin.gz
```

For custom snapshots or output paths, call the exporter directly from the repo
root:

```bash
uv run python website/python/export_model.py --snapshot checkpoints-rebel/rebel_latest.pt --out website/public/models/rebel_latest
```

Export promoted postflop curriculum checkpoints:

```bash
uv run python website/python/export_model.py \
  --curriculum \
  --flop-snapshot checkpoints-rebel/S_flop.pt \
  --turn-snapshot checkpoints-rebel/S_turn.pt \
  --river-snapshot checkpoints-rebel/S_river.pt \
  --out website/public/models/curriculum
```

This writes `website/public/models/curriculum/model_set.json` plus one existing
BetterFFN export per stage. `VITE_MODEL_MANIFEST_URL` may point to either
`.../model.json` or `.../model_set.json`. Curriculum model sets are postflop
only: flop, turn, and river public states are supported; preflop handoff is not
part of this website path.

Generate optional WebGPU all-in payoff assets. These are model-independent and
served from `/allin`, while the exported model manifest embeds references to
that shared asset root:

```bash
uv run python website/python/precompute_allin_assets.py \
  --out website/public/allin \
  --device cuda \
  --batch-size 1
uv run python website/python/export_model.py \
  --snapshot checkpoints-rebel/rebel_latest.pt \
  --out website/public/models/rebel_latest \
  --allin-manifest website/public/allin/allin_manifest.json
```

The flop asset generator writes canonical int16 table shards plus lookup files.
The browser loads only the one canonical flop table needed for the current
street-local solve.

Run the exported-model evaluator directly:

```bash
npm run eval -- --manifest public/models/rebel_latest/model.json --spot 1
```

Benchmark the exported-model evaluator on Node/Dawn:

```bash
npm run bench -- --manifest public/models/rebel_latest/model.json --spot 1 --warmups 1 --runs 5
```

Exported manifests include the checkpoint `search:` CFR configuration. The
browser, CLI, and benchmark default to the exported `depth`, scheduled
`iterations`, and `cfr_avg` values; pass `--iterations`, `--depth`,
`--cfr-avg`, or `--no-cfr-avg` to override them for experiments.

Run the legacy Python-fixture evaluator directly:

```bash
npm run eval -- --snapshot checkpoints-rebel/rebel_latest.pt --spot 1 --iterations 8
```

`WEBGPU_BACKEND` defaults to `vulkan`. Set it if you need another Dawn backend.

For the browser spot solver, export a single model into
`website/public/models/rebel_latest` or a curriculum set into
`website/public/models/curriculum`, then run `npm run dev` or
`npm run build && npm run preview`. The default local app still loads
`/models/rebel_latest/model.json`; set
`VITE_MODEL_MANIFEST_URL=/models/curriculum/model_set.json` to load the
curriculum registry. Each stage manifest and weights blob gets its own
IndexedDB cache key.

Upload assets to R2:

```bash
npm run upload:assets -- --source-model public/models/rebel_latest --latest
npm run upload:assets -- --source-model-set public/models/curriculum/model_set.json --latest
```

The curriculum upload stages all three model manifests and weights under the
immutable `models/<version>/` prefix. With `--latest`, it also publishes
`models/curriculum_latest/model_set.json`.

The same build/export flow also enables `website/public/benchmark.html`,
which times repeated browser evaluations with configurable spots, CFR settings,
warmups, and run counts.
