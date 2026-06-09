## Directory summary
Isolated TypeScript/WebGPU CFR evaluator for exported BetterFFN models, with Node/Dawn parity tests, a Vite/Solid browser spot solver, and static benchmark assets.

### Source files
- `README.md`: WebGPU evaluator overview, single-model and curriculum model-set shapes, commands, and export/deploy flow.
- `package.json`: Node scripts and dependencies for build, typecheck, tests, export, eval, and benchmark commands.
- `package-lock.json`: Locked Node dependency graph.
- `biome.json`: Biome formatter/linter configuration for TypeScript, TSX, JSON, and deployment scripts.
- `tsconfig.json`: TypeScript compiler configuration.
- `tsconfig.app.json`: Vite/Solid typecheck configuration for TSX browser app files.
- `vite.config.ts`: Solid plugin and app build output configuration.
- `.env.production.example`: Example public Vite production asset manifest URL for a deployed single model or curriculum model set on R2.
- `index.html`: Vite application HTML entry point for the spot solver.
- `bench_spots_root.json`: Tracked deterministic root-PBS benchmark spot set for the Node/Dawn spot benchmark harnesses. `bench_spots.json` is an ignored generated spot sample when extracted locally.
- `.gitignore`: WebGPU subproject ignore rules.

### Subdirectories
- `src/`: TypeScript evaluator, WebGPU buffer helpers, WGSL kernels, browser/Node single-model and curriculum loaders, HUNL env port, card/belief/cache helpers, Solid app files, and CLI/benchmark entry points.
- `python/`: Python reference and model export utilities that bridge PyTorch checkpoints to the WebGPU runtime.
- `public/`: Static Vite assets, including benchmark HTML and exported model artifacts under `models/`.
- `scripts/`: TypeScript deployment helpers for pruning Pages output and uploading model/all-in assets to R2.
- `cloudflare/`: Cloudflare R2/Pages deployment configuration files.
- `tests/`: Node tests comparing WebGPU output with Python references.
- `dist/`: Generated TypeScript build output; do not edit as source.
- `node_modules/`: Installed Node dependencies; do not edit.
