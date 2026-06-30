## Directory summary
Isolated TypeScript/WebGPU CFR evaluator for exported BetterFFN models.

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
- `bench_spots_root.json`: Tracked deterministic benchmark spot set.
- `.gitignore`: WebGPU subproject ignore rules.

### Subdirectories
- `src/`: TypeScript evaluator, kernels, loaders, CLI, benchmarks, and UI source.
- `python/`: Python reference and model export utilities.
- `public/`: Static Vite assets and exported model artifacts.
- `scripts/`: TypeScript deployment helpers for pruning Pages output and uploading model/all-in assets to R2.
- `cloudflare/`: Cloudflare R2/Pages deployment configuration files.
- `tests/`: Node tests comparing WebGPU output with Python references.
- `dist/`: Generated TypeScript build output; do not edit as source.
- `node_modules/`: Installed Node dependencies; do not edit.
