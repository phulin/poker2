## Directory summary
Isolated TypeScript/WebGPU CFR evaluator for exported BetterFFN models, with Node/Dawn parity tests, a Vite/Solid browser spot solver, and static benchmark assets.

### Source files
- `README.md`: WebGPU evaluator overview, shape, commands, and export flow.
- `package.json`: Node scripts and dependencies for build, typecheck, tests, export, eval, and benchmark commands.
- `package-lock.json`: Locked Node dependency graph.
- `tsconfig.json`: TypeScript compiler configuration.
- `tsconfig.app.json`: Vite/Solid typecheck configuration for TSX browser app files.
- `vite.config.ts`: Solid plugin and app build output configuration.
- `index.html`: Vite application HTML entry point for the spot solver.
- `.gitignore`: WebGPU subproject ignore rules.

### Subdirectories
- `src/`: TypeScript evaluator, WebGPU buffer helpers, WGSL kernels, browser/Node loaders, HUNL env port, card/belief/cache helpers, Solid app files, and CLI/benchmark entry points.
- `python/`: Python reference and model export utilities that bridge PyTorch checkpoints to the WebGPU runtime.
- `public/`: Static Vite assets, including benchmark HTML and exported model artifacts under `models/`.
- `tests/`: Node tests comparing WebGPU output with Python references.
- `dist/`: Generated TypeScript build output; do not edit as source.
- `node_modules/`: Installed Node dependencies; do not edit.
