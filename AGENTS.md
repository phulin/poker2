This is a repository for a neural-network-based bot for heads-up no-limit Texas Hold 'Em. The main work right now is on a ReBeL-style PBS and CFR approach.

## General instructions
- We're using PyTorch for all our learning code. It's very important all our code be performant. That means no for loops over tensors, be intentional about what's on GPU/CPU, and avoid CPU-GPU syncs at all costs.

## Specific instructions
- Use `uv run` to run python scripts using the venv (which is in `.venv`).

## Progressive disclosure
- This codebase uses progressive disclosure. Each directory should have `AGENTS.md` with a brief summary of the directory and its source files and subdirectories. Keep these updated.

## Root summary
This repository trains and evaluates neural-network-based heads-up no-limit Texas Hold 'Em agents. The Python package lives under `src/p2`, with Hydra configs, tests, profiling scripts, and an isolated TypeScript/WebGPU CFR evaluator alongside it.

### Top-level source files
- `README.md`: Project overview, model families, training entry points, and links to additional docs.
- `pyproject.toml`: Python package metadata, dependencies, uv/PyTorch indexes, setuptools package discovery, and tool configuration.
- `uv.lock`: Locked Python dependency graph for `uv`.

### Top-level directories
- `src/`: Python source root for the `p2` package.
- `conf/`: Hydra training and model configuration files.
- `tests/`: Python unit and integration tests.
- `scripts/`: Reusable benchmark and profiling scripts.
- `benchmarks/`: Focused performance microbenchmarks for tensor/CFR kernels.
- `webgpu_cfr/`: TypeScript/WebGPU evaluator and browser demo for exported BetterFFN CFR inference.
- `outputs/`, `wandb/`: Hydra and Weights & Biases run artifacts; do not edit as source.
- `.agents/`, `.codex/`: Local agent/tooling metadata.
