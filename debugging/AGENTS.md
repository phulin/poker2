## Directory summary
Debugging, repro, inspection, and profiling scripts for investigating training, CFR, tensor environments, and model behavior.

### Source files
- `bench_better_ffn_step_hotpath.py`: Benchmarks BetterFFN forward and chance-node value hot paths.
- `bench_rebel_train_step.py`: Benchmarks compiled ReBeL trainer `train_step` timing with run-like overrides and logging/eval side effects disabled.
- `bench_envs.py`: Environment benchmark script.
- `debug_checkpoint_size.py`: Checkpoint size inspection.
- `debug_data_generation.py`: ReBeL data generation debugging.
- `debug_embedding_norms.py`: Embedding norm diagnostics.
- `debug_rebel_preflop.py`: ReBeL preflop debugging.
- `downsample_trace.py`: Trace downsampling helper.
- `inspect_transformer_sequence.py`: Transformer token/sequence inspection.
- `print_preflop_grids.py`: Prints preflop range grids.
- `profile_cfr_trainer.py`: CFR trainer profiling.
- `profile_sparse_methods.py`: Sparse method profiling.
- `repro_sparse_repeat_interleave.py`: Sparse repeat-interleave repro.

### Subdirectories
There are no child source directories.
