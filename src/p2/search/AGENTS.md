## Directory summary
CFR/DCFR search, public-belief state handling, ReBeL data generation, evaluator orchestration, chance-node logic, and fused Triton kernels.

### Source files
- `allin_payoff.py`: All-in call payoff lookup and writeback helpers.
- `cfr_evaluator.py`: Shared CFR evaluator types, interfaces, and training-data export logic.
- `cfr_manager.py`: High-level CFRManager orchestration.
- `chance_node_helper.py`: Chance-node expansion and chance-target evaluation helpers.
- `dcfr.py`: Standalone DCFR utilities and regret matching.
- `end_of_street_distillation.py`: End-of-street value distillation batch builder.
- `fused_cfr_triton.py`: Triton kernels and graph utilities for fused CFR.
- `fused_preflop_sparse_cfr_evaluator.py`: CUDA/Triton compact preflop sparse CFR evaluator.
- `fused_sparse_cfr_evaluator.py`: Fused sparse CFR evaluator implementation.
- `postflop_spot_sampler.py`: Postflop public-root sampler utilities.
- `preflop_belief_sampler.py`: Preflop public-belief sampler utilities.
- `preflop_live_pair_distillation.py`: Live-pair E-preflop distillation batch builder.
- `preflop_sparse_cfr_evaluator.py`: Compact preflop sparse CFR evaluator.
- `rebel_data_generator.py`: ReBeL public-belief training data generator.
- `rebel_data_source.py`: Data-source boundary for ReBeL trainer batches, including live, pregenerated, hybrid holdout, and finite pregenerated-bootstrap streams.
- `rebel_solved_dataset.py`: Tensor-only solved-example dataset reader/writer.
- `sparse_cfr_evaluator.py`: Reference sparse CFR evaluator implementation.
- `subgame_constructor_triton.py`: Triton kernels for sparse subgame construction.

### Subdirectories
There are no child source directories.
