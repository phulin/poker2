## Directory summary
CFR/DCFR search, public-belief state handling, ReBeL data generation, evaluator orchestration, chance-node logic, and fused Triton kernels.

### Source files
- `cfr_evaluator.py`: PublicBeliefState, exploitability stats, hand-rank data, and evaluator interface.
- `dcfr.py`: Standalone DCFR utilities and regret matching.
- `cfr_manager.py`: High-level CFRManager orchestration.
- `chance_node_helper.py`: Chance-node expansion and board/deck helper logic.
- `rebel_data_generator.py`: ReBeL public-belief training data generator.
- `rebel_cfr_evaluator.py`: ReBeL-specific CFR evaluator.
- `sparse_cfr_evaluator.py`: Sparse CFR evaluator implementation.
- `fused_sparse_cfr_evaluator.py`: Sparse evaluator with fused operations, graph-friendly paths, BetterFFN static leaf-feature caching, and combined leaf-belief gather paths.
- `fused_cfr_triton.py`: Triton kernels and graph runner utilities for fused DCFR updates, reach/belief/average-policy propagation, showdown EV, and related reductions.

### Subdirectories
There are no child source directories.
