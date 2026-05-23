## Directory summary
CFR/DCFR search, public-belief state handling, ReBeL data generation, evaluator orchestration, chance-node logic, and fused Triton kernels.

### Source files
- `cfr_evaluator.py`: PublicBeliefState, exploitability stats, hand-rank data, evaluator interface, and shared split policy/value model dispatch.
- `allin_payoff.py`: Preflop/flop/turn all-in call payoff table generation, lookup, eager references, and fused Triton writeback kernels.
- `dcfr.py`: Standalone DCFR utilities and regret matching.
- `cfr_manager.py`: High-level CFRManager orchestration.
- `chance_node_helper.py`: Chance-node expansion and board/deck helper logic, including post-chance value-head evaluation for street-value targets.
- `rebel_data_generator.py`: ReBeL public-belief training data generator.
- `sparse_cfr_evaluator.py`: Sparse CFR evaluator implementation used as the reference path, including split policy/value encoder setup.
- `fused_sparse_cfr_evaluator.py`: Sparse evaluator with fused operations, persistent-buffer subgame construction, graph-friendly paths, BetterFFN static leaf-feature caching, split value-model leaf evaluation, combined leaf-belief gather paths, and fused all-in terminal payoff writeback.
- `fused_cfr_triton.py`: Triton kernels and graph runner utilities for fused DCFR updates, reach/belief/average-policy propagation, showdown EV, and related reductions.
- `subgame_constructor_triton.py`: Triton kernels for fused sparse same-street subgame expansion, child-row construction, constructor masks, root hand legality, and init-buffer writes.

### Subdirectories
There are no child source directories.
