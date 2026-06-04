## Directory summary
CFR/DCFR search, public-belief state handling, ReBeL data generation, evaluator orchestration, chance-node logic, and fused Triton kernels.

### Source files
- `cfr_evaluator.py`: PublicBeliefState, exploitability stats, hand-rank data, evaluator interface, shared split policy/value model dispatch, optional closing-leaf value model routing, sparse warm-start prior/regret extraction helpers, sparse-only delayed DCFR-hybrid PCFR/SAPCFR predictive regret extraction, and per-root leaf target-source provenance.
- `allin_payoff.py`: Preflop/flop/turn all-in call payoff table generation, lookup, eager references, and fused Triton writeback kernels.
- `dcfr.py`: Standalone DCFR utilities and regret matching.
- `end_of_street_distillation.py`: Builds value-only end-of-street distillation batches by evaluating frozen next-street value nets through chance-node target helpers.
- `cfr_manager.py`: High-level CFRManager orchestration.
- `chance_node_helper.py`: Chance-node expansion and board/deck helper logic, including post-chance value-head evaluation for street-value targets.
- `postflop_spot_sampler.py`: Random legal postflop public-root samplers for heads-up flop/turn/river street-start and legal-prefix roots, closed-street chance-target roots with randomized pre-chance beliefs, texture-stratified boards, tensorized strength-ordered and named-shape board-legal belief mixtures, and conservative pot/SPR templates.
- `rebel_data_generator.py`: ReBeL public-belief training data generator with optional legacy pre-chance value augmentation, replay-buffer insertion, continuation sampling, and batched diagnostic syncing.
- `rebel_data_source.py`: Data-source boundary for ReBeL trainer batches, wrapping live CFR generation, bounded pregenerated solved datasets, or hybrid live-training-plus-holdout metrics through replay-buffer or direct dataset sampling with step-windowed dataset selection, optional shuffled row sampling, and checkpointed cursor plus manifest-state validation.
- `rebel_solved_dataset.py`: Tensor-only bounded solved-example dataset streaming writer/reader for postflop value and policy `RebelBatch` shards with manifest validation, street/depth/target-source/root-source and leaf-source coverage metadata, normalized included-street names, wrapped sampling, optional compressed float storage, and optional pinned/async shard prefetch.
- `sparse_cfr_evaluator.py`: Sparse CFR evaluator implementation used as the reference path, including split policy/value encoder setup, configurable warm-start prior controls, PCFR/SAPCFR enablement, and initial PBSEnv-backed multiway tree construction.
- `preflop_sparse_cfr_evaluator.py`: Guarded sparse CFR boundary for the `S_preflop`/`S_0` arbitrary preflop public-state model, requiring PBSEnv street-0 roots and generation-time continuation-depth value sampling.
- `fused_sparse_cfr_evaluator.py`: Sparse evaluator with fused operations, persistent-buffer subgame construction, graph-friendly paths, BetterFFN static leaf-feature caching, split value-model leaf evaluation, combined leaf-belief gather paths, fused DCFR-hybrid SAPCFR policy extraction, and fused all-in terminal payoff writeback.
- `fused_cfr_triton.py`: Triton kernels and graph runner utilities for fused DCFR and SAPDCFR regret updates, reach/belief/average-policy propagation, showdown EV, and related reductions.
- `subgame_constructor_triton.py`: Triton kernels for fused sparse same-street subgame expansion, child-row construction, constructor masks, root hand legality, and init-buffer writes.

### Subdirectories
There are no child source directories.
