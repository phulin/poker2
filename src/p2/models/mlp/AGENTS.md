## Directory summary
Flat feature encoders and MLP/TRM model family used heavily by ReBeL-style CFR training.

### Source files
- `__init__.py`: Package exports for ReBeL and Better MLP model classes.
- `better_features.py`: Policy/value feature context enums and chance-phase labels.
- `better_feature_encoder.py`: BetterFFN policy and street-boundary value feature encoders.
- `better_ffn.py`: Residual feed-forward BetterFFN model family, compact preflop token mixers, and optional postflop value-architecture branches.
- `better_trm.py`: Recursive trunk model with iterative refinement.
- `mlp_features.py`: MLP feature dataclass and suit-permutation helpers.
- `preflop_token_mixer_graph.py`: CUDA graph helpers for compact preflop token mixers.
- `preflop_token_mixer_mpk.py`: Experimental Mirage/MPK wrapper for the compact preflop token mixer.
- `rebel_feature_encoder.py`: ReBeL public-belief feature encoder.
- `rebel_ffn.py`: ReBeL feed-forward model and config.
- `turn_range_equity.py`: Turn-board range-equity baseline utilities, including reusable board/runout/rank cache construction and belief-dependent equity reduction.

### Subdirectories
There are no child source directories.
