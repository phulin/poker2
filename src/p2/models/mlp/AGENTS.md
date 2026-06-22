## Directory summary
Flat feature encoders and MLP/TRM model family used heavily by ReBeL-style CFR training.

### Source files
- `__init__.py`: Package exports for ReBeL and Better MLP model classes.
- `better_features.py`: Policy/value feature context enums, chance-phase labels, current and legacy context layout helpers.
- `better_feature_encoder.py`: BetterFFN policy and street-boundary value feature encoders for richer public-belief inputs, including compact 169-hand preflop encoders whose value features keep same-street betting context and an old-checkpoint legacy context mode.
- `better_ffn.py`: Residual feed-forward Better models, including the legacy combined model, compact 169-hand preflop policy/value variants, compact preflop game/player-token transformer variants with branch-specific policy/value token stacks, plus split `BetterPolicyFFN`, `BetterStreetValueFFN`, and `BetterSplitFFN` wrapper with explicit policy/value hand-dimension validation, static feature-base helpers, multiway policy blocker aggregation, optional legacy context-width compatibility, and optional low-rank rank/suit board interactions.
- `better_trm.py`: Recursive trunk model with iterative refinement.
- `mlp_features.py`: MLP feature dataclass and indexing/suit-permutation helpers for flattened `P * hand_dim` belief layouts, defaulting to 1326 and supporting compact 169 preflop features.
- `rebel_feature_encoder.py`: ReBeL public-belief feature encoder.
- `rebel_ffn.py`: ReBeL feed-forward model and config.

### Subdirectories
There are no child source directories.
