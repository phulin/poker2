## Directory summary
Flat feature encoders and MLP/TRM model family used heavily by ReBeL-style CFR training.

### Source files
- `__init__.py`: Package exports for ReBeL and Better MLP model classes.
- `better_features.py`: Policy/value feature context enums, chance-phase labels, and layout helpers.
- `better_feature_encoder.py`: BetterFFN policy and street-boundary value feature encoders for richer public-belief inputs.
- `better_ffn.py`: Residual feed-forward Better models, including the legacy combined model plus split `BetterPolicyFFN`, `BetterStreetValueFFN`, and `BetterSplitFFN` wrapper with static feature-base helpers, multiway policy blocker aggregation, and optional low-rank rank/suit board interactions.
- `better_trm.py`: Recursive trunk model with iterative refinement.
- `mlp_features.py`: MLP feature dataclass and indexing/suit-permutation helpers for flattened `P * 1326` belief layouts.
- `rebel_feature_encoder.py`: ReBeL public-belief feature encoder.
- `rebel_ffn.py`: ReBeL feed-forward model and config.

### Subdirectories
There are no child source directories.
