## Directory summary
Flat feature encoders and MLP/TRM model family used heavily by ReBeL-style CFR training.

### Source files
- `__init__.py`: Package marker.
- `better_features.py`: Feature context enums and layout helpers.
- `better_feature_encoder.py`: BetterFFN feature encoder for richer public-belief inputs.
- `better_ffn.py`: Residual feed-forward model for policy/value prediction, including a static feature-base helper used by fused CFR leaf evaluation and optional low-rank rank/suit board interaction features.
- `better_trm.py`: Recursive trunk model with iterative refinement.
- `mlp_features.py`: MLP feature dataclass and indexing helpers.
- `rebel_feature_encoder.py`: ReBeL public-belief feature encoder.
- `rebel_ffn.py`: ReBeL feed-forward model and config.

### Subdirectories
There are no child source directories.
