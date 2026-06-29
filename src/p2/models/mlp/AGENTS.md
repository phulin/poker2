## Directory summary
Flat feature encoders and MLP/TRM model family used heavily by ReBeL-style CFR training.

### Source files
- `__init__.py`: Package exports for ReBeL and Better MLP model classes.
- `better_features.py`: Policy/value feature context enums and chance-phase labels.
- `better_feature_encoder.py`: BetterFFN policy and street-boundary value feature encoders for richer public-belief inputs, including compact 169-hand preflop encoders whose value features keep same-street betting context.
- `better_ffn.py`: Residual feed-forward Better models, including the legacy combined model, compact 169-hand preflop policy/value variants, compact preflop game/player-token transformer and gated-token-mixer variants with optional learned slot-moment range pooling, conditional Triton eval fast paths for compact gated-token-mixer residual/next-RMSNorm stages, a compile-visible stack-level eval runner that fuses cross-block FFN residuals with the next token RMSNorm, a compiled Torch boundary for FFN residual-plus-next-token-RMSNorm under Dynamo, and branch-specific policy/value token stacks, plus split `BetterPolicyFFN`, `BetterStreetValueFFN`, and `BetterSplitFFN` wrapper with explicit policy/value hand-dimension validation, static feature-base helpers, multiway policy blocker aggregation, and optional low-rank rank/suit board interactions.
- `better_trm.py`: Recursive trunk model with iterative refinement.
- `mlp_features.py`: MLP feature dataclass and indexing/suit-permutation helpers for flattened `P * hand_dim` belief layouts, defaulting to 1326 and supporting compact 169 preflop features.
- `preflop_token_mixer_mpk.py`: Experimental Mirage/MPK wrapper for the staged compact preflop gated token mixer, import-safe without Mirage and expecting a custom MPK task for the token mixer plus gated residual.
- `rebel_feature_encoder.py`: ReBeL public-belief feature encoder.
- `rebel_ffn.py`: ReBeL feed-forward model and config.

### Subdirectories
There are no child source directories.
