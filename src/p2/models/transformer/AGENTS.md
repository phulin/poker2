## Directory summary
Transformer model family for sequence-based poker state modeling, including token builders, structured embeddings, custom attention, and heads.

### Source files
- `__init__.py`: Package marker.
- `tokens.py`: Special, context, game, card, and action token ID layout.
- `token_sequence_builder.py`: Builds transformer input sequences from tensor environment state.
- `structured_embedding_data.py`: Structured embedding dataclass for transformer inputs.
- `embeddings.py`: Fused poker embeddings and embedding-combination helpers.
- `attention.py`: Poker attention and transformer encoder layer.
- `rotary_attention.py`: Rotary positional embedding helpers and self-attention.
- `orthogonal_linear.py`: Orthogonal linear layers used by transformer components.
- `heads.py`: Policy, value, quantile value, and hand-range heads.
- `poker_transformer.py`: PokerTransformerV1 model and layer stack.
- `kv_cache_manager.py`: KV cache managers for self-play inference.
- `debug_utils.py`: Transformer state debugging and inspection utilities.

### Subdirectories
There are no child source directories.
