## Directory summary
Shared model interfaces, outputs, activation helpers, and model-family subpackages for CNN, MLP/TRM, and transformer architectures.

### Source files
- `__init__.py`: Package marker.
- `activation_utils.py`: Activation factory and SwiGLU module.
- `base_mlp_model.py`: Abstract base for MLP-style models.
- `model_output.py`: Shared model output dataclasses, including TRM latent state.
- `policy.py`: Categorical policy wrapper around model logits.
- `street_model_registry.py`: Model-like dispatcher that routes MLP feature batches to frozen street-specific postflop nets by `features.street`.

### Subdirectories
- `cnn/`: Convolutional encoders and SiameseConvNet policy/value model.
- `mlp/`: Flat ReBeL/Better feature encoders, feed-forward models, and TRM model.
- `transformer/`: Tokenization, embeddings, attention, heads, KV cache, and PokerTransformer model.
