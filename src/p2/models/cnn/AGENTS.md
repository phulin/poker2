## Directory summary
CNN model family for policy/value self-play using card and action tensor planes.

### Source files
- `__init__.py`: Package marker.
- `cards_encoder.py`: Converts card state into plane features.
- `actions_encoder.py`: Encodes heads-up action history into tensor features.
- `cnn_embedding_data.py`: Dataclass container for CNN embeddings.
- `state_encoder.py`: Combines environment state into CNN-ready embeddings.
- `siamese_convnet.py`: SiameseConvNetV1 trunks and policy/value heads.

### Subdirectories
There are no child source directories.
