## Directory summary
Static model-asset root served by Vite at `/models/`.

### Source files
There are no hand-authored source files here beyond this directory note.

### Subdirectories
- `rebel_latest/`: Generated single-model `model.json` and compressed fp16 `weights.bin.gz` from `npm run export:model`; treat as artifacts.
- `curriculum/`: Postflop curriculum `model_set.json` descriptor. Generate the referenced `S_flop/`, `S_turn/`, and `S_river/` stage artifacts with `website/python/export_model.py --curriculum`; treat generated stage directories as artifacts.
