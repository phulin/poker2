# Notes: 8k All-In HP Comparison

## Inputs
- Training data: `outputs/allin_training_data_512k_s65536_b4096_bc64/manifest.json`
  - 512,000 examples, 500 shards.
- Validation data: `outputs/allin_validation_data_4096_s16777216_b2097152_bc64/manifest.json`
  - 4,096 examples, 64 shards.
  - Regenerated with `sample_count=16,777,216`, `board_samples=2,097,152`.
  - `noise_report.json` overall all-entry target-target MSE: `1.0579054409093863e-05`.
  - Overall all-entry noise floor: `5.289527204546932e-06`.

## Candidate Matrix
- `mlp_cos4000_r015`: MLP, LR 0.015, AdamW LR 0.024, cosine floor 0.015, decay steps 4000.
- `mlp_cos8000_r015`: MLP, LR 0.015, AdamW LR 0.024, cosine floor 0.015, decay steps 8000.
- `mlp_linear8000_r015`: MLP, LR 0.015, AdamW LR 0.024, linear floor 0.015, decay steps 8000.
- `xf10_cos4000_r015`: player transformer, 10 layers, LR 0.015, AdamW LR 0.024, cosine floor 0.015, decay steps 4000.
- `xf10_cos8000_r015`: player transformer, 10 layers, LR 0.015, AdamW LR 0.024, cosine floor 0.015, decay steps 8000.
- `xf10_linear8000_r015`: player transformer, 10 layers, LR 0.015, AdamW LR 0.024, linear floor 0.015, decay steps 8000.

## Results
- Pending.
