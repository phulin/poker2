# 6-Player All-In LR Sweep Report

## Manifest
- Created: `outputs/allin_training_data_512k_s65536_b4096_bc64/manifest_players6_existing.json`
- Snapshot: 231 shards, 236,544 examples, players=6, hands=1326.
- Train split: `manifest_players6_train200.json`, 204,800 examples.
- Validation split: `manifest_players6_val31.json`, 31,744 examples.

## Sweep Setup
- Steps: 1,000
- Batch size: 512
- Players: 6
- Model: hidden_dim=1024, hand_dim=512, layers=6, film_rank=64
- Eval interval: 250
- W&B disabled
- Compile enabled

## Results
### Cosine
| candidate | lr | adamw lr | cosine steps | floor ratio | eval mse | eval mae | max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lr015_cos500_r015` | 0.015 | 0.0240 | 500 | 0.015 | 0.000231 | 0.00898 | 0.18398 |
| `lr015_cos1000_r015` | 0.015 | 0.0240 | 1000 | 0.015 | 0.000226 | 0.00883 | 0.18805 |
| `lr012_cos1000_r015` | 0.012 | 0.0192 | 1000 | 0.015 | 0.000226 | 0.00884 | 0.18832 |
| `lr010_cos2000_r03` | 0.010 | 0.0160 | 2000 | 0.030 | 0.000307 | 0.01084 | 0.18703 |
| `lr008_cos4000_r05` | 0.008 | 0.0128 | 4000 | 0.050 | 0.000348 | 0.01174 | 0.22027 |

### Linear
| candidate | lr | adamw lr | linear steps | floor ratio | eval mse | eval mae | max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lr015_linear500_r015` | 0.015 | 0.0240 | 500 | 0.015 | 0.000230 | 0.00896 | 0.18410 |
| `lr015_linear1000_r015` | 0.015 | 0.0240 | 1000 | 0.015 | 0.000226 | 0.00884 | 0.19035 |
| `lr015_linear2000_r015` | 0.015 | 0.0240 | 2000 | 0.015 | 0.000304 | 0.01057 | 0.19187 |

### Stable Warmdown
All rows use `lr_decay=stable_warmdown`, `lr_warmdown_start_step=800`, and train for 1,000 steps.

| candidate | initial lr | final lr | adamw lr | eval mse | eval mae | max abs |
|---|---:|---:|---:|---:|---:|---:|
| `lr012_wd800_final00018` | 0.012 | 0.00018 | 0.0192 | 0.000232 | 0.00902 | 0.18741 |
| `lr015_wd800_final000225` | 0.015 | 0.000225 | 0.0240 | 0.000232 | 0.00902 | 0.18576 |
| `lr015_wd800_final00045` | 0.015 | 0.00045 | 0.0240 | 0.000232 | 0.00902 | 0.18589 |
| `lr018_wd800_final00027` | 0.018 | 0.00027 | 0.0288 | 0.000231 | 0.00899 | 0.18073 |
| `lr018_wd800_final00054` | 0.018 | 0.00054 | 0.0288 | 0.000231 | 0.00900 | 0.18529 |

### 2,000-Step Cosine
| candidate | lr | adamw lr | cosine steps | floor ratio | step 1000 mse | step 1000 mae | step 2000 mse | step 2000 mae | max abs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lr015_cos1000_r015_steps2000` | 0.015 | 0.0240 | 1000 | 0.015 | 0.000226 | 0.00883 | 0.000224 | 0.00877 | 0.18292 |
| `lr015_cos2000_r015_steps2000` | 0.015 | 0.0240 | 2000 | 0.015 | 0.000292 | 0.01054 | 0.000223 | 0.00874 | 0.18066 |

## Recommendation
Use Muon LR `0.015`, AdamW LR `0.024`, cosine floor ratio `0.015`, and cosine decay over `1000` steps for this 1,000-step bs=512 6-player run shape.

The lower-LR extended schedules underfit over 1,000 steps. Extending the current LR's cosine decay from 500 to 1000 steps is a small but consistent improvement at final validation.

Linear decay does not improve the recommendation. Linear 1000 ties cosine 1000 on rounded MSE, but cosine 1000 has slightly better MAE. Linear 2000 is too slow for this budget.

Stable warmdown from step 800 does not improve the 1,000-step result. The best warmdown row is still behind cosine 1000 on MAE.

For a 2,000-step budget, cosine 2000 is slightly best at the final eval, but cosine 1000 is much better at step 1000 and nearly tied by step 2000. Use cosine 2000 only if the run will actually go to 2,000 steps.
