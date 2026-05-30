# Notes: 6-Player All-In LR Sweep

## Shards
- Directory: `outputs/allin_training_data_512k_s65536_b4096_bc64`
- Sidecar manifest: `outputs/allin_training_data_512k_s65536_b4096_bc64/manifest_players6_existing.json`
- Captured complete shards: 231 (`shard_000000.pt` through `shard_000230.pt`)
- Examples: 236,544
- Tensor shape per shard: `beliefs` and `allin_values` are `[1024, 6, 1326]`
- Manifest config inferred from directory name and shard contents:
  - `players=6`
  - `sample_count=65536`
  - `board_samples=4096`
  - `board_chunk=64`
  - `tuple_tries=4`
- Train split manifest: `manifest_players6_train200.json`, 200 shards, 204,800 examples
- Validation split manifest: `manifest_players6_val31.json`, 31 shards, 31,744 examples

## Smoke Tests
- `uv run python -m p2.allin.train players=6 batch_size=512 steps=5 ... compile_model=false`
- Loaded manifest and trained successfully.
- Model size: 37,713,921 trainable parameters.
- First step included load/warmup: 1.58s.
- Subsequent step times observed: 0.18-0.35s with logging every step and no compile.

## Sweep Results
- Training-speed fix before final sweep:
  - `p2.allin.train` intended to call `torch.set_float32_matmul_precision("high")`, but compared `torch.device("cuda")` to `"cuda"`.
  - Patched to `device.type == "cuda"` before collecting final sweep results.
- Candidate matrix:
  - `lr015_cos500_r015`: Muon LR 0.015, AdamW LR 0.024, cosine floor 0.015, decay over 500 steps.
  - `lr015_cos1000_r015`: Muon LR 0.015, AdamW LR 0.024, cosine floor 0.015, decay over 1000 steps.
  - `lr012_cos1000_r015`: Muon LR 0.012, AdamW LR 0.0192, cosine floor 0.015, decay over 1000 steps.
  - `lr010_cos2000_r03`: Muon LR 0.010, AdamW LR 0.016, cosine floor 0.030, decay over 2000 steps.
  - `lr008_cos4000_r05`: Muon LR 0.008, AdamW LR 0.0128, cosine floor 0.050, decay over 4000 steps.
- Logs will be written under `allin_lr_sweep/logs/`.
- Completed 1,000-step sweep results:
  - `lr015_cos500_r015`: final eval MSE 0.000231, MAE 0.00898, max_abs 0.18398.
  - `lr015_cos1000_r015`: final eval MSE 0.000226, MAE 0.00883, max_abs 0.18805.
  - `lr012_cos1000_r015`: final eval MSE 0.000226, MAE 0.00884, max_abs 0.18832.
  - `lr010_cos2000_r03`: final eval MSE 0.000307, MAE 0.01084, max_abs 0.18703.
  - `lr008_cos4000_r05`: final eval MSE 0.000348, MAE 0.01174, max_abs 0.22027.
- Best final MSE/MAE: `lr015_cos1000_r015`.
- `lr012_cos1000_r015` is effectively tied on MSE but slightly worse on MAE.
- Extended schedules beyond the run length underfit within 1,000 steps for the tested lower LRs.

## Linear Decay Results
- Candidate matrix:
  - `lr015_linear500_r015`: Muon LR 0.015, AdamW LR 0.024, linear floor 0.015, decay over 500 steps.
  - `lr015_linear1000_r015`: Muon LR 0.015, AdamW LR 0.024, linear floor 0.015, decay over 1000 steps.
  - `lr015_linear2000_r015`: Muon LR 0.015, AdamW LR 0.024, linear floor 0.015, decay over 2000 steps.
- Completed linear sweep results:
  - `lr015_linear500_r015`: final eval MSE 0.000230, MAE 0.00896, max_abs 0.18410.
  - `lr015_linear1000_r015`: final eval MSE 0.000226, MAE 0.00884, max_abs 0.19035.
  - `lr015_linear2000_r015`: final eval MSE 0.000304, MAE 0.01057, max_abs 0.19187.
- Linear 1000 ties cosine 1000 on rounded MSE but is slightly worse on MAE.
- Linear 2000 underfits over the 1,000-step budget.

## Stable Warmdown Results
- Candidate matrix:
  - `lr012_wd800_final00018`: initial Muon LR 0.012, final Muon LR 0.00018, AdamW LR 0.0192.
  - `lr015_wd800_final000225`: initial Muon LR 0.015, final Muon LR 0.000225, AdamW LR 0.024.
  - `lr015_wd800_final00045`: initial Muon LR 0.015, final Muon LR 0.00045, AdamW LR 0.024.
  - `lr018_wd800_final00027`: initial Muon LR 0.018, final Muon LR 0.00027, AdamW LR 0.0288.
  - `lr018_wd800_final00054`: initial Muon LR 0.018, final Muon LR 0.00054, AdamW LR 0.0288.
- All hold initial LR through step 799 and cosine-warmdown from step 800 to step 1000.
- Completed stable warmdown results:
  - `lr012_wd800_final00018`: final eval MSE 0.000232, MAE 0.00902, max_abs 0.18741.
  - `lr015_wd800_final000225`: final eval MSE 0.000232, MAE 0.00902, max_abs 0.18576.
  - `lr015_wd800_final00045`: final eval MSE 0.000232, MAE 0.00902, max_abs 0.18589.
  - `lr018_wd800_final00027`: final eval MSE 0.000231, MAE 0.00899, max_abs 0.18073.
  - `lr018_wd800_final00054`: final eval MSE 0.000231, MAE 0.00900, max_abs 0.18529.
- Best warmdown did not beat cosine 1000 from the 1,000-step sweep.

## 2,000-Step Cosine Results
- Candidate matrix:
  - `lr015_cos1000_r015_steps2000`: LR 0.015, AdamW LR 0.024, floor 0.015, decay over 1000 steps, train for 2000 steps.
  - `lr015_cos2000_r015_steps2000`: LR 0.015, AdamW LR 0.024, floor 0.015, decay over 2000 steps, train for 2000 steps.
- Completed 2,000-step cosine results:
  - `lr015_cos1000_r015_steps2000`: step 1000 eval MSE 0.000226, MAE 0.00883; step 2000 eval MSE 0.000224, MAE 0.00877.
  - `lr015_cos2000_r015_steps2000`: step 1000 eval MSE 0.000292, MAE 0.01054; step 2000 eval MSE 0.000223, MAE 0.00874.
- Cosine 2000 is slightly best at step 2000 but much worse at step 1000.
