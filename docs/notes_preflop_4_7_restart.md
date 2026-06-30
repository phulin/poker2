# Notes: Preflop 4-7 Restart Monitor

## Active Run
- Host process tree:
  - `16503` fish wrapper
  - `16506` `uv run`
  - `16509` Python trainer
  - `16567` wandb core
  - `16582` wandb gpu stats
  - `16741` torch compile worker
- Active log:
  `outputs/preflop_backward_induction/gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5/logs/resume_20260628_231918_from1200_fixed_hydra.log`
- Original command includes old LR overrides:
  `train.learning_rate=0.00105 train.learning_rate_final=0.0001575 train.adamw_learning_rate=0.000875 train.muon_compile_step=false preflop_buckets.train_batch_size=256 preflop_buckets.policy_train_batch_size=null`.
- Latest observed 8-11 progress at discovery: `bucket_step=2417`, near the expected end around `ceil(5,000,000 / 2048) = 2442`.

## Replacement Settings
- `train.learning_rate = 0.006`
- `train.learning_rate_final = 0.0009`
- `train.lr_schedule = wsd`
- `train.lr_wsd_decay_fraction = 0.6`
- `train.adamw_learning_rate = 0.004`

## Monitoring Log
- 2026-06-29: Identified active run and confirmed it is still in `actions_8_11`.
- Replacement should set:
  - `preflop_buckets.train_bucket=actions_4_7`
  - `preflop_buckets.base_checkpoint=.../actions_8_11/checkpoints/specialist_final.pt`
  - `preflop_buckets.output_dir=outputs/preflop_backward_induction/gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5`
  - keep `preflop_buckets.train_batch_size=256`
  - keep `preflop_buckets.policy_train_batch_size=null`
  - use new Hydra defaults for LR unless explicit overrides are needed: `0.006`, `0.0009`, AdamW `0.004`, WSD `0.6`.
- 2026-06-29 03:39 UTC: active run completed `actions_8_11` and saved `actions_8_11/checkpoints/specialist_final.pt`.
- 2026-06-29 03:42 UTC: detected `actions_4_7/checkpoints/resolved_config.json`; inherited run had entered actions_4_7.
- 2026-06-29 03:42 UTC: sent SIGTERM to inherited trainer process group `16503`; process table confirmed stopped.
- 2026-06-29 03:45 UTC: first detached launch via `nohup ... &` failed to leave a surviving process and created an empty log.
- 2026-06-29 03:46 UTC: replacement launched successfully via `setsid -f`.
  - `uv` PID: `27817`
  - trainer PID: `27823`
  - log: `outputs/preflop_backward_induction/gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5/logs/restart_4_7_20260629_0346_lr006_adamw004.log`
  - W&B run: `qbb17d1a`
- Resolved config check confirmed `actions_4_7`, base checkpoint `actions_8_11/checkpoints/specialist_final.pt`, LR `0.006`, final `0.0009`, WSD `0.6`, AdamW `0.004`, `muon_compile_step=false`, train batch `256`, policy batch `null`.
- 2026-06-29 03:56 UTC interval 1:
  - Process alive: `uv` 27817, trainer 27823.
  - Log still has only W&B startup lines.
  - Health counters show real work, not a dead process: trainer CPU ~99.6%, GPU util 100%, GPU memory ~13.4GB.
  - Interpretation: likely still in quiet initial setup/validation before first printed `actions_4_7` progress.
- 2026-06-29 04:07 UTC interval 2:
  - Process alive at elapsed ~21:52.
  - Log still has only W&B startup lines.
  - Trainer CPU ~99.2%, GPU util 100%, GPU memory ~11.8GB.
  - Interpretation unchanged: active work before first printed progress; no exception seen.
- 2026-06-29 04:18 UTC interval 3:
  - Process alive at elapsed ~32:51.
  - Log still has only W&B startup lines.
  - Trainer CPU ~99.0%, GPU util 100%, GPU memory ~11.7GB.
  - `py-spy dump --pid 27823 --native` showed main thread in `cudaGraphLaunch` from:
    `_build_validation_cache -> _solve_public_state_batch -> evaluate_cfr -> CUDAGraph.replay`.
  - Interpretation: not stuck; running initial validation cache CFR work before first training progress line.
- 2026-06-29 04:29 UTC interval 4:
  - Process alive at elapsed ~43:37.
  - Log still has only W&B startup lines.
  - Trainer CPU ~99.0%, GPU util 100%, GPU memory ~12.1GB.
  - Interpretation unchanged: slow initial validation, no exception.
- 2026-06-29 04:39 UTC interval 5:
  - Process alive at elapsed ~54:12.
  - Initial validation completed and wrote:
    `outputs/preflop_backward_induction/validation_cache/actions_4_7/validation_n4096_cfr10000_f37c529fa6a02ea9.pt`
  - Validation solved `roots=4,096 value=4,096 policy=161,947`.
  - Log warning: RMSNorm input bf16 vs weight float prevents fused implementation. Non-fatal; same model path can still train.
  - GPU snapshot at the instant of check was lower (23%) because validation had just completed or phase changed.
- 2026-06-29 04:50 UTC interval 6:
  - Process alive at elapsed ~1:04:36.
  - GPU util 100%, GPU memory ~16.9GB.
  - First actual training progress printed:
    `actions_4_7: progress ... roots=10,240 ... step=20 bucket_step=20 bucket_train_step=20 ... nodes=124,264 ... roots/s=15.62`.
  - Interpretation: replacement run has gotten through validation and is training.
- 2026-06-29 05:00 UTC interval 7:
  - Process alive at elapsed ~1:15:00.
  - Progress advanced to `roots=20,480`, `bucket_step=40`, `bucket_train_step=40`, `roots/s=18.83`.
  - Saved in-progress checkpoint at `bucket_step=50`, `roots=25,600`.
  - GPU snapshot at instant was 0%, likely phase/checkpoint timing; no error in log.
- 2026-06-29 05:11 UTC interval 8:
  - Process alive at elapsed ~1:25:23.
  - GPU util 100%, GPU memory ~15.2GB.
  - Progress advanced to `roots=30,208`, `bucket_step=59`, `bucket_train_step=59`, `roots/s=18.59`.
  - No errors in log.
- 2026-06-29 05:21 UTC interval 9:
  - Process alive at elapsed ~1:35:45.
  - GPU util 100%, GPU memory ~16.8GB.
  - Progress advanced through:
    - `roots=40,448`, `bucket_step=79`, `roots/s=18.17`
    - `roots=50,176`, `bucket_step=98`, `roots/s=19.13`
  - No errors in log.
- 2026-06-29 05:32 UTC interval 10:
  - Process alive at elapsed ~1:46:12.
  - Progress advanced to `roots=60,416`, `bucket_step=118`, `bucket_train_step=118`, `roots/s=17.77`.
  - Saved in-progress checkpoint at `bucket_step=100`, `roots=51,200`.
  - No errors in log. GPU snapshot at instant was 0%, likely phase/checkpoint timing.
