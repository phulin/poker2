# Notes: Launch S_turn from tq39uf58 Settings

## Sources

### W&B run `tq39uf58`
- Local directory: `wandb/run-20260707_192221-tq39uf58`
- Started: `2026-07-07T19:22:21.349869Z`
- Commit: `05005e9b467630bc16385c5f70bb344b0bfffbcd`
- Program: `-m p2.cli.train_rebel_curriculum`
- Args:
  - `--config-name config_rebel_curriculum_turn`
  - `curriculum.stages=[turn]`
  - `num_steps=5000`
  - `curriculum.substeps.turn.num_steps=5000`
  - `checkpoint_dir=checkpoints-rebel-curriculum-sturn-5k-turnbase-newposneg-initfix-val4096-eturn100k-fp32pair-wandb`
  - `curriculum.promote_dir=checkpoints-rebel-curriculum-sturn-5k-turnbase-newposneg-initfix-val4096-eturn100k-fp32pair-wandb/promoted`
  - `curriculum.substeps.turn.closing_checkpoint=checkpoints-rebel-curriculum-eturn-100k-turneq-posneg-noblockers-lr0p02-linear-b1024-from-3ytaa643-mlp-b96-belief128-wandb/promoted/E_turn.pt`
  - `use_wandb=true`
  - `+wandb_name=sturn-5k-turnbase-newposneg-initfix-val4096-eturn100k-fp32pair`
  - `wandb_tags=[rebel,cfr,postflop-curriculum,turn,S_turn,turneq,newposneg,initfix,fp32pair,expandable-segments,val4096]`
  - `+validation_set={enabled:true,dataset:outputs/rebel_postflop/turn_val_4096_5kit_eturn100k_allincutoff_fp32pair_v2_20260707,interval:50,batch_size:1024,max_examples:null}`
  - `+search.allin_call_terminal_abstraction=true`
  - `search.sparse=true`
  - `search.sparse_fused=true`

### New distilled E_turn candidates
- Completed/promoted 300k checkpoint:
  - `checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-wandb-20260708/t001_lr0p01_300000st_b1024/promoted/E_turn.pt`
  - Timestamp: `2026-07-09 21:01:34 UTC`
- Interrupted plus50k continuation:
  - `checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-plus50k-lr0p001-wandb-20260709/distill_E_turn/rebel_latest.pt`
  - Latest saved step: `320200`
  - W&B run `owqc9mq4` ended with `KeyboardInterrupt` around step `320258`, no promoted `E_turn.pt`.

## Synthesized Findings

- Use the completed/promoted 300k `E_turn.pt` for the new S_turn run; it is the newest fully promoted distilled E_turn artifact.
- Keep `tq39uf58` settings otherwise, including validation set and search flags.
- Use unique output names with `eturn300k` to avoid clobbering `tq39uf58`.
- First non-tmux/background launch did not survive.
- First tmux attempt `59k2vrnb` failed because Triton could not find `libcuda.so.1`; `nvidia-smi` worked but `ldconfig` did not list `libcuda`.
- Relaunched with `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`.

### Launched run
- tmux session: `sturn_eturn300k_v2`
- W&B run id: `j9k53hh3`
- W&B URL: `https://wandb.ai/phulin-self/poker-rebel-postflop-curriculum/runs/j9k53hh3`
- Checkpoint dir: `checkpoints-rebel-curriculum-sturn-5k-turnbase-newposneg-initfix-val4096-eturn300k-fp32pair-v2-wandb`
- Output dir: `outputs/2026-07-10/02-02-12`
- Tmux log: `outputs/training_logs/sturn_5k_turnbase_newposneg_initfix_val4096_eturn300k_fp32pair_v2_20260710.tmux.log`
- Step 0 completed:
  - `loss=0.0137`
  - `policy=0.00768`
  - `value=0.00598`
  - `exploit=0.01307`
  - `exploit_mbbg=1927.23`
