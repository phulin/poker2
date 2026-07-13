# Result: Launch S_turn from tq39uf58 Settings

Started the new 5k-step S_turn run in tmux.

- tmux session: `sturn_eturn300k_v2`
- W&B run id: `j9k53hh3`
- W&B URL: `https://wandb.ai/phulin-self/poker-rebel-postflop-curriculum/runs/j9k53hh3`
- Checkpoint dir: `checkpoints-rebel-curriculum-sturn-5k-turnbase-newposneg-initfix-val4096-eturn300k-fp32pair-v2-wandb`
- Closing checkpoint: `checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-wandb-20260708/t001_lr0p01_300000st_b1024/promoted/E_turn.pt`
- Tmux log: `outputs/training_logs/sturn_5k_turnbase_newposneg_initfix_val4096_eturn300k_fp32pair_v2_20260710.tmux.log`

Step 0 completed successfully:

```text
[Step 00000/5000] loss=0.0137 policy=0.00768 value=0.00598 exploit=0.01307 exploit_mbbg=1927.23 street=2.0000 time=195.93s total=3.3m
```

Note: failed first W&B/tmux attempt `59k2vrnb` was caused by Triton not finding `libcuda.so.1`. The active `v2` run sets `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`.
