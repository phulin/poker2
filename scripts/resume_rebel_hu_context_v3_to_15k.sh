#!/usr/bin/env bash
set -euo pipefail
cd /home/user/poker2
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=/tmp/uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Continues checkpoints-rebel-hu-context-v3 (wandb hu-context-v3-refold-fix) from
# its finished 10k steps out to 15k, writing to a fresh checkpoint_dir so the 10k
# artifacts stay intact. resume_from restores model/optimizer/RNG plus the replay
# buffers next to it (checkpoints-rebel-hu-context-v3/rebel_replay_buffers.pt).
#
# LRs: the schedule is cosine over num_steps with t = step/num_steps, and the
# AdamW group is always learning_rate_start-relative (adamw_lr = adamw_start *
# lr_now/lr_start). The 10k run ended at muon 1e-3 / adamw 2e-4. Resuming at
# t = 10000/15000 = 2/3, cosine gives lr_final + 0.25*(lr_start - lr_final), so
# scaling every configured LR by 0.34 makes step 10000 land exactly on
# 1e-3 / 2e-4 and step 15000 on 2e-4 / 4e-5 (1/5 of the resume point).
uv run python -m p2.cli.train_rebel \
  train.optimizer=muon \
  train.episodes_per_step=40 \
  train.batch_size=2048 \
  train.value_reuse_goal=2 \
  num_envs=512 \
  train.learning_rate=3.4e-3 \
  train.learning_rate_final=2e-4 \
  num_steps=15000 \
  trueskill.snapshot_frac=0.02 \
  model.num_hidden_layers=3 \
  model.num_value_layers=3 \
  model.num_policy_layers=5 \
  train.adamw_learning_rate=6.8e-4 \
  model.hidden_dim=512 \
  search.value_targets_from_final_policy=false \
  search.iterations=300 \
  search.iterations_final=300 \
  train.policy_loss_type=mse \
  'search.depth=5' \
  'search.allin_by_depth=[true,true,true,true,false]' \
  'search.bet_bins_by_depth=[[0.25,0.5,0.75,1,1.5],[0.5,1],[1],[1],[]]' \
  'search.warm_start_type=model' \
  'search.warm_start_iterations=10' \
  'search.warm_start_multiplier=10' \
  'search.dcfr_beta_final=-2' \
  'model.ffn_dim=1024' \
  'model.policy_rank=128' \
  'model.range_hidden_dim=256' \
  'model.policy_hand_bias_rank=32' \
  'train.policy_capacity_factor=10' \
  'train.replay_buffer_batches=16' \
  'train.policy_logit_l2_coef=1e-6' \
  checkpoint_dir=checkpoints-rebel-hu-context-v3-to15k \
  +resume_from=checkpoints-rebel-hu-context-v3/rebel_latest.pt
