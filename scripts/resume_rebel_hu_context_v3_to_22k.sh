#!/usr/bin/env bash
set -euo pipefail
cd /home/user/poker2
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=/tmp/uv-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Continues checkpoints-rebel-hu-context-v3-to15k from its finished 15k steps out
# to 22k, writing to a fresh checkpoint_dir so the 15k artifacts stay intact.
# resume_from restores model/optimizer/RNG plus the replay buffers next to it
# (checkpoints-rebel-hu-context-v3-to15k/rebel_replay_buffers.pt).
#
# LRs: the schedule is cosine over num_steps with t = step/num_steps, and the
# AdamW group is always learning_rate-relative (adamw_lr = adamw_start *
# lr_now/lr_start), so the adamw/muon ratio of 0.2 is preserved by construction.
#
# The 15k run ended at muon 2e-4 / adamw 4e-5 (cosine at t = 1). This run must
# pick up at exactly that LR and finish at half of it, so with num_steps=22000
# the resume point sits at t = 15000/22000 rather than at the top of the curve.
# Solving lr_final + 0.5*(lr_start - lr_final)*(1 + cos(pi*t)) = 2e-4 with
# lr_final = 1e-4 gives lr_start = 5.35389e-4. That lands step 15000 on exactly
# 2e-4 / 4e-5 and step 22000 on 1e-4 / 2e-5.
#
# Checkpoints: every 500 steps, and economize_checkpoints=false so every one of
# them is kept. The whole point is to end this run with a real ladder of
# snapshots -- the v3 lineage kept only its last few steps, which is why there
# was nothing early enough to evaluate against.
uv run python -m p2.cli.train_rebel \
  train.optimizer=muon \
  train.episodes_per_step=40 \
  train.batch_size=2048 \
  train.value_reuse_goal=2 \
  num_envs=512 \
  train.learning_rate=5.35389e-4 \
  train.learning_rate_final=1e-4 \
  num_steps=22000 \
  trueskill.snapshot_frac=0.02 \
  model.num_hidden_layers=3 \
  model.num_value_layers=3 \
  model.num_policy_layers=5 \
  train.adamw_learning_rate=1.07078e-4 \
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
  checkpoint_interval=500 \
  economize_checkpoints=false \
  checkpoint_dir=checkpoints-rebel-hu-context-v3-to22k \
  +resume_from=checkpoints-rebel-hu-context-v3-to15k/rebel_latest.pt
