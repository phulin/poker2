#!/usr/bin/env bash
set -euo pipefail
cd /home/user/poker2
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=/tmp/uv-cache

# Reproduces wandb run o6vbd6ax (poker-rebel-cfr), num_steps 5000 -> 10000.
# First block: the original 18 explicit overrides from o6vbd6ax's launch.
# Second block: recoverable defaults that drifted since 2026-05-29 (commit 61cba4ef),
#   pinned back to o6vbd6ax's values. Includes solve-to-end-of-street search
#   (depth 5; allin_by_depth ends in false; bet_bins_by_depth ends in []).
# NOTE: some old model architecture params were removed in a refactor and cannot
#   be pinned via overrides, so the network is not byte-identical to o6vbd6ax.
uv run python -m p2.cli.train_rebel \
  train.optimizer=muon \
  train.episodes_per_step=40 \
  train.batch_size=2048 \
  train.value_reuse_goal=2 \
  num_envs=512 \
  train.learning_rate=1e-2 \
  train.learning_rate_final=1e-3 \
  num_steps=10000 \
  trueskill.snapshot_frac=0.02 \
  model.num_hidden_layers=3 \
  model.num_value_layers=3 \
  model.num_policy_layers=5 \
  train.adamw_learning_rate=0.002 \
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
  'train.policy_logit_l2_coef=1e-6'
