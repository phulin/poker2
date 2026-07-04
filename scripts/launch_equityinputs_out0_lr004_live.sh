#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=/tmp/uv-cache
export HYDRA_FULL_ERROR=1

LR=${LR:-0.04}
LR_FINAL=${LR_FINAL:-0.004}
LR_TAG=${LR_TAG:-lr004}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints-rebel-curriculum-sapcfr-80-40-300it-8000-val-ctx41-live-board96-belief128-r96-equityinputs-out0-${LR_TAG}-random-wandb}
WANDB_NAME=${WANDB_NAME:-river_sapcfr_80_40_300it_8000_val_ctx41_live_board96_belief128_r96_equityinputs_out0_${LR_TAG}_random}

exec uv run python -m p2.cli.train_rebel_curriculum \
  --config-name config_rebel_curriculum_river \
  num_steps=8000 \
  curriculum.substeps.river.num_steps=8000 \
  checkpoint_interval=200 \
  +log_interval=25 \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  curriculum.promote_dir="${CHECKPOINT_DIR}/promoted" \
  use_wandb=true \
  +wandb_name="${WANDB_NAME}" \
  data.mode=live \
  data.live_root_source=random_river \
  data.belief_mode=mixed \
  data.belief_profile=actions_12_end \
  validation_set.enabled=true \
  validation_set.dataset=outputs/rebel_postflop/river_val_8192_10k_sapdcfr_nowarm_ctx41_20260630 \
  validation_set.interval=50 \
  validation_set.batch_size=1024 \
  train.replay_buffer_batches=32 \
  train.batch_size=2048 \
  train.episodes_per_step=5 \
  train.value_reuse_goal=2 \
  train.learning_rate="${LR}" \
  train.learning_rate_final="${LR_FINAL}" \
  train.lr_schedule=cosine \
  train.optimizer=muon \
  train.adamw_learning_rate="${LR}" \
  train.value_coef=1 \
  train.permutation_coef=0.1 \
  train.grad_clip=1 \
  ++train.policy_grad_clip=10 \
  trueskill.enabled=false \
  model.hidden_dim=384 \
  model.ffn_dim=768 \
  model.range_hidden_dim=192 \
  model.num_value_layers=6 \
  model.board_interaction_dim=96 \
  ++model.belief_low_rank_dim=128 \
  model.compile=default \
  ++model.value_river_range_equity_baseline=true \
  ++model.value_river_range_equity_pos_scale=0.8543022528460094 \
  ++model.value_river_range_equity_neg_scale=0.4753640305061305 \
  ++model.value_river_range_equity_intercept=-0.010797645393242563 \
  ++model.value_river_range_equity_blockers=true \
  ++model.value_river_range_equity_rank_bins=96 \
  ++model.value_river_range_equity_feature_head=true \
  ++model.value_river_range_equity_trunk_context=true \
  ++model.value_output_init_scale=0.0 \
  search.depth=5 \
  'search.bet_bins_by_depth=[[0.25,0.5,0.75,1.0,1.5],[0.5,1.0],[1.0],[1.0],[]]' \
  'search.allin_by_depth=[true,true,true,true,false]' \
  search.cfr_type=sapcfr \
  search.iterations=300 \
  search.iterations_final=300 \
  search.sapcfr_alpha=2 \
  search.predictive_cfr_delay=40 \
  search.predictive_cfr_dcfr_hybrid=true \
  search.dcfr_plus_delay=80 \
  search.warm_start_iterations=15 \
  search.warm_start_type=model_br \
  search.warm_start_multiplier=2 \
  search.sparse=true \
  search.sparse_fused=true \
  search.cfr_plus=false \
  search.cfr_avg=false
