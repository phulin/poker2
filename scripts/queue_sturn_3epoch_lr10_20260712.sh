#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t sturn_3ep_cheap 2>/dev/null; do
  sleep 30
done

cd /home/user/poker2
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=/tmp/uv-cache

uv run python scripts/run_sturn_pregen_sweep.py \
  cheap_turn_lr10_cosine \
  --hard-validation outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711 \
  --dataset outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711 \
  --output-root outputs/sturn_3epoch_cheap_optimizer_sweep_bs2048_20260712 \
  --seed 42 \
  --matched-validation-examples 0 \
  --train-cycle-examples 1024000 \
  --batch-size 2048 \
  --steps 1500
