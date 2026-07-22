#!/usr/bin/env bash
set -euo pipefail

cd /home/user/poker2

export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=/tmp/uv-cache

uv run python scripts/run_sturn_pregen_sweep.py \
  no_teb_prod_lr2_cosine \
  no_teb_prod_lr4_cosine \
  no_teb_prod_lr8_cosine \
  no_teb_cold_out0p00 \
  no_teb_cold_out0p03 \
  no_teb_cold_out0p10 \
  no_teb_cold_out0p30 \
  --hard-validation outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711 \
  --dataset outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711 \
  --seed 42 \
  --matched-validation-examples 0 \
  --train-cycle-examples 1024000 \
  --output-root outputs/sturn_3epoch_no_teb_production_lr_sweep_bs2048_20260713 \
  --batch-size 2048 \
  --steps 1500
