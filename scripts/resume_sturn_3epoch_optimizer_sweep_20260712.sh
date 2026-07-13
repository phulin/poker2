#!/usr/bin/env bash
set -euo pipefail

export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export UV_CACHE_DIR=/tmp/uv-cache

common=(
  --hard-validation outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711
  --dataset outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711
  --seed 42
  --matched-validation-examples 0
  --train-cycle-examples 1024000
)

uv run python scripts/run_sturn_pregen_sweep.py \
  cheap_turn_lr40_wsd \
  cheap_turn_lr40_cosine_warmup100 \
  "${common[@]}" \
  --output-root outputs/sturn_3epoch_cheap_optimizer_sweep_bs2048_20260712 \
  --batch-size 2048 \
  --steps 1500

for spec in "1024 3000 lr20" "1024 3000 lr40" "4096 750 lr40" "4096 750 lr80"; do
  read -r batch_size steps lr <<<"${spec}"
  experiment="cheap_turn_${lr}_cosine"
  if [[ "${lr}" == "lr40" ]]; then
    experiment="baseline"
  fi
  uv run python scripts/run_sturn_pregen_sweep.py \
    "${experiment}" \
    "${common[@]}" \
    --output-root "outputs/sturn_3epoch_cheap_batch_sweep_bs${batch_size}_${lr}_20260712" \
    --batch-size "${batch_size}" \
    --steps "${steps}"
done
