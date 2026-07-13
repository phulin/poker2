#!/usr/bin/env bash
set -euo pipefail

cd /home/user/poker2

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run python scripts/run_sturn_pregen_sweep.py \
  turn_equity_input_blockers \
  turn_equity_input_blockers_second \
  turn_equity_input_blockers_refit \
  --dataset outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711 \
  --hard-validation outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711 \
  --matched-validation-examples 0 \
  --seed 42 \
  --output-root outputs/sturn_equity_input_500step_sweep_20260712
