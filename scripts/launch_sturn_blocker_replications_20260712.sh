#!/usr/bin/env bash
set -euo pipefail

cd /home/user/poker2

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset=outputs/rebel_postflop/sturn_value_500steps_1024000_300it_eturn300k_20260711
validation=outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711

for seed in 43 44; do
  uv run python scripts/run_sturn_pregen_sweep.py \
    baseline \
    turn_blockers \
    second_moment_blockers \
    turn_blockers_refit \
    second_moment_blockers_refit \
    --dataset "${dataset}" \
    --hard-validation "${validation}" \
    --matched-validation-examples 0 \
    --seed "${seed}" \
    --output-root "outputs/sturn_blocker_replication_seed${seed}_20260712"
done
