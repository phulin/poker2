#!/usr/bin/env bash
set -euo pipefail

cd /home/user/poker2

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

holdout="outputs/rebel_postflop/turn_holdout_32768_300it_eturn300k_seed777_20260711"
while [[ ! -f "${holdout}/manifest.json" ]]; do
  sleep 30
done

uv run python scripts/generate_paired_sturn_targets.py \
  --output-root outputs/rebel_postflop/_smoke_paired_sturn_512_300it_20260711 \
  --examples 512 \
  --batch-size 512 \
  --iterations 300 \
  --solve-seeds 9001 \
  --repeat-300-seeds 9002

uv run python scripts/generate_paired_sturn_targets.py \
  --output-root outputs/rebel_postflop/paired_sturn_4096_300_1000_5000it_eturn300k_20260711 \
  --examples 4096 \
  --batch-size 512 \
  --iterations 300,1000,5000 \
  --solve-seeds 9001 \
  --repeat-300-seeds 9002

uv run python scripts/run_sturn_pregen_sweep.py \
  turn_blockers \
  turn_equity_feature_head \
  cross_range64 \
  second_moment_blockers \
  turn_pair_direct \
  --hard-validation "${holdout}" \
  --matched-validation-examples 0 \
  --output-root outputs/sturn_pregen_500step_structural_sweep_20260711
