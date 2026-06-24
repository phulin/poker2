# Preflop Value Sweep Report

## Cleanup
- Removed mistaken partial rollout output: `outputs/preflop_policy_states/eroymcd2_unique_buckets_20m_n5_cap5m_20260622_rerun`.

## Source Changes
- Commit `8a2842f`: extended `scripts/sweep_preflop_value_lr_bs.py` with schedule, WSD fraction, AdamW LR, policy-head Muon LR, weight decay, grad clip, value-loss type, and fixed-shuffle options.

## Best Result
- Best 400-CFR one-epoch value run: WSD schedule, final ratio `0.1`, WSD decay fraction `0.5`, Muon LR `0.01`, AdamW LR `0.005`, batch size `256`, Huber loss.
- Validation value loss: `0.0095354964`.

## Key Comparisons
- Original LR/BS best: `0.0138450533`.
- Schedule sweep best with AdamW `0.01`: WSD-50, `0.0099833640`.
- AdamW/schedule sweep best: WSD-50 + AdamW `0.005`, `0.0096445081`.
- Muon LR fine sweep confirmed LR `0.01` as best among `0.004, 0.006, 0.008, 0.01, 0.012, 0.016`.
- Batch sweep found `256` best; larger batches degraded.
- Huber and MSE were identical in this run.
- 500-CFR targets produced no improvement because 400- and 500-CFR emitted targets were bit-identical, including float32 8,192-root diagnostics.
