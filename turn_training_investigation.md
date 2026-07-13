# Turn Training Investigation

## Bottom Line

The recent turn runs look bad primarily because the recent `S_turn` target distribution is much harder and less matched than the river setup, not because the turn equity baseline is broken and not because there is an obvious turn-only architecture bug.

The strongest local explanation is:

1. Recent `S_turn` targets are CFR root backups for a harder turn distribution than the river validation distribution.
2. Direct CUDA stratification shows that validation loss concentrates in large-pot and high-local-exploitability roots.
3. Recent live training uses lower-quality 300-iteration CFR targets, while validation uses 5000-iteration fixed targets.
4. The current 100k/turneq `E_turn` closing model is in the target-generation path and has nontrivial residual error, so bootstrapping can still add noise. But direct stratification did not show "more closing leaves = higher loss" inside the current validation sets.
5. The recent `S_turn` configuration also changed initialization/source, replay size, CFR settings, and model/data representation versus the older low-loss turn run, so the recent results are not evidence that the turn value function is intrinsically unlearnable.

I would treat this as a hard-target/distribution and solve-quality problem first, with closing-net bootstrapping as a plausible contributor rather than the sole explanation.

## Evidence

### River Comparison

Recent river models train and validate much lower:

- `river_sapcfr_80_40_300it_3000...`: final training value loss around `0.00087`; validation value loss around `0.00117`.
- Stronger July river run: validation value loss around `0.00079`.
- A river pos/neg run reached training value loss around `0.00049`.

These runs use similar value-model machinery, which weakens the case for a generic MLP/value-head architecture failure.

### Recent Turn Runs

Recent turn validation-enabled runs plateau much higher:

- No turn equity baseline, `sturn-5k-teboff-initfix-val4096-eturn100k`: validation loss started near `0.00323` and was still around `0.00241` near step 1050.
- Turn equity baseline, `sturn-5k-turnbase-oldposneg-expandable-val4096-eturn100k`: validation loss started near `0.00260` and was around `0.00237` near step 1600.
- Another turneq run plateaued around `0.00244-0.00270`.
- The short fp32-pair/new-posneg run reached about `0.00225` early, but did not continue to river-like values.

Because both turneq-on and turneq-off runs show the same qualitative failure mode, the turn equity baseline is not the first-order cause.

Direct CUDA validation reproduced the gap:

- Turneq `S_turn`, step 1599, on `turn_val_4096_5kit_eturn100k_allincutoff_20260706`: value loss `0.002366`, pot-relative RMSE `0.493`.
- No-turneq `S_turn`, step 999, on `turn_val_4096_5kit_eturn100k_allincutoff_fp32pair_v2_20260707`: value loss `0.002669`, pot-relative RMSE `0.608`.
- Promoted upstream `S_river`, step 8000, on `river_val_8192_10k_sapdcfr_nowarm_ctx41_20260630`: value loss `0.000682`, pot-relative RMSE `0.268`.

### Closing Net In The Target Path

The recent turn validation manifests show:

- `target_model.role=closing_leaf`.
- `target_model` and `closing_leaf_checkpoint` point to the promoted `E_turn` checkpoint distilled from `S_river`.
- All root value targets are CFR backups.
- Leaf source counts are roughly `275k` exact terminal leaves and `101k` closing-net leaves, so about 27% of leaves are directly bootstrapped from `E_turn`.

In `src/p2/search/cfr_evaluator.py`, root value targets are produced from CFR average values. The root labels are therefore not direct `E_turn` predictions, but `E_turn` values enter the search at leaves and can also alter the strategy found by CFR. That makes closing-net error a plausible source of nonlocal label noise.

However, direct stratification of recent turn validation errors does not support a simple "closing leaves cause the loss" explanation. In both recent turn variants, per-root loss was negatively correlated with closing-leaf count/fraction. This is probably because the validation set's closing fraction is narrow, roughly 21-29%, and covaries with other tree-shape variables.

The measured hard axes were pot size and local exploitability:

- Turneq run: pot quartile value losses rose from about `0.00123` to `0.00399`; local-exploitability quartiles rose from about `0.00116` to `0.00414`.
- No-turneq run: pot quartile value losses rose from about `0.00150` to `0.00459`; the highest local-exploitability quartile was `0.00370`.

Belief entropy also separates the examples, but in the opposite direction from "diffuse beliefs are harder." High-loss states have lower belief entropy:

- Turneq run: per-example loss correlation with normalized belief entropy was about `-0.358`. Mean normalized entropy by loss quartile fell from `0.764` in the lowest-loss quartile to `0.564` in the highest-loss quartile.
- No-turneq run: correlation was about `-0.422`. Mean normalized entropy by loss quartile fell from `0.809` to `0.518`.
- Binning by entropy gives the same story. In the no-turneq run, the lowest-entropy quartile had value loss about `0.00552`; the highest-entropy quartile had value loss about `0.000668`.

The old turn run's embedded config predates explicit `belief_mode`/`belief_profile`; the closest current-code equivalent is the legacy `random` board-legal postflop belief sampler. Sampling current `random_turn` roots with `belief_mode=random` gives normalized belief entropy mean `0.890`, q10 `0.794`, q25 `0.830`, and median `0.936`. Limiting the current validation sets to those old-random-like high-entropy slices greatly lowers loss:

- Turneq full val: `0.002366`. Entropy >= old-random q10: `0.001297`; entropy >= old-random q25: `0.001263`; entropy >= old-random median: `0.001145`.
- No-turneq full val: `0.002669`. Entropy >= old-random q10: `0.000775`; entropy >= old-random q25: `0.000718`; entropy >= old-random median: `0.000598`.
- The complementary low-entropy slices are much worse: turneq entropy < q10 is `0.002954`; no-turneq entropy < q10 is `0.003710`.

The older June `S_turn` checkpoint does not transfer to these current validation targets, even on the high-entropy old-random-like slices. Using the first 15 scalar context fields, which matches the old model's context encoder, its current-target losses are:

- On `turn_val_4096_5kit_eturn100k_allincutoff_20260706`: full val `0.07849`; entropy >= old-random q10 `0.07551`; entropy >= old-random q25 `0.07581`.
- On `turn_val_4096_5kit_eturn100k_allincutoff_fp32pair_v2_20260707`: full val `0.08643`; entropy >= old-random q10 `0.08495`; entropy >= old-random q25 `0.08525`.

That is a cross-target-distribution result, not an in-distribution score for the old run. The old checkpoint was trained with the older 25k actual `E_turn` closer and old target setup; no old turn validation set is present on disk to measure its in-distribution validation loss.

So the strongest direct diagnostic says the recent turn model is weak on big-pot, low-belief-entropy, high-CFR-instability roots. Closing-net noise may help create those targets, but it is not isolated by leaf count alone.

### E_turn Quality

The current promoted `E_turn` distilled from `S_river` over turn chance reached value loss around `0.000121` and pot-relative RMSE around `0.117`.

Older `E_turn` distillation runs used by the low-loss June turn setup reached about `1.8e-5` to `1.9e-5` value loss. This is a large quality difference in the model being used to close turn searches.

### Older Turn Control

The older `turn-2000-fused-grouped-allin-turn` run reached training/fresh losses around `7e-5` to `1e-4`. That argues against "turn is impossible."

But it is not directly comparable to the recent runs. It used:

- `from_net=S_river` for policy/model source.
- An older lower-loss `E_turn` closing model.
- `replay_buffer_batches=128` instead of 32.
- 400 CFR iterations instead of 300.
- Different predictive/warm-start settings.
- No recent turneq/belief-low-rank setup.

The older run shows that the current problem is likely setup drift rather than an inherent inability to fit turn values.

I attempted to directly score that older checkpoint on the current turn validation set. It is not a valid control without a feature conversion layer: the old checkpoint uses a 15-wide context encoder, while the current validation features use context width 41. That difference should not be overinterpreted as a root cause, but it does mean the old checkpoint cannot be directly evaluated on the new validation shards by the current value evaluator.

## Architecture Assessment

I did not find a static code-level architecture bug that explains the turn-only loss gap.

- `src/p2/stages/curriculum.py` correctly maps `E_*` to pre-chance heads and `S_*` to post-chance heads.
- Recent `S_turn` value initialization from `E_turn` is intentional fallback behavior, not an accidental missing checkpoint.
- `src/p2/models/mlp/better_ffn.py` shares most value computation between streets, with street-specific heads and optional turn/river range-equity baselines.
- The river models using the same family can reach much lower losses.

Capacity could still be a secondary issue because turn values require learning a continuation through many possible rivers. The evidence does not make it the primary cause.

## Confidence

High confidence:

- Turneq is not the root cause.
- Closing-net bootstrapping is in the recent turn target path.
- Recent turn runs have a large validation gap versus river.
- The directly measured recent-turn validation error concentrates on large-pot, low-belief-entropy, and high-local-exploitability roots.
- The recent setup differs materially from the old low-loss turn setup.
- No obvious architecture bug surfaced in the traced code paths.

Medium/high confidence:

- The main root cause is difficult/mismatched turn targets, especially big-pot/high-instability roots, with 300-iteration training targets being compared against 5000-iteration validation targets.

Not proven from existing artifacts:

- The exact irreducible label-noise floor on identical turn roots.
- How much of the validation loss is due to closing-net error versus CFR-iteration mismatch versus distribution/generalization.
- Whether closing-net error is the cause of the high-local-exploitability targets; leaf-count stratification alone does not prove that.

## Recommended Next Checks

1. Generate a fixed-root turn validation set at 300 CFR iterations, matching train settings, and evaluate the recent checkpoints. If loss falls near training loss, the current 5000-iteration validation target mismatch is a major factor.
2. Re-solve the same turn roots with old `E_turn`, new `E_turn`, and exact/stronger closure where available. Compare root target deltas and validation loss by closing-leaf fraction.
3. Re-run a recent `S_turn` control with the older proven knobs: `from_net=S_river`, the lower-loss `E_turn`, larger replay buffer, and 400 CFR iterations.
4. Stratify validation examples further by SPR, all-in status, bet history/action depth, and board texture. Pot, low belief entropy, and local exploitability already predict high error; closing-leaf fraction alone does not.
5. Evaluate the promoted `E_turn` on actual closing leaves sampled from `S_turn` searches, not only on its own distillation distribution.
