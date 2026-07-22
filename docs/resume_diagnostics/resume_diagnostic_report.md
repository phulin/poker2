# ReBeL Resume Diagnostic Report

## CORRECTION (2026-07-21) — supersedes the has_folded conclusion below

The `has_folded` mechanism blamed below is **no longer reachable in 2-player (HU)
runs.** Commit `dab97193` ("Compact heads-up context and track aggression",
2026-07-18 20:20 — a few minutes *after* this report was written) removed
`FOLDED` (and `ACTED_THIS_ROUND`, `REL_POS_TO_ACTOR`) from the HU `PlayerContext`
schema. `context_schemas(2)` returns `PlayerContext`, which has no `FOLDED`
field, and `BetterFeatureEncoder` reads `has_folded` only under
`if hasattr(player_schema, "FOLDED")` — False for HU. So the uninitialized-
`has_folded` read does not occur for the 2-player model. (It is still live for
`MultiwayPlayerContext`, 3+ players.)

The current-code analogue of that bug is **`last_aggressive_amount`**: the same
`dab97193` commit added this env-state tensor, the HU scalar context consumes it
in both the policy and value feature paths (`ScalarContext.LAST_AGGRESSIVE_AMOUNT`,
`better_feature_encoder.py:166,205`), but it is **missing from
`_ENV_STATE_FIELDS`** in `rebel_data_generator.py`. Therefore it is not saved into
`current_pbs` and not restored on resume — the reconstructed root subgames get a
reset/garbage aggression feature. This explains the transient post-resume value-
loss spike (it decays over ~10 steps as the corrupted `current_pbs` subgames
terminate and refill via `_new_pbs` → `env.reset()`, which sets the field
correctly). Fix: add `"last_aggressive_amount"` to `_ENV_STATE_FIELDS`.

Note: a separate, larger *persistent* shift (raw exploitability and value-target
magnitude dropping ~3x and holding for 150+ steps, against a rock-stable
3000-step pre-crash baseline) is NOT explained by this field bug and is not yet
localized. The env's own RNG (stacks/cards/button) is still un-checkpointed and
un-seeded, but a fresh draw from the same stack distribution should average out,
so it cannot by itself explain a persistent mean shift. Re-run
`scripts/probe_rebel_resume_equivalence.py` after fixing the field to localize.

## Original Conclusion (Superseded for HU)

The replay buffers were restored successfully. The persistent fresh-value-loss
shift after resume is caused by allocator-sensitive live CFR inputs, not by an
overwritten policy replay buffer or a changed policy checkpoint.

The optimized same-street child writer omits `env.has_folded`, claiming fused CFR
does not read it. The current BetterFFN policy feature encoder does read it for
every node. Consequently, every non-root CFR child gets an uninitialized folded
feature. Restarting with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
changes allocator layout and reused memory contents, so the same checkpoint can
enter a different policy/CFR/continuation-state regime immediately.

## Evidence

1. Independent loads matched exactly for model weights, current PBS environment
   and beliefs, trainer/buffer/environment/global RNG hashes, replay positions,
   and representative value and policy replay rows.
2. Duplicate production generation diverged from that identical state: 2220
   versus 2236 fresh value rows, then different PBS and replay states.
3. The first meaningful staged divergence occurs in policy/belief initialization,
   before warm start, CFR iterations, replay sampling, or a model update.
4. With deterministic algorithms, toggling only PyTorch's uninitialized-memory
   fill changed fresh target absolute mean from about `0.160` to `0.104`.
5. At subgame construction, the only fill-sensitive environment field consumed
   by the active BetterFFN feature path was `has_folded`. The optimized writer
   does not write it; the legacy writer does.
6. The legacy-writer control reduced duplicate target-distribution variation to
   `0.15696` versus `0.15426` absolute mean. Residual later-solve divergence is
   ordinary nondeterministic GPU arithmetic amplified by CFR branching.

`board_onehot` and `hole_onehot` were also allocator-sensitive, but current
BetterFFN features reconstruct cards from compact indices and do not consume
those tensors.

## Other resume gaps

The checkpoint still omits global PyTorch and environment-owned RNG state, and a
post-checkpoint preflop analyzer consumes `trainer.rng`. The replay sidecar step
is also not checked against the main checkpoint. These should be repaired for
strict resume equivalence, but they do not explain this immediate experiment:
the probes began with matching hashes and diverged inside live CFR generation.

The W&B step 2864 was not proof that training resumed an old trajectory. W&B
already contained step 2864, rejected replayed logs for 2850-2863 as
non-monotonic, and then accepted the new 2864 point.

## Original Recommended Repair (Superseded for HU)

Make the optimized child writer copy `has_folded` for both players, add a test
that poisons destination storage before construction and compares all consumed
child fields with the legacy writer, then repeat the deterministic fill/no-fill
and duplicate-generation probes. Separately checkpoint every RNG stream, isolate
the analyzer RNG, and validate replay-sidecar step equality.
