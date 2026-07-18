# Notes: Two-Player Context Cleanup

## Initial requirements

- Add `last_aggressive_amount` to `HUNLTensorEnv`.
- Remove folded, can-call, acted-this-round, and redundant two-player context features for two-player models.

## Findings

## Chosen heads-up schema

- Scalar policy fields: actions-round, pot, min-raise, log stack depth, log pot,
  max committed, last aggressive amount, number of legal actions, can-fold,
  can-raise, can-all-in.
- Scalar value fields: the same layout, with chance-phase replacing
  actions-round.
- Per-player fields: stack, committed, SPR, log committed, all-in, is-actor,
  position relative to button, to-call/scale, to-call/pot, and stack after call.
- Removed only for heads-up: actor, actor position, unopened/check-to-actor,
  can-call, folded, acted-this-round, and position relative to actor.
- Multi-player schemas retain the existing 15 scalar and 13 per-player fields.
- Heads-up context length changes from 41 to 31. Existing BetterFFN checkpoints
  therefore have incompatible first-layer shapes and must not be resumed.

## State propagation sites

`last_aggressive_amount` must be allocated/initialized/reset/copied by
`HUNLTensorEnv`, included in CPU and Triton gathers, carried through both
same-street child writers, reset at street closure, and exposed through the fused
evaluator's active environment view.
