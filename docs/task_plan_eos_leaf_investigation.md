# Task Plan: EOS Leaf Value Investigation

## Goal
Determine whether the BTN limp/check versus raise/call EOS values are caused by inconsistent model inputs, out-of-distribution leaves, or biased/distorted EOS distillation targets.

## Phases
- [x] Phase 1: Plan and setup
- [x] Phase 2: Trace model/input wiring for EOS and all-in/value sources
- [x] Phase 3: Extract exact leaf env, beliefs, features, and model values for target BTN leaves
- [x] Phase 4: Recompute or approximate the underlying heads-up distillation targets for those leaves
- [x] Phase 5: Compare against dataset distribution and summarize likely cause
- [x] Phase 6: Re-test the same leaf states under in-distribution uniform/random beliefs
- [x] Phase 7: Check ReBeL paper against our policy/value target extraction

## Key Questions
1. Are the EOS leaf inputs consistent with the model's training inputs: street, pot/scale, stack/scale, committed/scale, to_act/button, live-player/fold state, and beliefs?
2. Do the underlying heads-up projected values for the exact leaves support EOS preferring limp/check over raise/call?
3. Is the EOS distillation target construction biased for 3-live-player preflop-closed states, especially by uniformly averaging live HU pairs?
4. Are these leaf states in distribution relative to the extracted street-closed training data?
5. Does the 6p EOS student match the frozen 2p teacher more closely when the public state is fixed but beliefs are sampled from the training belief distribution?
6. Are policy targets extracted with PBS beliefs that are consistent with the target policy, per ReBeL?

## Decisions Made
- Use the saved even-20k BTN tree as the primary reproducer because it has the requested stack scale and corrected EOS checkpoint.

## Errors Encountered
- None yet.

## Status
**Complete** - exact leaf probes, dataset distribution scan, in-domain belief probe, and ReBeL target consistency check are recorded in `notes_eos_leaf_investigation.md`.
