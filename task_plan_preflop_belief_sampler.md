# Task Plan: Preflop Belief Sampler

## Goal
Implement a reusable sampler that approximates the observed preflop continuation belief shape while deliberately covering hard belief tails, with tests against the observed cascade distribution.

## Phases
- [x] Phase 1: Inspect existing belief generation and test layout
- [x] Phase 2: Implement sampler module
- [x] Phase 3: Add parity and coverage tests
- [x] Phase 4: Run focused validation
- [x] Phase 5: Add entropy histogram/CDF parity tests
- [x] Phase 6: Add native 1326-combo GPU-vectorized sampling support

## Key Questions
1. Can histogram-matched synthetic beliefs reproduce the cascade quantiles without needing to run CFR?
2. Does the full mixture cover broad, medium, sharp, and near-delta beliefs?
3. Are outputs normalized and shaped for compact 169-hand preflop models?

## Decisions Made
- Put reusable code under `src/p2/search/preflop_belief_sampler.py`.
- Keep CFR-free parity tests by hardcoding the observed cascade profile from the saved 1024-root run.
- Treat entropy as a first-class histogram target. Summary max/AA parity alone is not enough to catch the spiky-belief failure mode.
- Use a capped low-support endpoint for histogram sampling so entropy can be tuned without creating non-target classes larger than the sampled max class.
- Make `hand_dim` explicit on the sampler API. `169` uses compact class priors; `1326` samples combo beliefs natively with uniform combo prior and AA as an aggregate six-combo mass.
- For `1326`, map observed compact entropy profiles by normalized entropy fraction so compact uniform tails become combo-uniform tails at `log(1326)`.
- Replace variable-support and mixed-component CPU syncs with tensor masks/scatters so the sampling path can run on CUDA without row-count or support-size round trips.

## Errors Encountered
- A delta-to-uniform leftover-mass interpolation matched entropy but corrupted the max-class histogram by making the leftover delta the true largest class.
- Using compact-uniform as the global minimum max mass made the early `actions_0_3` entropy tail too uniform. Keep the prior-based minimum for ordinary rows and only force exact uniform for observed uniform entropy tails.
- One compact-only reshape remained in random mode after generalizing the API; fixed it to use `hand_dim`.

## Status
**Complete** - generic 2p/6p preflop belief sampler supports native 169-class and 1326-combo rows with focused max/AA/entropy histogram tests.
