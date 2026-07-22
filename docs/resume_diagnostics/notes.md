# Notes: ReBeL Resume Equivalence

## CORRECTION (2026-07-21)

The `has_folded` findings below are stale for 2-player runs: commit `dab97193`
removed `FOLDED` from the HU `PlayerContext` schema, so the encoder never reads
`has_folded` in HU (see `resume_diagnostic_report.md` correction). The current
resume gap in the feature path is `last_aggressive_amount` — added by the same
commit, consumed by the HU scalar context, but missing from `_ENV_STATE_FIELDS`,
so it is not restored into `current_pbs`. A separate persistent exploitability/
value-magnitude regime shift after resume remains unexplained (env RNG is still
un-checkpointed but that alone should average out).

## Preserved observations

- No training process was alive when diagnostics began.
- The retained full checkpoint and replay sidecar are both at internal step 3449.
- The prior resume loaded step 2849 and reported replay buffers restored before starting step 2850.
- Original internal step 2850 value loss was `0.00031`; replayed internal step 2850 was `0.00041`, so divergence began immediately rather than at the first W&B-visible step.
- Raw fresh loss increased while pot-relative MSE remained near its prior range.
- The resumed fresh stream showed more low-SPR/all-in states, but causality is not yet established.

## Known resume gaps

- The preflop analyzer runs after checkpointing and consumes `trainer.rng`; resume restores the pre-analyzer state without replaying the analyzer.
- Global PyTorch CPU/CUDA RNG states are not checkpointed.
- The environment owns an RNG distinct from `trainer.rng`; its state is not checkpointed.
- Replay sidecar payload contains a step but load does not validate it against the main checkpoint.

## Controlled experiments

- Two independent checkpoint loads produced identical hashes for model, trainer RNG, buffer RNG, environment RNG, global RNGs, current PBS, and representative rows from both replay buffers.
- Main checkpoint step and replay sidecar step both equal 3449 in the retained artifact.
- Two first-batch generation runs began from identical hashes but diverged during live CFR generation. The first solve returned the same row count; later solves returned different counts and produced different targets, next PBS state, replay contents, and buffer RNG state.
- Two full training steps from identical restored state produced fresh losses `0.00173765` and `0.00152202`; fresh pot-relative MSE was `0.03757` and `0.02417`. Model hashes diverged after one update step.
- Trainer/global/environment RNG hashes remained equal after the duplicate full steps. Buffer RNG diverged because differing generated row counts changed replay insertion/decimation behavior. This points to nondeterministic GPU CFR arithmetic/branching rather than missing checkpoint RNG as the immediate cause.
- Caveat: the first experiment matrix used the correct nested model/search/train/data settings but defaulted flat `num_steps` to 2000 when reading the grouped resolved config. This changed TrueSkill allocation and LR scheduling. Production-exact repeats use the checkpoint-embedded config.
- Production-exact duplicate generation started from identical state but produced
  2220 versus 2236 fresh value rows and different next PBS/replay states.
- Deterministic algorithms plus PyTorch's default uninitialized-memory fill made
  duplicate generation bit-identical, but changed target absolute mean from about
  `0.157` to `0.104`. Deterministic mode with the fill disabled returned to about
  `0.160`. The fill toggle only changes the contents returned by empty allocations.
- Stage probes localized the first meaningful divergence to
  `initialize_policy_and_beliefs`, before warm start or CFR iterations.
- The optimized same-street child writer deliberately omits `has_folded`, but
  `BetterPolicyFeatureEncoder` consumes `has_folded` as a policy feature. Root
  rows are gathered from the PBS; child rows therefore read uninitialized GPU
  memory.
- `board_onehot` and `hole_onehot` also remain unwritten, but are unused by the
  current BetterFFN feature path and are not causal.
- The existing legacy writer initializes `has_folded`. With that writer, the two
  production generations had close target statistics (`0.15696` versus
  `0.15426` target absolute mean), while later solves still diverged slightly due
  normal nondeterministic GPU arithmetic.
