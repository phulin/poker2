# Task Plan: Streaming Epoch Value Buffer

## Goal
Add a bounded double-buffered value replay mode that fills the first block before training, samples sealed blocks for exact shuffled epochs, fills the next block concurrently, and leaves policy replay unchanged.

## Phases
- [x] Phase 1: Establish requirements and inspect current replay/trainer paths
- [x] Phase 2: Design configuration and buffer state machine
- [x] Phase 3: Implement buffer and live-data integration
- [x] Phase 4: Add focused tests and configuration defaults
- [x] Phase 5: Verify and document behavior

## Key Questions
1. Which data-source methods own insertion and minimum-fill behavior?
2. How should generation shortfall or overfill be represented without losing examples?
3. How should exact epoch state survive checkpoints?

## Decisions Made
- Value-only feature; policy replay remains random.
- Targets are stationary and need no target-version tracking.
- First read block must be completely generated before the first optimizer update.
- Use fixed preallocated tensor blocks and shuffled index schedules; no disk storage.

## Errors Encountered
- Live-source state shape broke legacy random-replay tests: preserve the original generator-only state when rational generation is disabled.
- Existing bootstrap-pregenerated staging test leaves the value buffer empty; this also failed before the streaming feature and is outside the supported live/hybrid epoch mode.
- Initial live smoke test used a depth-zero root that was already terminal: changed the focused test to depth one, after which the complete fill/train/swap path passed.

## Status
**Complete** - implementation, defaults, tests, metrics, and documentation are in place.
