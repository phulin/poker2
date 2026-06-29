# Task Plan: Belief Sampler Wiring

## Goal
Wire the observed-shape belief sampler into new training and distillation runs for compact 169 preflop beliefs and native 1326 postflop beliefs, then commit the changes.

## Phases
- [x] Phase 1: Create plan and identify call sites
- [x] Phase 2: Add config fields and execution plumbing
- [x] Phase 3: Wire sampler into preflop bucket and distillation paths
- [x] Phase 4: Wire sampler into 1326 postflop spot/distillation paths
- [x] Phase 5: Add/update tests and AGENTS summaries
- [x] Phase 6: Run validation and commit

## Key Questions
1. Which new-run paths still call ad hoc uniform/random beliefs?
2. Which paths need compact 169 output versus native 1326 output?
3. How should 1326 postflop beliefs handle board-blocked private combos?

## Decisions Made
- Keep the sampler API as the source of truth and dispatch from existing training helpers rather than duplicating belief-generation logic.
- Preserve old `uniform`/`random` modes while making `mixed` the default for new Hydra preflop bucket and postflop random-root runs.
- Use `preflop_buckets.belief_profile=auto` to map bucket labels to observed profiles, including `actions_12_15 -> actions_12_end`.
- For native postflop 1326 beliefs, mask sampled combo beliefs by board legality and fall back to allowed-uniform rows if a near-delta sample is fully board-blocked.

## Errors Encountered
- Initial targeted ruff run found unused imports in two touched scripts; removed them and reran successfully.

## Status
**Complete** - sampler wiring is validated and ready to commit.
