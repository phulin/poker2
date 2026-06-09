# Task Plan: Prevent Multiway Flop Handoffs

## Goal
Modify preflop search/state handling so every non-terminal preflop street transition entering the flop has at most two live, non-all-in players, making 2p E_preflop adapters valid for all new-street closing leaves.

## Phases
- [ ] Phase 1: Define abstraction semantics for choosing surviving players
- [ ] Phase 2: Implement preflop-only closure normalization
- [ ] Phase 3: Integrate with all-in and folded-value ownership
- [ ] Phase 4: Add assertions and logging
- [ ] Phase 5: Add tests and verification probes

## Key Questions
1. Which two players survive a multiway preflop round close?
2. Should normalization happen in environment state transition or legal action masks?
3. How are folded players' net values assigned after forced fold normalization?

## Decisions Proposed
- Prefer forced fold normalization at preflop round closure over legal-action masking.
- Do not mutate ordinary action legality to be future-aware.
- Treat the forced fold as an abstraction boundary, with explicit stats and tests.

## Status
Draft plan for discussion.
