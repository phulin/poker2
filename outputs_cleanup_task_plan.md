# Task Plan: Outputs Retention Audit

## Goal
Classify every top-level family under `outputs/` as active/valuable, superseded, reproducible, or disposable, with exact reclaim estimates and dependency safeguards.

## Phases
- [x] Phase 1: Define audit criteria and protect active/dependent datasets
- [x] Phase 2: Inventory every output family by size, age, contents, and references
- [x] Phase 3: Identify supersession chains, failed/smoke runs, duplicates, and caches
- [x] Phase 4: Review classifications and deliver deletion tiers

## Key Questions
1. Which outputs are inputs to current configs, scripts, manifests, or checkpoints?
2. Which outputs are superseded by newer variants or are failed/smoke/profiling artifacts?
3. How much space can be reclaimed without losing restart, initialization, or meaningful comparison assets?

## Decisions Made
- No deletion is authorized by this audit request.
- Treat recent July 11-13 artifacts and manifest dependencies conservatively until proven superseded.
- Preserve compact result/config metadata even when bulk generated tensors are candidates.

## Errors Encountered
- The default planning files belong to an earlier task; use uniquely named audit files instead.
- One broad reference-extraction command had mismatched shell quoting; rerun with a simpler pattern and no postprocessing.

## Status
**Complete** - Findings and deletion tiers are in `outputs_cleanup_audit.md`.
