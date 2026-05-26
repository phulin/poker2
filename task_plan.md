# Task Plan: WebGPU CFR Asset Deployment

## Goal
Make the app fetch model/all-in assets from R2, cache the right all-in tables, generate flop tables only offline, and prepare/deploy the versioned assets for holdem.computer.

## Phases
- [x] Phase 1: Inspect current app loading, caching, build, and asset state
- [x] Phase 2: Implement runtime URL/config/cache behavior
- [x] Phase 3: Add production build/upload scripts and metadata handling
- [x] Phase 4: Verify locally
- [x] Phase 5: Attempt R2/Pages deployment and report any external blockers

## Key Questions
1. What current IndexedDB cache can be extended for preflop/flop tables?
2. How does the current manifest embed all-in table paths?
3. Which Wrangler project/bucket resources already exist?

## Decisions Made
- Use R2-hosted `model.json` as the base URL for model weights.
- Store model weights under `models/<model_version>/...`.
- Store all-in payoff assets separately under `allin/<allin_version>/...` because they are model-independent win-rate tables.
- Let model manifests reference all-in assets with explicit R2 URLs or root-relative R2 paths instead of nesting all-in assets under model directories.

## Errors Encountered
- Wrangler CLI is not installed locally, so R2/Pages deployment will require installing or otherwise providing Wrangler credentials/tooling.
- Initial R2 upload script omitted `--remote`; the first object puts went to local R2 emulation. Fixed script and reran remote uploads successfully.
- Initial CORS file used S3-style keys; Wrangler requires Cloudflare lowercase keys under `allowed`. Fixed and applied.
- Full flop table generation failed locally because CUDA is unavailable.
- R2 custom domain attach requires the `holdem.computer` zone ID, which is not available from the repo or Wrangler Pages listing.
- Wrangler Pages CLI created/deployed the project but does not expose custom-domain attach.

## Status
**Completed with external blockers** - Code, bucket, CORS, asset upload, and Pages preview deploy are done. Remaining custom-domain attachment and full flop asset generation need Cloudflare zone ID/dashboard/API access and a CUDA machine.
