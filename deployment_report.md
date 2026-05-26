# WebGPU CFR Deployment Report

## Completed
- Added configurable app model URL via `VITE_MODEL_MANIFEST_URL`.
- Added production `.env.production` pointing at `https://assets.holdem.computer/models/rebel_296_4000/model.json`.
- Added IndexedDB all-in table cache:
  - preflop table cached persistently
  - last 5 flop tables retained
- Changed flop generation behavior:
  - online: fetch/prefetch flop table assets
  - offline: generate flop table with WebGPU fallback
- Added flop prefetch when the UI has a complete flop.
- Added TypeScript deployment scripts:
  - `scripts/prune-pages-assets.ts`
  - `scripts/upload_assets.ts`
- Production build removes `dist/app/models` so Pages only ships app/static files.
- Created R2 bucket `p2-webgpu-cfr-assets`.
- Applied R2 CORS policy from `webgpu_cfr/cloudflare/r2-cors.json`.
- Uploaded remote R2 assets:
  - `models/rebel_296_4000/model.json`
  - `models/rebel_296_4000/weights.bin.gz`
  - `allin/holdem_v1/allin_manifest.json`
  - `allin/holdem_v1/preflop.i16`
- Verified remote R2 downloads of `model.json` and `preflop.i16`.
- Created Pages project `holdem-computer`.
- Deployed Pages preview: `https://c0afc2f2.holdem-computer.pages.dev`.

## Verification
- `npm run build` passed.
- `npm run upload:assets -- --dry-run` passed.
- `WEBGPU_BACKEND=metal node --test --import tsx tests/all_in_tables.test.ts` passed.

## Blocked
- Full flop shard generation is blocked locally because CUDA is unavailable.
- `assets.holdem.computer` R2 custom-domain attach requires the Cloudflare zone ID for `holdem.computer`.
- Wrangler confirms no custom domains are currently connected to `p2-webgpu-cfr-assets`.
- `holdem.computer` Pages custom-domain attach is not exposed by Wrangler Pages CLI; use the Cloudflare dashboard/API.

## Production URL To Use
`https://assets.holdem.computer/models/rebel_296_4000/model.json`
