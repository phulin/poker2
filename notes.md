# Notes: WebGPU CFR Asset Deployment

## Local Findings
- The app hardcodes `MODEL_MANIFEST_URL = "/models/rebel_latest/model.json"` in `website/src/App.tsx`.
- The current model cache stores decoded weights in IndexedDB under `p2-webgpu-cfr-model-cache`.
- There is no all-in table IndexedDB cache yet.
- Current local production assets are:
  - `website/public/models/rebel_latest/model.json`
  - `website/public/models/rebel_latest/weights.bin.gz`
  - `website/public/models/rebel_latest/allin/preflop.i16`
  - `website/public/models/rebel_latest/allin/allin_manifest.json`
- No flop shard files are present locally under `website/public/models/rebel_latest/allin`.
- `website/python/precompute_allin_assets.py` can generate canonical flop assets, but full flop generation requires CUDA unless `--limit` is used.
- Wrangler is not currently installed in this workspace (`wrangler` not found; no local `node_modules/.bin/wrangler`).
- Wrangler 4.94.0 was later installed as a dev dependency.
- R2 bucket `p2-webgpu-cfr-assets` was created.
- R2 CORS was applied from `website/cloudflare/r2-cors.json`.
- Remote R2 uploads completed for:
  - `models/rebel_296_4000/model.json`
  - `models/rebel_296_4000/weights.bin.gz`
  - `allin/holdem_v1/allin_manifest.json`
  - `allin/holdem_v1/preflop.i16`
- Remote R2 verification downloaded `model.json` and `preflop.i16` back successfully.
- Pages project `holdem-computer` was created and deployed to `https://c0afc2f2.holdem-computer.pages.dev`.
- Full flop asset generation is blocked on this machine because CUDA is unavailable.

## Layout Decision
- Model files should use `models/<model_version>/model.json` and `models/<model_version>/weights.bin.gz`.
- All-in assets should use a model-independent prefix such as `allin/<allin_version>/preflop.i16` and `allin/<allin_version>/flop/...`.
- Embedded `manifest.allIn` paths must therefore be absolute asset URLs or root-relative to the asset origin, not relative `allin/...` paths under the model directory.

## Remaining External Inputs
- R2 custom domain attachment for `assets.holdem.computer` needs the Cloudflare zone ID for `holdem.computer`.
- Pages custom domain attachment for `holdem.computer` is not exposed by Wrangler Pages CLI; use Cloudflare dashboard/API.
