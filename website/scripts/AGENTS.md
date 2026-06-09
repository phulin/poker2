## Directory Summary
Deployment and asset-management utilities for the WebGPU CFR app.

### Source Files
- `prune-pages-assets.ts`: Removes generated model/table assets from the Vite Pages output so Cloudflare Pages only ships app code and small static files.
- `upload_assets.ts`: Stages single-model or curriculum model-set weights/manifests and model-independent all-in table assets into R2 object prefixes, then uploads them with Wrangler.
