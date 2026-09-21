# Studio Playground

A standalone, client-side-only interactive demo published at `/studio/` on the docs
site (`https://foreblocks.laioseman.com/studio/`). It reimplements VMD/EWT/EMD
decomposition and outlier-detection algorithms in pure JavaScript so visitors can try
them in the browser without installing anything, and shows the equivalent
`foreblocks`/`foretools` Python code for what they configured (see `src/lib/pipeline.js`).

This is **not** the same thing as [`apps/webui/`](../webui) — that is a
separate, local, Python-backed node editor (FastAPI execution backend) meant to be
self-hosted against a real `foreblocks` install via the `foreblocks-studio` command. See
[docs/webui.md](../../docs/webui.md) for that tool.

## Structure

- `src/` — Vite + React source (`foreblocks-studio.jsx`, `components/`, `lib/`)
- `dist/` — build output, gitignored; rebuilt on every docs-site deploy, not committed

## Building

```bash
scripts/build_studio_web.sh
```

Builds into `dist/` (sibling of `src/`) and publishes to `site/studio/` at the repo
root. `.github/workflows/docs.yml` runs this same script during CI as part of the
published docs site.

## Local development

```bash
cd src
npm install
npm run dev   # http://localhost:5176
```
